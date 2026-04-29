"""Convert official Talkie PyTorch checkpoints into MLX-LM model directories."""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.resources import files
from pathlib import Path

import mlx.core as mx
import torch
from huggingface_hub import hf_hub_download

from talkie_mlx.chat_template import CHAT_TEMPLATE
from talkie_mlx.config import MODEL_SPECS, Style, target_vocab_size

LINEAR_SUFFIXES = (
    "attn_query.weight",
    "attn_key.weight",
    "attn_value.weight",
    "attn_resid.weight",
    "mlp_gate.weight",
    "mlp_linear.weight",
    "mlp_resid.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--output", required=True, help="Output MLX-LM model directory.")
    parser.add_argument("--checkpoint", help="Optional local .pt/.ckpt checkpoint path.")
    parser.add_argument("--vocab", help="Optional local vocab.txt path.")
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Weight quantization bits; 0 disables.",
    )
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--q-mode", default="affine")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--n-head", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--no-pad-special-vocab",
        action="store_true",
        help="Do not grow the vocab to include Talkie special tokens. Mainly useful for tests.",
    )
    return parser.parse_args()


def load_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def infer_n_layer(state_dict: dict[str, torch.Tensor]) -> int:
    indices = []
    for name in state_dict:
        parts = name.split(".")
        if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices) + 1 if indices else 40


def is_linear_weight(name: str) -> bool:
    return name == "lm_head" or any(name.endswith(suffix) for suffix in LINEAR_SUFFIXES)


def to_mx(tensor: torch.Tensor, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    arr = mx.array(tensor.detach().float().cpu().numpy())
    return arr.astype(dtype)


def quantize_mx_weight(
    weight: mx.array,
    group_size: int,
    bits: int,
    mode: str,
) -> dict[str, mx.array]:
    q_weight, scales, *biases = mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)
    mx.eval(q_weight, scales, *biases)
    out = {"weight": q_weight, "scales": scales}
    if biases:
        out["biases"] = biases[0]
    return out


def add_weight(
    weights: dict[str, mx.array],
    name: str,
    tensor: torch.Tensor,
    *,
    quantize: bool,
    group_size: int,
    bits: int,
    mode: str,
) -> None:
    if quantize:
        q = quantize_mx_weight(to_mx(tensor, mx.float32), group_size, bits, mode)
        base = name.rsplit(".weight", 1)[0]
        for suffix, value in q.items():
            weights[f"{base}.{suffix}"] = value
    else:
        weights[name] = to_mx(tensor, mx.bfloat16)
        mx.eval(weights[name])


def write_tokenizer_files(out_dir: Path, vocab_path: str | Path, style: Style) -> None:
    shutil.copyfile(vocab_path, out_dir / "vocab.txt")
    shutil.copyfile(
        files("talkie_mlx").joinpath("tokenization_talkie.py"),
        out_dir / "tokenization_talkie.py",
    )
    tokenizer_config = {
        "tokenizer_class": "TalkieTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_talkie.TalkieTokenizer", None]},
        "style": style,
        "model_max_length": 2048,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "clean_up_tokenization_spaces": False,
    }
    special_tokens_map = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    if style == "it":
        tokenizer_config["chat_template"] = CHAT_TEMPLATE
        tokenizer_config["additional_special_tokens"] = [
            "<|end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
        ]
        special_tokens_map["additional_special_tokens"] = tokenizer_config[
            "additional_special_tokens"
        ]
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2) + "\n"
    )


def write_model_card(out_dir: Path, repo_id: str, style: Style) -> None:
    mode = "instruction-tuned chat" if style == "it" else "base completion"
    (out_dir / "README.md").write_text(
        f"""---
library_name: mlx
pipeline_tag: text-generation
base_model: {repo_id}
tags:
- mlx
- talkie
---

# Talkie MLX export

This directory was converted from `{repo_id}` with `talkie-mlx`.

It contains custom MLX-LM model code and a tokenizer shim. Load with:

```python
from mlx_lm import load, generate

model, tokenizer = load("path/to/this/export", tokenizer_config={{"trust_remote_code": True}})
text = generate(model, tokenizer, prompt="The year 1960 will", max_tokens=128)
```

Variant: {mode}.
""",
        encoding="utf-8",
    )


def convert_checkpoint(
    *,
    model_name: str,
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    output: str | Path,
    q_bits: int = 4,
    q_group_size: int = 64,
    q_mode: str = "affine",
    max_seq_len: int = 2048,
    n_head: int = 40,
    head_dim: int = 128,
    pad_special_vocab: bool = True,
) -> Path:
    spec = MODEL_SPECS[model_name]
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = load_state_dict(checkpoint_path)
    checkpoint_vocab = int(state_dict["embed.weight"].shape[0])
    vocab_size = target_vocab_size(spec.style, checkpoint_vocab, pad_special_vocab)
    n_embd = int(state_dict["embed.weight"].shape[1])
    n_layer = infer_n_layer(state_dict)
    quantize = q_bits > 0

    weights: dict[str, mx.array] = {}
    for name, tensor in state_dict.items():
        if name.startswith("cos") or name.startswith("sin"):
            continue
        out_name = "lm_head.weight" if name == "lm_head" else name

        if name == "embed.weight" and tensor.shape[0] < vocab_size:
            embed = to_mx(tensor, dtype=mx.bfloat16)
            pad = mx.random.normal((vocab_size - tensor.shape[0], tensor.shape[1])).astype(
                mx.bfloat16
            ) * 0.02
            weights[out_name] = mx.concatenate([embed, pad], axis=0)
            mx.eval(weights[out_name])
        elif name == "lm_head" and tensor.shape[0] < vocab_size:
            head = to_mx(tensor, dtype=mx.float32)
            pad = mx.random.normal((vocab_size - tensor.shape[0], tensor.shape[1])).astype(
                mx.float32
            ) * 0.02
            merged = mx.concatenate([head, pad], axis=0)
            if quantize:
                q = quantize_mx_weight(merged, q_group_size, q_bits, q_mode)
                for suffix, value in q.items():
                    weights[f"lm_head.{suffix}"] = value
            else:
                weights[out_name] = merged.astype(mx.bfloat16)
                mx.eval(weights[out_name])
        elif is_linear_weight(name):
            add_weight(
                weights,
                out_name,
                tensor,
                quantize=quantize,
                group_size=q_group_size,
                bits=q_bits,
                mode=q_mode,
            )
        else:
            weights[out_name] = to_mx(tensor, dtype=mx.float32)
            mx.eval(weights[out_name])

    config: dict[str, object] = {
        "model_type": "talkie",
        "model_file": "modeling_talkie_mlx.py",
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "style": spec.style,
        "eos_token_id": 65_535,
    }
    if quantize:
        config["quantization"] = {
            "group_size": q_group_size,
            "bits": q_bits,
            "mode": q_mode,
        }
        config["quantization_config"] = config["quantization"]

    channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
    inv_freq = 1.0 / (1_000_000.0 ** (channel_range / head_dim))
    t = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = t[:, None] * inv_freq[None, :]
    weights["cos"] = mx.cos(freqs).astype(mx.bfloat16)[None, :, None, :]
    weights["sin"] = mx.sin(freqs).astype(mx.bfloat16)[None, :, None, :]
    mx.eval(weights["cos"], weights["sin"])

    shutil.copyfile(
        files("talkie_mlx").joinpath("modeling_talkie_mlx.py"),
        out_dir / "modeling_talkie_mlx.py",
    )
    write_tokenizer_files(out_dir, vocab_path, spec.style)
    write_model_card(out_dir, spec.repo_id, spec.style)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(out_dir / "model.safetensors", weights, metadata={"format": "mlx"})
    return out_dir


def main() -> int:
    args = parse_args()
    spec = MODEL_SPECS[args.model]
    checkpoint_path = args.checkpoint or hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.checkpoint_filename,
        cache_dir=args.cache_dir,
    )
    vocab_path = args.vocab or hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.vocab_filename,
        cache_dir=args.cache_dir,
    )
    out_dir = convert_checkpoint(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        output=args.output,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        q_mode=args.q_mode,
        max_seq_len=args.max_seq_len,
        n_head=args.n_head,
        head_dim=args.head_dim,
        pad_special_vocab=not args.no_pad_special_vocab,
    )
    print(f"Saved MLX-LM Talkie model to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
