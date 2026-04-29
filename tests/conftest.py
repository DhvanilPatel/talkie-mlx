import base64

import pytest
import torch

from talkie_mlx.convert import convert_checkpoint


def write_byte_vocab(path):
    with path.open("w", encoding="utf-8") as handle:
        for value in range(256):
            token = base64.b64encode(bytes([value])).decode("ascii")
            handle.write(f"{token} {value}\n")


def make_tiny_state(vocab_size=128, n_embd=128):
    n_mlp = 384
    state = {
        "embed.weight": torch.randn(vocab_size, n_embd),
        "lm_head": torch.randn(vocab_size, n_embd),
        "lm_head_gain.w_g": torch.ones(1),
        "blocks.0.attn_gain.a_g": torch.ones(1),
        "blocks.0.mlp_gain.a_g": torch.ones(1),
        "blocks.0.embed_skip.a_g": torch.zeros(1),
        "blocks.0.attn.head_gain.head_g": torch.ones(4),
    }
    for suffix in ["attn_query", "attn_key", "attn_value", "attn_resid"]:
        state[f"blocks.0.attn.{suffix}.weight"] = torch.randn(n_embd, n_embd)
    state["blocks.0.mlp.mlp_gate.weight"] = torch.randn(n_mlp, n_embd)
    state["blocks.0.mlp.mlp_linear.weight"] = torch.randn(n_mlp, n_embd)
    state["blocks.0.mlp.mlp_resid.weight"] = torch.randn(n_embd, n_mlp)
    return state


@pytest.fixture
def tiny_export(tmp_path):
    def _convert(**overrides):
        checkpoint = tmp_path / "tiny.ckpt"
        vocab = tmp_path / "vocab.txt"
        output = tmp_path / "out"
        torch.save(make_tiny_state(), checkpoint)
        write_byte_vocab(vocab)

        options = {
            "model_name": "talkie-1930-13b-base",
            "checkpoint_path": checkpoint,
            "vocab_path": vocab,
            "output": output,
            "q_bits": 0,
            "max_seq_len": 16,
            "n_head": 4,
            "head_dim": 32,
            "pad_special_vocab": False,
        }
        options.update(overrides)
        return convert_checkpoint(**options)

    return _convert
