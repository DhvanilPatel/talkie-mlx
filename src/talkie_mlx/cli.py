"""Small chat and JSONL batch CLIs for exported Talkie MLX-LM directories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def load_talkie(path: str):
    return load(path, tokenizer_config={"trust_remote_code": True})


def maybe_chat_prompt(tokenizer, prompt: str, mode: str) -> str:
    if mode == "chat":
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("chat mode requires an instruction-tuned export with a chat template")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Talkie MLX prompts from a JSONL file.")
    parser.add_argument(
        "--model",
        required=True,
        help="Exported Talkie MLX-LM directory or HF repo.",
    )
    parser.add_argument("--input", default="prompts.jsonl")
    parser.add_argument("--output", default="outputs.jsonl")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def batch_main() -> int:
    args = batch_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model, tokenizer = load_talkie(args.model)
    sampler = make_sampler(temp=args.temp, top_k=args.top_k)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            prompt = row["prompt"]
            mode = row.get("mode", "completion")
            record = dict(row)
            try:
                start = time.perf_counter()
                text = ""
                tokens = 0
                for response in stream_generate(
                    model,
                    tokenizer,
                    maybe_chat_prompt(tokenizer, prompt, mode),
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                ):
                    text += response.text
                    tokens = response.generation_tokens
                elapsed = max(time.perf_counter() - start, 1e-9)
                record.update({"output": text, "tokens": tokens, "tok_s": tokens / elapsed})
            except Exception as exc:  # keep batch jobs moving
                record["error"] = str(exc)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)}", file=sys.stderr)
    return 0


def chat_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming Talkie MLX chat REPL.")
    parser.add_argument("--model", required=True, help="Exported Talkie instruction model.")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def chat_main() -> int:
    args = chat_args()
    model, tokenizer = load_talkie(args.model)
    if not getattr(tokenizer, "chat_template", None):
        print("chat requires an instruction-tuned export with a chat template.", file=sys.stderr)
        return 1
    sampler = make_sampler(temp=args.temp, top_k=args.top_k)
    messages: list[dict[str, str]] = []
    print("Enter /exit or Ctrl-D to quit.")
    while True:
        try:
            user_text = input("\nuser> ").strip()
        except EOFError:
            print()
            return 0
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        messages.append({"role": "user", "content": user_text})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("assistant> ", end="", flush=True)
        reply = ""
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
        ):
            print(response.text, end="", flush=True)
            reply += response.text
        print()
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(batch_main())
