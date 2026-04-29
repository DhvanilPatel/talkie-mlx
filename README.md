# talkie-mlx

[![CI](https://github.com/DhvanilPatel/talkie-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/DhvanilPatel/talkie-mlx/actions/workflows/ci.yml)

Unofficial MLX support for the Talkie-1930 model family on Apple Silicon.

Talkie-1930 is a 13B language model trained on pre-1931 English text. The
official checkpoints are public, but they are PyTorch checkpoints and do not
currently load directly through `mlx-lm`. This repository provides:

- a Talkie decoder implementation for MLX-LM custom model loading
- a PyTorch checkpoint to MLX-LM converter
- a small Hugging Face tokenizer shim for Talkie's `tiktoken` vocabulary
- command line tools for chat and JSONL batch prompting
- tests that exercise the model, tokenizer, and converter on tiny fixtures

This is not an official Talkie release and not upstream `mlx-lm` support. It is
a practical compatibility layer.

## Status

Tested locally on an M3 Pro with 64GB unified memory:

- `talkie-1930-13b-base`, 4-bit affine: about 8.8GB on disk
- `talkie-1930-13b-it`, 4-bit affine: about 8.8GB on disk
- generation speed: roughly 20-22 tok/s in local batch runs

Your speed and memory use will vary. The converter downloads the original
Talkie checkpoints from Hugging Face; those source files are large.

## Install

Using `uv`:

```bash
git clone https://github.com/DhvanilPatel/talkie-mlx.git
cd talkie-mlx
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Convert a Talkie Checkpoint

Instruction-tuned chat model:

```bash
talkie-mlx-convert \
  --model talkie-1930-13b-it \
  --output ./talkie-1930-13b-it-4bit-mlx \
  --q-bits 4 \
  --q-group-size 64
```

Base completion model:

```bash
talkie-mlx-convert \
  --model talkie-1930-13b-base \
  --output ./talkie-1930-13b-base-4bit-mlx \
  --q-bits 4 \
  --q-group-size 64
```

The exported directory is an MLX-LM model directory. It includes:

- `model.safetensors`
- `config.json`
- `modeling_talkie_mlx.py`
- `vocab.txt`
- `tokenization_talkie.py`
- tokenizer metadata

Converted weights are intentionally ignored by git.

## Use With mlx-lm

The exported model uses MLX-LM's `model_file` custom architecture hook and a
custom tokenizer shim. Pass `trust_remote_code` when loading the tokenizer:

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "./talkie-1930-13b-base-4bit-mlx",
    tokenizer_config={"trust_remote_code": True},
)

text = generate(
    model,
    tokenizer,
    prompt="The Twitter Company of 1960 will",
    max_tokens=160,
)
print(text)
```

For the instruction-tuned model:

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "./talkie-1930-13b-it-4bit-mlx",
    tokenizer_config={"trust_remote_code": True},
)

messages = [{"role": "user", "content": "Write about motor cars in the year 1960."}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(generate(model, tokenizer, prompt=prompt, max_tokens=160))
```

## CLI

Interactive chat against the instruction-tuned export:

```bash
talkie-mlx-chat \
  --model ./talkie-1930-13b-it-4bit-mlx \
  --temp 0.7 \
  --max-tokens 256
```

Batch prompting from JSONL:

```bash
talkie-mlx-batch \
  --model ./talkie-1930-13b-base-4bit-mlx \
  --input examples/prompts.jsonl \
  --output outputs.jsonl \
  --temp 0.7 \
  --max-tokens 200
```

Input rows are:

```json
{"prompt": "The Apple Computer Company of 1960 will", "mode": "completion"}
{"prompt": "Write a paragraph about radio in 1960.", "mode": "chat"}
```

Use `mode: "completion"` with the base model. Use `mode: "chat"` with the
instruction-tuned model.

## Development

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Build the package:

```bash
python -m build
```

The tests use tiny randomly initialized models and tiny tokenizer fixtures, so
they do not download or load the 13B checkpoints.

## What This Is Not

- It is not an official Talkie project.
- It does not include model weights.
- It does not make the original Hugging Face PyTorch checkpoints load directly
  with `mlx_lm.load(...)`.
- It is not yet an upstream `mlx-lm` architecture PR.

The intended path is:

1. Convert the official checkpoint once.
2. Load the exported MLX-LM directory with `mlx_lm.load(...)`.
3. Use normal MLX-LM generation utilities.

## License

This repository is Apache-2.0 licensed. See `LICENSE` and `NOTICE`.

The Talkie model weights and original PyTorch implementation are distributed by
the Talkie LM authors separately. As of April 29, 2026, the upstream repository
and the two Talkie model cards also declare Apache-2.0, but check the model
cards and upstream repository for the current license and use terms:

- https://github.com/talkie-lm/talkie
- https://huggingface.co/talkie-lm/talkie-1930-13b-base
- https://huggingface.co/talkie-lm/talkie-1930-13b-it
