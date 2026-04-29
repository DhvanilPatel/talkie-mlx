# Contributing

This is an unofficial compatibility layer for Talkie-1930 on MLX-LM.

## Development Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run the local checks before opening a pull request:

```bash
ruff check .
pytest
python -m build
```

The tests use tiny random fixtures. Do not add tests that download or commit the
13B checkpoints.

## Contributions and License

By contributing to this repository, you agree that your contribution is licensed
under the Apache License 2.0.

Do not commit converted weights, original checkpoints, generated media, or other
large binary artifacts. The `.gitignore` is intentionally strict about these
files.

## Scope

Good contributions include:

- converter fixes for official Talkie checkpoints
- tokenizer compatibility fixes
- MLX-LM loading and generation fixes
- small tests that run without network access
- documentation that makes the setup clearer

Out of scope:

- redistributing Talkie model weights in this repository
- adding custom non-Talkie architectures
- silently changing generation semantics without tests
