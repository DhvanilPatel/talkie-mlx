"""Known Talkie model variants and Hugging Face checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Style = Literal["base", "it"]

BASE_VOCAB_SIZE = 65_536
IT_VOCAB_SIZE = BASE_VOCAB_SIZE + 4


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    checkpoint_filename: str
    vocab_filename: str = "vocab.txt"
    style: Style = "base"


MODEL_SPECS: dict[str, ModelSpec] = {
    "talkie-1930-13b-base": ModelSpec(
        repo_id="talkie-lm/talkie-1930-13b-base",
        checkpoint_filename="final.ckpt",
        style="base",
    ),
    "talkie-1930-13b-it": ModelSpec(
        repo_id="talkie-lm/talkie-1930-13b-it",
        checkpoint_filename="rl-refined.pt",
        style="it",
    ),
}


def target_vocab_size(style: Style, checkpoint_vocab_size: int, pad_special_vocab: bool) -> int:
    """Return the exported vocab size for a checkpoint."""
    if not pad_special_vocab:
        return checkpoint_vocab_size
    expected = IT_VOCAB_SIZE if style == "it" else BASE_VOCAB_SIZE
    return max(checkpoint_vocab_size, expected)
