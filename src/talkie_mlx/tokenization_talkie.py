"""Hugging Face tokenizer shim for Talkie's tiktoken vocabulary.

The official Talkie tokenizer is a tiktoken BPE stored as ``vocab.txt``.  This
class adapts it to the minimal ``transformers`` tokenizer interface needed by
``mlx-lm``.  It intentionally keeps token strings simple: ordinary tokens are
represented internally by their integer IDs as strings, while Talkie special
tokens keep their literal names.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import PreTrainedTokenizer

BASE_VOCAB_SIZE = 65_536
IT_VOCAB_SIZE = BASE_VOCAB_SIZE + 4

PAT_STR = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

BASE_SPECIAL_TOKENS = {"<|endoftext|>": BASE_VOCAB_SIZE - 1}
IT_SPECIAL_TOKENS = {
    "<|endoftext|>": BASE_VOCAB_SIZE - 1,
    "<|end|>": BASE_VOCAB_SIZE,
    "<|user|>": BASE_VOCAB_SIZE + 1,
    "<|assistant|>": BASE_VOCAB_SIZE + 2,
    "<|system|>": BASE_VOCAB_SIZE + 3,
}


class TalkieTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.txt"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        style: str = "base",
        model_max_length: int = 2048,
        **kwargs: Any,
    ):
        self.vocab_file = str(vocab_file)
        self.style = style
        self.special_token_ids = (
            dict(IT_SPECIAL_TOKENS) if style == "it" else dict(BASE_SPECIAL_TOKENS)
        )
        self.special_id_tokens = {v: k for k, v in self.special_token_ids.items()}
        self._vocab_size = IT_VOCAB_SIZE if style == "it" else BASE_VOCAB_SIZE

        mergeable_ranks = load_tiktoken_bpe(self.vocab_file)
        mergeable_ranks = {
            key: value for key, value in mergeable_ranks.items() if value < BASE_VOCAB_SIZE - 1
        }
        self.encoding = tiktoken.Encoding(
            name=f"talkie-{style}",
            pat_str=PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_token_ids,
        )

        kwargs.setdefault("eos_token", "<|endoftext|>")
        kwargs.setdefault("pad_token", "<|endoftext|>")
        if style == "it":
            kwargs.setdefault(
                "additional_special_tokens",
                ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"],
            )
        super().__init__(model_max_length=model_max_length, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> dict[str, int]:
        vocab = {str(i): i for i in range(self._vocab_size)}
        vocab.update(self.special_token_ids)
        return vocab

    def _tokenize(self, text: str, **_: Any) -> list[str]:
        ids = self.encoding.encode(text, allowed_special="all")
        return [self._convert_id_to_token(token_id) for token_id in ids]

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.special_token_ids:
            return self.special_token_ids[token]
        try:
            return int(token)
        except ValueError:
            return self.unk_token_id if self.unk_token_id is not None else 0

    def _convert_id_to_token(self, index: int) -> str:
        if index in self.special_id_tokens:
            return self.special_id_tokens[index]
        return str(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        ids = [self._convert_token_to_id(token) for token in tokens]
        return self.encoding.decode(ids)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        name = "vocab.txt" if filename_prefix is None else f"{filename_prefix}-vocab.txt"
        out = save_path / name
        shutil.copyfile(self.vocab_file, out)
        return (str(out),)
