"""Talkie instruction-template helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

STOP_STRINGS = ("<|user|>", "<|assistant|>", "<|system|>", "<|end|>", "<|endoftext|>")


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


def format_chat(messages: list[Message | dict[str, str]]) -> str:
    """Format messages for the Talkie instruction-tuned model."""
    parts: list[str] = []
    for message in messages:
        role = message["role"] if isinstance(message, dict) else message.role
        content = message["content"] if isinstance(message, dict) else message.content
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role: {role!r}")
        parts.append(f"<|{role}|>{content}<|end|>")
    parts.append("<|assistant|>")
    return "".join(parts)


def format_prompt(prompt: str) -> str:
    return format_chat([Message(role="user", content=prompt)])


def truncate_at_stop(text: str) -> tuple[str, bool]:
    positions = [text.find(stop) for stop in STOP_STRINGS if stop in text]
    if not positions:
        return text, False
    stop_at = min(positions)
    return text[:stop_at], True


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>{{ message['content'] }}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
