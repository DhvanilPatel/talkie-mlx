"""MLX-LM custom model implementation for Talkie decoder checkpoints."""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "talkie"
    vocab_size: int = 65_536
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128
    max_seq_len: int = 2048
    style: str = "base"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1).astype(x.dtype)


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = mx.ones((n_head,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.head_g.astype(x.dtype).reshape(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = mx.ones((1,), dtype=mx.float32)

    def __call__(self, y: mx.array) -> mx.array:
        return y * self.w_g.astype(y.dtype)


class ActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = mx.array([init_value], dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.a_g.astype(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.n_head
        self.head_dim = args.head_dim
        self.rms_norm_eps = args.rms_norm_eps
        self.attn_query = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.attn_key = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.attn_value = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.attn_resid = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.head_gain = HeadGain(args.n_head)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        q = self.attn_query(x).reshape(batch, seq_len, self.n_head, self.head_dim)
        k = self.attn_key(x).reshape(batch, seq_len, self.n_head, self.head_dim)
        v = self.attn_value(x).reshape(batch, seq_len, self.n_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = self.head_gain(rms_norm(q, self.rms_norm_eps))
        k = rms_norm(k, self.rms_norm_eps)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        y = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.head_dim**-0.5,
            mask=mask,
        )
        y = mx.transpose(y, (0, 2, 1, 3)).reshape(batch, seq_len, -1)
        return self.attn_resid(y)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        n_mlp = int(round(((8 / 3) * args.n_embd) / 128) * 128)
        self.mlp_gate = nn.Linear(args.n_embd, n_mlp, bias=False)
        self.mlp_linear = nn.Linear(args.n_embd, n_mlp, bias=False)
        self.mlp_resid = nn.Linear(n_mlp, args.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp_resid(nn.silu(self.mlp_gate(x)) * self.mlp_linear(x))


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.rms_norm_eps = args.rms_norm_eps
        self.attn = CausalSelfAttention(args)
        self.attn_gain = ActGain((2 * args.n_layer) ** -0.5)
        self.mlp = MLP(args)
        self.mlp_gain = ActGain((2 * args.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)

    def __call__(
        self,
        e_x: mx.array,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | str | None,
        cache: Any | None,
    ) -> mx.array:
        x = x + self.attn_gain(self.attn(rms_norm(x, self.rms_norm_eps), cos, sin, mask, cache))
        x = x + self.mlp_gain(self.mlp(rms_norm(x, self.rms_norm_eps)))
        x = x + self.embed_skip(e_x)
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.embed = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = [Block(args) for _ in range(args.n_layer)]
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.lm_head_gain = WeightGain()

        channel_range = mx.arange(0, args.head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (args.rope_theta ** (channel_range / args.head_dim))
        t = mx.arange(args.max_seq_len, dtype=mx.float32)
        freqs = t[:, None] * inv_freq[None, :]
        self.cos = mx.cos(freqs).astype(mx.bfloat16)[None, :, None, :]
        self.sin = mx.sin(freqs).astype(mx.bfloat16)[None, :, None, :]

    def __call__(self, inputs: mx.array, cache: list[Any] | None = None) -> mx.array:
        if cache is None:
            cache = [None] * len(self.blocks)

        seq_len = inputs.shape[1]
        offset = cache[0].offset if cache and cache[0] is not None else 0
        if offset + seq_len > self.args.max_seq_len:
            raise ValueError(
                f"sequence length {offset + seq_len} exceeds max_seq_len {self.args.max_seq_len}"
            )

        cos = self.cos[:, offset : offset + seq_len]
        sin = self.sin[:, offset : offset + seq_len]

        x = self.embed(inputs).astype(mx.bfloat16)
        x = rms_norm(x, self.args.rms_norm_eps)
        e_x = x
        mask = create_attention_mask(x, cache[0])
        for block, block_cache in zip(self.blocks, cache, strict=True):
            x = block(e_x, x, cos, sin, mask, block_cache)
        x = rms_norm(x, self.args.rms_norm_eps)
        logits = self.lm_head(x).astype(mx.float32)
        return self.lm_head_gain(logits).astype(mx.float32)

    @property
    def layers(self):
        return self.blocks

    def make_cache(self):
        return [KVCache() for _ in self.blocks]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return weights
