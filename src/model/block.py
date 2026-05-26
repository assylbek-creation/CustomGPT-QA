"""Transformer block (pre-LN GPT-2 style).

Two residual sub-layers:
    x = x + Attn(LN(x))
    x = x + FFN(LN(x))

Pre-LN (LayerNorm *before* sublayer) is the modern variant — more stable
than the original post-LN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import GPTConfig
from src.model.attention import MultiHeadMaskedSelfAttention


class FeedForward(nn.Module):
    """Position-wise MLP: Linear -> GELU -> Linear -> Dropout. Expands 4x.

    Uses tanh-approximate GELU to match HuggingFace GPT-2's NewGELU activation,
    so we can load pretrained GPT-2 weights without an activation mismatch.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = MultiHeadMaskedSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
