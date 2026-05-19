"""Multi-head masked self-attention (the heart of the decoder).

Implements scaled dot-product attention by hand:
    Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) + mask ) @ V

Mask is a strict lower-triangular pattern: position t can attend to
positions <= t. This is what makes the model *causal* (autoregressive).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GPTConfig


class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # Single Linear that produces Q, K, V concatenated — 3x faster than three Linears.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Triangular mask: row t allows columns 0..t. Registered as a buffer so it
        # moves with .to(device) but is not a learnable parameter.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V and split into heads: (B, T, C) -> (B, n_head, T, head_dim)
        qkv = self.qkv(x)                                 # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)           # each (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product: (B, h, T, d) @ (B, h, d, T) -> (B, h, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Weighted sum of values: (B, h, T, T) @ (B, h, T, d) -> (B, h, T, d)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads -> (B, T, C)
        return self.resid_drop(self.proj(y))
