"""Token + learned positional embeddings.

x_tok: (B, T) integer token IDs
x_pos: (T,)   integer positions 0..T-1

Output: (B, T, n_embd) — embeddings summed + dropout applied.
GPT-2 uses *learned* positional embeddings (not sinusoidal).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import GPTConfig


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)            # (T,)
        return self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, n_embd)
