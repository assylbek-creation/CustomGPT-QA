"""GPT — decoder-only Transformer.

Architecture:
    tokens -> TokenPosEmbedding -> N x TransformerBlock -> LayerNorm -> LM head

Weight tying: the LM head shares weights with the token embedding matrix
(standard GPT-2 trick — halves the parameter count of the output layer).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GPTConfig
from src.model.block import TransformerBlock
from src.model.embeddings import TokenPositionalEmbedding


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenPositionalEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying.
        self.lm_head.weight = self.embed.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.embed.pos_emb.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is None:
            # Inference: only the last position is needed for next-token prediction.
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

        logits = self.lm_head(x)                                  # (B, T, vocab)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. idx: (B, T) starting tokens."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]              # crop to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)    # (B, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)     # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
