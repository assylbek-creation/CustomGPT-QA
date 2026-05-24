"""Evaluate a trained checkpoint: validation perplexity + sample generations.

Usage:
    python -m src.evaluate
    python -m src.evaluate --ckpt checkpoints/best.pt --n-samples 5
    python -m src.evaluate --prompts "Julius Caesar was" "The Roman Senate"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import tiktoken
import torch

from src.config import GPTConfig
from src.model.gpt import GPT


DEFAULT_PROMPTS = [
    "In ancient Rome, ",
    "Julius Caesar was ",
    "The Egyptian pharaoh ",
    "Alexander the Great ",
    "The Code of Hammurabi ",
]


def load_checkpoint(path: Path, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def perplexity(model: GPT, data: np.memmap, block_size: int, batch_size: int,
               n_batches: int, device: str) -> float:
    """Average CE over n_batches random chunks; ppl = exp(loss)."""
    losses = torch.zeros(n_batches)
    for k in range(n_batches):
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    return math.exp(losses.mean().item())


def sample(model: GPT, prompt: str, max_new_tokens: int, temperature: float,
           top_k: int, device: str) -> str:
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())


def resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/best.pt", type=Path)
    parser.add_argument("--data", default="data/processed", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-batches", type=int, default=50,
                        help="Number of random batches for perplexity estimate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"loading: {args.ckpt}")
    model, ckpt = load_checkpoint(args.ckpt, device)
    print(f"checkpoint step: {ckpt.get('step', '?')} | val_loss: {ckpt.get('val_loss', 0):.4f}")
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    val = np.memmap(args.data / "val.bin", dtype=np.uint16, mode="r")
    ppl = perplexity(model, val, model.cfg.block_size, args.batch_size,
                     args.n_batches, device)
    print(f"\nValidation perplexity ({args.n_batches} batches x {args.batch_size}): {ppl:.2f}")

    print(f"\nSamples (temp={args.temperature}, top_k={args.top_k}):")
    print("=" * 70)
    for p in args.prompts:
        text = sample(model, p, args.max_new_tokens, args.temperature, args.top_k, device)
        print(f"\n>>> {p!r}\n{text}\n")
        print("-" * 70)


if __name__ == "__main__":
    main()
