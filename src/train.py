"""Train CustomGPT on the tokenized corpus.

Usage:
    python -m src.train                     # default config
    python -m src.train --max-iters 2000    # shorter run
    python -m src.train --wandb             # enable Weights & Biases
    python -m src.train --device cpu        # force CPU
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tiktoken
import torch

from src.config import GPTConfig, TrainConfig
from src.model.gpt import GPT


def get_batch(
    data: np.memmap, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random contiguous chunks; targets = inputs shifted by 1."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def cosine_lr(step: int, tcfg: TrainConfig) -> float:
    """Linear warmup -> cosine decay to min_lr."""
    if step < tcfg.warmup_iters:
        return tcfg.learning_rate * (step + 1) / tcfg.warmup_iters
    if step > tcfg.max_iters:
        return tcfg.min_lr
    progress = (step - tcfg.warmup_iters) / (tcfg.max_iters - tcfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tcfg.min_lr + coeff * (tcfg.learning_rate - tcfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    splits: dict[str, np.memmap],
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def sample(model: GPT, prompt: str, max_new_tokens: int, device: str) -> str:
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    return enc.decode(out[0].tolist())


def resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed", type=Path)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Checkpoint to initialize model weights from (e.g. checkpoints/gpt2_init.pt)")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--sample-prompt", default="In ancient Rome, ",
                        help="Prompt used when generating samples every eval interval")
    args = parser.parse_args()

    gcfg = GPTConfig()
    tcfg = TrainConfig()
    if args.max_iters is not None:     tcfg.max_iters = args.max_iters
    if args.batch_size is not None:    tcfg.batch_size = args.batch_size
    if args.eval_interval is not None: tcfg.eval_interval = args.eval_interval
    if args.device is not None:        tcfg.device = args.device
    if args.out_dir is not None:       tcfg.out_dir = str(args.out_dir)
    device = resolve_device(tcfg.device)
    print(f"device: {device}")

    torch.manual_seed(tcfg.seed)

    train_data = np.memmap(args.data / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(args.data / "val.bin", dtype=np.uint16, mode="r")
    print(f"train: {len(train_data):,} tokens | val: {len(val_data):,} tokens")

    if args.init_from is not None:
        print(f"init from: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        gcfg = GPTConfig(**ckpt["config"])      # use the saved model shape
        model = GPT(gcfg).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        model = GPT(gcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        betas=(0.9, 0.95),
    )

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=tcfg.wandb_project,
            config={**asdict(gcfg), **asdict(tcfg)},
        )

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    splits = {"train": train_data, "val": val_data}

    t0 = time.time()
    for step in range(tcfg.max_iters + 1):
        lr = cosine_lr(step, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % tcfg.eval_interval == 0:
            losses = estimate_loss(model, splits, gcfg.block_size, tcfg.batch_size,
                                   tcfg.eval_iters, device)
            ppl = math.exp(min(losses["val"], 20))
            dt = time.time() - t0
            print(f"step {step:5d} | lr {lr:.2e} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | ppl {ppl:.2f} | {dt:.1f}s")

            if run is not None:
                sample_text = sample(model, args.sample_prompt, 80, device)
                run.log({
                    "step": step, "lr": lr,
                    "train/loss": losses["train"], "val/loss": losses["val"],
                    "val/perplexity": ppl,
                    "sample": wandb.Html(f"<pre>{sample_text}</pre>"),
                })

            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "config": asdict(gcfg),
                    "step": step,
                    "val_loss": best_val,
                }
                torch.save(ckpt, out_dir / "best.pt")

        if step == tcfg.max_iters:
            break

        x, y = get_batch(train_data, gcfg.block_size, tcfg.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

    print(f"\nDone. Best val loss: {best_val:.4f} -> {out_dir/'best.pt'}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
