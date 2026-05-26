"""Load HuggingFace GPT-2 pretrained weights into our custom GPT.

Why this works: our architecture *is* GPT-2 (token + learned-position
embeddings, pre-LN, multi-head causal self-attention with fused QKV,
4x GELU FFN, tied LM head). With the defaults in src/config.py
(n_layer=12, n_head=12, n_embd=768, block_size=1024, bias=True) every
HF tensor has a matching counterpart in our state_dict.

The only catch: HF uses Conv1D (linear-with-transposed-weight) for the
attention QKV/output projections and the MLP. We transpose those when
copying. PyTorch nn.Linear stores weight as (out, in); Conv1D stores
it as (in, out).

Usage:
    python -m src.load_gpt2                # writes checkpoints/gpt2_init.pt
    python -m src.load_gpt2 --variant gpt2-medium --out checkpoints/gpt2m.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from src.config import GPTConfig
from src.model.gpt import GPT


# HF Conv1D weights need to be transposed before loading into nn.Linear.
TRANSPOSED_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def hf_to_ours(name: str) -> str | None:
    """Translate one HuggingFace GPT-2 parameter name to our state_dict key.

    Returns None for keys we do not consume (e.g. the masked-bias buffer that
    HF stores inside attention — we have our own causal_mask buffer).
    """
    if name == "wte.weight":          return "embed.tok_emb.weight"
    if name == "wpe.weight":          return "embed.pos_emb.weight"
    if name == "ln_f.weight":         return "ln_f.weight"
    if name == "ln_f.bias":           return "ln_f.bias"

    if not name.startswith("h."):
        return None
    _, idx, *rest = name.split(".")
    tail = ".".join(rest)
    if tail == "ln_1.weight":             return f"blocks.{idx}.ln1.weight"
    if tail == "ln_1.bias":               return f"blocks.{idx}.ln1.bias"
    if tail == "ln_2.weight":             return f"blocks.{idx}.ln2.weight"
    if tail == "ln_2.bias":               return f"blocks.{idx}.ln2.bias"
    if tail == "attn.c_attn.weight":      return f"blocks.{idx}.attn.qkv.weight"
    if tail == "attn.c_attn.bias":        return f"blocks.{idx}.attn.qkv.bias"
    if tail == "attn.c_proj.weight":      return f"blocks.{idx}.attn.proj.weight"
    if tail == "attn.c_proj.bias":        return f"blocks.{idx}.attn.proj.bias"
    if tail == "mlp.c_fc.weight":         return f"blocks.{idx}.ffn.net.0.weight"
    if tail == "mlp.c_fc.bias":           return f"blocks.{idx}.ffn.net.0.bias"
    if tail == "mlp.c_proj.weight":       return f"blocks.{idx}.ffn.net.2.weight"
    if tail == "mlp.c_proj.bias":         return f"blocks.{idx}.ffn.net.2.bias"
    return None


CONFIGS_FOR = {
    "gpt2":        GPTConfig(n_layer=12, n_head=12, n_embd=768,  block_size=1024),
    "gpt2-medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
}


def load(variant: str = "gpt2") -> GPT:
    if variant not in CONFIGS_FOR:
        raise ValueError(f"unsupported variant {variant}; choose from {list(CONFIGS_FOR)}")
    cfg = CONFIGS_FOR[variant]

    print(f"Downloading HuggingFace {variant} weights...")
    hf = GPT2LMHeadModel.from_pretrained(variant)
    hf_sd = hf.transformer.state_dict()
    # The HF LM head is tied to wte — no extra copy needed because our model
    # also ties lm_head.weight to embed.tok_emb.weight.

    print(f"Building our GPT({variant}-shape) and copying weights...")
    model = GPT(cfg)
    our_sd = model.state_dict()

    n_copied, n_skipped = 0, 0
    for hf_name, hf_tensor in hf_sd.items():
        our_name = hf_to_ours(hf_name)
        if our_name is None:
            n_skipped += 1
            continue
        if any(hf_name.endswith(s) for s in TRANSPOSED_SUFFIXES):
            hf_tensor = hf_tensor.t().contiguous()
        assert hf_tensor.shape == our_sd[our_name].shape, (
            f"shape mismatch on {hf_name} -> {our_name}: "
            f"{tuple(hf_tensor.shape)} vs {tuple(our_sd[our_name].shape)}"
        )
        our_sd[our_name].copy_(hf_tensor)
        n_copied += 1
    print(f"  copied {n_copied} tensors, skipped {n_skipped} HF-internal keys")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="gpt2", choices=list(CONFIGS_FOR))
    parser.add_argument("--out", default="checkpoints/gpt2_init.pt", type=Path)
    args = parser.parse_args()

    model = load(args.variant)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": asdict(model.cfg),
        "step": 0,
        "val_loss": float("inf"),
        "init_from": f"huggingface/{args.variant}",
    }, args.out)
    print(f"\nSaved: {args.out} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")


if __name__ == "__main__":
    main()
