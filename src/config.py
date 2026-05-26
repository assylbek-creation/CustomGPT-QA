"""Hyperparameters for CustomGPT-QA.

Sized for Apple Silicon (unified memory). Override per-run via CLI flags
in train.py or by importing and mutating before instantiating the model.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Defaults match GPT-2 small (124M params) so HuggingFace weights load cleanly."""
    vocab_size: int = 50257  # tiktoken GPT-2 BPE
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True        # GPT-2 uses bias in every Linear and LayerNorm


@dataclass
class TrainConfig:
    batch_size: int = 8      # smaller because the model is now 124M
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-5   # fine-tuning LR (was 3e-4 for from-scratch)
    min_lr: float = 3e-6
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "mps"      # "mps" | "cuda" | "cpu"
    seed: int = 1337
    wandb_project: str = "customgpt-qa"
    out_dir: str = "checkpoints"
