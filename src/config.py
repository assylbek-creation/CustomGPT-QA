"""Hyperparameters for CustomGPT-QA.

Sized for Apple Silicon (unified memory). Override per-run via CLI flags
in train.py or by importing and mutating before instantiating the model.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # tiktoken GPT-2 BPE
    block_size: int = 256    # context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False       # bias in Linear/LayerNorm (False = a touch faster)


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "mps"      # "mps" | "cuda" | "cpu"
    seed: int = 1337
    wandb_project: str = "customgpt-qa"
    out_dir: str = "checkpoints"
