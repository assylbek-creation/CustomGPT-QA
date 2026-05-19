"""GPT-2 BPE tokenizer (tiktoken) + corpus → train.bin/val.bin encoder.

Output format: contiguous uint16 token IDs (matches nanoGPT). Load with
    np.memmap("train.bin", dtype=np.uint16, mode="r")

Usage:
    python -m src.tokenizer --encode
    python -m src.tokenizer --encode --raw data/raw --out data/processed --val-frac 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm


ENCODING_NAME = "gpt2"
VOCAB_SIZE = 50257  # tiktoken gpt2; fits in uint16


def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def encode(text: str) -> list[int]:
    return get_encoder().encode_ordinary(text)


def decode(ids: list[int]) -> str:
    return get_encoder().decode(ids)


def encode_corpus(raw_dir: Path, out_dir: Path, val_frac: float = 0.1) -> tuple[int, int]:
    """Concatenate every .txt in raw_dir, encode with GPT-2 BPE, split into
    train.bin and val.bin in out_dir. Returns (n_train_tokens, n_val_tokens).
    """
    enc = get_encoder()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {raw_dir} — run src.data --download first")

    all_ids: list[int] = []
    for f in tqdm(files, desc="Encoding"):
        text = f.read_text(encoding="utf-8")
        all_ids.extend(enc.encode_ordinary(text))
        all_ids.append(enc.eot_token)  # end-of-text between articles

    arr = np.array(all_ids, dtype=np.uint16)
    n_val = int(len(arr) * val_frac)
    val, train = arr[:n_val], arr[n_val:]

    train.tofile(out_dir / "train.bin")
    val.tofile(out_dir / "val.bin")
    return len(train), len(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--raw", default="data/raw", type=Path)
    parser.add_argument("--out", default="data/processed", type=Path)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    if not args.encode:
        parser.error("nothing to do — pass --encode")

    n_train, n_val = encode_corpus(args.raw, args.out, args.val_frac)
    print(f"\ntrain.bin: {n_train:,} tokens")
    print(f"val.bin:   {n_val:,} tokens")
    print(f"vocab:     {VOCAB_SIZE}")


if __name__ == "__main__":
    main()
