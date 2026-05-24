"""Split corpus .txt files into overlapping token-window chunks.

Each chunk carries:
    text:    decoded chunk string
    source:  filename it came from (e.g. "julius_caesar.txt")
    pos:     token offset within the source

Token-based windows (not characters) keep chunks model-friendly: the same
GPT-2 BPE we train on is reused here so we always know exactly how many
tokens we'll feed back into the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tiktoken
from tqdm import tqdm


DEFAULT_CHUNK_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 64


@dataclass
class Chunk:
    text: str
    source: str
    pos: int


def chunk_text(text: str, encoder: tiktoken.Encoding,
               chunk_tokens: int, overlap_tokens: int) -> list[tuple[int, str]]:
    """Return list of (start_pos, chunk_text). Overlap stride = chunk - overlap."""
    ids = encoder.encode_ordinary(text)
    stride = chunk_tokens - overlap_tokens
    chunks: list[tuple[int, str]] = []
    for start in range(0, max(1, len(ids) - overlap_tokens), stride):
        end = min(start + chunk_tokens, len(ids))
        chunks.append((start, encoder.decode(ids[start:end])))
        if end == len(ids):
            break
    return chunks


def build_chunks(raw_dir: Path, chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                 overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> list[Chunk]:
    encoder = tiktoken.get_encoding("gpt2")
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {raw_dir} — run src.data --download first")

    out: list[Chunk] = []
    for f in tqdm(files, desc="Chunking"):
        text = f.read_text(encoding="utf-8")
        for pos, chunk in chunk_text(text, encoder, chunk_tokens, overlap_tokens):
            out.append(Chunk(text=chunk, source=f.name, pos=pos))
    return out
