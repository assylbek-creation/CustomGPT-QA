"""FAISS index build + query.

Build:
    python -m src.rag.index --build
Query:
    python -m src.rag.index --query "Who was Cleopatra?"

Persists three files in data/vector_store/:
    index.faiss   — FAISS IndexFlatIP (cosine via normalized vectors)
    chunks.json   — list of {text, source, pos} aligned with vector row order
    meta.json     — {model_name, dim, chunk_tokens, overlap_tokens}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from src.rag.chunker import Chunk, DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS, build_chunks
from src.rag.embedder import EMBED_DIM, Embedder


INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


def build(raw_dir: Path, out_dir: Path, chunk_tokens: int, overlap_tokens: int,
          model_name: str | None = None) -> int:
    chunks = build_chunks(raw_dir, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    print(f"Built {len(chunks)} chunks")

    embedder = Embedder() if model_name is None else Embedder(model_name)
    print(f"Embedding with {embedder.model._model_card_vars.get('model_name', 'MiniLM')}...")
    vecs = embedder.encode([c.text for c in chunks], show_progress=True)
    assert vecs.shape == (len(chunks), EMBED_DIM)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    (out_dir / CHUNKS_FILE).write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False)
    )
    (out_dir / META_FILE).write_text(json.dumps({
        "model_name": model_name or "sentence-transformers/all-MiniLM-L6-v2",
        "dim": EMBED_DIM,
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "n_chunks": len(chunks),
    }, indent=2))
    return len(chunks)


class Retriever:
    """Loaded once, queried many times (e.g. from the Streamlit GUI)."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.index = faiss.read_index(str(store_dir / INDEX_FILE))
        chunks_raw = json.loads((store_dir / CHUNKS_FILE).read_text())
        self.chunks: list[Chunk] = [Chunk(**c) for c in chunks_raw]
        self.meta = json.loads((store_dir / META_FILE).read_text())
        self.embedder = Embedder(self.meta["model_name"])

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.embedder.encode([query])
        scores, idxs = self.index.search(q, k)
        return [(self.chunks[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--raw", default="data/raw", type=Path)
    parser.add_argument("--store", default="data/vector_store", type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()

    if not args.build and not args.query:
        parser.error("pass --build or --query")

    if args.build:
        n = build(args.raw, args.store, args.chunk_tokens, args.overlap_tokens)
        print(f"\nIndex built: {n} chunks -> {args.store}")

    if args.query:
        r = Retriever(args.store)
        print(f"\nTop-{args.k} for: {args.query!r}\n" + "=" * 70)
        for chunk, score in r.search(args.query, k=args.k):
            preview = chunk.text.replace("\n", " ")[:160]
            print(f"[{score:+.3f}] {chunk.source} @ pos {chunk.pos}")
            print(f"  {preview}...\n")


if __name__ == "__main__":
    main()
