"""Sentence-transformer wrapper used by the RAG index.

Uses all-MiniLM-L6-v2 — 384-dim, ~80 MB, fast on CPU. Vectors are L2-
normalized so a FAISS inner-product index gives cosine similarity.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so IP == cosine
        )
        return vecs.astype(np.float32)
