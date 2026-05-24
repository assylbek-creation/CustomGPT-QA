"""End-to-end RAG: retrieve top-k chunks -> prompt -> GPT.generate().

Usage:
    python -m src.rag.pipeline --question "Who was Cleopatra?"
    python -m src.rag.pipeline --question "..." --k 4 --max-new-tokens 120
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import torch

from src.config import GPTConfig
from src.model.gpt import GPT
from src.rag.chunker import Chunk
from src.rag.index import Retriever


PROMPT_TEMPLATE = (
    "Answer the question using the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)


@dataclass
class RAGAnswer:
    answer: str          # only the newly generated continuation
    full_text: str       # prompt + answer (for debugging / display)
    sources: list[tuple[Chunk, float]]


def format_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    context = "\n---\n".join(f"[{c.source}] {c.text}" for c, _ in retrieved)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def load_model(ckpt_path: Path, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


class RAGPipeline:
    """One-shot pipeline holding the model + retriever for repeated queries."""

    def __init__(self, ckpt_path: Path, store_dir: Path, device: str = "mps"):
        self.device = resolve_device(device)
        self.model = load_model(ckpt_path, self.device)
        self.retriever = Retriever(store_dir)
        self.enc = tiktoken.get_encoding("gpt2")

    def answer(self, question: str, k: int = 4, max_new_tokens: int = 120,
               temperature: float = 0.8, top_k: int = 40) -> RAGAnswer:
        retrieved = self.retriever.search(question, k=k)
        prompt = format_prompt(question, retrieved)

        # Crop the prompt to leave room for max_new_tokens within block_size.
        prompt_ids = self.enc.encode_ordinary(prompt)
        budget = self.model.cfg.block_size - max_new_tokens
        if budget <= 0:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} >= block_size={self.model.cfg.block_size}"
            )
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_k=top_k)
        full = self.enc.decode(out[0].tolist())
        answer = self.enc.decode(out[0, len(prompt_ids):].tolist())
        return RAGAnswer(answer=answer, full_text=full, sources=retrieved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--ckpt", default="checkpoints/best.pt", type=Path)
    parser.add_argument("--store", default="data/vector_store", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    pipe = RAGPipeline(args.ckpt, args.store, device=args.device)
    res = pipe.answer(args.question, k=args.k, max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature, top_k=args.top_k)

    print(f"\nQuestion: {args.question}")
    print(f"\nRetrieved sources:")
    for c, s in res.sources:
        print(f"  [{s:+.3f}] {c.source} @ pos {c.pos}")
    print(f"\nAnswer:\n{res.answer}\n")


if __name__ == "__main__":
    main()
