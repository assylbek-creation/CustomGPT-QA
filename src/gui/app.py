"""Streamlit chatbot for CustomGPT-QA.

Run:
    streamlit run src/gui/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import tiktoken
import torch

from src.config import GPTConfig
from src.model.gpt import GPT
from src.rag.index import Retriever
from src.rag.pipeline import format_prompt


CKPT_PATH = Path("checkpoints/best.pt")
STORE_DIR = Path("data/vector_store")


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource(show_spinner="Loading GPT checkpoint...")
def load_model() -> tuple[GPT, str]:
    device = resolve_device()
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_retriever() -> Retriever:
    return Retriever(STORE_DIR)


@st.cache_resource
def load_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def generate(model: GPT, prompt_ids: list[int], device: str,
             max_new_tokens: int, temperature: float, top_k: int) -> list[int]:
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return out[0, len(prompt_ids):].tolist()


def main() -> None:
    st.set_page_config(page_title="CustomGPT-QA", page_icon="📜", layout="wide")
    st.title("CustomGPT-QA")
    st.caption(
        "Decoder-only Transformer (GPT-2 style) trained from scratch on "
        "Ancient History Wikipedia, optionally augmented with RAG retrieval."
    )

    if not CKPT_PATH.exists():
        st.error(f"Checkpoint not found at {CKPT_PATH}. Run `python -m src.train` first.")
        st.stop()
    if not STORE_DIR.exists() or not (STORE_DIR / "index.faiss").exists():
        st.warning(
            f"No FAISS index at {STORE_DIR}. RAG will be disabled. "
            "Build it with `python -m src.rag.index --build`."
        )

    model, device = load_model()
    enc = load_tokenizer()
    retriever = None
    if (STORE_DIR / "index.faiss").exists():
        retriever = load_retriever()

    with st.sidebar:
        st.subheader("Sampling")
        max_new_tokens = st.slider("max_new_tokens", 16, 240, 120, 8)
        temperature = st.slider("temperature", 0.1, 1.5, 0.8, 0.05)
        top_k = st.slider("top_k", 1, 200, 40, 1)

        st.subheader("RAG")
        rag_available = retriever is not None
        use_rag = st.toggle(
            "Use retrieval", value=rag_available, disabled=not rag_available,
            help="When on, top-k chunks from the vector store are prepended to the prompt.",
        )
        k = st.slider("top_k chunks", 1, 8, 4, 1, disabled=not use_rag)

        st.divider()
        st.caption(f"device: `{device}`")
        st.caption(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Retrieved sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**[{s['score']:+.3f}] {s['source']}** @ pos {s['pos']}")
                        st.text(s["preview"])

    if question := st.chat_input("Ask something about ancient history..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            sources_payload: list[dict] | None = None
            if use_rag and retriever is not None:
                retrieved = retriever.search(question, k=k)
                prompt = format_prompt(question, retrieved)
                sources_payload = [
                    {
                        "score": float(score),
                        "source": chunk.source,
                        "pos": chunk.pos,
                        "preview": chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
                    }
                    for chunk, score in retrieved
                ]
            else:
                prompt = question

            prompt_ids = enc.encode_ordinary(prompt)
            budget = model.cfg.block_size - max_new_tokens
            if len(prompt_ids) > budget:
                prompt_ids = prompt_ids[-budget:]

            with st.spinner("Generating..."):
                new_ids = generate(model, prompt_ids, device,
                                   max_new_tokens, temperature, top_k)
            answer = enc.decode(new_ids).strip() or "_(empty generation)_"
            st.markdown(answer)

            if sources_payload:
                with st.expander("Retrieved sources"):
                    for s in sources_payload:
                        st.markdown(f"**[{s['score']:+.3f}] {s['source']}** @ pos {s['pos']}")
                        st.text(s["preview"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_payload,
            })


if __name__ == "__main__":
    main()
