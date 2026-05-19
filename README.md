# CustomGPT-QA

Building a decoder-only Transformer (GPT-2 style) **from scratch** in PyTorch,
trained on an Ancient History Wikipedia corpus, and served through a
Retrieval-Augmented Generation (RAG) chatbot with a Streamlit GUI.

## Stack

- **Python** ≥ 3.10
- **PyTorch** (Apple Silicon `mps` backend, CPU fallback)
- **tiktoken** — GPT-2 BPE tokenizer
- **FAISS** + **sentence-transformers** — vector store for RAG
- **Streamlit** — chatbot UI
- **Weights & Biases** — training metrics

## Project layout

```
src/
  config.py            # hyperparameters
  tokenizer.py         # tiktoken wrapper
  data.py              # corpus loader, batching
  model/
    attention.py       # multi-head masked self-attention
    block.py           # transformer block (Attn + FFN + LN + residual)
    embeddings.py      # token + positional embeddings
    gpt.py             # full GPT model + generate()
  train.py             # training loop with W&B
  evaluate.py          # perplexity, samples
  rag/
    chunker.py         # split corpus into chunks
    embedder.py        # sentence-transformers wrapper
    index.py           # FAISS build + query
    pipeline.py        # retrieve → prompt → generate
  gui/
    app.py             # Streamlit chatbot
notebooks/             # architecture walkthrough + demos
data/                  # raw + processed corpus, vector store (gitignored)
checkpoints/           # saved weights (gitignored)
```

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the corpus
python -m src.data --download

# 3. Train the model
python -m src.train

# 4. Build the RAG index
python -m src.rag.index --build

# 5. Launch the chatbot
streamlit run src/gui/app.py
```

## Status

Phase 0 — repository scaffold.
