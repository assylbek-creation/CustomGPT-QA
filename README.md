# CustomGPT-QA

Decoder-only Transformer (GPT-2 architecture) implemented **from scratch** in
PyTorch, initialized from pretrained GPT-2 weights, fine-tuned on an Ancient
History Wikipedia corpus, and served through a Retrieval-Augmented Generation
(RAG) chatbot with a Streamlit GUI.

## Results

| Setup                                          | Params | Val perplexity | Notes                                              |
| ---------------------------------------------- | -----: | -------------: | -------------------------------------------------- |
| 30M from-scratch (5000 iters)                  |    30M |            274 | Architecture-correctness baseline                  |
| 124M GPT-2 init, **zero-shot**                 |   124M |             29 | Confirms weight-loader maps every tensor correctly |
| 124M GPT-2 init + fine-tune (200 iters, best)  |   124M |         **26** | Production checkpoint                              |

Architecture walkthrough with math + code: see
[`notebooks/01_architecture_walkthrough.ipynb`](notebooks/01_architecture_walkthrough.ipynb).

## Stack

- **Python** ≥ 3.10, **PyTorch** (Apple Silicon `mps` backend, CPU fallback)
- **tiktoken** — GPT-2 BPE tokenizer
- **transformers** — HuggingFace, only for loading pretrained GPT-2 weights
- **FAISS** + **sentence-transformers** — vector store for RAG
- **Streamlit** — chatbot UI
- **Weights & Biases** — optional training metrics

## Project layout

```
src/
  config.py            # GPTConfig / TrainConfig dataclasses
  tokenizer.py         # tiktoken wrapper, corpus -> train.bin/val.bin
  data.py              # Wikipedia downloader + cleaner
  load_gpt2.py         # HuggingFace GPT-2 weights -> our state_dict
  train.py             # training loop (AdamW, cosine LR, --init-from)
  evaluate.py          # perplexity + sample generation
  model/
    attention.py       # multi-head masked self-attention (by-hand SDPA)
    block.py           # pre-LN transformer block (Attn + tanh-GELU FFN)
    embeddings.py      # token + learned positional embeddings
    gpt.py             # full GPT module + autoregressive generate()
  rag/
    chunker.py         # token-window chunker over data/raw/*.txt
    embedder.py        # all-MiniLM-L6-v2 wrapper
    index.py           # FAISS IndexFlatIP build + Retriever class
    pipeline.py        # retrieve -> prompt -> GPT.generate()
  gui/
    app.py             # Streamlit chatbot with RAG toggle
notebooks/             # architecture walkthrough
data/                  # raw, processed, vector_store (gitignored)
checkpoints/           # saved weights (gitignored)
```

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download + tokenize the Ancient History corpus
python -m src.data --download
python -m src.tokenizer --encode

# 3. Load pretrained GPT-2 weights into our architecture
python -m src.load_gpt2                                    # -> checkpoints/gpt2_init.pt

# 4. Fine-tune on Ancient History
python -m src.train --init-from checkpoints/gpt2_init.pt \
    --max-iters 600 --eval-interval 100 --batch-size 4    # -> checkpoints/best.pt

# 5. Evaluate (perplexity + samples)
python -m src.evaluate

# 6. Build the FAISS index for RAG
python -m src.rag.index --build

# 7. Launch the chatbot (http://localhost:8501)
streamlit run src/gui/app.py
```

## From-scratch training (no GPT-2 init)

To skip the pretrained-weight load and train from random init:

```bash
python -m src.train --max-iters 5000
```

The model is the same — only the starting weights differ.

## Status

Phases 0–7 complete: scaffold, data, model, training, eval, RAG, GUI, and
architecture walkthrough notebook.
