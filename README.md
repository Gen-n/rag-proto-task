# RAG Pipeline Prototype (Final Interview Exercise)

## Overview

This project is a **minimal Retrieval-Augmented Generation (RAG) prototype** built as part of a final interview technical exercise.

The goal of the project is **not to build a full-featured chatbot**, but to clearly demonstrate:

- correct **document ingestion and chunking**
- reproducible **embedding generation**
- **similarity-based retrieval** from a vector store
- **LLM-based answer generation grounded strictly in retrieved context**
- **explicit source citations** for every factual statement
- **basic evaluation and testability** of RAG outputs

The implementation prioritizes **clarity, correctness, determinism, and debuggability** over feature completeness.

---

## Why RAG?

Large Language Models alone may hallucinate or answer from training data.  
RAG mitigates this by:

1. Retrieving relevant document chunks using vector similarity
2. Passing only retrieved context to the LLM
3. Enforcing answers that are **grounded and cited**

This prototype is designed to make the entire pipeline **transparent and evaluatable**.

---

## Architecture (High-Level)

User Question  
→ Vector Similarity Search (ChromaDB)  
→ Top-K Retrieved Chunks (with metadata)  
→ LLM Prompt (Context + Question)  
→ Answer with `[Source N]` citations  

---

## Project Structure

```
rag-proto/
├── app.py                 # Streamlit application
├── rag/
│   ├── ingest.py          # Document ingestion
│   ├── chunk.py           # Text chunking
│   ├── index.py           # Vector index creation
│   ├── retrieve.py        # Top-K retrieval
│   └── generate.py        # Answer generation + citations
├── data/
│   └── Attention_is_all_you_need.pdf
├── chroma_db/             # Persisted vector store
├── eval/
│   ├── dataset.jsonl      # Evaluation questions
│   └── run_eval.py        # Offline RAG evaluation
├── tests/
│   └── test_citations.py  # Unit tests for citation logic
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Tech Stack

- Python 3.10+
- Streamlit
- ChromaDB
- SentenceTransformers (MiniLM)
- Groq API (`llama-3.1-8b-instant`)
- pypdf
- pytest

---

## Environment Variables

This project uses the **Groq API** for LLM inference.

Generate a free API key at:  
https://console.groq.com/keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Notes:
- The LLM client is **lazily initialized**
- Unit tests can run **without** an API key
- Answer generation and evaluation **require** the key

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to run tests:
```bash
pip install -r requirements-dev.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Unit Tests

Run all unit tests:

```bash
pytest -q
```

Run a specific test file:

```bash
pytest -q tests/test_citations.py
```

---

## Offline Evaluation (Eval)

The project includes a lightweight offline evaluation script to assess:

- answer generation
- citation coverage
- basic accuracy proxies

Run evaluation:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --k 3
```

Change Top-K retrieval:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --k 5
```

---

## Example Evaluation Prompts

- What architecture is proposed in the paper "Attention Is All You Need"?
- How many layers are used in the Transformer encoder and decoder?
- What are the main components of the Transformer encoder?

Expected behavior:
- answers grounded strictly in retrieved context
- visible retrieved chunks
- explicit `[Source N]` citations including document metadata

---

## Notes

- This repository is intentionally minimal
- The focus is on **RAG correctness, traceability, and evaluation**
- No production hardening or UI polish is included
