# RAG Pipeline Prototype (Final Interview Exercise)

## Overview

This project is a **minimal Retrieval-Augmented Generation (RAG) prototype** built as part of a final interview technical exercise.

The goal of the project is **not to build a full-featured chatbot**, but to clearly demonstrate:

- correct **document ingestion and chunking**
- reproducible **embedding generation**
- **similarity-based retrieval** from a vector store
- **LLM-based answer generation grounded in retrieved context**
- explicit **citation of retrieved chunks** (evaluation focus)

The implementation prioritizes **clarity, correctness, and debuggability** over feature completeness.

---

## Why RAG?

Large Language Models alone may hallucinate or answer from training data.  
RAG mitigates this by:

1. Retrieving relevant document chunks based on similarity
2. Passing only retrieved context to the LLM
3. Requiring answers to be **explicitly grounded and cited**

This prototype is designed to make that pipeline **transparent and evaluatable**.

---

## Architecture (High-Level)

User Question  
→ Vector Similarity Search (ChromaDB)  
→ Top-K Retrieved Chunks (with metadata)  
→ LLM Prompt (Context + Question)  
→ Answer with [Source N] citations  

---

## Tech Stack

- Python 3.10+
- Streamlit
- ChromaDB
- SentenceTransformers (MiniLM)
- Groq API (llama-3.1-8b-instant)
- pypdf

---

## Environment Variables

Create a `.env` file in the project root.

### Required

```env
GROQ_API_KEY=your_groq_api_key_here
```

The application will fail on startup if `GROQ_API_KEY` is not set.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Example Evaluation Prompts

- What architecture is proposed in the paper "Attention Is All You Need"?
- How many layers are used in the Transformer encoder and decoder?
- What are the main components of the Transformer encoder?

Expected behavior:
- grounded answers
- visible retrieved chunks
- citations with source, page, and chunk id
