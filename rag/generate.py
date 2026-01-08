"""
Answer Generation Module
Generates answers using LLM with source citations
"""

import os
import re
from typing import List, Dict, Any, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))

SYSTEM_PROMPT = """You are a strict retrieval-based QA system.

You MUST follow these rules:
- Answer ONLY using the provided CONTEXT.
- Do NOT use outside knowledge.
- Do NOT speculate, summarize, or guess.
- If the answer is not explicitly stated in the CONTEXT, respond EXACTLY with:
"I don't know based on the provided context."

- Do NOT mention relevance scores.
- Do NOT mention missing access to documents.

CITATIONS (STRICT):
- Every factual statement MUST include a citation in the exact format: [Source N]
- If you cite multiple sources, DO NOT group them in one bracket.
  Correct: [Source 1][Source 2]
  Incorrect: [Source 1, Source 2]
- Do not invent sources. Use only sources provided in the context.
"""


class AnswerGenerator:
    """Generates answers using LLM with citations"""

    def __init__(self, model: str = LLM_MODEL, client: Optional[Groq] = None):
        """
        Initialize the answer generator

        Args:
            model: Groq model to use
            client: Optional injected Groq client (useful for tests)
        """
        self.model = model
        self.client = client  # lazy init if None

    # def _ensure_client(self) -> None:
    #     """Initialize Groq client lazily (keeps unit tests independent of API keys)."""
    #     if self.client is not None:
    #         return
    #
    #     api_key = os.getenv("GROQ_API_KEY")
    #     if not api_key:
    #         raise ValueError("GROQ_API_KEY not found in environment variables")
    #
    #     self.client = Groq(api_key=api_key)

    def _ensure_client(self) -> None:
        """Initialize Groq client lazily (keeps unit tests independent of API keys)."""
        if self.client is not None:
            return

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required to generate answers. "
                "Set it in a .env file or as an environment variable."
            )

        self.client = Groq(api_key=api_key)

    @staticmethod
    def _normalize_grouped_citations(answer: str) -> str:
        """
        Normalize grouped citations like:
          [Source 1, Source 2] -> [Source 1][Source 2]
        """
        return re.sub(
            r"\[Source (\d+)(?:,\s*Source (\d+))+\]",
            lambda m: "".join(f"[Source {n}]" for n in re.findall(r"\d+", m.group(0))),
            answer,
        )

    @staticmethod
    def _strip_invalid_sources(answer: str, max_source: int) -> str:
        """Remove citations out of range: [Source N] where N > max_source."""
        return re.sub(
            r"\[Source (\d+)\]",
            lambda m: m.group(0) if int(m.group(1)) <= max_source else "",
            answer,
        )

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer based on a query and retrieved documents
        """
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "citations": [],
            }

        context = self._prepare_context(retrieved_docs)
        prompt = self._create_prompt(query, context)

        # Ensure Groq client only when we actually need it
        self._ensure_client()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )

            answer = response.choices[0].message.content or ""

            max_source = len(retrieved_docs)

            # Normalize and strip invalid citations (deterministic eval contract)
            answer = self._normalize_grouped_citations(answer)
            answer = self._strip_invalid_sources(answer, max_source=max_source)

            citations = self._extract_citations(answer, retrieved_docs)

            return {
                "answer": answer,
                "citations": citations,
                "model": self.model,
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "citations": [],
            }

    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []

        for idx, doc in enumerate(retrieved_docs, 1):
            md = doc.get("metadata", {}) or {}

            source = doc.get("source", md.get("source", "Unknown"))
            page = doc.get("page", md.get("page"))
            chunk_id = doc.get("chunk_id", md.get("chunk_id"))
            chunk_index = doc.get("chunk_index", md.get("chunk_index"))
            content = doc.get("content", "")

            locator = f"{source}"
            if page is not None:
                locator += f", page={page}"
            if chunk_id:
                locator += f", chunk_id={chunk_id}"
            elif chunk_index is not None:
                locator += f", chunk={chunk_index}"

            context_parts.append(f"[Source {idx}] {locator}\n{content}\n")

        return "\n---\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for the LLM"""
        return f"""CONTEXT:
{context}

INSTRUCTIONS:
- Answer the QUESTION using ONLY the CONTEXT.
- If the answer is not explicitly stated in the CONTEXT, reply exactly:
"I don't know based on the provided context."
- For every factual statement, add citations like [Source 1] or [Source 2].

QUESTION: {query}

ANSWER:"""

    def _extract_citations(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and format citations from the answer"""
        citations: List[Dict[str, Any]] = []

        matches = re.finditer(r"\[Source (\d+)\]", answer)
        cited_sources = {int(m.group(1)) for m in matches}

        for source_num in sorted(cited_sources):
            if source_num <= len(retrieved_docs):
                doc = retrieved_docs[source_num - 1]
                md = doc.get("metadata", {}) or {}

                citations.append(
                    {
                        "source_number": source_num,
                        "source": doc.get("source", md.get("source", "Unknown")),
                        "page": doc.get("page", md.get("page")),
                        "chunk_id": doc.get("chunk_id", md.get("chunk_id")),
                        "chunk_index": doc.get("chunk_index", md.get("chunk_index")),
                    }
                )

        return citations

    def generate_answer_with_streaming(self, query: str, retrieved_docs: List[Dict[str, Any]]):
        """
        Generate an answer with streaming
        NOTE: Streaming yields text chunks only; citations are not parsed here.
        """
        if not retrieved_docs:
            yield "I couldn't find any relevant information to answer your question."
            return

        context = self._prepare_context(retrieved_docs)
        prompt = self._create_prompt(query, context)

        # Ensure Groq client for streaming path as well
        self._ensure_client()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            yield f"Error generating answer: {str(e)}"
