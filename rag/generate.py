"""
Answer Generation Module
Generates answers using LLM with source citations
"""

import os
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configuration
LLM_MODEL = os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.3))
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 500))
SYSTEM_PROMPT = """You are a strict retrieval-based QA system.

You MUST follow these rules:
- Answer ONLY using the provided CONTEXT.
- Do NOT use outside knowledge.
- Do NOT speculate, summarize, or guess.
- If the answer is not explicitly stated in the CONTEXT, respond EXACTLY with:
"I don't know based on the provided context."

- Do NOT mention relevance scores.
- Do NOT mention missing access to documents.
- Every factual statement MUST include a citation like [Source N].
"""


class AnswerGenerator:
    """Generates answers using LLM with citations"""

    def __init__(self, model: str = LLM_MODEL):
        """
        Initialize the answer generator

        Args:
            model: Groq model to use
        """
        self.model = model
        api_key = os.getenv('GROQ_API_KEY')

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)

    def generate_answer(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate an answer based on query and retrieved documents

        Args:
            query: User's question
            retrieved_docs: List of retrieved documents with content and metadata

        Returns:
            Dictionary with answer and citations
        """
        if not retrieved_docs:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'citations': []
            }

        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)

        # Create prompt
        prompt = self._create_prompt(query, context)

        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )

            answer = response.choices[0].message.content

            import re
            def strip_invalid_sources(text: str) -> str:
                return re.sub(
                    r'\[Source (\d+)\]',
                    lambda m: m.group(0) if int(m.group(1)) <= max_source else '',
                    text
                )

            answer = strip_invalid_sources(answer)

            # Extract citations
            citations = self._extract_citations(answer, retrieved_docs)

            return {
                'answer': answer,
                'citations': citations,
                'model': self.model
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'citations': []
            }

    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Prepare context from retrieved documents

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.get('source', 'Unknown')
            page = doc.get('page') or doc.get('metadata', {}).get('page')
            chunk_id = doc.get('chunk_id') or doc.get('metadata', {}).get('chunk_id')
            chunk_index = doc.get('chunk_index') or doc.get('metadata', {}).get('chunk_index')
            content = doc.get('content', '')

            locator = f"{source}"
            if page is not None:
                locator += f", page={page}"
            if chunk_id:
                locator += f", chunk_id={chunk_id}"
            elif chunk_index is not None:
                locator += f", chunk={chunk_index}"

            context_part = f"[Source {idx}] {locator}\n{content}\n"
            context_parts.append(context_part)

        return "\n---\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create the prompt for the LLM

        Args:
            query: User's question
            context: Prepared context from documents

        Returns:
            Formatted prompt
        """
        prompt = f"""CONTEXT:
{context}

INSTRUCTIONS:
- Answer the QUESTION using ONLY the CONTEXT.
- If the answer is not explicitly stated in the CONTEXT, reply exactly:
"I don't know based on the provided context."
- For every factual statement, add citations like [Source 1] or [Source 2].

QUESTION: {query}

ANSWER:"""

        return prompt

    def _extract_citations(
        self, 
        answer: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract and format citations from the answer

        Args:
            answer: Generated answer
            retrieved_docs: Original retrieved documents

        Returns:
            List of citation dictionaries
        """
        citations = []

        # Look for [Source N] patterns in the answer
        import re
        source_pattern = r'\[Source (\d+)\]'
        matches = re.finditer(source_pattern, answer)

        cited_sources = set()
        for match in matches:
            source_num = int(match.group(1))
            cited_sources.add(source_num)

        # Format citations for cited sources
        for source_num in sorted(cited_sources):
            if source_num <= len(retrieved_docs):
                doc = retrieved_docs[source_num - 1]

                md = doc.get('metadata', {}) or {}
                citations.append({
                    'source_number': source_num,
                    'source': doc.get('source', md.get('source', 'Unknown')),
                    'page': doc.get('page', md.get('page')),
                    'chunk_id': doc.get('chunk_id', md.get('chunk_id')),
                    'chunk_index': doc.get('chunk_index', md.get('chunk_index')),
                })

        return citations

    def generate_answer_with_streaming(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]]
    ):
        """
        Generate an answer with streaming (for future enhancement)

        Args:
            query: User's question
            retrieved_docs: List of retrieved documents

        Yields:
            Answer chunks as they're generated
        """
        if not retrieved_docs:
            yield "I couldn't find any relevant information to answer your question."
            return

        context = self._prepare_context(retrieved_docs)
        prompt = self._create_prompt(query, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error generating answer: {str(e)}"
