"""
Document Chunking Module
Implements various text chunking strategies
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))


class Chunk:
    """Represents a text chunk with metadata"""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Chunk(length={len(self.content)}, metadata={self.metadata})"


class DocumentChunker:
    """Handles document chunking with various strategies"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Any], strategy: str = 'fixed') -> List[Chunk]:
        """
        Chunk a list of documents using specified strategy

        Args:
            documents: List of Document objects
            strategy: Chunking strategy ('fixed', 'recursive', 'semantic')

        Returns:
            List of Chunk objects
        """
        all_chunks = []

        for doc in documents:
            if strategy == 'fixed':
                chunks = self._fixed_size_chunking(doc.content, doc.metadata)
            elif strategy == 'recursive':
                chunks = self._recursive_chunking(doc.content, doc.metadata)
            elif strategy == 'semantic':
                chunks = self._semantic_chunking(doc.content, doc.metadata)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")

            all_chunks.extend(chunks)

        return all_chunks

    def _fixed_size_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Simple fixed-size chunking with overlap

        Args:
            text: Text to chunk
            metadata: Document metadata

        Returns:
            List of chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = len(chunks)
                chunk_metadata['chunk_start'] = start
                chunk_metadata['chunk_end'] = end

                source = chunk_metadata.get('source', 'unknown')
                page = chunk_metadata.get('page', 'na')
                chunk_metadata['chunk_id'] = f"{source}:p{page}:c{chunk_metadata['chunk_index']}"

                chunks.append(Chunk(content=chunk_text, metadata=chunk_metadata))

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _recursive_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Recursive chunking that tries to split on natural boundaries

        Args:
            text: Text to chunk
            metadata: Document metadata

        Returns:
            List of chunks
        """
        # Separators in order of preference
        separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']

        return self._recursive_split(text, metadata, separators, 0)

    def _recursive_split(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        separators: List[str],
        chunk_index: int
    ) -> List[Chunk]:
        """
        Recursively split text using separators
        """
        chunks = []

        if len(text) <= self.chunk_size:
            if text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = chunk_index
                chunks.append(Chunk(content=text, metadata=chunk_metadata))
            return chunks

        # Try each separator
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""

                for part in parts:
                    # Add separator back
                    part_with_sep = part + separator if part != parts[-1] else part

                    if len(current_chunk) + len(part_with_sep) <= self.chunk_size:
                        current_chunk += part_with_sep
                    else:
                        if current_chunk.strip():
                            chunk_metadata = metadata.copy()
                            chunk_metadata['chunk_index'] = len(chunks)
                            chunks.append(Chunk(content=current_chunk, metadata=chunk_metadata))

                        current_chunk = part_with_sep

                # Add remaining chunk
                if current_chunk.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_index'] = len(chunks)
                    chunks.append(Chunk(content=current_chunk, metadata=chunk_metadata))

                return chunks

        # If no separator found, fall back to fixed size
        return self._fixed_size_chunking(text, metadata)

    def _semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Semantic chunking based on paragraph and sentence structure

        Args:
            text: Text to chunk
            metadata: Document metadata

        Returns:
            List of chunks
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_index'] = len(chunks)
                    chunks.append(Chunk(content=current_chunk, metadata=chunk_metadata))

                    source = chunk_metadata.get('source', 'unknown')
                    page = chunk_metadata.get('page', 'na')
                    chunk_metadata['chunk_id'] = f"{source}:p{page}:c{chunk_metadata['chunk_index']}"

                    # Add overlap from previous chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + '\n\n' + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = len(chunks)
            chunks.append(Chunk(content=current_chunk, metadata=chunk_metadata))

        return chunks
