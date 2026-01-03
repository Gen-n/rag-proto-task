"""
Document Retrieval Module
Handles similarity search and document retrieval
"""

from typing import List, Dict, Any, Optional


class DocumentRetriever:
    """Retrieves relevant documents based on query"""

    def __init__(self, indexer):
        """
        Initialize the retriever

        Args:
            indexer: VectorIndexer instance
        """
        self.indexer = indexer

    def retrieve(
        self, 
        query: str, 
        k: int = 3, 
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of retrieved documents with metadata
        """
        # Search using the indexer
        results = self.indexer.search(query, k=k)

        # Filter by score threshold if provided
        if score_threshold is not None:
            results = [r for r in results if r['score'] >= score_threshold]

        # Add ranking information
        for idx, result in enumerate(results):
            result['rank'] = idx + 1

        return results

    def retrieve_with_reranking(
        self, 
        query: str, 
        k: int = 3, 
        initial_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with two-stage retrieval and reranking

        Args:
            query: Search query
            k: Final number of documents to return
            initial_k: Initial number of documents to retrieve

        Returns:
            List of reranked documents
        """
        # First stage: retrieve more documents
        initial_results = self.indexer.search(query, k=initial_k)

        # Second stage: rerank based on multiple factors
        reranked_results = self._rerank_results(query, initial_results)

        # Return top k
        return reranked_results[:k]

    def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on multiple factors

        Args:
            query: Original query
            results: Initial retrieval results

        Returns:
            Reranked results
        """
        # Simple reranking based on:
        # 1. Similarity score (already available)
        # 2. Document length (prefer medium-length chunks)
        # 3. Source diversity

        query_lower = query.lower()

        for result in results:
            content = result['content'].lower()

            # Base score from similarity
            score = result['score']

            # Bonus for exact query term matches
            query_terms = query_lower.split()
            term_matches = sum(1 for term in query_terms if term in content)
            term_bonus = (term_matches / len(query_terms)) * 0.1

            # Bonus for medium-length chunks (not too short, not too long)
            length = len(result['content'])
            if 300 <= length <= 1500:
                length_bonus = 0.05
            else:
                length_bonus = 0

            # Calculate final score
            result['rerank_score'] = score + term_bonus + length_bonus

        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return results

    def retrieve_by_metadata(
        self, 
        query: str,
        metadata_filters: Dict[str, Any],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with metadata filtering

        Args:
            query: Search query
            metadata_filters: Dictionary of metadata filters
            k: Number of documents to retrieve

        Returns:
            Filtered and retrieved documents
        """
        # Retrieve more documents initially
        results = self.indexer.search(query, k=k*3)

        # Filter by metadata
        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})

            # Check if all filters match
            match = True
            for key, value in metadata_filters.items():
                if metadata.get(key) != value:
                    match = False
                    break

            if match:
                filtered_results.append(result)

                # Stop if we have enough results
                if len(filtered_results) >= k:
                    break

        return filtered_results

    def get_document_context(
        self, 
        query: str, 
        k: int = 3,
        context_window: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with expanded context

        Args:
            query: Search query
            k: Number of documents to retrieve
            context_window: Additional characters to include around matches

        Returns:
            Documents with expanded context
        """
        results = self.retrieve(query, k=k)

        # For now, just return the results as-is
        # In a full implementation, this would fetch adjacent chunks
        # from the original document to provide more context

        return results
