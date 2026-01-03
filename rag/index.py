"""
Vector Indexing Module
Handles embeddings generation and ChromaDB indexing
"""

import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Configuration
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'rag_documents')


class VectorIndexer:
    """Manages vector embeddings and ChromaDB storage"""

    def __init__(self, persist_directory: str = "chroma_db", embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize the vector indexer

        Args:
            persist_directory: Directory to persist ChromaDB
            embedding_model: Name of the embedding model to use
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"ChromaDB collection '{COLLECTION_NAME}' initialized with {self.collection.count()} documents")

    def add_documents(self, chunks: List[Any]) -> bool:
        """
        Add document chunks to the vector store

        Args:
            chunks: List of Chunk objects

        Returns:
            Success status
        """
        if not chunks:
            return False

        # Prepare data
        documents = []
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)

            # Generate unique ID
            chunk_id = chunk.metadata.get('chunk_id')
            if not chunk_id:
                source = chunk.metadata.get('source', 'unknown')
                page = chunk.metadata.get('page', 'na')
                chunk_idx = chunk.metadata.get('chunk_index', idx)
                chunk_id = f"{source}:p{page}:c{chunk_idx}"

            ids.append(chunk_id)

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()

        # Add to ChromaDB
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(documents)} chunks to vector store")
            return True

        except Exception as e:
            print(f"Error adding documents to ChromaDB: {str(e)}")
            return False

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of dictionaries containing document content, metadata, and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        formatted_results = []

        if results['documents'] and len(results['documents']) > 0:
            for idx in range(len(results['documents'][0])):
                md = results['metadatas'][0][idx] or {}
                formatted_results.append({
                    'content': results['documents'][0][idx],
                    'metadata': md,
                    'distance': results['distances'][0][idx],
                    'score': 1 - results['distances'][0][idx],
                    'source': md.get('source', 'Unknown'),
                    'page': md.get('page'),
                    'chunk_id': md.get('chunk_id'),
                    'chunk_index': md.get('chunk_index'),
                })

        return formatted_results

    def clear_database(self) -> bool:
        """
        Clear all documents from the database

        Returns:
            Success status
        """
        try:
            # Delete the collection
            self.client.delete_collection(name=COLLECTION_NAME)

            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            print("Database cleared successfully")
            return True

        except Exception as e:
            print(f"Error clearing database: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Returns:
            Dictionary with collection statistics
        """
        return {
            'total_documents': self.collection.count(),
            'collection_name': COLLECTION_NAME,
            'embedding_model': self.embedding_model_name
        }
