"""
RAG Prototype - Streamlit UI
Main application for document ingestion and question answering
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from rag.ingest import DocumentIngester
from rag.chunk import DocumentChunker
from rag.index import VectorIndexer
from rag.retrieve import DocumentRetriever
from rag.generate import AnswerGenerator

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="RAG Prototype",
    layout="wide"
)

# Initialize session state
if 'indexer' not in st.session_state:
    st.session_state.indexer = VectorIndexer(persist_directory=str(CHROMA_DIR))
if 'retriever' not in st.session_state:
    st.session_state.retriever = DocumentRetriever(st.session_state.indexer)
if 'generator' not in st.session_state:
    try:
        st.session_state.generator = AnswerGenerator()
        st.session_state.generator_init_error = None
    except Exception as e:
        st.session_state.generator = None
        st.session_state.generator_init_error = str(e)
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = 0


def main():
    st.title("🤖 RAG Prototype")
    st.markdown("### Retrieval-Augmented Generation System")

    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'html'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                process_documents(uploaded_files)

        st.divider()

        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Documents Indexed", st.session_state.documents_indexed)

        # Clear database
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.session_state.indexer.clear_database():
                st.session_state.documents_indexed = 0
                st.success("Database cleared!")
                st.rerun()

    # Main area for Q&A
    if st.session_state.generator is None:
        st.error(
            "LLM generator is not initialized. "
            "Check GROQ_API_KEY in your environment (.env) and restart the app.\n\n"
            f"Error: {st.session_state.get('generator_init_error')}"
        )
        return
    st.header("💬 Ask Questions")

    if st.session_state.documents_indexed == 0:
        st.info("👈 Please upload and process documents first")
    else:
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            k_docs = st.number_input("Top K results", min_value=1, max_value=10, value=3)

        if question:
            with st.spinner("Searching and generating answer..."):
                answer_with_citations(question, k_docs)


def process_documents(uploaded_files):
    """Process uploaded documents and add to vector store"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    ingester = DocumentIngester()
    chunker = DocumentChunker()

    total_chunks = 0

    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")

        # Save uploaded file temporarily
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Ingest document
        try:
            documents = ingester.load_document(str(file_path))

            # Chunk documents
            chunks = chunker.chunk_documents(documents, strategy="fixed")

            # Index chunks
            st.session_state.indexer.add_documents(chunks)

            total_chunks += len(chunks)
            st.session_state.documents_indexed += 1

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        progress_bar.progress((idx + 1) / len(uploaded_files))

    status_text.empty()
    progress_bar.empty()
    st.success(f"✅ Processed {len(uploaded_files)} documents ({total_chunks} chunks)")
    st.rerun()


def answer_with_citations(question: str, k: int = 3):
    """Retrieve relevant documents and generate answer with citations"""

    # Retrieve relevant documents
    retrieved_docs = st.session_state.retriever.retrieve(question, k=k)

    st.write("DEBUG retrieved_docs[0] keys:", list(retrieved_docs[0].keys()) if retrieved_docs else None)
    st.write("DEBUG retrieved_docs[0] metadata:", retrieved_docs[0].get("metadata") if retrieved_docs else None)
    st.write("DEBUG retrieved_docs[0] page:", retrieved_docs[0].get("page") if retrieved_docs else None)
    st.write("DEBUG retrieved_docs[0] chunk_id:", retrieved_docs[0].get("chunk_id") if retrieved_docs else None)

    if not retrieved_docs:
        st.warning("No relevant documents found.")
        return

    # Display retrieved documents
    with st.expander("📄 Retrieved Documents", expanded=False):
        for idx, doc in enumerate(retrieved_docs, start=1):
            score = doc.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"

            source = doc.get("source", "Unknown")
            page = doc.get("page") or (doc.get("metadata") or {}).get("page")
            chunk_id = doc.get("chunk_id") or (doc.get("metadata") or {}).get("chunk_id")
            chunk_index = doc.get("chunk_index") or (doc.get("metadata") or {}).get("chunk_index")

            locator_parts = [f"source={source}"]
            if page is not None:
                locator_parts.append(f"page={page}")
            if chunk_id:
                locator_parts.append(f"chunk_id={chunk_id}")
            elif chunk_index is not None:
                locator_parts.append(f"chunk={chunk_index}")

            locator = ", ".join(locator_parts)

            st.markdown(f"**Result {idx}** (similarity: {score_str})")
            st.caption(locator)

            content = doc.get("content", "")
            preview = content[:500] + "..." if len(content) > 500 else content
            st.text(preview)
            st.divider()

    # Generate answer
    answer_data = st.session_state.generator.generate_answer(question, retrieved_docs)

    # Display answer
    st.markdown("### 💡 Answer")
    st.markdown(answer_data['answer'])

    # Display citations
    if answer_data.get('citations'):
        st.markdown("### 📚 Citations")
        for c in answer_data['citations']:
            source = c.get("source", "Unknown")
            page = c.get("page")
            chunk_id = c.get("chunk_id")
            chunk_index = c.get("chunk_index")

            parts = [f"**Source:** {source}"]
            if page is not None:
                parts.append(f"**Page:** {page}")
            if chunk_id:
                parts.append(f"**Chunk ID:** {chunk_id}")
            elif chunk_index is not None:
                parts.append(f"**Chunk:** {chunk_index}")

            st.markdown(" — ".join(parts))
            st.divider()
    else:
        st.info("No citations were detected in the answer. Consider tightening the prompt or increasing Top K.")


if __name__ == "__main__":
    main()
