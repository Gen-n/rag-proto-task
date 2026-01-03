"""
Document Ingestion Module
Handles loading documents from various formats (PDF, TXT, HTML)
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from bs4 import BeautifulSoup


class Document:
    """Simple document class to store content and metadata"""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class DocumentIngester:
    """Loads documents from various file formats"""

    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.html', '.htm']

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path

        Args:
            file_path: Path to the document

        Returns:
            List of Document objects
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Route to appropriate loader
        if suffix == '.pdf':
            return self._load_pdf(path)
        elif suffix == '.txt':
            return self._load_txt(path)
        elif suffix in ['.html', '.htm']:
            return self._load_html(path)

        return []

    def _load_pdf(self, path: Path) -> List[Document]:
        """Load PDF document"""
        documents = []

        try:
            reader = PdfReader(str(path))

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if text.strip():  # Only add non-empty pages
                    doc = Document(
                        content=text,
                        metadata={
                            'source': path.name,
                            'doc_id': path.stem,
                            'page': page_num + 1,
                            'format': 'pdf',
                            'total_pages': len(reader.pages)
                        }
                    )
                    documents.append(doc)

        except Exception as e:
            raise Exception(f"Error loading PDF {path.name}: {str(e)}")

        return documents

    def _load_txt(self, path: Path) -> List[Document]:
        """Load plain text document"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            doc = Document(
                content=text,
                metadata={
                    'source': path.name,
                    'format': 'txt'
                }
            )

            return [doc]

        except UnicodeDecodeError:
            # Try with different encoding
            with open(path, 'r', encoding='latin-1') as f:
                text = f.read()

            doc = Document(
                content=text,
                metadata={
                    'source': path.name,
                    'format': 'txt',
                    'encoding': 'latin-1'
                }
            )

            return [doc]

        except Exception as e:
            raise Exception(f"Error loading TXT {path.name}: {str(e)}")

    def _load_html(self, path: Path) -> List[Document]:
        """Load HTML document"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'lxml')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Extract title if available
            title = soup.title.string if soup.title else path.name

            doc = Document(
                content=text,
                metadata={
                    'source': path.name,
                    'title': title,
                    'format': 'html'
                }
            )

            return [doc]

        except Exception as e:
            raise Exception(f"Error loading HTML {path.name}: {str(e)}")

    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory

        Args:
            directory_path: Path to directory containing documents

        Returns:
            List of Document objects
        """
        path = Path(directory_path)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        all_documents = []

        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path.name}: {str(e)}")

        return all_documents
