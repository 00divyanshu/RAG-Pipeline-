import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

logger = logging.getLogger(__name__)

def load_documents_from_directory(directory_path: Path) -> List[Document]:
    """
    Loads all PDF documents from the specified directory.
    Attaches relative file path and page metadata to each document.
    """
    directory = Path(directory_path)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created empty documents directory at {directory}")
        return []

    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {directory}: {[f.name for f in pdf_files]}")
    loader = PyPDFDirectoryLoader(str(directory))
    documents = loader.load()

    # Standardize metadata: ensure source is the filename, not full system path
    for doc in documents:
        full_source = doc.metadata.get("source", "")
        if full_source:
            doc.metadata["filename"] = Path(full_source).name
            # Keep 1-indexed page number for human readability in citations
            if "page" in doc.metadata:
                doc.metadata["page_number"] = doc.metadata["page"] + 1

    logger.info(f"Loaded {len(documents)} pages across {len(pdf_files)} document(s).")
    return documents

def load_single_pdf(file_path: Path) -> List[Document]:
    """Loads a single PDF file."""
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Valid PDF file not found at: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    for doc in documents:
        doc.metadata["filename"] = path.name
        if "page" in doc.metadata:
            doc.metadata["page_number"] = doc.metadata["page"] + 1
    return documents

