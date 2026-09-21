import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits documents into smaller semantic chunks with overlapping boundaries.
    Preserves all document metadata (filename, page number, etc.).
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} page(s) into {len(chunks)} chunk(s) (chunk_size={chunk_size}, overlap={chunk_overlap}).")
    return chunks

