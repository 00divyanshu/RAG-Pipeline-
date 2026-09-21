import logging
import shutil
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

def get_embeddings(model_name: str, base_url: str) -> OllamaEmbeddings:
    """Returns an OllamaEmbeddings instance."""
    return OllamaEmbeddings(
        model=model_name,
        base_url=base_url,
    )

def get_vector_store(
    persist_directory: Path,
    embeddings: OllamaEmbeddings,
    collection_name: str = "pdf_rag_collection",
) -> Chroma:
    """Initializes or loads a persistent Chroma vector store."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

def index_documents(
    documents: List[Document],
    persist_directory: Path,
    embeddings: OllamaEmbeddings,
    collection_name: str = "pdf_rag_collection",
    recreate: bool = False,
) -> Chroma:
    """
    Ingests and indexes document chunks into the Chroma vector database.
    If recreate is True, wipes any existing database at persist_directory first.
    """
    persist_path = Path(persist_directory)
    if recreate and persist_path.exists():
        logger.info(f"Recreating vector store collection '{collection_name}' at {persist_path}...")
        try:
            existing_vs = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_path),
            )
            existing_vs.delete_collection()
            logger.info("Previous collection deleted successfully via Chroma API.")
        except Exception as e:
            logger.warning(f"Could not delete collection via Chroma API: {e}. Attempting folder cleanup...")
            try:
                shutil.rmtree(persist_path, ignore_errors=True)
            except Exception:
                pass

    persist_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Indexing {len(documents)} document chunks into ChromaDB at {persist_path}...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_path),
    )
    logger.info("Indexing completed successfully.")
    return vector_store

def get_retriever(
    vector_store: Chroma,
    search_type: str = "similarity",
    k: int = 4,
):
    """
    Returns a configured retriever from the vector store.
    search_type can be 'similarity' or 'mmr' (Maximal Marginal Relevance).
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

