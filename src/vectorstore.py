import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from src import config

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

def get_embeddings(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
):
    """
    Returns an Embeddings instance.
    Uses GoogleGenerativeAIEmbeddings with Google Gemini Cloud embeddings.
    """
    google_key = config.GOOGLE_API_KEY
    if not google_key:
        raise ValueError(
            "Google Gemini API Key is missing! "
            "Please configure GOOGLE_API_KEY in your Streamlit Cloud Secrets or via the Admin panel."
        )
    os.environ["GOOGLE_API_KEY"] = google_key
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    model = model_name or config.GEMINI_EMBEDDING_MODEL
    logger.info(f"Using Google Gemini embeddings: {model}")
    embed_kwargs: Dict[str, Any] = {"model": model, "google_api_key": google_key}
    return GoogleGenerativeAIEmbeddings(**embed_kwargs)

def get_pinecone_index():
    """Initializes and returns a Pinecone serverless Index object."""
    from pinecone import Pinecone, ServerlessSpec
    
    api_key = config.PINECONE_API_KEY
    if not api_key:
        raise ValueError(
            "Pinecone API Key is missing! "
            "Please configure PINECONE_API_KEY in your Streamlit Cloud Secrets or via the sidebar."
        )
    pc = Pinecone(api_key=api_key)
    index_name = config.PINECONE_INDEX_NAME
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone serverless index '{index_name}' (dimension 3072)...")
        pc.create_index(
            name=index_name,
            dimension=3072,  # Gemini embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)

def get_vector_store(
    persist_directory: Optional[Path] = None,
    embeddings = None,
    collection_name: str = "pdf_rag_collection",
    namespace: Optional[str] = None,
):
    """
    Initializes or loads a vector store.
    If PINECONE_API_KEY is configured, returns a PineconeVectorStore (100% cloud).
    Otherwise returns a local persistent Chroma instance.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    use_pinecone = (config.VECTOR_DB_PROVIDER == "pinecone" or config.is_cloud_environment())
    if use_pinecone:
        if not config.PINECONE_API_KEY:
            raise ValueError(
                "Pinecone API Key is missing! "
                "Please configure PINECONE_API_KEY in your Streamlit Cloud Secrets or via the sidebar."
            )
        from langchain_pinecone import PineconeVectorStore
        index = get_pinecone_index()
        logger.info(f"Connected to cloud Pinecone index '{config.PINECONE_INDEX_NAME}' (namespace='{namespace}').")
        return PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
    else:
        persist_path = Path(persist_directory or config.CHROMA_PERSIST_DIR)
        persist_path.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )

def index_documents(
    documents: List[Document],
    persist_directory: Optional[Path] = None,
    embeddings = None,
    collection_name: str = "pdf_rag_collection",
    recreate: bool = False,
    namespace: Optional[str] = None,
):
    """
    Ingests and indexes document chunks into the configured vector database
    (Pinecone cloud or local Chroma) with user namespace isolation.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    use_pinecone = (config.VECTOR_DB_PROVIDER == "pinecone" or config.is_cloud_environment())
    if use_pinecone:
        if not config.PINECONE_API_KEY:
            raise ValueError(
                "Pinecone API Key is missing! "
                "Please configure PINECONE_API_KEY in your Streamlit Cloud Secrets or via the sidebar."
            )
        from langchain_pinecone import PineconeVectorStore
        index = get_pinecone_index()
        if recreate:
            logger.info(f"Wiping vectors in cloud Pinecone index '{config.PINECONE_INDEX_NAME}' (namespace='{namespace}')...")
            try:
                if namespace:
                    index.delete(delete_all=True, namespace=namespace)
                else:
                    index.delete(delete_all=True)
            except Exception as e:
                if "not found" in str(e).lower() or "404" in str(e):
                    logger.info(f"Namespace '{namespace}' already empty or does not exist.")
                else:
                    logger.warning(f"Error resetting Pinecone index: {e}")

        logger.info(f"Indexing {len(documents)} document chunks into cloud Pinecone (namespace='{namespace}')...")
        try:
            os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
            vector_store = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=config.PINECONE_INDEX_NAME,
                namespace=namespace,
            )
            logger.info("Cloud indexing completed successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Pinecone indexing error: {e}")
            from src.error_logger import record_error
            record_error(
                service="Pinecone Cloud Index",
                user_message="Document chunk indexing into Pinecone failed",
                exception=e,
            )
            raise e
    else:
        persist_path = Path(persist_directory or config.CHROMA_PERSIST_DIR)
        persist_path.mkdir(parents=True, exist_ok=True)

        if recreate:
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

        logger.info(f"Indexing {len(documents)} document chunks into ChromaDB at {persist_path}...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_path),
        )
        logger.info("Indexing completed successfully.")
        return vector_store

def delete_document_by_name(
    vector_store,
    filename: str,
    docs_dir: Optional[Path] = None,
    namespace: Optional[str] = None,
) -> bool:
    """
    Removes all chunks associated with `filename` from the vector store
    (both Pinecone and Chroma supported, scoped to namespace) and deletes the physical file if present.
    """
    success = False
    try:
        class_name = type(vector_store).__name__
        if class_name == "PineconeVectorStore":
            logger.info(f"Deleting '{filename}' from Pinecone cloud index (namespace='{namespace}')...")
            try:
                if namespace:
                    vector_store._index.delete(filter={"filename": filename}, namespace=namespace)
                else:
                    vector_store._index.delete(filter={"filename": filename})
                success = True
            except Exception as pe:
                if "not found" in str(pe).lower() or "404" in str(pe):
                    logger.info(f"Namespace '{namespace}' not found in Pinecone, skipping vector deletion.")
                    success = True
                else:
                    raise pe
        else:
            # Chroma or standard collection
            logger.info(f"Deleting '{filename}' from vector database...")
            res = vector_store.get(where={"filename": filename})
            if res and res.get("ids"):
                vector_store.delete(ids=res["ids"])
                logger.info(f"Deleted {len(res['ids'])} chunk(s) for '{filename}'.")
            success = True
    except Exception as e:
        logger.error(f"Error deleting vectors for '{filename}': {e}")
        try:
            from src.error_logger import record_error
            record_error(
                service="Pinecone Cloud Index",
                user_message=f"Failed to delete document vectors for '{filename}'",
                exception=e,
            )
        except Exception:
            pass

    # Remove physical file if docs_dir is specified
    if docs_dir:
        try:
            target_path = Path(docs_dir) / filename
            if target_path.exists():
                target_path.unlink(missing_ok=True)
                logger.info(f"Deleted local file at {target_path}")
        except Exception as e:
            logger.warning(f"Could not delete local file '{filename}': {e}")

    return success

def list_indexed_documents(
    vector_store,
    docs_dir: Optional[Path] = None,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a summary list of all distinct ingested documents and their chunk counts
    across Pinecone cloud or local Chroma, scoped by user namespace.
    Example return: [{'filename': 'sample.pdf', 'chunks': 5}]
    """
    counts: Dict[str, int] = {}
    try:
        class_name = type(vector_store).__name__
        if class_name == "PineconeVectorStore" and hasattr(vector_store, "_index"):
            # Pinecone serverless index query scoped to namespace
            query_kwargs = {"vector": [0.0] * 3072, "top_k": 10000, "include_metadata": True}
            if namespace:
                query_kwargs["namespace"] = namespace
            res = vector_store._index.query(**query_kwargs)
            for m in res.matches:
                if m.metadata and "filename" in m.metadata:
                    fn = m.metadata["filename"]
                    counts[fn] = counts.get(fn, 0) + 1
        elif hasattr(vector_store, "get"):
            # Chroma database
            data = vector_store.get()
            for meta in data.get("metadatas", []):
                if meta and "filename" in meta:
                    fn = meta["filename"]
                    counts[fn] = counts.get(fn, 0) + 1
    except Exception as e:
        logger.warning(f"Could not fetch document counts from vector store: {e}")
        try:
            from src.error_logger import record_error
            record_error(
                service="Vector Store Query",
                user_message="Failed to fetch indexed documents list",
                exception=e,
            )
        except Exception:
            pass

    # Also include any local files not yet counted
    if docs_dir and Path(docs_dir).exists():
        for f in Path(docs_dir).glob("*.pdf"):
            if f.name not in counts:
                counts[f.name] = 0

    return [{"filename": fn, "chunks": count} for fn, count in sorted(counts.items())]

def get_retriever(
    vector_store,
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
