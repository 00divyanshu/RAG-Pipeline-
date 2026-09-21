import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

def _get_config_val(key: str, default: str = "") -> str:
    """Gets configuration from Streamlit secrets, environment variables, or default."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

# API Keys & Cloud Credentials
GOOGLE_API_KEY = _get_config_val("GOOGLE_API_KEY", "") or _get_config_val("GEMINI_API_KEY", "")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PINECONE_API_KEY = _get_config_val("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = _get_config_val("PINECONE_INDEX_NAME", "pdf-rag")

# Provider Detection (defaults to Gemini if GOOGLE_API_KEY is present, else Ollama)
DEFAULT_LLM_PROVIDER = "gemini" if GOOGLE_API_KEY else "ollama"
LLM_PROVIDER = _get_config_val("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()

DEFAULT_EMBED_PROVIDER = "gemini" if GOOGLE_API_KEY else "ollama"
EMBEDDING_PROVIDER = _get_config_val("EMBEDDING_PROVIDER", DEFAULT_EMBED_PROVIDER).lower()

# Vector DB Provider: "pinecone" if PINECONE_API_KEY is set, else "chroma"
DEFAULT_VECTOR_DB = "pinecone" if PINECONE_API_KEY else "chroma"
VECTOR_DB_PROVIDER = _get_config_val("VECTOR_DB_PROVIDER", DEFAULT_VECTOR_DB).lower()

# Model Names
GEMINI_LLM_MODEL = _get_config_val("GEMINI_LLM_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = _get_config_val("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

OLLAMA_BASE_URL = _get_config_val("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = _get_config_val("LLM_MODEL", "qwen2.5-coder:7b")
OLLAMA_EMBEDDING_MODEL = _get_config_val("EMBEDDING_MODEL", "nomic-embed-text")

# Active Models based on provider
LLM_MODEL = GEMINI_LLM_MODEL if LLM_PROVIDER == "gemini" else OLLAMA_LLM_MODEL
EMBEDDING_MODEL = GEMINI_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "gemini" else OLLAMA_EMBEDDING_MODEL

# Directories
DOCS_DIR = BASE_DIR / _get_config_val("DOCS_DIR", "data/docs")
CHROMA_PERSIST_DIR = BASE_DIR / _get_config_val("CHROMA_PERSIST_DIR", "chroma_db")

# Text Splitting & Retrieval Parameters
CHUNK_SIZE = int(_get_config_val("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_get_config_val("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(_get_config_val("RETRIEVER_K", "4"))
