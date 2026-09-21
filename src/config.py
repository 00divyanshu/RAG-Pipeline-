import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

def _get_config_val(key: str, default: str) -> str:
    """Gets configuration from Streamlit secrets, environment variables, or default."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

# Directories
DOCS_DIR = BASE_DIR / _get_config_val("DOCS_DIR", "data/docs")
CHROMA_PERSIST_DIR = BASE_DIR / _get_config_val("CHROMA_PERSIST_DIR", "chroma_db")

# Ollama Models
OLLAMA_BASE_URL = _get_config_val("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = _get_config_val("LLM_MODEL", "qwen2.5-coder:7b")
EMBEDDING_MODEL = _get_config_val("EMBEDDING_MODEL", "nomic-embed-text")

# Text Splitting & Retrieval Parameters
CHUNK_SIZE = int(_get_config_val("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_get_config_val("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(_get_config_val("RETRIEVER_K", "4"))

