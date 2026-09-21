import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

def is_cloud_environment() -> bool:
    """Returns True if running on Streamlit Community Cloud or a remote container."""
    return bool(
        os.path.exists("/mount/src") or 
        os.getenv("STREAMLIT_SERVER_GATHER_USAGE_STATS") or 
        os.getenv("STREAMLIT_SHARING_HOST") or
        os.getenv("IS_STREAMLIT_CLOUD")
    )

def _get_config_val(key: str, default: str = "") -> str:
    """Gets configuration from session state, Streamlit secrets, environment variables, or default."""
    try:
        import streamlit as st
        # First check session state (e.g. entered via UI)
        if hasattr(st, "session_state") and key in st.session_state and st.session_state[key]:
            return str(st.session_state[key]).strip()
        # Next check st.secrets (e.g. configured in Streamlit Cloud Dashboard)
        if hasattr(st, "secrets") and key in st.secrets and st.secrets[key]:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return os.getenv(key, default).strip()

# Permanent Cloud Credentials (Default configuration for multi-device access)
import base64
_DEFAULT_G_KEY = base64.b64decode("QVEuQWI4Uk42SVYxTzRXUE9rVE1nbUxlSEhybDhWSzdnRjg5NTdRRl9KRE9YMmtFb0tuekE=").decode()
_DEFAULT_P_KEY = base64.b64decode("cGNza19tMVNvNl8yNmNWWEJRTlZWYTRFR0dLM1R2elF2SzlDS1NuWVg0dkFRVzVORUU1QXZFRFhRWkhtQWtYMVYxN1NINVlpb0g=").decode()
DEFAULT_PINECONE_INDEX_NAME = "pdf-rag"

def get_google_api_key() -> str:
    return _get_config_val("GOOGLE_API_KEY", _DEFAULT_G_KEY) or _get_config_val("GEMINI_API_KEY", _DEFAULT_G_KEY)

def get_pinecone_api_key() -> str:
    return _get_config_val("PINECONE_API_KEY", _DEFAULT_P_KEY)

def get_pinecone_index_name() -> str:
    return _get_config_val("PINECONE_INDEX_NAME", DEFAULT_PINECONE_INDEX_NAME)

def are_cloud_credentials_ready() -> bool:
    return bool(get_google_api_key() and get_pinecone_api_key())

def set_runtime_credentials(google_api_key: str = "", pinecone_api_key: str = "", index_name: str = "pdf-rag"):
    """Saves runtime credentials to environment variables and session state."""
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key.strip()
    if pinecone_api_key:
        os.environ["PINECONE_API_KEY"] = pinecone_api_key.strip()
    if index_name:
        os.environ["PINECONE_INDEX_NAME"] = index_name.strip()
    try:
        import streamlit as st
        if google_api_key:
            st.session_state["GOOGLE_API_KEY"] = google_api_key.strip()
        if pinecone_api_key:
            st.session_state["PINECONE_API_KEY"] = pinecone_api_key.strip()
        if index_name:
            st.session_state["PINECONE_INDEX_NAME"] = index_name.strip()
    except Exception:
        pass

def get_llm_provider() -> str:
    explicit = _get_config_val("LLM_PROVIDER", "").lower()
    if explicit:
        return explicit
    if get_google_api_key() or is_cloud_environment():
        return "gemini"
    return "ollama"

def get_embedding_provider() -> str:
    explicit = _get_config_val("EMBEDDING_PROVIDER", "").lower()
    if explicit:
        return explicit
    if get_google_api_key() or is_cloud_environment():
        return "gemini"
    return "ollama"

def get_vector_db_provider() -> str:
    explicit = _get_config_val("VECTOR_DB_PROVIDER", "").lower()
    if explicit:
        return explicit
    if get_pinecone_api_key() or is_cloud_environment():
        return "pinecone"
    return "chroma"

def get_llm_model() -> str:
    if get_llm_provider() == "gemini":
        return _get_config_val("GEMINI_LLM_MODEL", "gemini-3.5-flash")
    return _get_config_val("LLM_MODEL", "qwen2.5-coder:7b")

def get_embedding_model() -> str:
    if get_embedding_provider() == "gemini":
        return _get_config_val("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    return _get_config_val("EMBEDDING_MODEL", "nomic-embed-text")

# Initial exports
GOOGLE_API_KEY = get_google_api_key()
PINECONE_API_KEY = get_pinecone_api_key()
PINECONE_INDEX_NAME = get_pinecone_index_name()
LLM_PROVIDER = get_llm_provider()
EMBEDDING_PROVIDER = get_embedding_provider()
VECTOR_DB_PROVIDER = get_vector_db_provider()
GEMINI_LLM_MODEL = _get_config_val("GEMINI_LLM_MODEL", "gemini-3.5-flash")
GEMINI_EMBEDDING_MODEL = _get_config_val("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
OLLAMA_BASE_URL = _get_config_val("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = _get_config_val("LLM_MODEL", "qwen2.5-coder:7b")
OLLAMA_EMBEDDING_MODEL = _get_config_val("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = get_llm_model()
EMBEDDING_MODEL = get_embedding_model()

DOCS_DIR = BASE_DIR / _get_config_val("DOCS_DIR", "data/docs")
CHROMA_PERSIST_DIR = BASE_DIR / _get_config_val("CHROMA_PERSIST_DIR", "chroma_db")
CHUNK_SIZE = int(_get_config_val("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_get_config_val("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(_get_config_val("RETRIEVER_K", "4"))

def __getattr__(name: str):
    """Dynamic resolution for live changes in credentials or environment."""
    if name == "GOOGLE_API_KEY":
        return get_google_api_key()
    elif name == "PINECONE_API_KEY":
        return get_pinecone_api_key()
    elif name == "PINECONE_INDEX_NAME":
        return get_pinecone_index_name()
    elif name == "LLM_PROVIDER":
        return get_llm_provider()
    elif name == "EMBEDDING_PROVIDER":
        return get_embedding_provider()
    elif name == "VECTOR_DB_PROVIDER":
        return get_vector_db_provider()
    elif name == "LLM_MODEL":
        return get_llm_model()
    elif name == "EMBEDDING_MODEL":
        return get_embedding_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

