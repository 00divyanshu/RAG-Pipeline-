# Linux / Streamlit Community Cloud compatibility for ChromaDB SQLite
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import requests
import streamlit as st
from pathlib import Path
from src import config
from src.loader import load_documents_from_directory
from src.chunker import split_documents
from src.vectorstore import (
    get_embeddings,
    get_vector_store,
    index_documents,
    get_retriever,
)
from src.rag_chain import RAGPipeline

st.set_page_config(
    page_title="Local PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .citation-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin-top: 6px;
        border-radius: 4px;
        font-size: 0.88em;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #2e7d32;
        color: white;
        font-size: 0.75em;
        font-weight: bold;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

def is_ollama_reachable(base_url: str) -> bool:
    """Checks if the configured Ollama instance is online and responding."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

@st.cache_resource(show_spinner=False)
def load_rag_pipeline(base_url: str, llm_model: str, embedding_model: str, retriever_k: int):
    """Initializes and caches the RAG pipeline."""
    try:
        embeddings = get_embeddings(embedding_model, base_url)
        vector_store = get_vector_store(config.CHROMA_PERSIST_DIR, embeddings)
        retriever = get_retriever(vector_store, search_type="similarity", k=retriever_k)
        return RAGPipeline(
            retriever=retriever,
            llm_model=llm_model,
            base_url=base_url,
        )
    except Exception as e:
        return None

def main():
    st.title("📚 Local PDF RAG Assistant")
    st.caption("100% Local & Private | Powered by LangChain, Ollama & ChromaDB")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your local AI knowledge assistant. Ask me anything about the documents in your knowledge base, and I'll provide grounded answers with exact source citations.",
                "citations": []
            }
        ]

    # Sidebar: Document Management & Settings
    with st.sidebar:
        st.header("⚙️ Configuration")

        with st.expander("🌐 Server & Model Settings", expanded=False):
            ollama_url = st.text_input(
                "Ollama Base URL",
                value=config.OLLAMA_BASE_URL,
                help="Local: http://localhost:11434 | Remote/Cloud: ngrok or Cloudflare tunnel URL",
            )
            selected_llm = st.text_input("LLM Model", value=config.LLM_MODEL)
            selected_embed = st.text_input("Embedding Model", value=config.EMBEDDING_MODEL)
            selected_k = st.slider("Retriever Top-K", min_value=1, max_value=10, value=config.RETRIEVER_K)

        # Ollama connection indicator
        if is_ollama_reachable(ollama_url):
            st.success(f"🟢 Ollama Online (`{ollama_url}`)")
        else:
            st.warning(f"⚠️ Ollama Unreachable at `{ollama_url}`")
            st.caption(
                "💡 **Streamlit Cloud Note:** Ollama runs on your local machine. "
                "To connect from Streamlit Cloud, expose your local Ollama with `ngrok http 11434` or Cloudflare Tunnel, "
                "and set the public URL in Streamlit Secrets or in the box above."
            )

        st.divider()
        st.header("📄 Upload & Index PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to include in the knowledge base."
        )

        col1, col2 = st.columns(2)
        with col1:
            recreate_db = st.checkbox("Fresh Index", value=False, help="Wipe previous database and re-index from scratch.")
        with col2:
            process_btn = st.button("📥 Ingest", type="primary", use_container_width=True)

        if process_btn:
            config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
            if uploaded_files:
                with st.spinner("Saving uploaded files..."):
                    for uploaded_file in uploaded_files:
                        save_path = config.DOCS_DIR / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    st.success(f"Saved {len(uploaded_files)} file(s)!")

            with st.spinner("Indexing documents into ChromaDB..."):
                docs = load_documents_from_directory(config.DOCS_DIR)
                if not docs:
                    st.warning("No PDF documents found in data/docs to index.")
                else:
                    chunks = split_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                    embeddings = get_embeddings(selected_embed, ollama_url)
                    index_documents(
                        documents=chunks,
                        persist_directory=config.CHROMA_PERSIST_DIR,
                        embeddings=embeddings,
                        recreate=recreate_db,
                    )
                    st.cache_resource.clear()
                    st.success(f"Successfully indexed {len(docs)} page(s) into {len(chunks)} chunks!")
                    st.rerun()

        st.divider()
        st.subheader("📁 Indexed Documents")
        if config.DOCS_DIR.exists():
            existing_pdfs = list(config.DOCS_DIR.glob("*.pdf"))
            if existing_pdfs:
                for pdf in existing_pdfs:
                    st.markdown(f"- 📄 `{pdf.name}`")
            else:
                st.caption("No PDFs currently in data/docs.")
        else:
            st.caption("Directory data/docs does not exist yet.")

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I help you today?",
                    "citations": []
                }
            ]
            st.rerun()

    # Main Chat Area
    pipeline = load_rag_pipeline(ollama_url, selected_llm, selected_embed, selected_k)

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📌 View {len(msg['citations'])} Source Citations", expanded=False):
                    for i, cite in enumerate(msg["citations"], 1):
                        st.markdown(f"""
                        <div class="citation-card">
                            <span class="badge">Citation #{i}</span><br>
                            <strong>File:</strong> <code>{cite['filename']}</code> | <strong>Page:</strong> {cite['page']}<br>
                            <em>"{cite['snippet']}"</em>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input
    if user_query := st.chat_input("Ask a question about your PDF documents..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query, "citations": []})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant generation
        with st.chat_message("assistant"):
            if pipeline is None:
                err_msg = "Error connecting to vector database or Ollama. Please ensure Ollama is running and documents are indexed."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg, "citations": []})
            else:
                with st.spinner("Searching document context and generating response..."):
                    try:
                        result = pipeline.ask(user_query)
                        answer = result["answer"]
                        citations = result["citations"]

                        st.markdown(answer)

                        if citations:
                            with st.expander(f"📌 View {len(citations)} Source Citations", expanded=False):
                                for i, cite in enumerate(citations, 1):
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <span class="badge">Citation #{i}</span><br>
                                        <strong>File:</strong> <code>{cite['filename']}</code> | <strong>Page:</strong> {cite['page']}<br>
                                        <em>"{cite['snippet']}"</em>
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations
                        })
                    except Exception as ex:
                        err_text = f"An error occurred while generating the answer: {str(ex)}"
                        st.error(err_text)
                        st.session_state.messages.append({"role": "assistant", "content": err_text, "citations": []})

if __name__ == "__main__":
    main()

