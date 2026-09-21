import os
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

@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    """Initializes and caches the RAG pipeline."""
    try:
        embeddings = get_embeddings(config.EMBEDDING_MODEL, config.OLLAMA_BASE_URL)
        vector_store = get_vector_store(config.CHROMA_PERSIST_DIR, embeddings)
        retriever = get_retriever(vector_store, search_type="similarity", k=config.RETRIEVER_K)
        return RAGPipeline(
            retriever=retriever,
            llm_model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
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
        st.info(f"**LLM:** `{config.LLM_MODEL}`\n\n"
                f"**Embeddings:** `{config.EMBEDDING_MODEL}`\n\n"
                f"**Retriever Top-K:** `{config.RETRIEVER_K}`")

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
                    embeddings = get_embeddings(config.EMBEDDING_MODEL, config.OLLAMA_BASE_URL)
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
    pipeline = load_rag_pipeline()

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

