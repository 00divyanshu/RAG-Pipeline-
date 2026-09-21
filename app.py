# Linux / Streamlit Community Cloud compatibility for ChromaDB SQLite
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

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
    delete_document_by_name,
    list_indexed_documents,
)
from src.rag_chain import RAGPipeline

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for a modern, responsive, mobile-ready UI
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .citation-card {
        background-color: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #4CAF50;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 6px;
        font-size: 0.88em;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        background-color: #2e7d32;
        color: white;
        font-size: 0.75em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .doc-item {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Initializes and caches the RAG pipeline and vector store."""
    try:
        embeddings = get_embeddings()
        vector_store = get_vector_store(config.CHROMA_PERSIST_DIR, embeddings)
        retriever = get_retriever(vector_store, search_type="similarity", k=config.RETRIEVER_K)
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_model=config.LLM_MODEL,
        )
        return pipeline, vector_store
    except Exception as e:
        st.error(f"Initialization notice: {str(e)}")
        return None, None

def main():
    st.title("⚡ AI Knowledge Assistant")
    st.caption("Cloud RAG Pipeline | Grounded Answers with Instant Streaming Citations")

    pipeline, vector_store = load_rag_system()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI knowledge assistant. Ask me anything about the documents in your knowledge base, and I will provide fast, grounded answers with exact source citations.",
                "citations": []
            }
        ]

    # Sidebar: Document Management (System configuration is locked and hidden)
    with st.sidebar:
        st.header("📄 Document Ingestion")
        st.caption("Upload PDF documents to expand your knowledge base.")

        uploaded_files = st.file_uploader(
            "Select PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to index into the knowledge base."
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            recreate_db = st.checkbox("Reset Index", value=False, help="Wipe previous index and rebuild from scratch.")
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

            with st.spinner("Indexing documents into database..."):
                docs = load_documents_from_directory(config.DOCS_DIR)
                if not docs:
                    st.warning("No PDF documents found in data/docs to index.")
                else:
                    st.cache_resource.clear()
                    chunks = split_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                    embeddings = get_embeddings()
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
        st.header("📁 Ingested Documents")
        st.caption("Manage existing documents in your knowledge base.")

        # Display ingested documents with single-file removal option
        indexed_docs = list_indexed_documents(vector_store, config.DOCS_DIR)
        if indexed_docs:
            for doc_info in indexed_docs:
                doc_name = doc_info["filename"]
                chunks_count = doc_info["chunks"]
                
                with st.container():
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"📄 **`{doc_name}`**\n\n*{chunks_count} chunk(s)*")
                    with c2:
                        if st.button("🗑️", key=f"del_{doc_name}", help=f"Remove '{doc_name}' from database"):
                            with st.spinner(f"Removing '{doc_name}'..."):
                                delete_document_by_name(vector_store, doc_name, config.DOCS_DIR)
                                st.cache_resource.clear()
                            st.toast(f"Removed '{doc_name}'!", icon="🗑️")
                            st.rerun()
                    st.markdown("<hr style='margin: 4px 0 10px 0; border: none; border-top: 1px solid rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)
        else:
            st.info("No documents currently in your knowledge base. Upload a PDF above to get started.")

        st.divider()
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I help you today?",
                    "citations": []
                }
            ]
            st.rerun()

    # Main Chat Area
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

    # Chat Input with Real-Time Streaming
    if user_query := st.chat_input("Ask a question about your documents..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query, "citations": []})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant generation with token streaming for ultra-fast response
        with st.chat_message("assistant"):
            if pipeline is None:
                err_msg = "Could not connect to AI service or vector database. Please check your credentials."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg, "citations": []})
            else:
                with st.spinner("Searching knowledge base..."):
                    context_str, citations, _ = pipeline.retrieve(user_query)

                if not context_str:
                    fallback_text = "I cannot find the answer to that in the provided documents."
                    st.markdown(fallback_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": fallback_text,
                        "citations": []
                    })
                else:
                    # Stream tokens in real-time
                    full_answer = st.write_stream(pipeline.stream_response(context_str, user_query))

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
                        "content": full_answer,
                        "citations": citations
                    })

if __name__ == "__main__":
    main()
