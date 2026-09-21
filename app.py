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
from src.error_logger import (
    record_error,
    get_recent_errors,
    get_unacknowledged_count,
    acknowledge_all_errors,
    clear_all_errors,
)

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
        record_error(
            service="System Initialization",
            user_message="Failed initializing RAG pipeline or vector store",
            exception=e,
        )
        return None, None

def main():
    st.title("⚡ AI Knowledge Assistant")
    st.caption("Cloud RAG Pipeline | Grounded Answers with Instant Streaming Citations")

    # Check if this session is the Host Device
    is_host = config.is_host_session()

    # Host Device Alert Banner: Only displayed on the host device!
    if is_host:
        unack = get_unacknowledged_count()
        if unack > 0:
            st.warning(
                f"🚨 **Host System Alert**: {unack} service disconnection / error event(s) captured from user sessions. "
                "Review the **Host Diagnostics & Error Logs** panel in the sidebar."
            )

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

    # Sidebar: Document Management & Host Administration
    with st.sidebar:
        # =========================================================================
        # 👑 HOST DEVICE ONLY: Administration, Diagnostics, and Credential Manager
        # =========================================================================
        if is_host:
            st.markdown("### 👑 Host Administration Mode")
            
            with st.expander("🛠️ Host Diagnostics & Error Logs", expanded=get_unacknowledged_count() > 0):
                st.markdown("**Cloud Connectivity Status:**")
                provider_display = "Groq LPU" if config.LLM_PROVIDER == "groq" else "Google Gemini"
                st.write(f"- LLM Engine: `🟢 Active ({provider_display}: {config.LLM_MODEL})`")
                st.write(f"- Pinecone Cloud DB: `🟢 Connected ({config.PINECONE_INDEX_NAME})`")

                if st.button("🔄 Test Live Cloud Connections", use_container_width=True):
                    with st.spinner("Pinging Pinecone serverless index..."):
                        try:
                            from src.vectorstore import get_pinecone_index
                            idx = get_pinecone_index()
                            stats = idx.describe_index_stats()
                            total_v = stats.get('total_vector_count', 0)
                            st.success(f"✅ Pinecone OK! Live Vector Count: {total_v}")
                        except Exception as e:
                            st.error(f"❌ Pinecone Disconnection: {e}")
                            record_error("Pinecone", "Host test connection failed", exception=e)

                st.divider()
                # Display system error logs
                errors = get_recent_errors(limit=8)
                st.markdown(f"**Recorded System Errors ({len(errors)}):**")
                if errors:
                    for err in errors:
                        st.markdown(f"**[{err['timestamp']}] ⚠️ {err['service']}**: {err['message']}")
                        if err.get("technical_details"):
                            with st.expander(f"View Traceback (Error #{err['id']})"):
                                st.code(err["technical_details"], language="text")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Acknowledge All", use_container_width=True):
                            acknowledge_all_errors()
                            st.rerun()
                    with c2:
                        if st.button("Clear Log", use_container_width=True):
                            clear_all_errors()
                            st.rerun()
                else:
                    st.info("No system errors recorded. All systems operating normally!")

            with st.expander("🔑 Cloud API Credentials", expanded=False):
                st.caption("Host-only API key configuration:")
                st_groq_key = st.text_input(
                    "Groq API Key (High-Speed LLM)",
                    value=config.GROQ_API_KEY,
                    type="password",
                )
                st_google_key = st.text_input(
                    "Google Gemini API Key (Embeddings)",
                    value=config.GOOGLE_API_KEY,
                    type="password",
                )
                st_pinecone_key = st.text_input(
                    "Pinecone API Key",
                    value=config.PINECONE_API_KEY,
                    type="password",
                )
                st_index_name = st.text_input(
                    "Pinecone Index Name",
                    value=config.PINECONE_INDEX_NAME,
                )

                if st.button("💾 Save Credentials", use_container_width=True):
                    if (st_groq_key or st_google_key) and st_pinecone_key:
                        config.set_runtime_credentials(
                            google_api_key=st_google_key,
                            pinecone_api_key=st_pinecone_key,
                            index_name=st_index_name,
                            groq_api_key=st_groq_key,
                        )
                        st.cache_resource.clear()
                        st.toast("Credentials updated successfully!", icon="✅")
                        st.rerun()

            st.divider()

        # =========================================================================
        # 📄 DOCUMENT MANAGEMENT SECTION (Clean & simple for all users)
        # =========================================================================
        st.header("📄 Document Ingestion")
        st.caption("Upload PDF documents to expand your cloud knowledge base.")

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
            try:
                config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
                if uploaded_files:
                    with st.spinner("Saving uploaded files..."):
                        for uploaded_file in uploaded_files:
                            save_path = config.DOCS_DIR / uploaded_file.name
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        st.success(f"Saved {len(uploaded_files)} file(s)!")

                with st.spinner("Indexing documents into cloud database..."):
                    docs = load_documents_from_directory(config.DOCS_DIR)
                    if not docs:
                        st.warning("No PDF documents found to index. Please upload a file first.")
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
            except Exception as e:
                record_error(
                    service="Document Ingestion",
                    user_message="Document upload or indexing failed",
                    exception=e,
                )
                if is_host:
                    st.error(f"Host Diagnostic Error: {e}")
                else:
                    st.error("⚠️ Document processing is temporarily unavailable. Please try again shortly.")

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
                            try:
                                with st.spinner(f"Removing '{doc_name}'..."):
                                    delete_document_by_name(vector_store, doc_name, config.DOCS_DIR)
                                    st.cache_resource.clear()
                                st.toast(f"Removed '{doc_name}'!", icon="🗑️")
                                st.rerun()
                            except Exception as del_err:
                                record_error(
                                    service="Document Deletion",
                                    user_message=f"Failed removing '{doc_name}'",
                                    exception=del_err,
                                )
                                if is_host:
                                    st.error(f"Host Diagnostic Error: {del_err}")
                                else:
                                    st.error("⚠️ Unable to remove file at this moment. Please try again.")
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

        # Host Access for Remote Admin Login
        if not is_host:
            st.divider()
            with st.expander("🔒 Host Access", expanded=False):
                st.caption("Enter Admin PIN to unlock Host Administration Mode on this device.")
                entered_pin = st.text_input("Admin PIN", type="password", key="admin_pin_input")
                if st.button("Unlock Host Mode", key="btn_unlock_host", use_container_width=True):
                    if entered_pin == config.HOST_ADMIN_PIN:
                        st.session_state.is_host_authenticated = True
                        st.toast("Host Mode Unlocked!", icon="👑")
                        st.rerun()
                    else:
                        st.error("Incorrect Admin PIN.")
        else:
            if st.session_state.get("is_host_authenticated"):
                st.divider()
                if st.button("🔒 Exit Host Mode", key="btn_lock_host", use_container_width=True):
                    st.session_state.is_host_authenticated = False
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
            active_key = config.GROQ_API_KEY if config.LLM_PROVIDER == "groq" else config.GOOGLE_API_KEY
            if not active_key:
                provider_title = "Groq" if config.LLM_PROVIDER == "groq" else "Google Gemini"
                record_error(
                    service=f"{provider_title} AI",
                    user_message=f"{provider_title} API Key is missing during chat interaction",
                )
                if is_host:
                    err_msg = f"⚠️ {provider_title} API Key is missing. Please enter your API key in the Host Administration section or Streamlit Cloud Secrets."
                    st.error(err_msg)
                else:
                    err_msg = "⚠️ The AI assistant service is temporarily unavailable. Please try again shortly."
                    st.info(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg, "citations": []})
            elif pipeline is None:
                record_error(
                    service="RAG System",
                    user_message="Pipeline is uninitialized (connection or config failure)",
                )
                if is_host:
                    err_msg = "⚠️ Could not connect to AI service or vector database. Review Host Diagnostics in the sidebar."
                    st.error(err_msg)
                else:
                    err_msg = "⚠️ We encountered a temporary connection issue. Please try again shortly."
                    st.info(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg, "citations": []})
            else:
                try:
                    with st.spinner("Searching knowledge base..."):
                        context_str, citations, _ = pipeline.retrieve(user_query)

                    if not context_str:
                        indexed_docs = list_indexed_documents(vector_store)
                        if indexed_docs:
                            doc_list_str = ", ".join([f"`{d['filename']}`" for d in indexed_docs if d.get('chunks', 0) > 0] or [f"`{d['filename']}`" for d in indexed_docs])
                            fallback_text = (
                                f"I couldn't find specific sections for that query, but I have access to these documents in your cloud knowledge base: {doc_list_str}.\n\n"
                                "Try asking:\n"
                                "- *'Summarize the document'*\n"
                                "- *'What are the key points?'*\n"
                                "- Or ask about specific topics inside them!"
                            )
                        else:
                            fallback_text = "No document chunks found in your database. Please upload and ingest a PDF in the sidebar to start asking questions."
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
                except Exception as query_err:
                    record_error(
                        service="Query & Retrieval",
                        user_message=f"Query failed: {user_query[:60]}",
                        exception=query_err,
                    )
                    if is_host:
                        st.error(f"Host Diagnostic Error: {query_err}")
                    else:
                        st.info("⚠️ We encountered a temporary connection issue. Please try again shortly.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "⚠️ We encountered a temporary connection issue. Please try again shortly.",
                        "citations": []
                    })

if __name__ == "__main__":
    main()
