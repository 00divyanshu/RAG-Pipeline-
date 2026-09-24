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
from typing import Optional, List, Dict, Any

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
    get_pinecone_index,
)
from src.rag_chain import RAGPipeline
from src.error_logger import (
    record_error,
    get_recent_errors,
    get_unacknowledged_count,
    acknowledge_all_errors,
    clear_all_errors,
)
from src.db import (
    init_db,
    register_user,
    authenticate_user,
    create_chat_session,
    get_user_chat_sessions,
    update_chat_session_title,
    delete_chat_session,
    add_chat_message,
    get_session_messages,
    record_user_document,
    get_user_documents,
    delete_user_document,
    get_admin_platform_stats,
)

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database schema and pre-seeded admin
try:
    init_db()
except Exception as _e:
    pass

# Custom styling for high-end modern, responsive UI
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .citation-card {
        background-color: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #10b981;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 8px;
        font-size: 0.88em;
    }
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        font-size: 0.75em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .admin-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .admin-stat {
        font-size: 2.2em;
        font-weight: 800;
        color: #34d399;
    }
    .user-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_user_rag_pipeline(user_namespace: str):
    """Initializes and caches a RAG pipeline scoped to the user's isolated namespace."""
    try:
        embeddings = get_embeddings()
        vector_store = get_vector_store(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embeddings=embeddings,
            namespace=user_namespace,
        )
        retriever = get_retriever(vector_store, search_type="similarity", k=config.RETRIEVER_K)
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_model=config.LLM_MODEL,
        )
        return pipeline, vector_store
    except Exception as e:
        record_error(
            service="System Initialization",
            user_message=f"Failed initializing pipeline for namespace {user_namespace}",
            exception=e,
        )
        return None, None

def render_auth_portal():
    """Renders the sleek Sign In & Sign Up authentication screen."""
    st.markdown("<div style='text-align: center; margin-top: 1.5rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    st.title("⚡ AI Knowledge Assistant")
    st.caption("Enterprise Multi-Tenant RAG | Dedicated Document Isolation & Persistent Neural Memory")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab_login:
            st.markdown("#### Access Your Workspace")
            with st.form("form_login"):
                login_username = st.text_input("Username", placeholder="e.g. johndoe or @dmin")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In 🚀", type="primary", use_container_width=True)

                if submit_login:
                    if not login_username or not login_password:
                        st.error("Please enter both username and password.")
                    else:
                        user = authenticate_user(login_username, login_password)
                        if user:
                            st.session_state.user = user
                            st.toast(f"Welcome back, @{user['username']}! 👋", icon="🎉")
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Please verify username and password.")

            st.markdown(
                "<div style='font-size: 0.82em; color: #888; text-align: center; margin-top: 10px;'>"
                "👑 <strong>Master Admin Credentials:</strong> Username: <code>@dmin</code> | Password: <code>@dmin0812</code>"
                "</div>",
                unsafe_allow_html=True,
            )

        with tab_register:
            st.markdown("#### Create Your Personal Account")
            st.caption("Each account receives an isolated, zero-bleed vector database index.")
            with st.form("form_register"):
                reg_username = st.text_input("Choose Username", placeholder="At least 3 characters")
                reg_password = st.text_input("Choose Password", type="password", placeholder="At least 4 characters")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-type password")
                submit_register = st.form_submit_button("Create Account ✨", type="primary", use_container_width=True)

                if submit_register:
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match. Please re-enter.")
                    else:
                        ok, msg, new_user = register_user(reg_username, reg_password)
                        if ok:
                            st.session_state.user = new_user
                            # Auto-create initial conversation session
                            sid = create_chat_session(new_user["id"], "Initial Chat")
                            add_chat_message(
                                sid,
                                "assistant",
                                f"Welcome @{new_user['username']}! 🥑 I am your AI Knowledge Assistant. Upload any PDF in the sidebar to begin querying your documents.",
                                [],
                            )
                            st.session_state.current_session_id = sid
                            st.toast("Account created successfully! 🚀", icon="✨")
                            st.rerun()
                        else:
                            st.error(msg)

def render_admin_suite(user: Dict[str, Any]):
    """Renders the streamlined, professional Master Admin Control Suite."""
    st.markdown("## 👑 Master Admin Control Center")
    st.caption("Real-Time Platform Analytics, Multi-Tenant User Management & Live Cloud Health")

    stats = get_admin_platform_stats()

    # 1. Platform KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="admin-card">
            <div style="color: #94a3b8; font-size: 0.85em; text-transform: uppercase;">Total Users</div>
            <div class="admin-stat">{stats['total_users']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="admin-card">
            <div style="color: #94a3b8; font-size: 0.85em; text-transform: uppercase;">Indexed Documents</div>
            <div class="admin-stat">{stats['total_docs']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="admin-card">
            <div style="color: #94a3b8; font-size: 0.85em; text-transform: uppercase;">Chat Sessions</div>
            <div class="admin-stat">{stats['total_chats']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="admin-card">
            <div style="color: #94a3b8; font-size: 0.85em; text-transform: uppercase;">Messages Exchanged</div>
            <div class="admin-stat">{stats['total_messages']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_users, tab_health, tab_telemetry, tab_creds = st.tabs([
        "👥 User Directory",
        "🩺 Cloud Health Checks",
        "🚨 System Telemetry Logs",
        "🔑 Runtime Credentials",
    ])

    with tab_users:
        st.markdown("### Registered Users & Usage Breakdown")
        if stats["users_list"]:
            import pandas as pd
            df = pd.DataFrame(stats["users_list"])
            df.columns = ["User ID", "Username", "Role", "Created At", "Uploaded Docs", "Chat Sessions"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No registered users found.")

    with tab_health:
        st.markdown("### Live Cloud Service Connectivity Checks")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"- **Groq LPU Engine**: `🟢 Active ({config.GROQ_LLM_MODEL})`")
            st.markdown(f"- **Pinecone Vector Database**: `🟢 Connected (Index: {config.PINECONE_INDEX_NAME})`")
        with col_c2:
            st.markdown(f"- **Gemini Cloud Embeddings**: `🟢 Active ({config.GEMINI_EMBEDDING_MODEL})`")
            st.markdown("- **Neon Serverless PostgreSQL**: `🟢 Connected (SSL Encrypted)`")

        if st.button("🔄 Execute Live Ping Test Across Cloud Stack", use_container_width=True):
            with st.spinner("Pinging Pinecone Serverless and Neon Database..."):
                try:
                    idx = get_pinecone_index()
                    idx_stats = idx.describe_index_stats()
                    total_v = idx_stats.get("total_vector_count", 0)
                    namespaces_count = len(idx_stats.get("namespaces", {}))
                    st.success(f"✅ Pinecone Serverless OK! Total Vectors: {total_v} across {namespaces_count} isolated tenant namespace(s).")
                except Exception as pc_err:
                    st.error(f"❌ Pinecone Error: {pc_err}")

                try:
                    from src.db import get_connection
                    conn, eng = get_connection()
                    conn.close()
                    st.success(f"✅ Neon DB OK! Engine: {eng.upper()} connection active.")
                except Exception as db_err:
                    st.error(f"❌ Database Error: {db_err}")

    with tab_telemetry:
        st.markdown("### Error Telemetry & Exception Logs")
        errors = get_recent_errors(limit=12)
        if errors:
            for err in errors:
                st.markdown(f"**[{err['timestamp']}] ⚠️ {err['service']}**: {err['message']}")
                if err.get("technical_details"):
                    with st.expander(f"Stacktrace (Error #{err['id']})"):
                        st.code(err["technical_details"], language="text")

            c_ack, c_clr = st.columns(2)
            with c_ack:
                if st.button("Acknowledge All Events", use_container_width=True):
                    acknowledge_all_errors()
                    st.toast("All error events acknowledged.")
                    st.rerun()
            with c_clr:
                if st.button("Clear Error History", use_container_width=True):
                    clear_all_errors()
                    st.toast("Error history purged.")
                    st.rerun()
        else:
            st.success("All systems operating at 100% nominal performance. Zero errors recorded.")

    with tab_creds:
        st.markdown("### Cloud API Credentials & Configuration")
        st.caption("Update cloud runtime keys instantly across all workers.")
        st_groq_key = st.text_input("Groq API Key (High-Speed LLM)", value=config.GROQ_API_KEY, type="password")
        st_google_key = st.text_input("Google Gemini API Key (Embeddings)", value=config.GOOGLE_API_KEY, type="password")
        st_pinecone_key = st.text_input("Pinecone API Key", value=config.PINECONE_API_KEY, type="password")
        st_index_name = st.text_input("Pinecone Index Name", value=config.PINECONE_INDEX_NAME)

        if st.button("💾 Save Cloud Credentials", use_container_width=True, type="primary"):
            if (st_groq_key or st_google_key) and st_pinecone_key:
                config.set_runtime_credentials(
                    google_api_key=st_google_key,
                    pinecone_api_key=st_pinecone_key,
                    index_name=st_index_name,
                    groq_api_key=st_groq_key,
                )
                st.cache_resource.clear()
                st.toast("Credentials saved successfully!", icon="✅")
                st.rerun()

def main():
    # If user is not authenticated, render login/signup
    if "user" not in st.session_state or st.session_state.user is None:
        render_auth_portal()
        return

    user = st.session_state.user
    user_id = user["id"]
    username = user["username"]
    is_admin = (user.get("role") == "admin" or username == config.ADMIN_USERNAME)
    user_namespace = f"user_{username}"
    user_docs_dir = config.DOCS_DIR / user_namespace
    user_docs_dir.mkdir(parents=True, exist_ok=True)

    # Manage chat session selection
    user_sessions = get_user_chat_sessions(user_id)
    if not user_sessions:
        init_sid = create_chat_session(user_id, "Welcome Discussion")
        add_chat_message(
            init_sid,
            "assistant",
            f"Hello @{username}! 🥑 I am your AI Knowledge Assistant. Upload any PDF in the sidebar to ask questions with precise source citations.",
            [],
        )
        st.session_state.current_session_id = init_sid
        user_sessions = get_user_chat_sessions(user_id)
    elif "current_session_id" not in st.session_state or st.session_state.current_session_id not in [s["id"] for s in user_sessions]:
        st.session_state.current_session_id = user_sessions[0]["id"]

    active_session_id = st.session_state.current_session_id

    # Sidebar: User Profile, Navigation, Sessions, Ingestion
    with st.sidebar:
        # Profile badge
        role_tag = "👑 Master Admin" if is_admin else "⚡ Pro User"
        st.markdown(f"""
        <div class="user-pill">
            <span style="font-size: 1.4em;">🥑</span>
            <div style="flex-grow: 1;">
                <div style="font-weight: 700; font-size: 0.95em; color: #10b981;">@{username}</div>
                <div style="font-size: 0.72em; color: #888;">{role_tag}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.user = None
            st.session_state.current_session_id = None
            st.rerun()

        # Admin View Navigation Switcher
        active_view = "workspace"
        if is_admin:
            st.divider()
            nav_choice = st.radio(
                "Mode Selector",
                ["💬 Knowledge Assistant", "👑 Master Admin Suite"],
                index=0,
                label_visibility="collapsed",
            )
            if nav_choice == "👑 Master Admin Suite":
                active_view = "admin"

        st.divider()

        # ==========================================
        # 💬 Chat Sessions Drawer
        # ==========================================
        st.markdown("### 💬 Chat History")
        if st.button("➕ New Conversation", use_container_width=True, type="secondary"):
            new_sid = create_chat_session(user_id, "New Chat")
            add_chat_message(
                new_sid,
                "assistant",
                f"Fresh conversation started! How can I assist you with your documents, @{username}? 🥑",
                [],
            )
            st.session_state.current_session_id = new_sid
            st.rerun()

        # List user's sessions
        for s in user_sessions[:12]:
            c_sess, c_del = st.columns([5, 1])
            is_active = (s["id"] == active_session_id)
            title = s["title"]
            if len(title) > 22:
                title = title[:20] + "..."
            icon_prefix = "👉 " if is_active else "💬 "
            with c_sess:
                if st.button(f"{icon_prefix}{title}", key=f"session_btn_{s['id']}", use_container_width=True):
                    st.session_state.current_session_id = s["id"]
                    st.rerun()
            with c_del:
                if st.button("🗑️", key=f"session_del_{s['id']}", help="Delete chat thread"):
                    delete_chat_session(s["id"], user_id)
                    if st.session_state.current_session_id == s["id"]:
                        st.session_state.current_session_id = None
                    st.rerun()

        st.divider()

        # ==========================================
        # 📄 Document Ingestion Drawer (Multi-Tenant)
        # ==========================================
        st.markdown("### 📄 My Document Vault")
        st.caption(f"Tenant namespace: `{user_namespace}`")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="All files uploaded are strictly encrypted and isolated to your account.",
        )

        col_reset, col_ingest = st.columns([1, 1])
        with col_reset:
            reset_user_db = st.checkbox("Reset Index", value=False, help="Wipe only your namespace vectors.")
        with col_ingest:
            process_btn = st.button("📥 Ingest", type="primary", use_container_width=True)

        if process_btn:
            try:
                if uploaded_files:
                    with st.spinner("Saving uploaded PDF files..."):
                        for uploaded_file in uploaded_files:
                            save_path = user_docs_dir / uploaded_file.name
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                with st.spinner(f"Ingesting into cloud namespace '{user_namespace}'..."):
                    docs = load_documents_from_directory(user_docs_dir)
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
                            recreate=reset_user_db,
                            namespace=user_namespace,
                        )
                        # Record documents in user DB
                        file_chunk_map = {}
                        for c in chunks:
                            fn = c.metadata.get("filename", "unknown.pdf")
                            file_chunk_map[fn] = file_chunk_map.get(fn, 0) + 1
                        for fn, count in file_chunk_map.items():
                            record_user_document(user_id, fn, count)

                        st.cache_resource.clear()
                        st.success(f"Indexed {len(docs)} pages into {len(chunks)} chunks!")
                        st.rerun()
            except Exception as e:
                record_error(
                    service="Document Ingestion",
                    user_message="Document upload or indexing failed",
                    exception=e,
                )
                st.error(f"⚠️ Document processing error: {e}")

        # List user's ingested documents
        user_docs = get_user_documents(user_id)
        if user_docs:
            st.markdown("**Your Ingested Files:**")
            pipeline_inst, vector_store_inst = get_user_rag_pipeline(user_namespace)
            for d in user_docs:
                d_name = d["filename"]
                d_chunks = d["chunks"]
                c_info, c_rm = st.columns([4, 1])
                with c_info:
                    st.markdown(f"📄 **`{d_name}`**\n\n*{d_chunks} chunk(s)*")
                with c_rm:
                    if st.button("🗑️", key=f"doc_del_{d_name}", help=f"Remove '{d_name}'"):
                        try:
                            with st.spinner(f"Removing '{d_name}'..."):
                                delete_document_by_name(
                                    vector_store_inst,
                                    d_name,
                                    docs_dir=user_docs_dir,
                                    namespace=user_namespace,
                                )
                                delete_user_document(user_id, d_name)
                                st.cache_resource.clear()
                            st.toast(f"Removed '{d_name}'!", icon="🗑️")
                            st.rerun()
                        except Exception as del_err:
                            st.error(f"Error removing document: {del_err}")
                st.markdown("<hr style='margin: 4px 0 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)
        else:
            st.info("No documents currently in your vault. Upload a PDF above to get started.")

    # Main Area Rendering
    if active_view == "admin":
        render_admin_suite(user)
        return

    # Assistant Workspace View
    pipeline, vector_store = get_user_rag_pipeline(user_namespace)

    st.title("⚡ AI Knowledge Assistant")
    st.caption(f"Tenant: `@{username}` | Active Memory: `{user_namespace}` | Grounded Answers with Source Citations")

    # Render Active Session Chat Messages from Neon DB
    messages = get_session_messages(active_session_id)
    for msg in messages:
        avatar = "👤" if msg["role"] == "user" else "🥑"
        with st.chat_message(msg["role"], avatar=avatar):
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

    # Chat Input with Real-Time Streaming & Under-The-Hood Status
    if user_query := st.chat_input("Ask a question about your uploaded documents..."):
        # Store user query into Neon DB
        add_chat_message(active_session_id, "user", user_query)

        # Auto-update session title if it's the first user question
        current_sess_info = [s for s in user_sessions if s["id"] == active_session_id]
        if current_sess_info and current_sess_info[0]["title"] in ["New Chat", "New Conversation", "Welcome Discussion", "Initial Chat"]:
            new_title = user_query[:32] + ("..." if len(user_query) > 32 else "")
            update_chat_session_title(active_session_id, new_title)

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🥑"):
            active_key = config.GROQ_API_KEY if config.LLM_PROVIDER == "groq" else config.GOOGLE_API_KEY
            if not active_key:
                err_msg = "⚠️ AI API Key is unconfigured. Please configure API keys in Master Admin Suite or .env."
                st.error(err_msg)
                add_chat_message(active_session_id, "assistant", err_msg)
            elif pipeline is None:
                err_msg = "⚠️ Pipeline failed to initialize. Please check Pinecone and Neon DB connections."
                st.error(err_msg)
                add_chat_message(active_session_id, "assistant", err_msg)
            else:
                try:
                    # Dynamic Under-The-Hood Intelligence Status
                    with st.status("🔮 Under the Hood Intelligence Engine...", expanded=True) as status_box:
                        st.write(f"🔍 Searching vector space in tenant namespace `{user_namespace}`...")
                        context_str, citations, raw_docs = pipeline.retrieve(user_query)

                        if raw_docs:
                            st.write(f"📑 Reading & scoring {len(raw_docs)} context chunk(s) across your documents...")
                        else:
                            st.write("📑 Scanning knowledge base for relevant context...")

                        provider_name = "Groq LPU" if config.LLM_PROVIDER == "groq" else "Google Gemini"
                        st.write(f"🧠 Synthesizing grounded answer with {provider_name} ({config.LLM_MODEL})...")
                        status_box.update(label="🚀 Intelligence synthesized successfully!", state="complete", expanded=False)

                    if not context_str:
                        user_docs_list = get_user_documents(user_id)
                        if user_docs_list:
                            doc_names_str = ", ".join([f"`{d['filename']}`" for d in user_docs_list])
                            fallback_msg = (
                                f"I couldn't find specific passages for that question in your vault. "
                                f"You currently have these documents indexed: {doc_names_str}.\n\n"
                                "💡 **Suggested queries:**\n"
                                "- *'Summarize the main themes'* \n"
                                "- *'What are the key takeaways?'* \n"
                                "- Or ask about specific topics contained in your uploaded PDFs!"
                            )
                        else:
                            fallback_msg = "Your knowledge vault is currently empty. Upload and ingest a PDF in the sidebar to start asking questions!"
                        st.markdown(fallback_msg)
                        add_chat_message(active_session_id, "assistant", fallback_msg)
                    else:
                        # Stream response tokens live to user
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

                        # Save assistant message with citations to Neon DB
                        add_chat_message(active_session_id, "assistant", full_answer, citations=citations)

                except Exception as query_err:
                    record_error(
                        service="Query & Retrieval",
                        user_message=f"Query failed: {user_query[:60]}",
                        exception=query_err,
                    )
                    st.error(f"⚠️ Query processing error: {query_err}")
                    add_chat_message(active_session_id, "assistant", f"⚠️ Error processing query: {query_err}")

if __name__ == "__main__":
    main()
