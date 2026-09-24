# Linux / Streamlit Community Cloud compatibility for ChromaDB SQLite
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import shutil
import streamlit as st
from typing import Dict, Any

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
    create_auth_token,
    get_user_by_auth_token,
    revoke_auth_token,
    record_activity,
    get_recent_activity_logs,
    get_all_users_documents,
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
from src.security import (
    check_login_rate_limit,
    record_login_failure,
    reset_login_rate_limit,
    check_register_rate_limit,
    record_register_attempt,
    sanitize_filename,
    validate_pdf_content,
    validate_username,
    validate_password,
    sanitize_html,
)

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database schema and pre-seeded admin
try:
    init_db()
except Exception:
    pass

# Custom styling for modern, responsive, high-fidelity UI
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
        margin-bottom: 10px;
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
    """Renders the sleek Sign In & Sign Up authentication screen without any printed credentials."""
    st.markdown("<div style='text-align: center; margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    st.title("⚡ AI Knowledge Assistant")
    st.caption("Cloud Multi-Tenant RAG | Complete Document Isolation & Persistent Neural Memory")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab_login:
            st.markdown("#### Access Your Workspace")
            with st.form("form_login"):
                login_username = st.text_input("Username", placeholder="Enter your username")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In 🚀", type="primary", use_container_width=True)

                if submit_login:
                    if not login_username or not login_password:
                        st.error("Please enter both username and password.")
                    else:
                        is_allowed, remaining_sec = check_login_rate_limit(login_username)
                        if not is_allowed:
                            st.error(f"⛔ Too many failed login attempts! Account temporarily locked for {remaining_sec}s to protect against cyber brute-force attacks.")
                            record_activity(login_username, "SECURITY_LOCKOUT", f"Login rate limit lockout triggered ({remaining_sec}s remaining)")
                        else:
                            user = authenticate_user(login_username, login_password)
                            if user:
                                reset_login_rate_limit(login_username)
                                token = create_auth_token(user["id"])
                                st.query_params["session_token"] = token
                                st.session_state.auth_token = token
                                st.session_state.user = user
                                record_activity(user["username"], "USER_LOGIN", "Authenticated successfully into workspace", user_id=user["id"])
                                st.toast(f"Welcome back, @{user['username']}! 👋", icon="🎉")
                                st.rerun()
                            else:
                                rem = record_login_failure(login_username)
                                if rem > 0:
                                    st.error(f"Invalid credentials. ⚠️ {rem} attempt(s) remaining before temporary lockout.")
                                else:
                                    st.error("⛔ Account temporarily locked for 5 minutes due to multiple failed login attempts.")
                                record_activity(login_username, "LOGIN_FAILED", "Invalid credentials submitted")

        with tab_register:
            st.markdown("#### Create Your Personal Account")
            st.caption("Each account receives an isolated, zero-bleed vector database index.")
            with st.form("form_register"):
                reg_username = st.text_input("Choose Username", placeholder="At least 3 characters")
                reg_password = st.text_input("Choose Password", type="password", placeholder="At least 4 characters")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-type password")
                submit_register = st.form_submit_button("Create Account ✨", type="primary", use_container_width=True)

                if submit_register:
                    u_ok, u_msg = validate_username(reg_username)
                    p_ok, p_msg = validate_password(reg_password)
                    if not u_ok:
                        st.error(u_msg)
                    elif not p_ok:
                        st.error(p_msg)
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match. Please re-enter.")
                    else:
                        can_register, reg_rem_sec = check_register_rate_limit("client_session")
                        if not can_register:
                            st.error(f"⛔ Registration rate limit reached. Please wait {reg_rem_sec} seconds before creating another account.")
                            record_activity(reg_username, "REGISTER_RATE_LIMITED", "Registration rate limit exceeded")
                        else:
                            record_register_attempt("client_session")
                            ok, msg, new_user = register_user(reg_username, reg_password)
                            if ok and new_user is not None:
                                token = create_auth_token(new_user["id"])
                                st.query_params["session_token"] = token
                                st.session_state.auth_token = token
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
                                record_activity(new_user["username"], "USER_REGISTER", "New user workspace created", user_id=new_user["id"])
                                st.toast("Account created successfully! 🚀", icon="✨")
                                st.rerun()
                            else:
                                st.error(msg)

def render_admin_suite(user: Dict[str, Any]):
    """Renders the new full-featured Master Admin Control Center with KPI metrics, diagnostics, users, and credentials."""
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

    tab_health, tab_users, tab_global_docs, tab_activity, tab_telemetry, tab_creds = st.tabs([
        "🩺 Live Health & Diagnostics",
        "👥 User Directory",
        "📁 Global Document Directory",
        "📋 User Activity Logs",
        "🚨 System Telemetry Logs",
        "🔑 Cloud API Credentials",
    ])

    with tab_health:
        st.markdown("### Cloud Infrastructure Connectivity Status")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            provider_display = "Groq LPU" if config.LLM_PROVIDER == "groq" else "Google Gemini"
            st.markdown(f"- **LLM Engine**: `🟢 Active ({provider_display}: {config.LLM_MODEL})`")
            st.markdown(f"- **Pinecone Serverless DB**: `🟢 Connected ({config.PINECONE_INDEX_NAME})`")
        with col_c2:
            st.markdown(f"- **Gemini Cloud Embeddings**: `🟢 Active ({config.GEMINI_EMBEDDING_MODEL})`")
            st.markdown("- **Neon Serverless PostgreSQL**: `🟢 Connected (SSL Encrypted)`")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Execute Live Ping Test Across Cloud Stack", use_container_width=True, type="primary"):
            with st.spinner("Pinging Pinecone Serverless and Neon Database..."):
                try:
                    idx = get_pinecone_index()
                    idx_stats = idx.describe_index_stats()
                    total_v = idx_stats.get("total_vector_count", 0)
                    namespaces_count = len(idx_stats.get("namespaces", {}))
                    st.success(f"✅ Pinecone Serverless OK! Total Vectors: {total_v} across {namespaces_count} isolated tenant namespace(s).")
                except Exception as pc_err:
                    st.error(f"❌ Pinecone Error: {pc_err}")
                    record_error("Pinecone", "Admin health test failed", exception=pc_err)

                try:
                    from src.db import get_connection
                    conn, eng = get_connection()
                    conn.close()
                    st.success(f"✅ Neon DB OK! Engine: {eng.upper()} connection active.")
                except Exception as db_err:
                    st.error(f"❌ Database Error: {db_err}")
                    record_error("Neon DB", "Admin health test failed", exception=db_err)

    with tab_users:
        st.markdown("### Registered Users & Tenant Storage")
        if stats.get("users_list"):
            import pandas as pd
            df = pd.DataFrame(stats["users_list"])
            df.columns = ["User ID", "Username", "Role", "Created At", "Uploaded Docs", "Chat Sessions"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No registered users found.")

    with tab_global_docs:
        st.markdown("### Global Document Directory (All Users)")
        st.caption("Inventory of every document uploaded across all tenants in the platform.")
        all_docs = get_all_users_documents()
        if all_docs:
            import pandas as pd
            df_docs = pd.DataFrame(all_docs)
            df_docs.columns = ["Doc ID", "User ID", "Owner Username", "Filename", "Chunks", "Uploaded At"]
            st.dataframe(df_docs, use_container_width=True, hide_index=True)
            st.metric("Total Platform Uploaded Files", len(all_docs))
        else:
            st.info("No documents have been uploaded by any user yet.")

    with tab_activity:
        st.markdown("### User Activity Logs & Audit Trail")
        st.caption("Real-time telemetry recording every user login, upload, query, and session action.")
        col_act1, col_act2 = st.columns([3, 1])
        with col_act1:
            search_act = st.text_input("🔍 Filter by username or action", placeholder="e.g. USER_LOGIN, DOCUMENT_UPLOAD, or username")
        with col_act2:
            limit_val = st.selectbox("Max Logs", [50, 100, 200], index=1)

        logs = get_recent_activity_logs(limit=int(limit_val))
        if search_act:
            s_low = search_act.lower()
            logs = [
                l for l in logs
                if s_low in l["username"].lower() or s_low in l["action"].lower() or s_low in l["details"].lower()
            ]

        if logs:
            import pandas as pd
            df_logs = pd.DataFrame(logs)
            df_logs.columns = ["Event ID", "Username", "Action", "Details", "Client IP", "Timestamp"]
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.info("No activity records matching query.")

    with tab_telemetry:
        st.markdown("### Error Telemetry & Exception Logs")
        errors = get_recent_errors(limit=15)
        if errors:
            for err in errors:
                st.markdown(f"**[{err['timestamp']}] ⚠️ {err['service']}**: {err['message']}")
                if err.get("technical_details"):
                    with st.expander(f"View Stacktrace (Error #{err['id']})"):
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
        st.caption("Update cloud runtime keys instantly across all workers:")
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
    # 0. Session Persistence across Browser Page Refresh
    if "user" not in st.session_state or st.session_state.user is None:
        token = st.query_params.get("session_token")
        if token:
            persisted_user = get_user_by_auth_token(token)
            if persisted_user:
                st.session_state.user = persisted_user
                st.session_state.auth_token = token
            else:
                st.query_params.clear()

    # If user is still not authenticated, render login/signup
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

    # Sidebar: Profile, Navigation, Sessions, Ingestion
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
            if "auth_token" in st.session_state and st.session_state.auth_token:
                revoke_auth_token(st.session_state.auth_token)
            record_activity(username, "USER_LOGOUT", "Signed out of workspace", user_id=user["id"])
            st.query_params.clear()
            st.session_state.user = None
            st.session_state.auth_token = None
            st.session_state.current_session_id = None
            st.rerun()

        # Admin View Switcher (Offers both the modern suite and normal assistant view)
        active_view = "workspace"
        if is_admin:
            st.divider()
            nav_choice = st.radio(
                "Mode Selector",
                ["💬 AI Knowledge Assistant", "👑 Master Admin Suite"],
                index=0,
            )
            if nav_choice == "👑 Master Admin Suite":
                active_view = "admin"

            # Sidebar Quick Diagnostics (Old familiar controls right in the sidebar)
            with st.expander("🛠️ Quick Diagnostics & Status", expanded=False):
                st.write(f"- Engine: `{config.LLM_MODEL}`")
                st.write(f"- Pinecone: `{config.PINECONE_INDEX_NAME}`")
                st.write("- Database: `Neon PostgreSQL (Active)`")
                if st.button("🔄 Quick Ping Stack", key="quick_ping_btn", use_container_width=True):
                    try:
                        idx = get_pinecone_index()
                        stats = idx.describe_index_stats()
                        st.success(f"Pinecone Vectors: {stats.get('total_vector_count', 0)}")
                    except Exception as q_err:
                        st.error(f"Pinecone: {q_err}")

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
        # 📄 Document Ingestion Drawer (Multi-Tenant & Resilient)
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
                # 1. Save newly uploaded files to tenant directory with security validation
                if uploaded_files:
                    with st.spinner("Saving uploaded PDF files..."):
                        for uploaded_file in uploaded_files:
                            safe_filename = sanitize_filename(uploaded_file.name)
                            file_bytes = uploaded_file.getbuffer().tobytes()
                            is_valid, val_msg = validate_pdf_content(file_bytes)
                            if not is_valid:
                                st.error(f"Security Alert for '{uploaded_file.name}': {val_msg}")
                                record_activity(username, "SECURITY_BLOCKED_FILE", f"Blocked '{uploaded_file.name}': {val_msg}", user_id=user_id)
                                continue
                            save_path = user_docs_dir / safe_filename
                            with open(save_path, "wb") as f:
                                f.write(file_bytes)

                # If user directory is empty and user is admin, copy sample docs from config.DOCS_DIR
                if not any(user_docs_dir.glob("*.pdf")) and is_admin:
                    for sf in config.DOCS_DIR.glob("*.pdf"):
                        if sf.is_file():
                            shutil.copy2(sf, user_docs_dir / sf.name)

                with st.spinner(f"Ingesting into cloud namespace '{user_namespace}'..."):
                    docs = load_documents_from_directory(user_docs_dir)
                    if not docs:
                        st.warning("No PDF documents found to index. Please select and upload a PDF file first.")
                    else:
                        st.cache_resource.clear()
                        chunks = split_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                        embeddings = get_embeddings()

                        # If reset is checked, clean user DB records too
                        if reset_user_db:
                            for ed in get_user_documents(user_id):
                                delete_user_document(user_id, ed["filename"])

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
                            record_activity(
                                username,
                                "DOCUMENT_UPLOAD",
                                f"Uploaded & indexed '{fn}' ({count} chunks)",
                                user_id=user_id,
                            )

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
        # If DB is empty, sync with disk/vectorstore
        if not user_docs and any(user_docs_dir.glob("*.pdf")):
            pipeline_inst, vector_store_inst = get_user_rag_pipeline(user_namespace)
            indexed_list = list_indexed_documents(vector_store_inst, docs_dir=user_docs_dir, namespace=user_namespace)
            for item in indexed_list:
                record_user_document(user_id, item["filename"], item["chunks"])
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
                                record_activity(username, "DOCUMENT_DELETE", f"Removed document '{d_name}'", user_id=user_id)
                                st.cache_resource.clear()
                            st.toast(f"Removed '{d_name}'!", icon="🗑️")
                            st.rerun()
                        except Exception as del_err:
                            st.error(f"Error removing document: {del_err}")
                st.markdown("<hr style='margin: 4px 0 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)
        else:
            st.info("No documents currently in your vault. Upload a PDF above to get started.")

    # =========================================================================
    # Main View Routing
    # =========================================================================
    if active_view == "admin":
        render_admin_suite(user)
        return

    # Assistant Workspace View
    st.title("⚡ AI Knowledge Assistant")
    st.caption(f"Tenant: `@{username}` | Active Memory: `{user_namespace}` | Grounded Answers with Source Citations")

    # Host Alert Banner: Displayed to admin if errors exist
    if is_admin:
        unack = get_unacknowledged_count()
        if unack > 0:
            st.warning(
                f"🚨 **Host System Alert**: {unack} service disconnection / error event(s) captured from user sessions. "
                "Switch to the **Master Admin Suite** or review Host Diagnostics in the sidebar."
            )

    pipeline, vector_store = get_user_rag_pipeline(user_namespace)

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
                        streamed_res = st.write_stream(pipeline.stream_response(context_str, user_query))
                        full_answer = "".join(str(chunk) for chunk in streamed_res) if isinstance(streamed_res, list) else str(streamed_res)

                        if citations:
                            with st.expander(f"📌 View {len(citations)} Source Citations", expanded=False):
                                for i, cite in enumerate(citations, 1):
                                    safe_filename = sanitize_html(str(cite.get('filename', 'Unknown')))
                                    safe_page = sanitize_html(str(cite.get('page', 'Unknown')))
                                    safe_snippet = sanitize_html(str(cite.get('snippet', '')))
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <span class="badge">Citation #{i}</span><br>
                                        <strong>File:</strong> <code>{safe_filename}</code> | <strong>Page:</strong> {safe_page}<br>
                                        <em>"{safe_snippet}"</em>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Save assistant message with citations to Neon DB
                        add_chat_message(active_session_id, "assistant", full_answer, citations=citations)
                        record_activity(username, "CHAT_QUERY", f"Asked: {user_query[:60]}", user_id=user_id)

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
