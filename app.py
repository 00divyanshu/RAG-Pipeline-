# Linux / Streamlit Community Cloud compatibility for ChromaDB SQLite
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st

from src import config
from src.loader import load_documents_from_directory, SUPPORTED_EXTENSIONS
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
    validate_document_content,
    validate_username,
    validate_password,
    sanitize_html,
)

# Page configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database schema and pre-seeded admin
try:
    init_db()
except Exception:
    pass

# Initialize theme and authentication session state
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "dark"
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "guest_messages" not in st.session_state:
    st.session_state.guest_messages = []
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

def apply_app_theme(theme_choice: str):
    """
    Applies dynamic Dark, Light, or Device/System CSS styling across all components
    using native st.html to prevent raw markdown code block leakage.
    Ensures text inversion (solid black in Light mode, crisp white in Dark mode)
    and removes GitHub codebase exposure while preserving sidebar toggle controls.
    """
    if theme_choice == "light":
        css_payload = """
<style>
/* Privacy: Hide GitHub link, deploy button, hamburger menu, and status widget */
#MainMenu, 
.stDeployButton, 
div[data-testid="stStatusWidget"],
div[data-testid="stToolbarActions"],
a[href*="github.com"],
footer { 
    display: none !important; 
    visibility: hidden !important; 
}
header[data-testid="stHeader"] { 
    background: transparent !important; 
    pointer-events: none !important;
}
div[data-testid="stToolbar"] { 
    background: transparent !important; 
    pointer-events: none !important;
}
button[data-testid="stExpandSidebarButton"] {
    display: flex !important;
    visibility: visible !important;
    pointer-events: auto !important;
    z-index: 99999 !important;
    position: fixed !important;
    top: 50px !important;
    left: 8px !important;
    width: 38px !important;
    height: 38px !important;
    cursor: pointer !important;
    opacity: 0 !important;
}
div[data-testid="stSidebarHeader"] {
    position: relative !important;
    display: flex !important;
    justify-content: flex-end !important;
    padding: 12px 14px 0 14px !important;
    min-height: 40px !important;
}
button[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    cursor: pointer !important;
    color: #1f1f1f !important;
    border-radius: 8px !important;
}
button[data-testid="stSidebarCollapseButton"]:hover {
    background: #e0e4eb !important;
}

/* Collapsed Sidebar Rail - Light Mode */
.collapsed-rail {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 52px;
    background: #f0f4f9;
    border-right: 1px solid #dfe3e7;
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 16px 0;
    z-index: 99990;
    box-sizing: border-box;
    user-select: none;
}
.collapsed-rail .rail-top, .collapsed-rail .rail-bottom {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}
.collapsed-rail .rail-item {
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    cursor: pointer;
    color: #1f1f1f;
    margin-bottom: 8px;
    transition: background 0.15s ease, transform 0.1s ease;
}
.collapsed-rail .rail-item:hover {
    background: #dfe3e7;
    color: #000000;
}
.collapsed-rail .rail-logo {
    font-size: 1.4rem;
    cursor: pointer;
    margin-bottom: 12px;
}
.collapsed-rail .rail-free-space {
    flex-grow: 1;
    width: 100%;
    cursor: pointer;
}
.collapsed-rail .rail-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #059669, #10b981);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.85rem;
}


/* Light Theme Variables */
:root {
    --app-bg: radial-gradient(circle at 50% 30%, #eef4ff 0%, #f4f7fc 50%, #ffffff 100%);
    --app-surface: #ffffff;
    --app-sidebar: #f0f4f9;
    --app-text: #1f1f1f;
    --app-text-muted: #444746;
    --app-border: #dfe3e7;
    --app-pill: #e9eef6;
    --app-accent: #059669;
}
.stApp {
    background: var(--app-bg) !important;
    color: #1f1f1f !important;
}
.stApp * {
    color: #1f1f1f !important;
}
section[data-testid="stSidebar"] {
    background-color: #f0f4f9 !important;
    border-right: 1px solid #dfe3e7 !important;
}
section[data-testid="stSidebar"] * {
    color: #1f1f1f !important;
}
.stChatMessage {
    background-color: #ffffff !important;
    border: 1px solid #dfe3e7 !important;
    border-radius: 18px;
    color: #1f1f1f !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stChatMessage * {
    color: #1f1f1f !important;
}
.citation-card {
    background-color: #f1f5f9 !important;
    border-left: 3px solid #059669 !important;
    color: #1e293b !important;
    padding: 10px 14px;
    margin-top: 8px;
    border-radius: 8px;
    font-size: 0.88em;
}
.citation-card * {
    color: #1e293b !important;
}
.hero-title {
    color: #1f1f1f !important;
    font-weight: 500;
    font-size: 2.3rem;
    letter-spacing: -0.02em;
    margin-top: 0.5rem;
    margin-bottom: 0.4rem;
}
.hero-sub {
    color: #444746 !important;
    font-size: 1.05rem;
}
.suggestion-item {
    background: #ffffff !important;
    border: 1px solid #e0e4eb !important;
    border-radius: 16px;
    padding: 12px 18px;
    margin-bottom: 8px;
    color: #1f1f1f !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.suggestion-item * {
    color: #1f1f1f !important;
}
.user-profile-card {
    background: #ffffff;
    border: 1px solid #dfe3e7;
    border-radius: 24px;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.user-profile-card * {
    color: #1f1f1f !important;
}
div[data-baseweb="input"] input {
    color: #1f1f1f !important;
    background-color: #ffffff !important;
}
.badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 12px;
    background: linear-gradient(135deg, #059669, #10b981);
    color: white !important;
    font-size: 0.75em;
    font-weight: 600;
}
.admin-card {
    background: #ffffff;
    border: 1px solid #dfe3e7;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    margin-bottom: 10px;
}
.admin-stat {
    font-size: 2.2em;
    font-weight: 800;
    color: #059669;
}
.avatar-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #059669, #10b981);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85em;
    font-weight: 700;
    color: white !important;
    flex-shrink: 0;
}
.hero-container {
    text-align: center;
    margin: 3.5rem auto 2rem auto;
    max-width: 680px;
}
</style>
"""
    elif theme_choice == "dark":
        css_payload = """
<style>
/* Privacy: Hide GitHub link, deploy button, hamburger menu, and status widget */
#MainMenu, 
.stDeployButton, 
div[data-testid="stStatusWidget"],
div[data-testid="stToolbarActions"],
a[href*="github.com"],
footer { 
    display: none !important; 
    visibility: hidden !important; 
}
header[data-testid="stHeader"] { 
    background: transparent !important; 
    pointer-events: none !important;
}
div[data-testid="stToolbar"] { 
    background: transparent !important; 
    pointer-events: none !important;
}
button[data-testid="stExpandSidebarButton"] {
    display: flex !important;
    visibility: visible !important;
    pointer-events: auto !important;
    z-index: 99999 !important;
    position: fixed !important;
    top: 50px !important;
    left: 8px !important;
    width: 38px !important;
    height: 38px !important;
    cursor: pointer !important;
    opacity: 0 !important;
}
div[data-testid="stSidebarHeader"] {
    position: relative !important;
    display: flex !important;
    justify-content: flex-end !important;
    padding: 12px 14px 0 14px !important;
    min-height: 40px !important;
}
button[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    cursor: pointer !important;
    color: #f0f4f9 !important;
    border-radius: 8px !important;
}
button[data-testid="stSidebarCollapseButton"]:hover {
    background: #282a2c !important;
}

/* Collapsed Sidebar Rail - Dark Mode */
.collapsed-rail {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 52px;
    background: #131314;
    border-right: 1px solid #282a2c;
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 16px 0;
    z-index: 99990;
    box-sizing: border-box;
    user-select: none;
}
.collapsed-rail .rail-top, .collapsed-rail .rail-bottom {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}
.collapsed-rail .rail-item {
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    cursor: pointer;
    color: #f0f4f9;
    margin-bottom: 8px;
    transition: background 0.15s ease, transform 0.1s ease;
}
.collapsed-rail .rail-item:hover {
    background: #282a2c;
    color: #ffffff;
}
.collapsed-rail .rail-logo {
    font-size: 1.4rem;
    cursor: pointer;
    margin-bottom: 12px;
}
.collapsed-rail .rail-free-space {
    flex-grow: 1;
    width: 100%;
    cursor: pointer;
}
.collapsed-rail .rail-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #059669, #10b981);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.85rem;
}


/* Dark Theme Variables */
:root {
    --app-bg: radial-gradient(circle at 50% 30%, #17243c 0%, #0d121c 55%, #07090e 100%);
    --app-surface: #1e1f20;
    --app-sidebar: #131314;
    --app-text: #f0f4f9;
    --app-text-muted: #c4c7c5;
    --app-border: #37393b;
    --app-pill: #282a2c;
    --app-accent: #10b981;
}
.stApp {
    background: var(--app-bg) !important;
    color: #f0f4f9 !important;
}
section[data-testid="stSidebar"] {
    background-color: #131314 !important;
    border-right: 1px solid #282a2c !important;
}
.stChatMessage {
    background-color: #1e1f20 !important;
    border: 1px solid #282a2c !important;
    border-radius: 18px;
    color: #f0f4f9 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
.citation-card {
    background-color: rgba(255, 255, 255, 0.04) !important;
    border-left: 3px solid #10b981 !important;
    color: #e2e8f0 !important;
    padding: 10px 14px;
    margin-top: 8px;
    border-radius: 8px;
    font-size: 0.88em;
}
.hero-title {
    color: #f0f4f9 !important;
    font-weight: 500;
    font-size: 2.3rem;
    letter-spacing: -0.02em;
    margin-top: 0.5rem;
    margin-bottom: 0.4rem;
}
.hero-sub {
    color: #c4c7c5 !important;
    font-size: 1.05rem;
}
.suggestion-item {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px;
    padding: 12px 18px;
    margin-bottom: 8px;
    color: #e3e3e3 !important;
}
.user-profile-card {
    background: #1e1f20;
    border: 1px solid #37393b;
    border-radius: 24px;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 12px;
    background: linear-gradient(135deg, #059669, #10b981);
    color: white !important;
    font-size: 0.75em;
    font-weight: 600;
}
.admin-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    margin-bottom: 10px;
}
.admin-stat {
    font-size: 2.2em;
    font-weight: 800;
    color: #34d399;
}
.avatar-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #059669, #10b981);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85em;
    font-weight: 700;
    color: white !important;
    flex-shrink: 0;
}
.hero-container {
    text-align: center;
    margin: 3.5rem auto 2rem auto;
    max-width: 680px;
}
</style>
"""
    else:
        # Device / System theme
        css_payload = """
<style>
/* Privacy: Hide GitHub link, deploy button, hamburger menu, and status widget */

#MainMenu, 
.stDeployButton, 
div[data-testid="stStatusWidget"],
div[data-testid="stToolbarActions"],
a[href*="github.com"],
footer { 
    display: none !important; 
    visibility: hidden !important; 
}
header[data-testid="stHeader"] { 
    background: transparent !important; 
    pointer-events: none !important;
}
div[data-testid="stToolbar"] { 
    background: transparent !important; 
    pointer-events: none !important;
}
button[data-testid="stExpandSidebarButton"] {
    display: flex !important;
    visibility: visible !important;
    pointer-events: auto !important;
    z-index: 99999 !important;
    position: fixed !important;
    top: 50px !important;
    left: 8px !important;
    width: 38px !important;
    height: 38px !important;
    cursor: pointer !important;
    opacity: 0 !important;
}
div[data-testid="stSidebarHeader"] {
    position: relative !important;
    display: flex !important;
    justify-content: flex-end !important;
    padding: 12px 14px 0 14px !important;
    min-height: 40px !important;
}
button[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    cursor: pointer !important;
    border-radius: 8px !important;
}

/* Collapsed Sidebar Rail Base */
.collapsed-rail {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 52px;
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 16px 0;
    z-index: 99990;
    box-sizing: border-box;
    user-select: none;
}
.collapsed-rail .rail-top, .collapsed-rail .rail-bottom {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}
.collapsed-rail .rail-item {
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    cursor: pointer;
    margin-bottom: 8px;
    transition: background 0.15s ease, transform 0.1s ease;
}
.collapsed-rail .rail-logo {
    font-size: 1.4rem;
    cursor: pointer;
    margin-bottom: 12px;
}
.collapsed-rail .rail-free-space {
    flex-grow: 1;
    width: 100%;
    cursor: pointer;
}
.collapsed-rail .rail-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #059669, #10b981);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.85rem;
}


@media (prefers-color-scheme: light) {
    .stApp { background: radial-gradient(circle at 50% 30%, #eef4ff 0%, #f4f7fc 50%, #ffffff 100%) !important; color: #1f1f1f !important; }
    .stApp * { color: #1f1f1f !important; }
    section[data-testid="stSidebar"] { background-color: #f0f4f9 !important; border-right: 1px solid #dfe3e7 !important; }
    section[data-testid="stSidebar"] * { color: #1f1f1f !important; }
    .stChatMessage { background-color: #ffffff !important; border: 1px solid #dfe3e7 !important; color: #1f1f1f !important; }
    .stChatMessage * { color: #1f1f1f !important; }
    .citation-card { background-color: #f1f5f9 !important; border-left: 3px solid #059669 !important; color: #1e293b !important; }
    .citation-card * { color: #1e293b !important; }
    .hero-title { color: #1f1f1f !important; }
    .hero-sub { color: #444746 !important; }
    .collapsed-rail { background: #f0f4f9; border-right: 1px solid #dfe3e7; }
    .collapsed-rail .rail-item { color: #1f1f1f; }
    .collapsed-rail .rail-item:hover { background: #dfe3e7; color: #000000; }
    button[data-testid="stSidebarCollapseButton"] { color: #1f1f1f !important; }
    button[data-testid="stSidebarCollapseButton"]:hover { background: #e0e4eb !important; }
}
@media (prefers-color-scheme: dark) {
    .stApp { background: radial-gradient(circle at 50% 30%, #17243c 0%, #0d121c 55%, #07090e 100%) !important; color: #f0f4f9 !important; }
    section[data-testid="stSidebar"] { background-color: #131314 !important; border-right: 1px solid #282a2c !important; }
    .stChatMessage { background-color: #1e1f20 !important; border: 1px solid #282a2c !important; color: #f0f4f9 !important; }
    .citation-card { background-color: rgba(255, 255, 255, 0.04) !important; border-left: 3px solid #10b981 !important; color: #e2e8f0 !important; }
    .hero-title { color: #f0f4f9 !important; }
    .hero-sub { color: #c4c7c5 !important; }
    .collapsed-rail { background: #131314; border-right: 1px solid #282a2c; }
    .collapsed-rail .rail-item { color: #f0f4f9; }
    .collapsed-rail .rail-item:hover { background: #282a2c; color: #ffffff; }
    button[data-testid="stSidebarCollapseButton"] { color: #f0f4f9 !important; }
    button[data-testid="stSidebarCollapseButton"]:hover { background: #282a2c !important; }
}

.badge { display: inline-block; padding: 3px 9px; border-radius: 12px; background: linear-gradient(135deg, #059669, #10b981); color: white !important; font-size: 0.75em; font-weight: 600; }
.admin-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 16px; text-align: center; margin-bottom: 10px; }
.admin-stat { font-size: 2.2em; font-weight: 800; color: #34d399; }
.avatar-circle { width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #059669, #10b981); display: flex; align-items: center; justify-content: center; font-size: 0.85em; font-weight: 700; color: white !important; flex-shrink: 0; }
.hero-container { text-align: center; margin: 3.5rem auto 2rem auto; max-width: 680px; }
</style>
"""

    st.html(css_payload)

apply_app_theme(st.session_state.app_theme)

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

@st.dialog("🔐 Sign In / Create Account")
def show_auth_modal(initial_tab: int = 0):
    """Displays in-page modal dialog for Sign In and Account Creation without screen disruption."""
    tab_login, tab_register = st.tabs(["🔐 Sign In", "✨ Create Account"])

    with tab_login:
        st.markdown("#### Access Your Workspace")
        st.caption("Sign in to query private documents, save conversation history, and access your vault.")
        with st.form("modal_login_form"):
            login_username = st.text_input("Username", placeholder="Enter username")
            login_password = st.text_input("Password", type="password", placeholder="Enter password")
            submit_login = st.form_submit_button("Sign In 🚀", type="primary", use_container_width=True)

            if submit_login:
                if not login_username or not login_password:
                    st.error("Please enter both username and password.")
                else:
                    is_allowed, remaining_sec = check_login_rate_limit(login_username)
                    if not is_allowed:
                        st.error(f"⛔ Too many failed login attempts! Account temporarily locked for {remaining_sec}s.")
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
        with st.form("modal_register_form"):
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
                            sid = create_chat_session(new_user["id"], "Initial Chat")
                            add_chat_message(
                                sid,
                                "assistant",
                                f"Welcome @{new_user['username']}! 🥑 I am your AI Assistant. Attach documents below to begin querying your data.",
                                [],
                            )
                            st.session_state.current_session_id = sid
                            record_activity(new_user["username"], "USER_REGISTER", "New user workspace created", user_id=new_user["id"])
                            st.toast("Account created successfully! 🚀", icon="✨")
                            st.rerun()
                        else:
                            st.error(msg)

def render_admin_suite(user: Dict[str, Any]):
    """Renders the full-featured Master Admin Control Center with KPI metrics, diagnostics, users, and credentials."""
    st.markdown("## 👑 Master Admin Control Center")
    st.caption("Real-Time Platform Analytics, Multi-Tenant User Management & Live Cloud Health")

    stats = get_admin_platform_stats()

    # 1. Platform KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="admin-card">
            <div style="font-size: 0.85em; text-transform: uppercase;">Total Users</div>
            <div class="admin-stat">{stats['total_users']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="admin-card">
            <div style="font-size: 0.85em; text-transform: uppercase;">Indexed Documents</div>
            <div class="admin-stat">{stats['total_docs']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="admin-card">
            <div style="font-size: 0.85em; text-transform: uppercase;">Chat Sessions</div>
            <div class="admin-stat">{stats['total_chats']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="admin-card">
            <div style="font-size: 0.85em; text-transform: uppercase;">Messages Exchanged</div>
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

def get_format_icon(filename: str) -> str:
    """Returns a visual format emoji for document types."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "📕"
    elif ext == ".docx":
        return "📘"
    elif ext == ".csv":
        return "📊"
    elif ext in [".txt", ".md"]:
        return "📄"
    return "📁"

def render_collapsed_sidebar_rail(username: str, role_label: str, initial_letter: str):
    """
    Renders the persistent collapsed icon rail on the left side of the screen.
    Includes Avocado logo, toggle button, new chat, search, document vault, settings,
    and profile avatar. Clicking ANYWHERE in the free space of the vertical rail line
    expands the sidebar.
    """
    rail_html = f"""
    <div id="collapsed-sidebar-rail" class="collapsed-rail">
      <div class="rail-top">
        <div class="rail-item rail-logo" id="rail-logo-btn" title="AI Assistant">🥑</div>
        <div class="rail-item" id="rail-toggle-btn" title="Expand Sidebar (click anywhere on line)">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/>
          </svg>
        </div>
        <div class="rail-item" id="rail-new-chat-btn" title="New chat">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>
          </svg>
        </div>
        <div class="rail-item" id="rail-search-btn" title="Recent chats">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
          </svg>
        </div>
        <div class="rail-item" id="rail-vault-btn" title="Document Vault">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>
          </svg>
        </div>
      </div>
      
      <!-- Free space along the sidebar line: clicking anywhere here opens the sidebar -->
      <div class="rail-free-space" id="rail-free-space" title="Click anywhere along this line to open sidebar"></div>
      
      <div class="rail-bottom">
        <div class="rail-item" id="rail-settings-btn" title="Settings &amp; Preferences">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
          </svg>
        </div>
        <div class="rail-item rail-avatar" id="rail-profile-btn" title="{username} ({role_label})">
          {initial_letter}
        </div>
      </div>
    </div>

    <script>
    (function() {{
        function getSidebar() {{
            return document.querySelector('section[data-testid="stSidebar"]');
        }}

        function isSidebarExpanded() {{
            const sb = getSidebar();
            if (!sb) return false;
            return sb.getAttribute('aria-expanded') === 'true';
        }}

        function openSidebar() {{
            const expBtn = document.querySelector('button[data-testid="stExpandSidebarButton"]');
            if (expBtn) {{
                expBtn.click();
                return;
            }}
            const sb = getSidebar();
            if (sb && sb.getAttribute('aria-expanded') === 'false') {{
                const collapseBtn = sb.querySelector('button[data-testid="stSidebarCollapseButton"]');
                if (collapseBtn) {{
                    collapseBtn.click();
                    return;
                }}
            }}
        }}

        function triggerNewChat() {{
            openSidebar();
            setTimeout(() => {{
                const btns = Array.from(document.querySelectorAll('section[data-testid="stSidebar"] button'));
                const newChatBtn = btns.find(b => b.textContent && b.textContent.includes('New chat'));
                if (newChatBtn) newChatBtn.click();
            }}, 150);
        }}

        function triggerSettings() {{
            openSidebar();
            setTimeout(() => {{
                const expanders = Array.from(document.querySelectorAll('section[data-testid="stSidebar"] details'));
                const settingsExp = expanders.find(e => e.textContent && e.textContent.includes('Settings'));
                if (settingsExp) {{
                    settingsExp.open = true;
                    settingsExp.scrollIntoView({{ behavior: 'smooth' }});
                }}
            }}, 150);
        }}

        function updateRailVisibility() {{
            const rail = document.getElementById('collapsed-sidebar-rail');
            if (!rail) return;
            const expanded = isSidebarExpanded();
            rail.style.display = expanded ? 'none' : 'flex';

            const main = document.querySelector('section.main');
            if (main) {{
                main.style.marginLeft = expanded ? '0px' : '52px';
            }}
        }}

        function attachRailListeners() {{
            const rail = document.getElementById('collapsed-sidebar-rail');
            if (!rail || rail.dataset.bound === 'true') return;
            rail.dataset.bound = 'true';

            const freeSpace = document.getElementById('rail-free-space');
            if (freeSpace) freeSpace.addEventListener('click', openSidebar);

            const toggleBtn = document.getElementById('rail-toggle-btn');
            if (toggleBtn) toggleBtn.addEventListener('click', openSidebar);

            const logoBtn = document.getElementById('rail-logo-btn');
            if (logoBtn) logoBtn.addEventListener('click', openSidebar);

            const newChat = document.getElementById('rail-new-chat-btn');
            if (newChat) newChat.addEventListener('click', triggerNewChat);

            const searchBtn = document.getElementById('rail-search-btn');
            if (searchBtn) searchBtn.addEventListener('click', openSidebar);

            const vaultBtn = document.getElementById('rail-vault-btn');
            if (vaultBtn) vaultBtn.addEventListener('click', openSidebar);

            const settingsBtn = document.getElementById('rail-settings-btn');
            if (settingsBtn) settingsBtn.addEventListener('click', triggerSettings);

            const profileBtn = document.getElementById('rail-profile-btn');
            if (profileBtn) profileBtn.addEventListener('click', triggerSettings);

            rail.addEventListener('click', function(e) {{
                if (e.target === rail || e.target.classList.contains('rail-top') || e.target.classList.contains('rail-bottom') || e.target.classList.contains('rail-free-space')) {{
                    openSidebar();
                }}
            }});
        }}

        attachRailListeners();
        updateRailVisibility();

        const observer = new MutationObserver(() => {{
            attachRailListeners();
            updateRailVisibility();
        }});
        observer.observe(document.body, {{ childList: true, subtree: true, attributes: true, attributeFilter: ['aria-expanded'] }});
        setInterval(updateRailVisibility, 250);
    }})();
    </script>
    """
    st.html(rail_html, unsafe_allow_javascript=True)

def main():

    # 0. Session Persistence across Browser Page Refresh
    if st.session_state.user is None:
        token = st.query_params.get("session_token")
        if token:
            persisted_user = get_user_by_auth_token(token)
            if persisted_user:
                st.session_state.user = persisted_user
                st.session_state.auth_token = token
            else:
                st.query_params.clear()

    user = st.session_state.user
    is_authenticated = user is not None
    username = user["username"] if is_authenticated else "Guest"
    user_id = user["id"] if is_authenticated else 0
    is_admin = is_authenticated and (user.get("role") == "admin" or username == config.ADMIN_USERNAME)
    user_namespace = f"user_{username}" if is_authenticated else "guest"
    user_docs_dir = config.DOCS_DIR / user_namespace
    user_docs_dir.mkdir(parents=True, exist_ok=True)

    # Manage user chat sessions
    user_sessions: List[Dict[str, Any]] = []
    if is_authenticated:
        user_sessions = get_user_chat_sessions(user_id)
        if not user_sessions:
            init_sid = create_chat_session(user_id, "Welcome Discussion")
            add_chat_message(
                init_sid,
                "assistant",
                f"Hello @{username}! 🥑 I am your AI Assistant. What can I help you explore today?",
                [],
            )
            st.session_state.current_session_id = init_sid
            user_sessions = get_user_chat_sessions(user_id)
        elif "current_session_id" not in st.session_state or st.session_state.current_session_id not in [s["id"] for s in user_sessions]:
            st.session_state.current_session_id = user_sessions[0]["id"]

    active_session_id = st.session_state.current_session_id

    # Presentation variables
    initial_letter = username[:1].upper() if username else "G"
    role_label = "Pro" if is_authenticated else "Guest"
    if is_admin:
        role_label = "Master Admin"

    # Persistent collapsed icon rail with full-height line click-to-open
    render_collapsed_sidebar_rail(username, role_label, initial_letter)

    # =========================================================================
    # SIDEBAR: Google Gemini-style Layout (Avocado, Mode Pills, Actions, Recents, User)
    # =========================================================================
    with st.sidebar:
        # Top Header: Avocado Logo + Brand Name (aligned with collapse button)
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: -38px; margin-bottom: 0.8rem; padding: 4px 0; height: 38px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.6rem;">🥑</span>
                <div style="font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em;">AI Assistant</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


        # Mode Pills: [ Chat ] [ Deep Insights ]
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.button("💬 Chat", use_container_width=True, key="mode_chat_pill", type="primary")
        with col_m2:
            st.button("⚡ Insights", use_container_width=True, key="mode_spark_pill", type="secondary")

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        # ✏️ New chat button (highlighted active pill)
        if st.button("✏️ New chat", use_container_width=True, type="secondary"):
            if is_authenticated:
                new_sid = create_chat_session(user_id, "New Chat")
                add_chat_message(
                    new_sid,
                    "assistant",
                    f"Fresh conversation started! What would you like to explore in your documents, @{username}? 🥑",
                    [],
                )
                st.session_state.current_session_id = new_sid
            else:
                st.session_state.guest_messages = []
            st.rerun()

        # 📂 Document Vault (collapsible drawer)
        with st.expander("📂 Document Vault", expanded=False):
            if is_authenticated:
                user_docs = get_user_documents(user_id)
                supported_files = [f for f in user_docs_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
                if not user_docs and supported_files:
                    _, vector_store_inst = get_user_rag_pipeline(user_namespace)
                    indexed_list = list_indexed_documents(vector_store_inst, docs_dir=user_docs_dir, namespace=user_namespace)
                    for item in indexed_list:
                        record_user_document(user_id, item["filename"], item["chunks"])
                    user_docs = get_user_documents(user_id)

                if user_docs:
                    _, vector_store_inst = get_user_rag_pipeline(user_namespace)
                    for d in user_docs:
                        d_name = d["filename"]
                        d_chunks = d["chunks"]
                        icon = get_format_icon(d_name)
                        c_info, c_rm = st.columns([4, 1])
                        with c_info:
                            st.markdown(f"{icon} **`{d_name}`**\n\n*{d_chunks} chunk(s)*")
                        with c_rm:
                            if st.button("🗑️", key=f"sidebar_doc_del_{d_name}", help=f"Remove '{d_name}'"):
                                try:
                                    with st.spinner(f"Removing '{d_name}'..."):
                                        delete_document_by_name(
                                            vector_store_inst,
                                            d_name,
                                            docs_dir=user_docs_dir,
                                            namespace=user_namespace,
                                        )
                                        delete_user_document(user_id, d_name)
                                        record_activity(username, "DOCUMENT_DELETE", f"Removed '{d_name}'", user_id=user_id)
                                        st.cache_resource.clear()
                                    st.toast(f"Removed '{d_name}'!", icon="🗑️")
                                    st.rerun()
                                except Exception as del_err:
                                    st.error(f"Error removing: {del_err}")
                        st.markdown("<hr style='margin: 4px 0 6px 0; border: none; border-top: 1px solid rgba(128,128,128,0.15);'/>", unsafe_allow_html=True)
                else:
                    st.info("Vault is empty. Click '+' beside the chat bar to attach documents.")
            else:
                st.caption("💡 Sign in to view and manage your uploaded files.")

        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

        # Recent Chats Section
        st.markdown("<div style='font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>Recent</div>", unsafe_allow_html=True)
        if is_authenticated and user_sessions:
            for s in user_sessions[:12]:
                c_sess, c_del = st.columns([5, 1])
                is_active = (s["id"] == active_session_id)
                title = s["title"]
                if len(title) > 20:
                    title = title[:18] + "..."
                icon_prefix = "👉 " if is_active else ""
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
        elif is_authenticated:
            st.caption("No previous chats. Start a new conversation!")
        else:
            st.caption("💡 Sign in to save and access previous chats across devices.")

        st.markdown("---")

        # Bottom Profile Bar (avatar + username + settings)
        active_view = "workspace"
        initial_letter = username[:1].upper() if username else "G"
        role_label = "Pro" if is_authenticated else "Guest"
        if is_admin:
            role_label = "Master Admin"

        st.markdown(f"""
        <div class="user-profile-card">
            <div class="avatar-circle">{initial_letter}</div>
            <div style="flex-grow: 1; min-width: 0;">
                <div style="font-weight: 600; font-size: 0.95rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{username}</div>
                <div style="font-size: 0.75rem; opacity: 0.7;">{role_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ Settings & Preferences", expanded=False):
            st.markdown("**Theme Mode**")
            theme_keys = {"🌙 Dark": "dark", "☀️ Light": "light", "💻 Device / System": "system"}
            current_theme_index = 0 if st.session_state.app_theme == "dark" else (1 if st.session_state.app_theme == "light" else 2)
            theme_choice = st.radio(
                "Theme Mode",
                ["🌙 Dark", "☀️ Light", "💻 Device / System"],
                index=current_theme_index,
                horizontal=False,
                label_visibility="collapsed",
            )
            selected_theme_code = theme_keys.get(theme_choice, "dark")
            if selected_theme_code != st.session_state.app_theme:
                st.session_state.app_theme = selected_theme_code
                st.rerun()

            # Mode Selector (if Admin)
            if is_admin:
                st.markdown("---")
                st.markdown("**Admin Controls**")
                nav_choice = st.radio(
                    "Interface View",
                    ["💬 AI Assistant", "👑 Master Admin Suite"],
                    index=0,
                )
                if nav_choice == "👑 Master Admin Suite":
                    active_view = "admin"

                if st.button("🔄 Quick Stack Ping", key="quick_ping_btn", use_container_width=True):
                    try:
                        idx = get_pinecone_index()
                        p_stats = idx.describe_index_stats()
                        st.success(f"Vectors: {p_stats.get('total_vector_count', 0)}")
                    except Exception as q_err:
                        st.error(f"Pinecone: {q_err}")

            st.markdown("---")

            # Sign In / Sign Out Button
            if is_authenticated:
                if st.button("🚪 Sign Out", use_container_width=True):
                    if "auth_token" in st.session_state and st.session_state.auth_token:
                        revoke_auth_token(st.session_state.auth_token)
                    record_activity(username, "USER_LOGOUT", "Signed out of workspace", user_id=user["id"])
                    st.query_params.clear()
                    st.session_state.user = None
                    st.session_state.auth_token = None
                    st.session_state.current_session_id = None
                    st.rerun()
            else:
                if st.button("🔐 Sign In / Sign Up", use_container_width=True, type="primary"):
                    show_auth_modal(0)

    # =========================================================================
    # MAIN CANVAS
    # =========================================================================
    if active_view == "admin" and is_authenticated:
        render_admin_suite(user)
        return

    # Check conversation messages
    messages: List[Dict[str, Any]] = []
    if is_authenticated and active_session_id:
        messages = get_session_messages(active_session_id)
    elif not is_authenticated:
        messages = st.session_state.guest_messages

    # Top right Auth status if Guest
    if not is_authenticated:
        top_c1, top_c2 = st.columns([5, 2])
        with top_c2:
            sb1, sb2 = st.columns(2)
            with sb1:
                if st.button("Sign In", type="primary", use_container_width=True):
                    show_auth_modal(0)
            with sb2:
                if st.button("Sign Up", type="secondary", use_container_width=True):
                    show_auth_modal(1)

    # Host Alert Banner for Admin
    if is_admin:
        unack = get_unacknowledged_count()
        if unack > 0:
            st.warning(
                f"🚨 **Host System Alert**: {unack} error event(s) recorded in telemetry. "
                "Switch to the **Master Admin Suite** in Settings to inspect."
            )

    pipeline, vector_store = get_user_rag_pipeline(user_namespace)

    # =========================================================================
    # HERO CANVAS: "Where should we start?" vs "Sign in or Sign up to get started"
    # =========================================================================
    if not messages:
        if is_authenticated:
            st.markdown("""
            <div class="hero-container">
                <div style="font-size: 3.2rem; margin-bottom: 0.5rem;">🥑</div>
                <h1 class="hero-title">Where should we start?</h1>
                <p class="hero-sub">Ask questions about your uploaded documents or explore insights.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hero-container">
                <div style="font-size: 3.2rem; margin-bottom: 0.5rem;">🥑</div>
                <h1 class="hero-title">Sign in or Sign up to get started</h1>
                <p class="hero-sub">Sign in to query private documents, save conversation history, and access your vault.</p>
            </div>
            """, unsafe_allow_html=True)

            col_act1, col_act2, col_act3 = st.columns([1, 2, 1])
            with col_act2:
                btn_in, btn_up = st.columns(2)
                with btn_in:
                    if st.button("🔐 Sign In", type="primary", use_container_width=True):
                        show_auth_modal(0)
                with btn_up:
                    if st.button("✨ Create Account", type="secondary", use_container_width=True):
                        show_auth_modal(1)

        # Gemini-style action suggestions
        st.markdown("<div style='max-width: 650px; margin: 1.5rem auto 1rem auto;'>", unsafe_allow_html=True)

        suggestions = [
            ("📄", "Create a structured summary from uploaded documents"),
            ("📊", "Analyze tabular data, CSVs, and department figures"),
            ("↳", "Find exact policy clauses and verified citations"),
            ("↳", "Reset my focus in 5 minutes"),
        ]

        for icon, text in suggestions:
            if st.button(f"{icon}  {text}", key=f"sug_{text[:15]}", use_container_width=True):
                if not is_authenticated:
                    show_auth_modal(0)
                else:
                    assert active_session_id is not None
                    add_chat_message(active_session_id, "user", text)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Render Conversation Messages
    for msg in messages:
        avatar = "👤" if msg["role"] == "user" else "🥑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📌 View {len(msg['citations'])} Source Citations", expanded=False):
                    for i, cite in enumerate(msg["citations"], 1):
                        safe_filename = sanitize_html(str(cite.get('filename', 'Unknown')))
                        safe_page = sanitize_html(str(cite.get('page', 'Unknown')))
                        safe_snippet = sanitize_html(str(cite.get('snippet', '')))
                        icon = get_format_icon(safe_filename)
                        st.markdown(f"""
                        <div class="citation-card">
                            <span class="badge">Citation #{i}</span><br>
                            <strong>{icon} File:</strong> <code>{safe_filename}</code> | <strong>Page/Section:</strong> {safe_page}<br>
                            <em>"{safe_snippet}"</em>
                        </div>
                        """, unsafe_allow_html=True)

    # =========================================================================
    # ATTACHMENT TRAY TOGGLE: Aligned directly with the Ask Question bar
    # =========================================================================
    if st.session_state.show_uploader:
        with st.container():
            st.markdown("""
            <div style="background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.18); border-radius: 16px; padding: 16px; margin-bottom: 12px;">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">📂 Document Ingestion Tray</div>
                <div style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 10px;">Select files to index into your private vault (PDF, DOCX, CSV, TXT, MD).</div>
            """, unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Upload documents to index into your private vault",
                type=["pdf", "docx", "csv", "txt", "md"],
                accept_multiple_files=True,
                key="chat_bar_uploader",
                label_visibility="collapsed",
            )

            c_reset, c_btn = st.columns([1, 2])
            with c_reset:
                reset_vault = st.checkbox("Reset Index", value=False, help="Wipe only your private vector namespace.")
            with c_btn:
                process_btn = st.button("📥 Ingest into Knowledge Vault", type="primary", use_container_width=True)

            if process_btn:
                if not is_authenticated:
                    show_auth_modal(0)
                elif not uploaded_files and not any(user_docs_dir.iterdir()):
                    st.warning("Please select at least one document to upload.")
                else:
                    try:
                        if uploaded_files:
                            with st.spinner("Validating and uploading files..."):
                                for uploaded_file in uploaded_files:
                                    safe_name = sanitize_filename(uploaded_file.name)
                                    file_bytes = uploaded_file.getbuffer().tobytes()
                                    is_valid, val_msg = validate_document_content(file_bytes, safe_name)
                                    if not is_valid:
                                        st.error(f"Security Alert for '{uploaded_file.name}': {val_msg}")
                                        record_activity(username, "SECURITY_BLOCKED_FILE", f"Blocked '{uploaded_file.name}': {val_msg}", user_id=user_id)
                                        continue
                                    save_path = user_docs_dir / safe_name
                                    with open(save_path, "wb") as f:
                                        f.write(file_bytes)

                        # If admin and user dir empty, copy sample docs from config.DOCS_DIR
                        if not any(user_docs_dir.iterdir()) and is_admin:
                            for sf in config.DOCS_DIR.iterdir():
                                if sf.is_file() and sf.suffix.lower() in SUPPORTED_EXTENSIONS:
                                    shutil.copy2(sf, user_docs_dir / sf.name)

                        with st.spinner(f"Indexing documents into cloud vault '{user_namespace}'..."):
                            docs = load_documents_from_directory(user_docs_dir)
                            if not docs:
                                st.warning("No supported documents found to index.")
                            else:
                                st.cache_resource.clear()
                                chunks = split_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                                embeddings = get_embeddings()

                                if reset_vault:
                                    for ed in get_user_documents(user_id):
                                        delete_user_document(user_id, ed["filename"])

                                index_documents(
                                    documents=chunks,
                                    persist_directory=config.CHROMA_PERSIST_DIR,
                                    embeddings=embeddings,
                                    recreate=reset_vault,
                                    namespace=user_namespace,
                                )

                                file_chunk_map: Dict[str, int] = {}
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
                                st.session_state.show_uploader = False
                                st.success(f"Indexed {len(docs)} section(s) into {len(chunks)} chunks across your files!")
                                st.rerun()
                    except Exception as e:
                        record_error(
                            service="Document Ingestion",
                            user_message="Document upload or indexing failed",
                            exception=e,
                        )
                        st.error(f"⚠️ Document processing error: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

    # '+' Attachment Trigger row right alongside/above chat input
    col_plus_btn, col_hint = st.columns([1, 8])
    with col_plus_btn:
        toggle_label = "➖ Close" if st.session_state.show_uploader else "➕ Attach"
        if st.button(toggle_label, key="toggle_attach_btn", help="Attach PDF, Word DOCX, CSV, TXT, MD"):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()
    with col_hint:
        if not st.session_state.show_uploader:
            st.caption("Click **➕ Attach** to upload and index documents into your private vault.")

    # =========================================================================
    # CHAT PROMPT INPUT BAR & QUERY PROCESSING
    # =========================================================================
    if user_query := st.chat_input("Ask a question or explore your documents..."):
        if not is_authenticated:
            show_auth_modal(0)
            st.info("💡 Please sign in or create an account to query your private knowledge vault.")
        else:
            assert active_session_id is not None
            # Store user query into Neon DB
            add_chat_message(active_session_id, "user", user_query)

            # Auto-update session title on first question
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
                        # Under-The-Hood Status Spinner
                        with st.status("🔮 Under the Hood Intelligence Engine...", expanded=True) as status_box:
                            st.write(f"🔍 Searching vector space in tenant vault `{user_namespace}`...")
                            context_str, citations, raw_docs = pipeline.retrieve(user_query)

                            if raw_docs:
                                st.write(f"📑 Reading & scoring {len(raw_docs)} context chunk(s) across your documents...")
                            else:
                                st.write("📑 Scanning knowledge vault for relevant context...")

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
                                    "- Or ask about specific topics contained in your uploaded documents!"
                                )
                            else:
                                fallback_msg = "Your knowledge vault is currently empty. Click '➕ Attach' above to upload documents!"
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
                                        icon = get_format_icon(safe_filename)
                                        st.markdown(f"""
                                        <div class="citation-card">
                                            <span class="badge">Citation #{i}</span><br>
                                            <strong>{icon} File:</strong> <code>{safe_filename}</code> | <strong>Page/Section:</strong> {safe_page}<br>
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
