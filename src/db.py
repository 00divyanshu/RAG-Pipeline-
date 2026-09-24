import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import bcrypt
from src import config

logger = logging.getLogger(__name__)

LOCAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "local_app.db"

def _is_postgres() -> bool:
    url = config.DATABASE_URL
    return bool(url and url.startswith("postgres"))

def get_connection():
    """
    Returns a database connection.
    Connects to Neon Serverless PostgreSQL if configured,
    otherwise falls back to SQLite for offline resilience.
    """
    if _is_postgres():
        try:
            import psycopg2
            # Clean connection params if needed
            db_url = config.DATABASE_URL
            # Strip redundant query params if causing issue
            if "channel_binding=" in db_url:
                db_url = db_url.split("&channel_binding=")[0]
            conn = psycopg2.connect(db_url, sslmode="require")
            return conn, "postgres"
        except Exception as e:
            logger.warning(f"Could not connect to Neon Postgres ({e}); falling back to local SQLite.")
    
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LOCAL_DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn, "sqlite"

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def init_db():
    """
    Initializes database tables and creates the master admin account if not present.
    """
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        if engine_type == "postgres":
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(64) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(32) DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id INT REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role VARCHAR(32) NOT NULL,
                content TEXT NOT NULL,
                citations JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS user_documents (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                filename VARCHAR(255) NOT NULL,
                chunk_count INT DEFAULT 0,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, filename)
            );
            """)
        else:
            cur.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS user_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, filename)
            );
            """)
        conn.commit()

        # Seed Master Admin account (@dmin / @dmin0812)
        admin_user = config.ADMIN_USERNAME
        admin_pass = config.ADMIN_PASSWORD
        cur.execute("SELECT id FROM users WHERE username = %s" if engine_type == "postgres" else "SELECT id FROM users WHERE username = ?", (admin_user,))
        if not cur.fetchone():
            hashed_admin = hash_password(admin_pass)
            if engine_type == "postgres":
                cur.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                    (admin_user, hashed_admin, "admin"),
                )
            else:
                cur.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (admin_user, hashed_admin, "admin"),
                )
            conn.commit()
            logger.info(f"Master admin '{admin_user}' initialized.")
    except Exception as e:
        logger.error(f"Error initializing DB schema: {e}")
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

# ---------------------------------------------------------
# User Authentication API
# ---------------------------------------------------------

def register_user(username: str, password: str, role: str = "user") -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Registers a new user account."""
    username = username.strip()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long.", None
    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters long.", None

    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        phash = hash_password(password)
        query = "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) RETURNING id, username, role" if engine_type == "postgres" else "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)"
        
        if engine_type == "postgres":
            cur.execute(query, (username, phash, role))
            row = cur.fetchone()
            user_data = {"id": row[0], "username": row[1], "role": row[2]}
        else:
            cur.execute(query, (username, phash, role))
            user_id = cur.lastrowid
            user_data = {"id": user_id, "username": username, "role": role}
        
        conn.commit()
        return True, "Account registered successfully!", user_data
    except Exception as e:
        conn.rollback()
        if "unique" in str(e).lower():
            return False, f"Username '{username}' is already taken. Please choose another.", None
        return False, f"Registration failed: {str(e)}", None
    finally:
        cur.close()
        conn.close()

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticates username and password against database."""
    username = username.strip()
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        query = "SELECT id, username, password_hash, role FROM users WHERE username = %s" if engine_type == "postgres" else "SELECT id, username, password_hash, role FROM users WHERE username = ?"
        cur.execute(query, (username,))
        row = cur.fetchone()
        if not row:
            return None
        
        user_id, uname, phash, role = row[0], row[1], row[2], row[3]
        if verify_password(password, phash):
            return {"id": user_id, "username": uname, "role": role}
        return None
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None
    finally:
        cur.close()
        conn.close()

# ---------------------------------------------------------
# Chat Session & Message Management
# ---------------------------------------------------------

def create_chat_session(user_id: int, title: str = "New Conversation") -> int:
    """Creates a fresh chat session for the given user."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        if engine_type == "postgres":
            cur.execute(
                "INSERT INTO chat_sessions (user_id, title) VALUES (%s, %s) RETURNING id",
                (user_id, title),
            )
            session_id = cur.fetchone()[0]
        else:
            cur.execute(
                "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
                (user_id, title),
            )
            session_id = cur.lastrowid
        conn.commit()
        return session_id
    finally:
        cur.close()
        conn.close()

def get_user_chat_sessions(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves all chat sessions for a user, sorted newest first."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        query = """
        SELECT id, title, created_at, updated_at 
        FROM chat_sessions 
        WHERE user_id = %s 
        ORDER BY updated_at DESC
        """ if engine_type == "postgres" else """
        SELECT id, title, created_at, updated_at 
        FROM chat_sessions 
        WHERE user_id = ? 
        ORDER BY updated_at DESC
        """
        cur.execute(query, (user_id,))
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "title": r[1],
                "created_at": str(r[2]),
                "updated_at": str(r[3]),
            }
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()

def update_chat_session_title(session_id: int, title: str):
    """Updates the title of a chat session."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        query = "UPDATE chat_sessions SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s" if engine_type == "postgres" else "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        cur.execute(query, (title[:60], session_id))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def delete_chat_session(session_id: int, user_id: int):
    """Deletes a chat session belonging to the user."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        query = "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s" if engine_type == "postgres" else "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?"
        cur.execute(query, (session_id, user_id))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def add_chat_message(session_id: int, role: str, content: str, citations: Optional[List] = None) -> int:
    """Appends a message to a chat session."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        cite_data = citations or []
        if engine_type == "postgres":
            cite_json = json.dumps(cite_data)
            cur.execute(
                "INSERT INTO chat_messages (session_id, role, content, citations) VALUES (%s, %s, %s, %s::jsonb) RETURNING id",
                (session_id, role, content, cite_json),
            )
            msg_id = cur.fetchone()[0]
            cur.execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s", (session_id,))
        else:
            cite_str = json.dumps(cite_data)
            cur.execute(
                "INSERT INTO chat_messages (session_id, role, content, citations) VALUES (?, ?, ?, ?)",
                (session_id, role, content, cite_str),
            )
            msg_id = cur.lastrowid
            cur.execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        conn.commit()
        return msg_id
    finally:
        cur.close()
        conn.close()

def get_session_messages(session_id: int) -> List[Dict[str, Any]]:
    """Fetches all messages in chronological order for a session."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        query = """
        SELECT id, role, content, citations, created_at 
        FROM chat_messages 
        WHERE session_id = %s 
        ORDER BY id ASC
        """ if engine_type == "postgres" else """
        SELECT id, role, content, citations, created_at 
        FROM chat_messages 
        WHERE session_id = ? 
        ORDER BY id ASC
        """
        cur.execute(query, (session_id,))
        rows = cur.fetchall()
        msgs = []
        for r in rows:
            cites = r[3]
            if isinstance(cites, str):
                try:
                    cites = json.loads(cites)
                except Exception:
                    cites = []
            elif not cites:
                cites = []
            msgs.append({
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "citations": cites,
                "created_at": str(r[4]),
            })
        return msgs
    finally:
        cur.close()
        conn.close()

# ---------------------------------------------------------
# User Ingested Document Tracking
# ---------------------------------------------------------

def record_user_document(user_id: int, filename: str, chunk_count: int):
    """Tracks document ingestion under the user account."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        if engine_type == "postgres":
            cur.execute("""
            INSERT INTO user_documents (user_id, filename, chunk_count)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, filename) DO UPDATE SET chunk_count = EXCLUDED.chunk_count, uploaded_at = CURRENT_TIMESTAMP
            """, (user_id, filename, chunk_count))
        else:
            cur.execute("""
            INSERT INTO user_documents (user_id, filename, chunk_count)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, filename) DO UPDATE SET chunk_count = excluded.chunk_count, uploaded_at = CURRENT_TIMESTAMP
            """, (user_id, filename, chunk_count))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def get_user_documents(user_id: int) -> List[Dict[str, Any]]:
    """Returns all documents uploaded by this user."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        query = "SELECT filename, chunk_count, uploaded_at FROM user_documents WHERE user_id = %s ORDER BY uploaded_at DESC" if engine_type == "postgres" else "SELECT filename, chunk_count, uploaded_at FROM user_documents WHERE user_id = ? ORDER BY uploaded_at DESC"
        cur.execute(query, (user_id,))
        rows = cur.fetchall()
        return [{"filename": r[0], "chunks": r[1], "uploaded_at": str(r[2])} for r in rows]
    finally:
        cur.close()
        conn.close()

def delete_user_document(user_id: int, filename: str):
    """Removes a document record for this user."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        query = "DELETE FROM user_documents WHERE user_id = %s AND filename = %s" if engine_type == "postgres" else "DELETE FROM user_documents WHERE user_id = ? AND filename = ?"
        cur.execute(query, (user_id, filename))
        conn.commit()
    finally:
        cur.close()
        conn.close()

# ---------------------------------------------------------
# Master Admin Platform Metrics
# ---------------------------------------------------------

def get_admin_platform_stats() -> Dict[str, Any]:
    """Returns platform-wide metrics for the Admin Dashboard."""
    conn, engine_type = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM user_documents")
        total_docs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM chat_sessions")
        total_chats = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM chat_messages")
        total_messages = cur.fetchone()[0]

        # User activity breakdown
        cur.execute("""
        SELECT u.id, u.username, u.role, u.created_at,
               COUNT(DISTINCT d.id) as doc_count,
               COUNT(DISTINCT s.id) as chat_count
        FROM users u
        LEFT JOIN user_documents d ON u.id = d.user_id
        LEFT JOIN chat_sessions s ON u.id = s.user_id
        GROUP BY u.id, u.username, u.role, u.created_at
        ORDER BY u.created_at DESC
        """)
        users_table = [
            {
                "id": r[0],
                "username": r[1],
                "role": r[2],
                "created_at": str(r[3])[:19],
                "docs": r[4],
                "chats": r[5],
            }
            for r in cur.fetchall()
        ]

        return {
            "total_users": total_users,
            "total_docs": total_docs,
            "total_chats": total_chats,
            "total_messages": total_messages,
            "users_list": users_table,
        }
    finally:
        cur.close()
        conn.close()

