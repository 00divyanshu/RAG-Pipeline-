import unittest
from src.db import (
    init_db,
    register_user,
    authenticate_user,
    create_chat_session,
    get_user_chat_sessions,
    add_chat_message,
    get_session_messages,
    delete_chat_session,
    record_user_document,
    get_user_documents,
    delete_user_document,
    get_admin_platform_stats,
)
from src import config

class TestDatabaseAndAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_admin_pre_seeded(self):
        """Test that master admin @dmin is pre-seeded and authenticates."""
        admin = authenticate_user(config.ADMIN_USERNAME, config.ADMIN_PASSWORD)
        self.assertIsNotNone(admin)
        self.assertEqual(admin["username"], config.ADMIN_USERNAME)
        self.assertEqual(admin["role"], "admin")

    def test_user_registration_and_auth(self):
        """Test creating a regular user and authenticating."""
        test_uname = f"test_user_{int(unittest.mock.time.time()) if hasattr(unittest.mock, 'time') else 9999}"
        import time
        test_uname = f"test_user_{int(time.time())}"
        
        ok, msg, user = register_user(test_uname, "password123")
        self.assertTrue(ok)
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], test_uname)
        self.assertEqual(user["role"], "user")

        # Authenticate with correct credentials
        auth_success = authenticate_user(test_uname, "password123")
        self.assertIsNotNone(auth_success)
        self.assertEqual(auth_success["id"], user["id"])

        # Authenticate with wrong password
        auth_fail = authenticate_user(test_uname, "wrongpass")
        self.assertIsNone(auth_fail)

        # Duplicate registration should fail
        dup_ok, dup_msg, dup_user = register_user(test_uname, "newpass")
        self.assertFalse(dup_ok)

    def test_chat_session_and_messages(self):
        """Test chat session creation, message storage, and retrieval."""
        import time
        test_uname = f"chat_tester_{int(time.time())}"
        _, _, user = register_user(test_uname, "secret123")
        user_id = user["id"]

        # Create session
        session_id = create_chat_session(user_id, "Test Discussion")
        self.assertIsInstance(session_id, int)

        # Add messages
        m1 = add_chat_message(session_id, "user", "What is the policy deductible?")
        m2 = add_chat_message(
            session_id,
            "assistant",
            "The deductible is $500.",
            citations=[{"filename": "doc.pdf", "page": 1, "snippet": "deductible is $500"}],
        )

        messages = get_session_messages(session_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(len(messages[1]["citations"]), 1)

        # Delete session
        delete_chat_session(session_id, user_id)
        remaining_sessions = [s for s in get_user_chat_sessions(user_id) if s["id"] == session_id]
        self.assertEqual(len(remaining_sessions), 0)

    def test_user_document_tracking(self):
        """Test tracking user ingested files and chunk counts."""
        import time
        test_uname = f"doc_tester_{int(time.time())}"
        _, _, user = register_user(test_uname, "docpass123")
        user_id = user["id"]

        record_user_document(user_id, "company_manual.pdf", 14)
        docs = get_user_documents(user_id)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "company_manual.pdf")
        self.assertEqual(docs[0]["chunks"], 14)

        delete_user_document(user_id, "company_manual.pdf")
        docs_after = get_user_documents(user_id)
        self.assertEqual(len(docs_after), 0)

    def test_platform_stats(self):
        """Verify admin dashboard stats query returns valid counts."""
        stats = get_admin_platform_stats()
        self.assertIn("total_users", stats)
        self.assertIn("total_docs", stats)
        self.assertIn("total_chats", stats)
        self.assertIn("users_list", stats)
        self.assertGreaterEqual(stats["total_users"], 1)

if __name__ == "__main__":
    unittest.main()
