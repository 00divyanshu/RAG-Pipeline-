"""
Defensive Security & Cyber Test Suite.
Validates rate limiting against brute force attacks, SQL injection resistance,
path traversal defense, XSS payload neutralization, file header validation,
and session token lifecycle.
"""

import unittest
import time
from src.security import (
    RateLimiter,
    sanitize_filename,
    validate_pdf_content,
    validate_document_content,
    validate_username,
    validate_password,
    sanitize_html,
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
)


class TestCyberDefenseSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    # =========================================================================
    # 1. Rate Limiting & Brute Force Defense Tests
    # =========================================================================

    def test_login_rate_limiter_blocks_brute_force(self):
        """Simulates automated credential stuffing / dictionary attack."""
        limiter = RateLimiter()
        target_user = "victim_account"

        # First 4 attempts should be allowed
        for i in range(4):
            allowed, rem_sec = limiter.check_login_rate_limit(target_user, max_attempts=5, window_seconds=60)
            self.assertTrue(allowed, f"Attempt {i+1} should be permitted")
            remaining = limiter.record_login_failure(target_user, max_attempts=5, window_seconds=60)
            self.assertEqual(remaining, 4 - i)

        # 5th attempt allowed to try, but fails
        allowed, _ = limiter.check_login_rate_limit(target_user, max_attempts=5, window_seconds=60)
        self.assertTrue(allowed)
        remaining = limiter.record_login_failure(target_user, max_attempts=5, window_seconds=60, lockout_seconds=120)
        self.assertEqual(remaining, 0)

        # 6th attempt MUST be blocked by rate limit lockout
        blocked, lockout_sec = limiter.check_login_rate_limit(target_user, max_attempts=5, window_seconds=60)
        self.assertFalse(blocked, "Brute force attempts must be locked out")
        self.assertGreater(lockout_sec, 0, "Lockout timer must be greater than 0")

    def test_login_rate_limiter_reset_on_success(self):
        """Verify that a successful login resets the failed attempts counter."""
        limiter = RateLimiter()
        user = "test_user_reset"
        limiter.record_login_failure(user, max_attempts=5)
        limiter.record_login_failure(user, max_attempts=5)

        # Successful authentication resets
        limiter.reset_login_rate_limit(user)
        allowed, _ = limiter.check_login_rate_limit(user, max_attempts=5)
        self.assertTrue(allowed)
        # Attempt counter should be fresh
        remaining = limiter.record_login_failure(user, max_attempts=5)
        self.assertEqual(remaining, 4)

    def test_registration_rate_limiter(self):
        """Verifies bot account mass creation prevention."""
        limiter = RateLimiter()
        client = "suspicious_bot_session"
        for _ in range(4):
            allowed, _ = limiter.check_register_rate_limit(client, max_registers=4, window_seconds=100)
            self.assertTrue(allowed)
            limiter.record_register_attempt(client)

        # 5th registration attempt must be blocked
        allowed, remaining_sec = limiter.check_register_rate_limit(client, max_registers=4, window_seconds=100)
        self.assertFalse(allowed, "Excessive registrations must be rate limited")
        self.assertGreater(remaining_sec, 0)

    # =========================================================================
    # 2. Path Traversal & File Injection Defense Tests
    # =========================================================================

    def test_path_traversal_filenames(self):
        """Tests that directory traversal sequences are completely neutralized."""
        malicious_inputs = [
            ("../../etc/passwd", "passwd.pdf"),
            ("..\\..\\Windows\\System32\\cmd.exe", "cmd.exe.pdf"),
            ("/var/log/syslog.pdf", "syslog.pdf"),
            ("C:\\Sensitive\\File.pdf", "File.pdf"),
            ("../../../../root/.ssh/id_rsa", "id_rsa.pdf"),
            ("test\x00malicious.pdf", "testmalicious.pdf"),
            ("normal_policy.pdf", "normal_policy.pdf"),
        ]
        for attack, expected in malicious_inputs:
            sanitized = sanitize_filename(attack)
            self.assertNotIn("..", sanitized, f"Path traversal found in {sanitized}")
            self.assertNotIn("/", sanitized, f"Forward slash found in {sanitized}")
            self.assertNotIn("\\", sanitized, f"Backslash found in {sanitized}")
            self.assertTrue(sanitized.endswith(".pdf"), "Must strictly enforce .pdf extension")

    def test_file_header_validation(self):
        """Rejects executable binaries and non-PDF files disguised with .pdf extensions."""
        # Executable binary header (MZ)
        fake_pdf_binary = b"MZ\x90\x00\x03\x00\x00\x00"
        ok, msg = validate_pdf_content(fake_pdf_binary)
        self.assertFalse(ok)
        self.assertIn("PDF magic header", msg)

        # HTML / Script disguised as PDF
        fake_pdf_html = b"<html><script>alert(1)</script></html>"
        ok, msg = validate_pdf_content(fake_pdf_html)
        self.assertFalse(ok)

        # Valid standard PDF
        valid_pdf_content = b"%PDF-1.4\n%real pdf header content"
        ok, _ = validate_pdf_content(valid_pdf_content)
        self.assertTrue(ok)

    # =========================================================================
    # 3. Cross-Site Scripting (XSS) Sanitization Tests
    # =========================================================================

    def test_xss_html_sanitization(self):
        """Verifies that script injection and HTML tag injections are escaped."""
        xss_payloads = [
            ("<script>alert('pwned')</script>", "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;"),
            ("<img src=x onerror=alert(1)>", "&lt;img src=x onerror=alert(1)&gt;"),
            ("hello & goodbye", "hello &amp; goodbye"),
            ("\"quote\" and 'single'", "&quot;quote&quot; and &#x27;single&#x27;"),
        ]
        for payload, expected in xss_payloads:
            escaped = sanitize_html(payload)
            self.assertEqual(escaped, expected)
            self.assertNotIn("<script>", escaped)
            self.assertNotIn("<img", escaped)

    # =========================================================================
    # 4. SQL Injection Resistance Tests
    # =========================================================================

    def test_sql_injection_in_authentication(self):
        """Verifies parameterized queries prevent SQL injection authentication bypass."""
        sqli_payloads = [
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT 1, 'admin', 'hash', 'admin'--",
            "'; DROP TABLE users;--",
        ]
        for payload in sqli_payloads:
            result = authenticate_user(payload, "any_password")
            self.assertIsNone(result, f"SQL injection payload '{payload}' should not authenticate")

    # =========================================================================
    # 5. Persistent Session Token Lifecycle Tests
    # =========================================================================

    def test_auth_token_lifecycle(self):
        """Tests token creation, retrieval across page refresh, and revocation."""
        # 1. Create a test user
        username = f"cyber_test_{int(time.time())}"
        ok, _, user = register_user(username, "cyberPass1234!")
        self.assertTrue(ok)
        self.assertIsNotNone(user)
        if not user:
            self.fail("User registration failed to return user data")
        user_id = user["id"]

        # 2. Issue session token (survives browser refresh)
        token = create_auth_token(user_id, days_valid=7)
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 20)

        # 3. Validate retrieval
        restored = get_user_by_auth_token(token)
        self.assertIsNotNone(restored)
        if not restored:
            self.fail("Token failed to retrieve user data")
        self.assertEqual(restored["id"], user_id)
        self.assertEqual(restored["username"], username)

        # 4. Invalidation / Revocation on logout
        revoke_auth_token(token)
        revoked_result = get_user_by_auth_token(token)
        self.assertIsNone(revoked_result, "Revoked token must not restore user session")

        # 5. Non-existent token
        fake_result = get_user_by_auth_token("non_existent_token_12345")
        self.assertIsNone(fake_result)

    # =========================================================================
    # 6. User Activity Logging & Admin Visibility Tests
    # =========================================================================

    def test_activity_logging_records_events(self):
        """Verifies user action audit trail recording."""
        test_uname = f"auditor_{int(time.time())}"
        record_activity(test_uname, "USER_LOGIN", "Test audit log entry", ip_address="127.0.0.1")
        record_activity(test_uname, "DOCUMENT_UPLOAD", "Uploaded test_file.pdf", ip_address="127.0.0.1")

        recent_logs = get_recent_activity_logs(limit=20)
        self.assertGreater(len(recent_logs), 0)
        found_login = any(l["username"] == test_uname and l["action"] == "USER_LOGIN" for l in recent_logs)
        found_upload = any(l["username"] == test_uname and l["action"] == "DOCUMENT_UPLOAD" for l in recent_logs)
        self.assertTrue(found_login, "USER_LOGIN activity must be recorded")
        self.assertTrue(found_upload, "DOCUMENT_UPLOAD activity must be recorded")

    def test_global_documents_directory_query(self):
        """Verifies all user documents can be queried across all tenants."""
        docs = get_all_users_documents()
        self.assertIsInstance(docs, list)
        for doc in docs:
            self.assertIn("filename", doc)
            self.assertIn("username", doc)
            self.assertIn("chunk_count", doc)

    def test_multi_format_document_validation(self):
        """Verifies validation and sanitization for PDF, DOCX, CSV, TXT, MD."""
        # 1. Valid Word DOCX (PK ZIP magic bytes)
        fake_docx = b"PK\x03\x04\x14\x00\x00\x00\x08\x00word/document.xml"
        ok, _ = validate_document_content(fake_docx, "report.docx")
        self.assertTrue(ok)

        # 2. Valid CSV content
        valid_csv = b"name,age,role\nAlice,30,Engineer\nBob,25,Designer\n"
        ok, _ = validate_document_content(valid_csv, "data.csv")
        self.assertTrue(ok)

        # 3. Disguised binary executable payload in CSV / TXT
        malicious_csv = b"MZ\x90\x00\x03\x00\x00\x00malicious binary"
        ok, msg = validate_document_content(malicious_csv, "fake.csv")
        self.assertFalse(ok)
        self.assertIn("Executable binary", msg)

        # 4. Null byte injection in text file
        null_byte_txt = b"normal text\x00hidden shellcode"
        ok, msg = validate_document_content(null_byte_txt, "notes.txt")
        self.assertFalse(ok)
        self.assertIn("null bytes", msg)

        # 5. Filename sanitization preserves multi-format extensions
        self.assertEqual(sanitize_filename("my_financial_report.docx"), "my_financial_report.docx")
        self.assertEqual(sanitize_filename("sales_2026.csv"), "sales_2026.csv")
        self.assertEqual(sanitize_filename("readme_guide.md"), "readme_guide.md")
        self.assertEqual(sanitize_filename("../../../etc/passwd.csv"), "passwd.csv")


if __name__ == "__main__":
    unittest.main()

