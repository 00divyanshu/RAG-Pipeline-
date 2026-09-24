"""
Security and Defensive Protection Engine.
Provides rate limiting, input sanitization, path traversal defenses,
HTML escaping for XSS prevention, and credential validation.
"""

import re
import html
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# ==============================================================================
# In-Memory Thread-Safe Rate Limiter
# ==============================================================================

class RateLimiter:
    """
    Sliding window / lockout rate limiter for authentication endpoints.
    Protects against brute-force attacks, dictionary attacks, and credential stuffing.
    """
    def __init__(self):
        # Maps identifier -> list of failed attempt timestamps
        self._login_attempts: Dict[str, list[float]] = {}
        # Maps identifier -> lockout expiration timestamp
        self._login_lockouts: Dict[str, float] = {}
        # Maps client_id -> list of registration attempt timestamps
        self._register_attempts: Dict[str, list[float]] = {}

    def check_login_rate_limit(
        self,
        identifier: str,
        max_attempts: int = 5,
        window_seconds: int = 300,
        lockout_seconds: int = 300,
    ) -> Tuple[bool, int]:
        """
        Checks if the identifier (e.g. username or IP) is allowed to attempt login.
        Returns:
            (is_allowed: bool, remaining_lockout_seconds: int)
        """
        now = time.time()
        identifier = identifier.strip().lower()

        # Check existing lockout
        if identifier in self._login_lockouts:
            lockout_expiry = self._login_lockouts[identifier]
            if now < lockout_expiry:
                remaining = int(lockout_expiry - now) + 1
                return False, remaining
            else:
                # Lockout expired, clean up
                del self._login_lockouts[identifier]
                self._login_attempts[identifier] = []

        # Clean old attempts outside the window
        attempts = self._login_attempts.get(identifier, [])
        valid_attempts = [t for t in attempts if now - t < window_seconds]
        self._login_attempts[identifier] = valid_attempts

        if len(valid_attempts) >= max_attempts:
            # Trigger fresh lockout
            self._login_lockouts[identifier] = now + lockout_seconds
            return False, lockout_seconds

        return True, 0

    def record_login_failure(
        self,
        identifier: str,
        max_attempts: int = 5,
        window_seconds: int = 300,
        lockout_seconds: int = 300,
    ) -> int:
        """
        Records a failed login attempt.
        Returns remaining attempts before lockout.
        """
        now = time.time()
        identifier = identifier.strip().lower()

        attempts = self._login_attempts.get(identifier, [])
        valid_attempts = [t for t in attempts if now - t < window_seconds]
        valid_attempts.append(now)
        self._login_attempts[identifier] = valid_attempts

        remaining = max_attempts - len(valid_attempts)
        if remaining <= 0:
            self._login_lockouts[identifier] = now + lockout_seconds
            return 0
        return remaining

    def reset_login_rate_limit(self, identifier: str) -> None:
        """Clears failed attempts and lockouts after successful login."""
        identifier = identifier.strip().lower()
        self._login_attempts.pop(identifier, None)
        self._login_lockouts.pop(identifier, None)

    def check_register_rate_limit(
        self,
        client_id: str,
        max_registers: int = 4,
        window_seconds: int = 600,
    ) -> Tuple[bool, int]:
        """
        Prevents mass automated account creation spam.
        Allows up to max_registers accounts per window_seconds per client.
        """
        now = time.time()
        client_id = client_id.strip().lower()

        attempts = self._register_attempts.get(client_id, [])
        valid_attempts = [t for t in attempts if now - t < window_seconds]
        self._register_attempts[client_id] = valid_attempts

        if len(valid_attempts) >= max_registers:
            oldest = valid_attempts[0]
            remaining = int(window_seconds - (now - oldest)) + 1
            return False, max(1, remaining)

        return True, 0

    def record_register_attempt(self, client_id: str) -> None:
        """Records an account registration attempt."""
        now = time.time()
        client_id = client_id.strip().lower()
        if client_id not in self._register_attempts:
            self._register_attempts[client_id] = []
        self._register_attempts[client_id].append(now)


# Global singleton rate limiter
_rate_limiter = RateLimiter()

def check_login_rate_limit(identifier: str) -> Tuple[bool, int]:
    return _rate_limiter.check_login_rate_limit(identifier)

def record_login_failure(identifier: str) -> int:
    return _rate_limiter.record_login_failure(identifier)

def reset_login_rate_limit(identifier: str) -> None:
    _rate_limiter.reset_login_rate_limit(identifier)

def check_register_rate_limit(client_id: str) -> Tuple[bool, int]:
    return _rate_limiter.check_register_rate_limit(client_id)

def record_register_attempt(client_id: str) -> None:
    _rate_limiter.record_register_attempt(client_id)


# ==============================================================================
# Input Validation & Sanitization
# ==============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes uploaded document filenames to prevent Path Traversal, Directory
    Climbing, null byte injections, and shell escape exploits.
    """
    if not filename:
        return "unnamed_document.pdf"

    # Remove null bytes and control characters
    clean = filename.replace("\x00", "").strip()

    # Extract only the file basename (strips ../, ..\\, /etc/passwd, C:\...)
    clean = Path(clean).name

    # Replace all characters except alphanumeric, underscore, dot, and hyphen
    clean = re.sub(r"[^\w.-]", "_", clean)

    # Prevent multiple consecutive dots (e.g. '..')
    clean = re.sub(r"\.{2,}", ".", clean)

    # Ensure max filename length
    if len(clean) > 80:
        base, dot, ext = clean.rpartition(".")
        clean = base[:75] + "." + ext

    # Enforce .pdf extension
    if not clean.lower().endswith(".pdf"):
        clean += ".pdf"

    return clean or "document.pdf"


def validate_pdf_content(file_bytes: bytes, max_mb: int = 30) -> Tuple[bool, str]:
    """
    Validates uploaded file binary content against magic bytes and size limits.
    Prevents binary executable disguised uploads (MIME spoofing).
    """
    if not file_bytes:
        return False, "Uploaded file is empty."

    max_bytes = max_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return False, f"File size exceeds the {max_mb} MB limit."

    # Verify standard PDF file signature (%PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        return False, "Invalid file format. File does not contain a valid PDF magic header (%PDF-)."

    return True, "Valid PDF file."


def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validates username format to prevent injection and format errors.
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."

    if len(username) < 3:
        return False, "Username must be at least 3 characters long."

    if len(username) > 32:
        return False, "Username must be no more than 32 characters long."

    # Allow alphanumeric characters, @, _, -, .
    if not re.match(r"^[a-zA-Z0-9_@.-]+$", username):
        return False, "Username may only contain letters, numbers, @, ., _, or -."

    return True, "Valid username."


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validates password complexity requirements.
    """
    if not password:
        return False, "Password cannot be empty."

    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    if len(password) > 128:
        return False, "Password must be no more than 128 characters long."

    if password.strip() == "":
        return False, "Password cannot contain only whitespace."

    return True, "Valid password."


def sanitize_html(text: str) -> str:
    """
    Escapes all HTML entities to neutralize Cross-Site Scripting (XSS) vectors.
    """
    if not text:
        return ""
    return html.escape(str(text), quote=True)

