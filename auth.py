"""Admin authentication: bcrypt verification, login throttling, server-side sessions.

Design notes
- Passwords are never stored. Only a bcrypt hash (from env/secrets) is used.
- Sessions live in SERVER memory, keyed by a random token. The browser (Streamlit
  session_state) only holds the opaque token, which is re-validated on every admin
  action. The token is not proof by itself: it must exist here and not be expired.
- Throttling is global (there is a single admin account) and survives page refreshes
  because it is not stored in Streamlit session state.
- This is prototype-grade auth: in-memory state is lost on restart and there is no
  MFA / per-user accounts. See README "Future work".
"""
import logging
import secrets
import threading
import time
from typing import Callable, Optional

import bcrypt

import config

logger = logging.getLogger(__name__)

# bcrypt only uses the first 72 bytes; bcrypt>=5 raises on longer input.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 12


class AuthError(Exception):
    """Login failed. The message is safe to show to the user."""


class LockedOutError(AuthError):
    pass


class NotAuthenticatedError(PermissionError):
    """Raised by require_admin when the session token is missing/invalid/expired."""


# ---------------------------------------------------------------- hashing helpers
def hash_password(password: str, rounds: int = 12) -> str:
    """Return a bcrypt hash (str). Used by scripts/generate_admin_hash.py."""
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=rounds)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """bcrypt.checkpw wrapper that returns False (never raises) on bad input."""
    if not password or not password_hash:
        return False
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, password_hash.encode("utf-8"))
    except ValueError:  # malformed hash
        logger.error("ADMIN_PASSWORD_HASH is not a valid bcrypt hash.")
        return False


# ---------------------------------------------------------------- manager
class AuthManager:
    def __init__(
        self,
        password_hash: Optional[str] = None,
        session_timeout_minutes: Optional[int] = None,
        max_failed_logins: Optional[int] = None,
        login_delay_seconds: Optional[float] = None,
        lockout_minutes: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._hash = config.ADMIN_PASSWORD_HASH if password_hash is None else password_hash
        self._timeout = 60 * (config.SESSION_TIMEOUT_MINUTES if session_timeout_minutes is None else session_timeout_minutes)
        self._max_failed = config.MAX_FAILED_LOGINS if max_failed_logins is None else max_failed_logins
        self._delay = config.LOGIN_DELAY_SECONDS if login_delay_seconds is None else login_delay_seconds
        self._lockout = 60 * (config.LOGIN_LOCKOUT_MINUTES if lockout_minutes is None else lockout_minutes)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._failed = 0
        self._locked_until = 0.0
        self._sessions: dict[str, float] = {}  # token -> last activity time
        if not self._hash:
            logger.warning("ADMIN_PASSWORD_HASH is not set: admin login is disabled.")

    # -- login / logout
    def login(self, password: str) -> str:
        """Return a session token or raise AuthError. Never logs the password."""
        with self._lock:
            now = self._clock()
            if now < self._locked_until:
                remaining = int(self._locked_until - now) + 1
                raise LockedOutError(f"Too many failed attempts. Try again in {remaining} seconds.")

        ok = verify_password(password, self._hash)  # outside the lock: bcrypt is slow

        if ok:
            with self._lock:
                self._failed = 0
                token = secrets.token_urlsafe(32)
                self._sessions[token] = self._clock()
            logger.info("Admin login succeeded.")
            return token

        with self._lock:
            self._failed += 1
            if self._failed >= self._max_failed:
                self._locked_until = self._clock() + self._lockout
                self._failed = 0
                logger.warning("Admin login locked out after repeated failures.")
        self._sleep(self._delay)  # slows brute force
        logger.warning("Admin login failed.")
        raise AuthError("Incorrect password.")

    def logout(self, token: Optional[str]) -> None:
        with self._lock:
            self._sessions.pop(token or "", None)

    # -- session checks
    def is_authenticated(self, token: Optional[str], refresh: bool = True) -> bool:
        """True if token is known and idle < timeout. Sliding expiry: a real user action
        (refresh=True) extends the session; background polling must pass refresh=False so an
        auto-refreshing page cannot keep a forgotten session alive forever."""
        if not token:
            return False
        with self._lock:
            last = self._sessions.get(token)
            if last is None:
                return False
            now = self._clock()
            if now - last > self._timeout:
                del self._sessions[token]
                return False
            if refresh:
                self._sessions[token] = now
            return True

    def require_admin(self, token: Optional[str], refresh: bool = True) -> None:
        """Call at the start of EVERY admin-only action (upload, delete, retry)."""
        if not self.is_authenticated(token, refresh=refresh):
            raise NotAuthenticatedError("Admin login required or session expired.")
