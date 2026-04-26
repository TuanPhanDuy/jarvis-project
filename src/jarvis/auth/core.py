"""JWT authentication + SQLite user store for JARVIS API.

Opt-in via JARVIS_AUTH_ENABLED=true. When disabled (default), all endpoints
are publicly accessible — existing behaviour is preserved.

Roles:
  admin  — full access including user management
  user   — chat, schedules, memory tools
  readonly — read endpoints only (health, schedules GET)

User DB: reports_dir/jarvis.db (shared SQLite file, users table).
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


# ── User model ────────────────────────────────────────────────────────────────

@dataclass
class User:
    user_id: int
    username: str
    role: str  # "admin" | "user" | "readonly"


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT    UNIQUE NOT NULL,
            hashed_password TEXT    NOT NULL,
            salt            TEXT    NOT NULL,
            role            TEXT    NOT NULL DEFAULT 'user',
            created_at      REAL    NOT NULL
        )
    """)
    conn.commit()
    return conn


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


# ── User CRUD ─────────────────────────────────────────────────────────────────

def create_user(db_path: Path, username: str, password: str, role: str = "user") -> User:
    salt = secrets.token_hex(16)
    hashed = _hash(password, salt)
    conn = _get_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO users (username, hashed_password, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, hashed, salt, role, time.time()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return User(user_id=row["id"], username=row["username"], role=row["role"])
    finally:
        conn.close()


def authenticate(db_path: Path, username: str, password: str) -> User | None:
    conn = _get_conn(db_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None
        if _hash(password, row["salt"]) != row["hashed_password"]:
            return None
        return User(user_id=row["id"], username=row["username"], role=row["role"])
    finally:
        conn.close()


def get_user(db_path: Path, username: str) -> User | None:
    conn = _get_conn(db_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return User(user_id=row["id"], username=row["username"], role=row["role"]) if row else None
    finally:
        conn.close()


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_token(user: User, secret: str, expire_minutes: int = 1440) -> str:
    try:
        import jwt
        payload = {
            "sub": user.username,
            "role": user.role,
            "uid": user.user_id,
            "exp": time.time() + expire_minutes * 60,
        }
        return jwt.encode(payload, secret, algorithm="HS256")
    except ImportError:
        raise RuntimeError("pyjwt is required for auth: pip install pyjwt")


def verify_token(token: str, secret: str) -> User | None:
    try:
        import jwt
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        if payload.get("exp", 0) < time.time():
            return None
        return User(user_id=payload["uid"], username=payload["sub"], role=payload["role"])
    except Exception:
        return None


def ensure_admin_exists(db_path: Path, default_password: str = "changeme") -> None:
    """Create default admin account if no users exist."""
    conn = _get_conn(db_path)
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    if count == 0:
        create_user(db_path, "admin", default_password, role="admin")
