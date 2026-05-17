"""Tests for auth/core.py: user CRUD, JWT, and FastAPI dependency."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.auth.core import (
    User,
    authenticate,
    create_token,
    create_user,
    ensure_admin_exists,
    get_user,
    make_auth_dependency,
    verify_token,
)


# ── User CRUD ─────────────────────────────────────────────────────────────────


def test_create_user_stores_hashed_password(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    user = create_user(db, "alice", "secret123")
    assert user.username == "alice"
    assert user.role == "user"
    assert isinstance(user.user_id, int)


def test_create_user_custom_role(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    user = create_user(db, "bob", "pass", role="admin")
    assert user.role == "admin"


def test_create_user_duplicate_raises(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    create_user(db, "alice", "pass1")
    with pytest.raises(Exception):
        create_user(db, "alice", "pass2")


def test_get_user_returns_user(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    create_user(db, "carol", "pw")
    user = get_user(db, "carol")
    assert user is not None
    assert user.username == "carol"


def test_get_user_missing_returns_none(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    assert get_user(db, "nobody") is None


# ── Authentication ────────────────────────────────────────────────────────────


def test_authenticate_correct_password_returns_user(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    create_user(db, "dave", "correct")
    user = authenticate(db, "dave", "correct")
    assert user is not None
    assert user.username == "dave"


def test_authenticate_wrong_password_returns_none(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    create_user(db, "eve", "right")
    assert authenticate(db, "eve", "wrong") is None


def test_authenticate_missing_user_returns_none(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    assert authenticate(db, "ghost", "any") is None


# ── JWT ───────────────────────────────────────────────────────────────────────


def test_create_token_and_verify_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    user = create_user(db, "frank", "pw")
    secret = "test-secret"
    token = create_token(user, secret, expire_minutes=60)
    recovered = verify_token(token, secret)
    assert recovered is not None
    assert recovered.username == "frank"
    assert recovered.role == "user"


def test_verify_expired_token_returns_none(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    user = create_user(db, "grace", "pw")
    token = create_token(user, "secret", expire_minutes=-1)
    assert verify_token(token, "secret") is None


def test_verify_invalid_token_returns_none() -> None:
    assert verify_token("not.a.valid.token", "secret") is None


def test_verify_wrong_secret_returns_none(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    user = create_user(db, "henry", "pw")
    token = create_token(user, "correct-secret")
    assert verify_token(token, "wrong-secret") is None


# ── ensure_admin_exists ───────────────────────────────────────────────────────


def test_ensure_admin_creates_on_empty_db(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    ensure_admin_exists(db, default_password="admin123")
    admin = authenticate(db, "admin", "admin123")
    assert admin is not None
    assert admin.role == "admin"


def test_ensure_admin_skips_if_users_exist(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    create_user(db, "existing", "pw")
    ensure_admin_exists(db, default_password="adminpw")
    # No "admin" user should have been created
    assert get_user(db, "admin") is None


# ── FastAPI dependency ────────────────────────────────────────────────────────


def test_make_auth_dependency_disabled_returns_none(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    dep = make_auth_dependency(db, "secret", auth_enabled=False)
    request = MagicMock()
    assert dep(request) is None


def test_make_auth_dependency_enabled_missing_header_raises_401(tmp_path: Path) -> None:
    from fastapi import HTTPException

    db = tmp_path / "auth.db"
    dep = make_auth_dependency(db, "secret", auth_enabled=True)
    request = MagicMock()
    request.headers.get.return_value = ""
    with pytest.raises(HTTPException) as exc_info:
        dep(request)
    assert exc_info.value.status_code == 401


def test_make_auth_dependency_enabled_invalid_token_raises_401(tmp_path: Path) -> None:
    from fastapi import HTTPException

    db = tmp_path / "auth.db"
    dep = make_auth_dependency(db, "secret", auth_enabled=True)
    request = MagicMock()
    request.headers.get.return_value = "Bearer invalid.token.here"
    with pytest.raises(HTTPException) as exc_info:
        dep(request)
    assert exc_info.value.status_code == 401


def test_make_auth_dependency_enabled_valid_token_returns_user(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    user = create_user(db, "ivan", "pw")
    token = create_token(user, "secret", expire_minutes=60)

    dep = make_auth_dependency(db, "secret", auth_enabled=True)
    request = MagicMock()
    request.headers.get.return_value = f"Bearer {token}"
    result = dep(request)
    assert result is not None
    assert result.username == "ivan"
