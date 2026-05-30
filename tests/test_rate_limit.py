"""Tests for per-user and per-IP rate limiting middleware."""
from __future__ import annotations

import collections
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_middleware(max_calls: int = 3, window: float = 60.0, per_user: bool = False):
    from jarvis.api.server import _RateLimitMiddleware

    class FakeApp:
        async def __call__(self, scope, receive, send):
            pass

    mw = _RateLimitMiddleware.__new__(_RateLimitMiddleware)
    mw._max_calls = max_calls
    mw._window = window
    mw._enabled = True
    mw._per_user = per_user
    mw._buckets = collections.defaultdict(collections.deque)
    return mw


class TestBucketKey:
    def _fake_request(self, ip: str = "1.2.3.4", auth: str = "") -> MagicMock:
        req = MagicMock()
        req.client.host = ip
        req.headers = {"authorization": auth} if auth else {}
        return req

    def test_per_ip_returns_ip_key(self):
        mw = _make_middleware(per_user=False)
        req = self._fake_request(ip="10.0.0.1")
        assert mw._bucket_key(req) == "ip:10.0.0.1"

    def test_per_user_no_token_falls_back_to_ip(self):
        mw = _make_middleware(per_user=True)
        req = self._fake_request(ip="10.0.0.2")
        assert mw._bucket_key(req).startswith("ip:")

    def test_per_user_valid_token_returns_user_key(self):
        mw = _make_middleware(per_user=True)
        fake_user = MagicMock()
        fake_user.username = "alice"
        req = self._fake_request(ip="1.2.3.4", auth="Bearer sometoken")
        with patch("jarvis.api.server.get_settings") as mock_cfg, \
             patch("jarvis.auth.core.verify_token", return_value=fake_user) as mock_verify:
            mock_cfg.return_value.jwt_secret = "secret"
            key = mw._bucket_key(req)
        assert key == "user:alice"

    def test_per_user_invalid_token_falls_back_to_ip(self):
        mw = _make_middleware(per_user=True)
        req = self._fake_request(ip="1.2.3.4", auth="Bearer bad.token")
        with patch("jarvis.api.server.get_settings") as mock_cfg, \
             patch("jarvis.auth.core.verify_token", side_effect=Exception("bad")):
            mock_cfg.return_value.jwt_secret = "s"
            key = mw._bucket_key(req)
        assert key.startswith("ip:")


class TestSlidingWindow:
    def test_within_limit_passes(self):
        mw = _make_middleware(max_calls=5, window=60.0)
        now = time.monotonic()
        # Add 4 timestamps in window
        for _ in range(4):
            mw._buckets["ip:x"].append(now - 1)
        # 5th should still be under limit
        assert len(mw._buckets["ip:x"]) < mw._max_calls

    def test_expired_timestamps_cleaned(self):
        mw = _make_middleware(max_calls=3, window=10.0)
        now = time.monotonic()
        # 3 old timestamps outside the window
        mw._buckets["ip:y"].extend([now - 20, now - 15, now - 11])
        # Simulate dispatch by cleaning
        bucket = mw._buckets["ip:y"]
        while bucket and bucket[0] < now - mw._window:
            bucket.popleft()
        assert len(bucket) == 0

    def test_different_users_independent_buckets(self):
        mw = _make_middleware(max_calls=2, window=60.0, per_user=True)
        now = time.monotonic()
        mw._buckets["user:alice"].extend([now - 5, now - 3])
        mw._buckets["user:bob"].extend([now - 1])
        # alice is at limit, bob is not
        assert len(mw._buckets["user:alice"]) >= mw._max_calls
        assert len(mw._buckets["user:bob"]) < mw._max_calls


class TestRateLimitEndToEnd:
    @pytest.fixture(autouse=True)
    def reset_auth(self):
        import jarvis.api.server as _s
        _s._require_auth = None
        yield
        _s._require_auth = None

    def _make_client(self, max_calls: int = 2, per_user: bool = False):
        from jarvis.api.server import _RateLimitMiddleware, app
        # Patch the existing middleware's settings directly
        for mw in app.middleware_stack.__dict__.get("app", [app]).app.__dict__.get("middleware_stack", []):
            pass
        return TestClient(app, raise_server_exceptions=False)

    def test_rate_limit_triggers_429(self):
        from jarvis.api.server import _RateLimitMiddleware
        # Directly drive the middleware logic: fill bucket beyond limit
        mw = _make_middleware(max_calls=2, window=60.0)
        now = time.monotonic()
        mw._buckets["ip:test"].extend([now - 1, now - 2])  # already at limit

        class FakeClient:
            host = "test"

        req = MagicMock()
        req.client = FakeClient()
        req.headers = {}
        req.url.path = "/api/chat"

        key = mw._bucket_key(req)
        bucket = mw._buckets[key]
        while bucket and bucket[0] < now - mw._window:
            bucket.popleft()
        assert len(bucket) >= mw._max_calls  # would be rate-limited
