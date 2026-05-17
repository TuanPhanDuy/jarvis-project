"""Tests for peer protocol retry/backoff logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_push_delta_retries_three_times_before_returning_false():
    """push_delta must attempt 3 times before giving up and returning False."""
    call_count = 0

    def _failing_urlopen(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise OSError("connection refused")

    with (
        patch("jarvis.peer.protocol.time.sleep"),  # skip real delays
        patch("urllib.request.urlopen", side_effect=_failing_urlopen),
    ):
        from jarvis.peer.protocol import push_delta
        result = push_delta("127.0.0.1", 9999, {"entities": [], "relationships": []}, timeout=1.0)

    assert result is False
    assert call_count == 4  # initial attempt + 3 retries (len(_RETRY_DELAYS) + 1)


def test_pull_delta_retries_three_times_before_returning_none():
    """pull_delta must attempt 3 times before giving up and returning None."""
    call_count = 0

    def _failing_urlopen(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise OSError("connection refused")

    with (
        patch("jarvis.peer.protocol.time.sleep"),
        patch("urllib.request.urlopen", side_effect=_failing_urlopen),
    ):
        from jarvis.peer.protocol import pull_delta
        result = pull_delta("127.0.0.1", 9999, since_ts=0.0, timeout=1.0)

    assert result is None
    assert call_count == 4


def test_push_delta_succeeds_on_first_attempt():
    """push_delta returns True immediately on success with no retries."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200

    with patch("urllib.request.urlopen", return_value=mock_resp):
        from jarvis.peer.protocol import push_delta
        result = push_delta("127.0.0.1", 8001, {"entities": [], "relationships": []})

    assert result is True


def test_push_delta_succeeds_on_second_attempt():
    """push_delta returns True when first attempt fails but second succeeds."""
    call_count = 0
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200

    def _flaky_urlopen(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("temporary failure")
        return mock_resp

    with (
        patch("jarvis.peer.protocol.time.sleep"),
        patch("urllib.request.urlopen", side_effect=_flaky_urlopen),
    ):
        from jarvis.peer.protocol import push_delta
        result = push_delta("127.0.0.1", 8001, {"entities": []})

    assert result is True
    assert call_count == 2
