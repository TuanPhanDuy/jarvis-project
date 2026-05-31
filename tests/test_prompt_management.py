"""Tests for prompt management (overrides + load_prompt integration)."""
from __future__ import annotations

import pytest

from jarvis.prompts.overrides import (
    clear_all_overrides,
    clear_override,
    get_override,
    list_overrides,
    set_override,
)
from jarvis.prompts.loader import load_prompt


@pytest.fixture(autouse=True)
def clean_overrides():
    """Ensure overrides don't leak between tests."""
    clear_all_overrides()
    yield
    clear_all_overrides()


class TestOverrideStore:
    def test_set_and_get(self):
        set_override("researcher", "You are a test researcher.")
        assert get_override("researcher") == "You are a test researcher."

    def test_get_nonexistent_returns_none(self):
        assert get_override("nonexistent") is None

    def test_key_normalised_lowercase(self):
        set_override("Researcher", "prompt")
        assert get_override("researcher") == "prompt"
        assert get_override("Researcher") == "prompt"

    def test_clear_existing_returns_true(self):
        set_override("coder", "code prompt")
        assert clear_override("coder") is True
        assert get_override("coder") is None

    def test_clear_nonexistent_returns_false(self):
        assert clear_override("ghost") is False

    def test_list_overrides_returns_all(self):
        set_override("researcher", "r")
        set_override("coder", "c")
        overrides = list_overrides()
        assert overrides == {"researcher": "r", "coder": "c"}

    def test_list_is_copy_not_reference(self):
        set_override("researcher", "r")
        overrides = list_overrides()
        overrides["researcher"] = "modified"
        assert get_override("researcher") == "r"

    def test_clear_all_removes_all(self):
        set_override("researcher", "r")
        set_override("coder", "c")
        count = clear_all_overrides()
        assert count == 2
        assert list_overrides() == {}

    def test_override_replaces_previous(self):
        set_override("researcher", "v1")
        set_override("researcher", "v2")
        assert get_override("researcher") == "v2"


class TestLoadPromptWithOverride:
    def test_override_takes_precedence_over_file(self):
        set_override("researcher", "OVERRIDDEN PROMPT")
        result = load_prompt("researcher")
        assert result == "OVERRIDDEN PROMPT"

    def test_file_used_when_no_override(self):
        result = load_prompt("researcher")
        assert "OVERRIDDEN" not in result
        assert len(result) > 10

    def test_after_clear_reverts_to_file(self):
        original = load_prompt("researcher")
        set_override("researcher", "temporary")
        clear_override("researcher")
        assert load_prompt("researcher") == original

    def test_override_respects_variable_substitution(self):
        set_override("researcher", "Hello {target}!")
        result = load_prompt("researcher", target="World")
        assert result == "Hello World!"

    def test_unknown_type_with_override_works(self):
        set_override("custom_agent", "Custom prompt text")
        result = load_prompt("custom_agent")
        assert result == "Custom prompt text"

    def test_unknown_type_without_override_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("definitely_not_a_real_agent")
