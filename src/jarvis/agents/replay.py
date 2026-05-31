"""Session replay — re-run saved user turns through an agent and diff the responses.

Useful for regression testing: after changing a system prompt or model, replay
a saved session to see which responses changed.

In stub mode (dry_run=True) the agent is replaced with an identity function
that echoes a sentinel instead of calling the real model. This validates the
replay scaffolding without model calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ReplayTurn:
    turn_index: int
    user_message: str
    original_response: str
    replayed_response: str
    changed: bool


def _extract_turns(messages: list[dict]) -> list[tuple[str, str]]:
    """Return (user_message, assistant_response) pairs in order."""
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "user":
            user_content = str(msg.get("content", ""))
            # Find the next assistant message
            j = i + 1
            while j < len(messages) and messages[j].get("role") != "assistant":
                j += 1
            if j < len(messages):
                asst_content = str(messages[j].get("content", ""))
                pairs.append((user_content, asst_content))
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    return pairs


def replay_session(
    messages: list[dict],
    run_turn_fn: Callable[[list[dict]], tuple[str, list[dict]]],
    dry_run: bool = False,
) -> list[ReplayTurn]:
    """Replay all user turns from a message history.

    Args:
        messages:     Full message history (alternating user/assistant).
        run_turn_fn:  Callable that takes a messages list and returns
                      (response_text, updated_messages). In practice this is
                      agent.run_turn but tests can pass any function.
        dry_run:      If True, skip run_turn_fn and return "<stub>" for each
                      replayed response.

    Returns:
        List of ReplayTurn, one per user→assistant exchange found in messages.
    """
    turns = _extract_turns(messages)
    results: list[ReplayTurn] = []
    context: list[dict] = []

    for idx, (user_msg, original_response) in enumerate(turns):
        context.append({"role": "user", "content": user_msg})

        if dry_run:
            replayed = "<stub>"
            context.append({"role": "assistant", "content": replayed})
        else:
            try:
                replayed, context = run_turn_fn(list(context))
            except Exception as exc:
                replayed = f"<error: {exc}>"
                context.append({"role": "assistant", "content": replayed})

        results.append(ReplayTurn(
            turn_index=idx,
            user_message=user_msg,
            original_response=original_response,
            replayed_response=replayed,
            changed=replayed != original_response,
        ))

    return results
