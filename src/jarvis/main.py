"""CLI entrypoint for JARVIS. Wires config → tools → agent → console."""
from __future__ import annotations

import argparse
import sys

from jarvis.config import get_settings
from jarvis.agents.planner import PlannerAgent
from jarvis.agents.researcher import ResearcherAgent
from jarvis.tools.registry import build_planner_registry, build_registry
from jarvis.utils.console import (
    get_user_input,
    listening_status,
    print_error,
    print_response,
    print_usage_summary,
    print_voice_input,
    print_voice_welcome,
    print_welcome,
    stream_chunk,
    stream_end,
    stream_start,
    thinking_status,
)


def _build_agent(settings, researcher_mode: bool = False) -> PlannerAgent | ResearcherAgent:
    agent_ref: list = []

    base_schemas, base_registry = build_registry(
        reports_dir=settings.reports_dir,
        get_messages=lambda: agent_ref[0].get_messages() if agent_ref else [],
        allowed_commands=settings.allowed_commands,
        vision_model=settings.vision_model,
    )

    if researcher_mode:
        agent = ResearcherAgent(
            model=settings.model,
            max_tokens=settings.max_tokens,
            tool_schemas=base_schemas,
            tool_registry=base_registry,
            max_search_calls=settings.max_search_calls,
        )
    else:
        planner_schemas, planner_registry = build_planner_registry(
            base_schemas=base_schemas,
            base_registry=base_registry,
            model=settings.model,
            max_tokens=settings.max_tokens,
        )
        agent = PlannerAgent(
            model=settings.model,
            max_tokens=settings.max_tokens,
            tool_schemas=planner_schemas,
            tool_registry=planner_registry,
        )

    agent_ref.append(agent)
    return agent


def _run_interactive(agent: PlannerAgent | ResearcherAgent) -> None:
    print_welcome()
    _status_ctx: list = []
    _streaming_started: list[bool] = [False]

    def on_thinking(msg: str) -> None:
        _streaming_started[0] = False
        ctx = thinking_status(msg)
        _status_ctx.clear()
        _status_ctx.append(ctx)
        ctx.__enter__()

    def _stop_spinner() -> None:
        if _status_ctx:
            _status_ctx[0].__exit__(None, None, None)
            _status_ctx.clear()

    def on_chunk(text: str) -> None:
        if not _streaming_started[0]:
            _stop_spinner()
            _streaming_started[0] = True
            stream_start()
        stream_chunk(text)

    def on_response(text: str) -> None:
        if _streaming_started[0]:
            stream_end()
        else:
            _stop_spinner()
            print_response(text)

    try:
        agent.run_conversation(
            on_response=on_response,
            on_thinking=on_thinking,
            get_input=get_user_input,
            on_chunk=on_chunk,
        )
    finally:
        print_usage_summary(agent.get_usage_summary())


def _run_voice(agent: PlannerAgent | ResearcherAgent, settings) -> None:
    from jarvis.voice import stt, tts

    if not stt.is_available():
        print_error(
            "Voice mode requires openai-whisper, pyaudio, and numpy.\n"
            "Install with: uv pip install openai-whisper pyaudio numpy"
        )
        return

    print_voice_welcome()
    _status_ctx: list = []
    _streaming_started: list[bool] = [False]

    def on_thinking(msg: str) -> None:
        _streaming_started[0] = False
        ctx = thinking_status(msg)
        _status_ctx.clear()
        _status_ctx.append(ctx)
        ctx.__enter__()

    def _stop_spinner() -> None:
        if _status_ctx:
            _status_ctx[0].__exit__(None, None, None)
            _status_ctx.clear()

    def on_chunk(text: str) -> None:
        if not _streaming_started[0]:
            _stop_spinner()
            _streaming_started[0] = True
            stream_start()
        stream_chunk(text)

    def on_response(text: str) -> None:
        if _streaming_started[0]:
            stream_end()
        else:
            _stop_spinner()
            print_response(text)
        tts.speak(
            text,
            engine=settings.tts_engine,
            elevenlabs_api_key=settings.elevenlabs_api_key,
            elevenlabs_voice_id=settings.elevenlabs_voice_id,
        )

    def get_input() -> str:
        with listening_status():
            transcribed = stt.record_and_transcribe()
        if transcribed:
            print_voice_input(transcribed)
        return transcribed

    try:
        agent.run_conversation(
            on_response=on_response,
            on_thinking=on_thinking,
            get_input=get_input,
            on_chunk=on_chunk,
        )
    finally:
        print_usage_summary(agent.get_usage_summary())


def _run_replay(session_id: str, settings, dry_run: bool = False) -> None:
    from jarvis.agents.replay import replay_session
    from jarvis.memory.sessions import load_sessions
    db_path = settings.reports_dir / "jarvis.db"
    rows = load_sessions(db_path, ttl_minutes=99999)
    row = next((r for r in rows if r["session_id"] == session_id), None)
    if not row:
        print_error(f"Session '{session_id}' not found in persisted sessions.")
        return
    messages = row["messages"]
    if not messages:
        from rich import print as rprint
        rprint("[yellow]Session has no messages to replay.[/yellow]")
        return

    if dry_run:
        turns = replay_session(messages, run_turn_fn=lambda m: ("<stub>", m), dry_run=True)
    else:
        agent = _build_agent(settings)
        turns = replay_session(messages, run_turn_fn=agent.run_turn)

    from rich import print as rprint
    from rich.table import Table
    from rich.console import Console
    console = Console()
    table = Table(title=f"Replay: {session_id[:16]}…", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("User message", max_width=40)
    table.add_column("Original", max_width=50)
    table.add_column("Replayed", max_width=50)
    table.add_column("Changed", width=8)
    for t in turns:
        changed_str = "[red]YES[/red]" if t.changed else "[green]no[/green]"
        table.add_row(
            str(t.turn_index),
            t.user_message[:80],
            t.original_response[:200],
            t.replayed_response[:200],
            changed_str,
        )
    console.print(table)
    changed = sum(1 for t in turns if t.changed)
    rprint(f"\n[bold]{changed}/{len(turns)}[/bold] turns changed.")


def _run_export_markdown(session_id: str, settings) -> None:
    from jarvis.memory.sessions import load_sessions, get_metadata
    from jarvis.tools.markdown_export import session_to_markdown
    db_path = settings.reports_dir / "jarvis.db"
    rows = load_sessions(db_path, ttl_minutes=99999)
    row = next((r for r in rows if r["session_id"] == session_id), None)
    if not row:
        print_error(f"Session '{session_id}' not found.")
        return
    meta = get_metadata(db_path, session_id)
    title = (meta or {}).get("title") or ""
    md = session_to_markdown(row["messages"], session_id=session_id, title=title)
    out_path = settings.reports_dir / f"session-{session_id[:8]}.md"
    out_path.write_text(md, encoding="utf-8")
    from rich import print as rprint
    rprint(f"[green]Exported to {out_path}[/green]")


def _run_one_shot(agent: ResearcherAgent, topic: str) -> None:
    messages = [{"role": "user", "content": f"Research this topic thoroughly: {topic}"}]
    with thinking_status(f"Researching: {topic}"):
        text, _ = agent.run_turn(messages)
    print_response(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JARVIS — local AI agent powered by Ollama"
    )
    parser.add_argument("--topic", metavar="TOPIC", help="Research a topic non-interactively and exit")
    parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    parser.add_argument("--researcher", action="store_true", help="Use ResearcherAgent directly")
    parser.add_argument("--ambient", action="store_true", help="Ambient wake-word voice loop")
    parser.add_argument(
        "--replay", metavar="SESSION_ID",
        help="Replay a saved session and print per-turn diff (--dry-run for stub mode)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use stub responses during --replay")
    parser.add_argument(
        "--export-markdown", metavar="SESSION_ID",
        help="Export a saved session as a Markdown file and exit",
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
    except Exception as e:
        print_error(f"Configuration error: {e}")
        sys.exit(1)

    if args.replay:
        _run_replay(args.replay, settings, dry_run=args.dry_run)
        return

    if args.export_markdown:
        _run_export_markdown(args.export_markdown, settings)
        return

    agent = _build_agent(settings, researcher_mode=args.researcher)

    if args.topic:
        _run_one_shot(agent, args.topic)
    elif args.ambient:
        from jarvis.voice.ambient import run_ambient_mode
        run_ambient_mode(agent, settings)
    elif args.voice:
        _run_voice(agent, settings)
    else:
        _run_interactive(agent)


if __name__ == "__main__":
    main()
