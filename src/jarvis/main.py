"""CLI entrypoint for JARVIS. Wires config → tools → agent → console."""
from __future__ import annotations

import argparse
import sys

import anthropic

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
    base_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    if settings.routing_strategy == "always_primary":
        client = base_client
    else:
        from jarvis.models.router import ModelRouter
        client = ModelRouter(  # type: ignore[assignment]
            primary=base_client,
            primary_model=settings.model,
            fast_model=settings.fast_model,
            strategy=settings.routing_strategy,
        )
    agent_ref: list = []

    # Base registry — all tools except delegate_task (used by sub-agents)
    base_schemas, base_registry = build_registry(
        tavily_api_key=settings.tavily_api_key,
        reports_dir=settings.reports_dir,
        get_messages=lambda: agent_ref[0].get_messages() if agent_ref else [],
        allowed_commands=settings.allowed_commands,
        anthropic_api_key=settings.anthropic_api_key,
    )

    if researcher_mode:
        agent = ResearcherAgent(
            client=client,
            model=settings.model,
            max_tokens=settings.max_tokens,
            tool_schemas=base_schemas,
            tool_registry=base_registry,
            max_search_calls=settings.max_search_calls,
        )
    else:
        # Planner registry adds delegate_task on top of the base tools
        planner_schemas, planner_registry = build_planner_registry(
            base_schemas=base_schemas,
            base_registry=base_registry,
            client=client,
            model=settings.model,
            max_tokens=settings.max_tokens,
        )
        agent = PlannerAgent(
            client=client,
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
    """Full hands-free voice loop: record → transcribe → agent → speak."""
    from jarvis.voice import stt, tts

    if not stt.is_available():
        print_error(
            "Voice mode requires openai-whisper, pyaudio, and numpy.\n"
            "Install with: pip install openai-whisper pyaudio numpy"
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


def _run_one_shot(agent: ResearcherAgent, topic: str) -> None:
    """Research a topic non-interactively and print the result."""
    messages = [{"role": "user", "content": f"Research this topic thoroughly: {topic}"}]
    with thinking_status(f"Researching: {topic}"):
        text, _ = agent.run_turn(messages)
    print_response(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JARVIS — AI research agent powered by Claude"
    )
    parser.add_argument(
        "--topic",
        metavar="TOPIC",
        help="Research a specific topic non-interactively and exit",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice mode: speak to JARVIS and hear responses aloud",
    )
    parser.add_argument(
        "--researcher",
        action="store_true",
        help="Use ResearcherAgent directly instead of the default PlannerAgent orchestrator",
    )
    parser.add_argument(
        "--ambient",
        action="store_true",
        help="Run in ambient mode: wake-word activated, continuous voice loop",
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
    except Exception as e:
        print_error(f"Configuration error: {e}\nCopy .env.example to .env and fill in your API keys.")
        sys.exit(1)

    agent = _build_agent(settings, researcher_mode=args.researcher)

    if args.topic:
        _run_one_shot(agent, args.topic)
    elif args.ambient:
        from jarvis.voice.ambient import run_ambient_mode
        from jarvis.voice import tts
        run_ambient_mode(agent, settings)
    elif args.voice:
        _run_voice(agent, settings)
    else:
        _run_interactive(agent)


if __name__ == "__main__":
    main()
