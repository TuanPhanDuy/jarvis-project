from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY")

    model: str = Field("claude-sonnet-4-6", alias="JARVIS_MODEL")
    max_tokens: int = Field(8096, alias="JARVIS_MAX_TOKENS")
    reports_dir: Path = Field(Path("reports"), alias="JARVIS_REPORTS_DIR")
    max_search_calls: int = Field(20, alias="JARVIS_MAX_SEARCH_CALLS")
    allowed_commands: list[str] = Field(
        default=["ls", "dir", "cat", "echo", "python", "python3", "git", "pwd", "whoami", "date"],
        alias="JARVIS_ALLOWED_COMMANDS",
    )

    # Voice settings
    tts_engine: str = Field("local", alias="JARVIS_TTS_ENGINE")  # "local" or "elevenlabs"
    elevenlabs_api_key: str | None = Field(None, alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field("Rachel", alias="JARVIS_ELEVENLABS_VOICE")
    whisper_model: str = Field("base", alias="JARVIS_WHISPER_MODEL")  # tiny|base|small|medium|large

    # API server settings
    api_host: str = Field("0.0.0.0", alias="JARVIS_API_HOST")
    api_port: int = Field(8000, alias="JARVIS_API_PORT")
    api_session_ttl_minutes: int = Field(60, alias="JARVIS_SESSION_TTL_MINUTES")

    # RabbitMQ
    rabbitmq_url: str = Field("amqp://guest:guest@localhost:5672/", alias="RABBITMQ_URL")
    rabbitmq_task_queue: str = Field("jarvis.tasks", alias="RABBITMQ_TASK_QUEUE")

    # Multi-model routing
    routing_strategy: str = Field("always_primary", alias="JARVIS_ROUTING_STRATEGY")
    fast_model: str = Field("claude-haiku-4-5-20251001", alias="JARVIS_FAST_MODEL")

    # Ollama local model
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.2", alias="OLLAMA_MODEL")

    # Auth
    auth_enabled: bool = Field(False, alias="JARVIS_AUTH_ENABLED")
    jwt_secret: str = Field("change-me-in-production", alias="JARVIS_JWT_SECRET")
    jwt_expire_minutes: int = Field(1440, alias="JARVIS_JWT_EXPIRE_MINUTES")

    # OpenTelemetry
    otel_enabled: bool = Field(False, alias="JARVIS_OTEL_ENABLED")
    otel_endpoint: str = Field("http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    # Wake word
    wake_word_key: str | None = Field(None, alias="PICOVOICE_ACCESS_KEY")

    # Action approval
    approval_threshold: str = Field("medium", alias="JARVIS_APPROVAL_THRESHOLD")
    approval_timeout_seconds: int = Field(60, alias="JARVIS_APPROVAL_TIMEOUT")

    # Proactive AI
    proactive_enabled: bool = Field(True, alias="JARVIS_PROACTIVE_ENABLED")
    idle_minutes: int = Field(30, alias="JARVIS_IDLE_MINUTES")

    # Self-improvement
    feedback_analyzer_enabled: bool = Field(True, alias="JARVIS_FEEDBACK_ANALYZER_ENABLED")

    # Digital Twin
    twin_enabled: bool = Field(False, alias="JARVIS_TWIN_ENABLED")

    # Peer coordination
    peer_enabled: bool = Field(False, alias="JARVIS_PEER_ENABLED")
    peer_port: int = Field(7474, alias="JARVIS_PEER_PORT")

    model_config = {"populate_by_name": True}


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
