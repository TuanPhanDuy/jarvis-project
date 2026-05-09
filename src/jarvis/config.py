from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Local Ollama model
    model: str = Field("qwen2.5:14b", alias="JARVIS_MODEL")
    vision_model: str = Field("llava:13b", alias="JARVIS_VISION_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")

    max_tokens: int = Field(8096, alias="JARVIS_MAX_TOKENS")
    reports_dir: Path = Field(Path("reports"), alias="JARVIS_REPORTS_DIR")
    max_search_calls: int = Field(20, alias="JARVIS_MAX_SEARCH_CALLS")
    allowed_commands: list[str] = Field(
        default=["ls", "dir", "cat", "echo", "python", "python3", "git", "pwd", "whoami", "date"],
        alias="JARVIS_ALLOWED_COMMANDS",
    )

    # API server
    api_host: str = Field("0.0.0.0", alias="JARVIS_API_HOST")
    api_port: int = Field(8000, alias="JARVIS_API_PORT")
    api_session_ttl_minutes: int = Field(60, alias="JARVIS_SESSION_TTL_MINUTES")

    # Auth
    auth_enabled: bool = Field(False, alias="JARVIS_AUTH_ENABLED")
    jwt_secret: str = Field("change-me-in-production", alias="JARVIS_JWT_SECRET")
    jwt_expire_minutes: int = Field(1440, alias="JARVIS_JWT_EXPIRE_MINUTES")

    # Action approval
    approval_threshold: str = Field("medium", alias="JARVIS_APPROVAL_THRESHOLD")
    approval_timeout_seconds: int = Field(60, alias="JARVIS_APPROVAL_TIMEOUT")

    # Per-tool timeout
    tool_timeout_seconds: int = Field(60, alias="JARVIS_TOOL_TIMEOUT")

    # Memory
    memory_retention_days: int = Field(90, alias="JARVIS_MEMORY_RETENTION_DAYS")

    # Agent turn timeout
    agent_turn_timeout_seconds: int = Field(120, alias="JARVIS_AGENT_TURN_TIMEOUT")

    # Proactive / event bus
    proactive_enabled: bool = Field(False, alias="JARVIS_PROACTIVE_ENABLED")
    idle_minutes: int = Field(30, alias="JARVIS_IDLE_MINUTES")

    # Peer coordination
    peer_enabled: bool = Field(False, alias="JARVIS_PEER_ENABLED")
    peer_port: int = Field(8001, alias="JARVIS_PEER_PORT")

    # Voice
    voice_enabled: bool = Field(False, alias="JARVIS_VOICE_ENABLED")
    tts_engine: str = Field("local", alias="JARVIS_TTS_ENGINE")
    elevenlabs_api_key: str | None = Field(None, alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field("Rachel", alias="JARVIS_ELEVENLABS_VOICE")
    whisper_model: str = Field("base", alias="JARVIS_WHISPER_MODEL")
    wake_word_key: str | None = Field(None, alias="PICOVOICE_ACCESS_KEY")

    # RabbitMQ
    rabbitmq_url: str = Field("amqp://guest:guest@localhost:5672/", alias="RABBITMQ_URL")
    rabbitmq_task_queue: str = Field("jarvis.tasks", alias="RABBITMQ_TASK_QUEUE")

    # OpenTelemetry
    otel_enabled: bool = Field(False, alias="JARVIS_OTEL_ENABLED")
    otel_endpoint: str = Field("http://localhost:4317", alias="OTL_EXPORTER_OTLP_ENDPOINT")

    # CORS / rate limiting
    cors_allowed_origins: list[str] = Field(["*"], alias="JARVIS_CORS_ORIGINS")
    rate_limit_enabled: bool = Field(False, alias="JARVIS_RATE_LIMIT_ENABLED")
    chat_rate_limit: str = Field("30/minute", alias="JARVIS_CHAT_RATE_LIMIT")

    # WebSocket heartbeat
    ws_heartbeat_seconds: int = Field(30, alias="JARVIS_WS_HEARTBEAT")

    model_config = {"populate_by_name": True}

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if not (1 <= v <= 100_000):
            raise ValueError(f"max_tokens must be between 1 and 100000, got {v}")
        return v

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"api_port must be between 1 and 65535, got {v}")
        return v

    @field_validator("approval_timeout_seconds")
    @classmethod
    def validate_approval_timeout(cls, v: int) -> int:
        if v < 5:
            raise ValueError(f"approval_timeout_seconds must be >= 5, got {v}")
        return v


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
