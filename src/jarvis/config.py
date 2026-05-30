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
    ollama_model: str = Field("llama3.2", alias="OLLAMA_MODEL")

    max_tokens: int = Field(8096, alias="JARVIS_MAX_TOKENS")
    fast_model: str = Field("qwen2.5:3b", alias="JARVIS_FAST_MODEL")
    routing_strategy: str = Field("always_primary", alias="JARVIS_ROUTING_STRATEGY")
    agent_model_map: dict[str, str] = Field(
        default={},
        alias="JARVIS_AGENT_MODELS",
        description="JSON map of agent-type → model override, e.g. '{\"coder\": \"codellama:7b\"}'",
    )
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

    # Agent reasoning enhancements
    reflection_enabled: bool = Field(False, alias="JARVIS_REFLECTION_ENABLED")
    confidence_gate_enabled: bool = Field(True, alias="JARVIS_CONFIDENCE_GATE")
    auto_graph_extraction: bool = Field(False, alias="JARVIS_AUTO_GRAPH_EXTRACTION")
    proactive_memory_enabled: bool = Field(False, alias="JARVIS_PROACTIVE_MEMORY")
    goal_verification_enabled: bool = Field(True, alias="JARVIS_GOAL_VERIFICATION")
    consensus_n_agents: int = Field(3, alias="JARVIS_CONSENSUS_N_AGENTS")

    # Context window budgeting
    context_budget_tokens: int = Field(4096, alias="JARVIS_CONTEXT_BUDGET_TOKENS")

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

    # Training pipeline
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    training_data_dir: Path = Field(Path("reports/training"), alias="JARVIS_TRAINING_DIR")
    training_base_model_mlx: str = Field(
        "mlx-community/Qwen2.5-14B-Instruct-4bit",
        alias="JARVIS_TRAINING_BASE_MODEL",
    )
    training_max_papers_per_source: int = Field(10, alias="JARVIS_TRAINING_MAX_PAPERS")
    training_lora_rank: int = Field(16, alias="JARVIS_TRAINING_LORA_RANK")
    training_lora_epochs: int = Field(3, alias="JARVIS_TRAINING_EPOCHS")
    training_target_pairs: int = Field(500, alias="JARVIS_TRAINING_TARGET_PAIRS")

    # Auto-eval and eval-triggered training
    auto_eval_enabled: bool = Field(False, alias="JARVIS_AUTO_EVAL")
    eval_check_cron: str = Field("0 22 * * 6", alias="JARVIS_EVAL_CRON")  # Saturday 22:00 UTC
    eval_pass_rate_threshold: float = Field(0.8, alias="JARVIS_EVAL_THRESHOLD")

    # Auto-training schedule
    auto_training_enabled: bool = Field(False, alias="JARVIS_AUTO_TRAINING")
    auto_training_topics: str = Field(
        "RLHF,transformers,constitutional AI,multimodal systems,memory systems",
        alias="JARVIS_TRAINING_TOPICS",
    )
    auto_crawl_cron: str = Field("0 1 * * *", alias="JARVIS_AUTO_CRAWL_CRON")    # daily 01:00 UTC
    auto_finetune_cron: str = Field("0 3 * * 0", alias="JARVIS_AUTO_FINETUNE_CRON")  # Sunday 03:00 UTC
    auto_training_model_name: str = Field("jarvis-ft", alias="JARVIS_FT_MODEL_NAME")
    auto_training_min_new_docs: int = Field(5, alias="JARVIS_TRAINING_MIN_NEW_DOCS")

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
