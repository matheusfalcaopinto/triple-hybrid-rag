"""Application configuration and settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "control-plane"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    secret_key: SecretStr = Field(default=SecretStr("change-me-in-production"))

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/control_plane"
    )
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Supabase (alternative to direct PostgreSQL)
    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = None

    # Redis (optional, for caching)
    redis_url: str | None = None

    # JWT Settings
    jwt_secret_key: SecretStr = Field(default=SecretStr("jwt-secret-change-me"))
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # Docker Settings (for runtime management)
    docker_host: str = "unix:///var/run/docker.sock"
    runtime_image: str = "voice-agent-runtime:latest"
    runtime_network: str = "control_plane_network"
    runtime_port_start: int = 18000
    runtime_port_end: int = 18999

    # Twilio Settings
    twilio_account_sid: str | None = None
    twilio_auth_token: SecretStr | None = None

    # OpenAI (for summary generation)
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"

    # Storage (for recordings)
    s3_bucket: str | None = None
    s3_access_key: SecretStr | None = None
    s3_secret_key: SecretStr | None = None
    s3_region: str = "us-east-1"
    s3_endpoint: str | None = None

    # Logging
    log_level: str = "INFO"

    # Polling intervals (seconds)
    runtime_health_poll_interval: int = 30
    sentiment_compute_interval: int = 60
    summary_generate_delay: int = 30

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
