"""Runtime schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RuntimeCreateRequest(BaseModel):
    """Create runtime request."""

    image: str = Field(default="voice-agent-runtime:latest")
    env: dict[str, str] | None = None


class RuntimeAgentInfo(BaseModel):
    """Runtime agent info from /info endpoint."""

    active_calls: int = 0
    total_calls_handled: int = 0
    models: dict[str, str] | None = None
    status: str | None = None


class RuntimeResponse(BaseModel):
    """Runtime response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    establishment_id: str
    status: str
    container_name: str
    container_id: str | None = None
    image_tag: str
    host_port: int
    base_url: str
    created_at: datetime | None = None


class RuntimeStatusResponse(BaseModel):
    """Runtime status response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    ready: bool
    ready_issues: list[str] | None = None
    last_checked_at: datetime | None = None
    agent_info: RuntimeAgentInfo | None = None
