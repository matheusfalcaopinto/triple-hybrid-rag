"""Agent schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """LLM model configuration."""

    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1024, ge=1, le=128000)


class ToolConfig(BaseModel):
    """Tool configuration."""

    enabled: bool = False
    config: dict[str, Any] | None = None


class ToolsConfig(BaseModel):
    """All tools configuration."""

    email: ToolConfig = Field(default_factory=lambda: ToolConfig(enabled=False))
    calendar: ToolConfig = Field(default_factory=lambda: ToolConfig(enabled=False))
    whatsapp_messaging: ToolConfig = Field(default_factory=lambda: ToolConfig(enabled=False))


class AgentCreate(BaseModel):
    """Create agent request."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class AgentResponse(BaseModel):
    """Agent response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    establishment_id: str
    name: str
    description: str | None = None
    is_active: bool = True
    active_version_id: str | None = None
    created_at: datetime | None = None


class AgentUpdate(BaseModel):
    """Update agent request."""

    name: str | None = None
    description: str | None = None
    is_active: bool | None = None


class AgentVersionCreate(BaseModel):
    """Create agent version request."""

    name: str = Field(..., min_length=1, max_length=100)  # e.g., "v1"
    prompt: str = Field(..., min_length=1)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


class AgentVersionResponse(BaseModel):
    """Agent version response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    name: str
    status: str  # draft, published, archived
    prompt: str
    llm_config: ModelConfig
    tools_config: ToolsConfig
    published_at: datetime | None = None
    created_at: datetime | None = None


class AgentVersionPublish(BaseModel):
    """Publish version response."""

    agent_id: str
    active_version_id: str
    previous_version_id: str | None = None


class AgentDeploymentResponse(BaseModel):
    """Agent deployment response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    version_id: str
    runtime_id: str
    status: str
    deployed_at: datetime | None = None


class AgentsListResponse(BaseModel):
    """List of agents."""

    items: list[AgentResponse]
