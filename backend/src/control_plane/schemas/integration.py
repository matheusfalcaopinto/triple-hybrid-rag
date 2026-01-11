"""Integration schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntegrationResponse(BaseModel):
    """Integration type response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    type: str
    name: str
    description: str | None = None
    icon_url: str | None = None
    category: str
    auth_type: str
    is_active: bool = True


class IntegrationConnectRequest(BaseModel):
    """Connect integration request."""

    type: str
    display_name: str
    auth: dict[str, Any] | None = None


class IntegrationConnectResponse(BaseModel):
    """Connect integration response."""

    integration_id: str
    type: str
    status: str
    auth_url: str | None = None


class IntegrationConnectionResponse(BaseModel):
    """Integration connection response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    establishment_id: str
    integration_id: str
    integration_type: str | None = None
    display_name: str
    status: str
    is_enabled: bool = True
    last_used_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime | None = None


class IntegrationConnectionUpdate(BaseModel):
    """Update integration connection request."""

    display_name: str | None = None
    is_enabled: bool | None = None
