"""Lead schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LeadCreate(BaseModel):
    """Create lead request."""

    phone_e164: str = Field(..., pattern=r"^\+[1-9]\d{1,14}$")
    name: str | None = None
    email: str | None = None
    source: str | None = None
    custom_fields: dict[str, Any] | None = None
    notes: str | None = None


class LeadResponse(BaseModel):
    """Lead response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    establishment_id: str
    phone_e164: str
    name: str | None = None
    email: str | None = None
    status: str
    source: str | None = None
    custom_fields: dict[str, Any] | None = None
    notes: str | None = None
    last_contacted_at: datetime | None = None
    total_calls: int = 0
    created_at: datetime | None = None


class LeadUpdate(BaseModel):
    """Update lead request."""

    name: str | None = None
    email: str | None = None
    status: str | None = Field(
        None,
        pattern="^(new|contacted|qualified|converted|not_interested|invalid|do_not_call)$",
    )
    custom_fields: dict[str, Any] | None = None
    notes: str | None = None


class LeadImportResponse(BaseModel):
    """Lead import response."""

    job_id: str
    status: str = "accepted"


class LeadsListResponse(BaseModel):
    """List of leads."""

    items: list[LeadResponse]
    total: int | None = None
    cursor: str | None = None
