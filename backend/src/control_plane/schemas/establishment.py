"""Establishment schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PhoneNumberResponse(BaseModel):
    """Phone number response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    e164: str
    provider: str
    routing_agent_id: str | None = None
    is_active: bool = True
    display_name: str | None = None


class BusinessHours(BaseModel):
    """Business hours configuration."""

    days: list[str]  # ["mon", "tue", ...]
    start: str  # "08:00"
    end: str  # "22:00"


class TelephonyPolicyResponse(BaseModel):
    """Telephony policy response."""

    establishment_id: str
    inbound_numbers: list[PhoneNumberResponse] = []
    handoff_e164: str | None = None
    max_call_duration_seconds: int = 900
    business_hours: list[BusinessHours] | None = None
    blacklist_e164: list[str] | None = None


class TelephonyPolicyUpdate(BaseModel):
    """Telephony policy update request."""

    handoff_e164: str | None = None
    max_call_duration_seconds: int | None = None
    business_hours: list[BusinessHours] | None = None
    blacklist_e164: list[str] | None = None


class EstablishmentCreate(BaseModel):
    """Create establishment request."""

    name: str = Field(..., min_length=1, max_length=255)
    timezone: str = "UTC"
    locale: str = "en-US"


class EstablishmentResponse(BaseModel):
    """Establishment response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    timezone: str
    locale: str
    is_active: bool = True
    billing_plan: str | None = None
    billing_status: str | None = None
    created_at: datetime | None = None


class EstablishmentUpdate(BaseModel):
    """Update establishment request."""

    name: str | None = None
    timezone: str | None = None
    locale: str | None = None
    is_active: bool | None = None


class EstablishmentsListResponse(BaseModel):
    """List of establishments."""

    items: list[EstablishmentResponse]
