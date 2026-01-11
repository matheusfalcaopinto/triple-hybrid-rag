"""Campaign schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CampaignPace(BaseModel):
    """Campaign pacing configuration."""

    calls_per_minute: int = Field(6, ge=1, le=60)
    max_concurrent: int = Field(10, ge=1, le=100)


class CampaignSchedule(BaseModel):
    """Campaign schedule configuration."""

    timezone: str = "UTC"
    days: list[str] = ["mon", "tue", "wed", "thu", "fri"]
    start: str = "09:00"
    end: str = "18:00"


class CampaignCreate(BaseModel):
    """Create campaign request."""

    name: str = Field(..., min_length=1, max_length=255)
    agent_id: str
    agent_version_id: str | None = None
    lead_filter: dict[str, Any] | None = None
    pace: CampaignPace = Field(default_factory=CampaignPace)
    schedule: CampaignSchedule = Field(default_factory=CampaignSchedule)


class CampaignResponse(BaseModel):
    """Campaign response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    establishment_id: str
    name: str
    agent_id: str
    agent_version_id: str | None = None
    status: str
    lead_filter: dict[str, Any] | None = None
    pace_config: CampaignPace
    schedule: CampaignSchedule
    total_leads: int = 0
    leads_contacted: int = 0
    leads_completed: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None


class CampaignUpdate(BaseModel):
    """Update campaign request."""

    name: str | None = None
    lead_filter: dict[str, Any] | None = None
    pace: CampaignPace | None = None
    schedule: CampaignSchedule | None = None


class CampaignsListResponse(BaseModel):
    """List of campaigns."""

    items: list[CampaignResponse]
