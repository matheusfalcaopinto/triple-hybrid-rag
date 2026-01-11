"""Campaign routes."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import RoleChecker, get_current_user
from control_plane.db.models.campaign import Campaign
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.campaign import (
    CampaignCreate,
    CampaignPace,
    CampaignResponse,
    CampaignsListResponse,
    CampaignSchedule,
    CampaignUpdate,
)
from control_plane.services.campaign_service import campaign_service

router = APIRouter()

# Role checkers
require_admin = RoleChecker(["admin"])
require_operator = RoleChecker(["admin", "operator"])


def _campaign_to_response(campaign: Campaign) -> CampaignResponse:
    """Convert Campaign model to response."""
    return CampaignResponse(
        id=campaign.id,
        establishment_id=campaign.establishment_id,
        name=campaign.name,
        agent_id=campaign.agent_id,
        agent_version_id=campaign.agent_version_id,
        status=campaign.status,
        lead_filter=campaign.lead_filter,
        pace_config=CampaignPace(**campaign.pace_config),
        schedule=CampaignSchedule(**campaign.schedule),
        total_leads=campaign.total_leads,
        leads_contacted=campaign.leads_contacted,
        leads_completed=campaign.leads_completed,
        started_at=campaign.started_at,
        completed_at=campaign.completed_at,
        created_at=campaign.created_at,
    )


@router.get("", response_model=CampaignsListResponse)
async def list_campaigns(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
    campaign_status: Annotated[str | None, Query(alias="status")] = None,
):
    """List campaigns for an establishment."""
    # Check access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None,
        )
        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this establishment",
            )

    campaigns = await campaign_service.get_campaigns(
        session, establishment_id, campaign_status
    )

    return CampaignsListResponse(
        items=[_campaign_to_response(c) for c in campaigns]
    )


@router.post("", response_model=CampaignResponse, status_code=status.HTTP_201_CREATED)
async def create_campaign(
    request: CampaignCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
):
    """Create a new campaign."""
    # Check admin access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None,
        )
        if not membership or membership.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )

    campaign = await campaign_service.create_campaign(
        session,
        establishment_id=establishment_id,
        name=request.name,
        agent_id=request.agent_id,
        agent_version_id=request.agent_version_id,
        lead_filter=request.lead_filter,
        pace_config=request.pace.model_dump(),
        schedule=request.schedule.model_dump(),
    )

    return _campaign_to_response(campaign)


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(
    campaign_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Get campaign details."""
    campaign = await campaign_service.get_campaign(session, campaign_id)

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found",
        )

    # Check access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == campaign.establishment_id),
            None,
        )
        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this campaign",
            )

    return _campaign_to_response(campaign)


@router.post("/{campaign_id}/start", response_model=CampaignResponse)
async def start_campaign(
    campaign_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Start a campaign."""
    campaign = await campaign_service.get_campaign(session, campaign_id)

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found",
        )

    # Check operator access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == campaign.establishment_id),
            None,
        )
        if not membership or membership.role not in ["admin", "operator"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator access required",
            )

    try:
        campaign = await campaign_service.start_campaign(session, campaign_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return _campaign_to_response(campaign)


@router.post("/{campaign_id}/pause", response_model=CampaignResponse)
async def pause_campaign(
    campaign_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Pause a running campaign."""
    campaign = await campaign_service.get_campaign(session, campaign_id)

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found",
        )

    # Check operator access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == campaign.establishment_id),
            None,
        )
        if not membership or membership.role not in ["admin", "operator"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator access required",
            )

    try:
        campaign = await campaign_service.pause_campaign(session, campaign_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return _campaign_to_response(campaign)
