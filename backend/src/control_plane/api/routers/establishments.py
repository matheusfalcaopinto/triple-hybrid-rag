"""Establishment routes."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from control_plane.auth.deps import RoleChecker, get_current_user
from control_plane.db.models.establishment import (
    Establishment,
    EstablishmentTelephonyPolicy,
    PhoneNumber,
)
from control_plane.db.models.user import EstablishmentUser, User
from control_plane.db.session import get_session
from control_plane.schemas.establishment import (
    BusinessHours,
    EstablishmentCreate,
    EstablishmentResponse,
    EstablishmentsListResponse,
    EstablishmentUpdate,
    PhoneNumberResponse,
    TelephonyPolicyResponse,
    TelephonyPolicyUpdate,
)

router = APIRouter()

# Role checkers
require_admin = RoleChecker(["admin"])
require_viewer = RoleChecker(["admin", "operator", "viewer"])


@router.get("", response_model=EstablishmentsListResponse)
async def list_establishments(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """List establishments the user has access to."""
    if user.is_superuser:
        # Superuser sees all
        result = await session.execute(select(Establishment))
        establishments = result.scalars().all()
    else:
        # Regular user sees only their establishments
        establishment_ids = [m.establishment_id for m in user.memberships]
        result = await session.execute(
            select(Establishment).where(Establishment.id.in_(establishment_ids))
        )
        establishments = result.scalars().all()

    return EstablishmentsListResponse(
        items=[
            EstablishmentResponse(
                id=e.id,
                name=e.name,
                timezone=e.timezone,
                locale=e.locale,
                is_active=e.is_active,
                billing_plan=e.billing_plan,
                billing_status=e.billing_status,
                created_at=e.created_at,
            )
            for e in establishments
        ]
    )


@router.post("", response_model=EstablishmentResponse, status_code=status.HTTP_201_CREATED)
async def create_establishment(
    request: EstablishmentCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Create a new establishment."""
    establishment = Establishment(
        id=f"est_{uuid4().hex[:12]}",
        name=request.name,
        timezone=request.timezone,
        locale=request.locale,
    )
    session.add(establishment)

    # Create default telephony policy
    policy = EstablishmentTelephonyPolicy(
        id=f"tp_{uuid4().hex[:12]}",
        establishment_id=establishment.id,
    )
    session.add(policy)

    # Add creator as admin
    membership = EstablishmentUser(
        id=f"eu_{uuid4().hex[:12]}",
        establishment_id=establishment.id,
        user_id=user.id,
        role="admin",
    )
    session.add(membership)

    await session.flush()

    return EstablishmentResponse(
        id=establishment.id,
        name=establishment.name,
        timezone=establishment.timezone,
        locale=establishment.locale,
        is_active=establishment.is_active,
        created_at=establishment.created_at,
    )


@router.get("/{establishment_id}", response_model=EstablishmentResponse)
async def get_establishment(
    establishment_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_viewer)],
):
    """Get establishment details."""
    result = await session.execute(
        select(Establishment).where(Establishment.id == establishment_id)
    )
    establishment = result.scalar_one_or_none()

    if not establishment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Establishment not found",
        )

    return EstablishmentResponse(
        id=establishment.id,
        name=establishment.name,
        timezone=establishment.timezone,
        locale=establishment.locale,
        is_active=establishment.is_active,
        billing_plan=establishment.billing_plan,
        billing_status=establishment.billing_status,
        created_at=establishment.created_at,
    )


@router.patch("/{establishment_id}", response_model=EstablishmentResponse)
async def update_establishment(
    establishment_id: Annotated[str, Path()],
    request: EstablishmentUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_admin)],
):
    """Update establishment details."""
    result = await session.execute(
        select(Establishment).where(Establishment.id == establishment_id)
    )
    establishment = result.scalar_one_or_none()

    if not establishment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Establishment not found",
        )

    if request.name is not None:
        establishment.name = request.name
    if request.timezone is not None:
        establishment.timezone = request.timezone
    if request.locale is not None:
        establishment.locale = request.locale
    if request.is_active is not None:
        establishment.is_active = request.is_active

    await session.flush()

    return EstablishmentResponse(
        id=establishment.id,
        name=establishment.name,
        timezone=establishment.timezone,
        locale=establishment.locale,
        is_active=establishment.is_active,
        billing_plan=establishment.billing_plan,
        billing_status=establishment.billing_status,
        created_at=establishment.created_at,
    )


@router.get("/{establishment_id}/telephony", response_model=TelephonyPolicyResponse)
async def get_telephony_policy(
    establishment_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_viewer)],
):
    """Get telephony policy for an establishment."""
    result = await session.execute(
        select(EstablishmentTelephonyPolicy).where(
            EstablishmentTelephonyPolicy.establishment_id == establishment_id
        )
    )
    policy = result.scalar_one_or_none()

    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Telephony policy not found",
        )

    # Get phone numbers
    result = await session.execute(
        select(PhoneNumber).where(PhoneNumber.establishment_id == establishment_id)
    )
    phone_numbers = result.scalars().all()

    return TelephonyPolicyResponse(
        establishment_id=establishment_id,
        inbound_numbers=[
            PhoneNumberResponse(
                id=p.id,
                e164=p.e164,
                provider=p.provider,
                routing_agent_id=p.routing_agent_id,
                is_active=p.is_active,
                display_name=p.display_name,
            )
            for p in phone_numbers
        ],
        handoff_e164=policy.handoff_e164,
        max_call_duration_seconds=policy.max_call_duration_seconds,
        business_hours=[
            BusinessHours(**bh) for bh in (policy.business_hours or [])
        ],
        blacklist_e164=policy.blacklist_e164,
    )


@router.patch("/{establishment_id}/telephony", response_model=TelephonyPolicyResponse)
async def update_telephony_policy(
    establishment_id: Annotated[str, Path()],
    request: TelephonyPolicyUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_admin)],
):
    """Update telephony policy."""
    result = await session.execute(
        select(EstablishmentTelephonyPolicy).where(
            EstablishmentTelephonyPolicy.establishment_id == establishment_id
        )
    )
    policy = result.scalar_one_or_none()

    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Telephony policy not found",
        )

    if request.handoff_e164 is not None:
        policy.handoff_e164 = request.handoff_e164
    if request.max_call_duration_seconds is not None:
        policy.max_call_duration_seconds = request.max_call_duration_seconds
    if request.business_hours is not None:
        policy.business_hours = [bh.model_dump() for bh in request.business_hours]
    if request.blacklist_e164 is not None:
        policy.blacklist_e164 = request.blacklist_e164

    await session.flush()

    # Get phone numbers
    result = await session.execute(
        select(PhoneNumber).where(PhoneNumber.establishment_id == establishment_id)
    )
    phone_numbers = result.scalars().all()

    return TelephonyPolicyResponse(
        establishment_id=establishment_id,
        inbound_numbers=[
            PhoneNumberResponse(
                id=p.id,
                e164=p.e164,
                provider=p.provider,
                routing_agent_id=p.routing_agent_id,
                is_active=p.is_active,
                display_name=p.display_name,
            )
            for p in phone_numbers
        ],
        handoff_e164=policy.handoff_e164,
        max_call_duration_seconds=policy.max_call_duration_seconds,
        business_hours=[
            BusinessHours(**bh) for bh in (policy.business_hours or [])
        ],
        blacklist_e164=policy.blacklist_e164,
    )
