"""User management routes."""

from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import RoleChecker, get_current_user
from control_plane.db.models.user import EstablishmentUser, User, UserInvitation
from control_plane.db.session import get_session
from control_plane.schemas.auth import (
    UserInviteRequest,
    UserInviteResponse,
    UserResponse,
    UserUpdate,
)

router = APIRouter()

# Role checkers
require_admin = RoleChecker(["admin"])


@router.get(
    "/establishments/{establishment_id}/users",
    response_model=list[UserResponse],
)
async def list_establishment_users(
    establishment_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """List users in an establishment."""
    # Check access (any role can view users)
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

    # Get users with memberships in this establishment
    result = await session.execute(
        select(User)
        .join(EstablishmentUser)
        .where(EstablishmentUser.establishment_id == establishment_id)
    )
    users = result.scalars().all()

    return [
        UserResponse(
            id=u.id,
            email=u.email,
            display_name=u.display_name,
            is_active=u.is_active,
            created_at=u.created_at,
        )
        for u in users
    ]


@router.post(
    "/establishments/{establishment_id}/users",
    response_model=UserInviteResponse,
    status_code=status.HTTP_201_CREATED,
)
async def invite_user(
    establishment_id: Annotated[str, Path()],
    request: UserInviteRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_admin)],
):
    """Invite a user to an establishment."""
    # Check if user already exists
    result = await session.execute(
        select(User).where(User.email == request.email)
    )
    existing_user = result.scalar_one_or_none()

    if existing_user:
        # Check if already a member
        result = await session.execute(
            select(EstablishmentUser).where(
                EstablishmentUser.user_id == existing_user.id,
                EstablishmentUser.establishment_id == establishment_id,
            )
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User is already a member of this establishment",
            )

        # Add existing user to establishment
        membership = EstablishmentUser(
            id=f"eu_{uuid4().hex[:12]}",
            establishment_id=establishment_id,
            user_id=existing_user.id,
            role=request.role,
        )
        session.add(membership)
        await session.flush()

        return UserInviteResponse(
            invitation_id=f"direct_{membership.id}",
            status="added",
        )

    # Create invitation for new user
    invitation = UserInvitation(
        id=f"inv_{uuid4().hex[:12]}",
        establishment_id=establishment_id,
        email=request.email,
        role=request.role,
        invited_by_user_id=user.id,
        token=uuid4().hex,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    session.add(invitation)
    await session.flush()

    # TODO: Send invitation email

    return UserInviteResponse(
        invitation_id=invitation.id,
        status="sent",
    )


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: Annotated[str, Path()],
    request: UserUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Update a user (self or admin)."""
    # Get target user
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    target_user = result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Check authorization
    is_self = current_user.id == user_id
    is_admin = current_user.is_superuser

    if not is_self and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify other users",
        )

    # Update allowed fields
    if request.display_name is not None:
        target_user.display_name = request.display_name

    # Role updates require admin
    if request.role is not None and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can change roles",
        )

    await session.flush()

    return UserResponse(
        id=target_user.id,
        email=target_user.email,
        display_name=target_user.display_name,
        is_active=target_user.is_active,
        created_at=target_user.created_at,
    )
