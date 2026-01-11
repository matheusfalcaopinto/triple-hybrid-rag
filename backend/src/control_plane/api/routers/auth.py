"""Authentication routes."""

from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from control_plane.auth.deps import get_current_user
from control_plane.auth.jwt import (
    create_access_token,
    get_password_hash,
    verify_password,
)
from control_plane.config import settings
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    LoginResponse,
    MeResponse,
    ResetPasswordRequest,
    UserMembership,
    UserResponse,
)
from control_plane.schemas.common import SuccessResponse

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Authenticate user and return access token."""
    # Find user by email
    result = await session.execute(
        select(User)
        .options(selectinload(User.memberships))
        .where(User.email == request.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    # Update last login
    await session.execute(
        update(User)
        .where(User.id == user.id)
        .values(last_login_at=datetime.now(timezone.utc))
    )

    # Build token payload
    memberships = [
        {"establishment_id": m.establishment_id, "role": m.role}
        for m in user.memberships
    ]
    token_data = {
        "sub": user.id,
        "email": user.email,
        "is_superuser": user.is_superuser,
        "memberships": memberships,
    }

    access_token = create_access_token(token_data)

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            is_active=user.is_active,
            default_establishment_id=user.default_establishment_id,
            created_at=user.created_at,
        ),
    )


@router.post("/logout", response_model=SuccessResponse)
async def logout(
    user: Annotated[User, Depends(get_current_user)],
):
    """Logout current user (client should discard token)."""
    # In a JWT system, we just return success
    # The client is responsible for discarding the token
    return SuccessResponse(message="Logged out successfully")


@router.post("/forgot-password", response_model=SuccessResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Request password reset email."""
    # Find user by email
    result = await session.execute(
        select(User).where(User.email == request.email)
    )
    user = result.scalar_one_or_none()

    # Always return success to prevent email enumeration
    if user:
        # In production, send password reset email here
        # For now, just log it
        reset_token = uuid4().hex
        # TODO: Store reset token and send email
        pass

    return SuccessResponse(
        message="If an account with that email exists, a password reset link has been sent"
    )


@router.post("/reset-password", response_model=SuccessResponse)
async def reset_password(
    request: ResetPasswordRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Reset password using reset token."""
    # TODO: Validate reset token and get user
    # For now, return not implemented
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Password reset not yet implemented",
    )


@router.get("/me", response_model=MeResponse)
async def get_me(
    user: Annotated[User, Depends(get_current_user)],
):
    """Get current user info."""
    memberships = [
        UserMembership(
            establishment_id=m.establishment_id,
            establishment_name=m.establishment.name if m.establishment else None,
            role=m.role,
        )
        for m in user.memberships
    ]

    return MeResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            is_active=user.is_active,
            default_establishment_id=user.default_establishment_id,
            created_at=user.created_at,
        ),
        memberships=memberships,
    )
