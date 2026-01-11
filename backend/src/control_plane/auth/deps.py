"""Authentication dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from control_plane.auth.jwt import TokenData, decode_access_token
from control_plane.db.models.user import EstablishmentUser, User
from control_plane.db.session import get_session

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    """Get the current authenticated user."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    # Fetch user from database
    result = await session.execute(
        select(User)
        .options(selectinload(User.memberships))
        .where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


async def get_token_data(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> TokenData:
    """Get token data without loading the full user."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenData.from_payload(payload)


def require_role(*allowed_roles: str):
    """Create a dependency that requires specific roles for an establishment."""
    
    async def check_role(
        user: Annotated[User, Depends(get_current_user)],
        establishment_id: str,
    ) -> User:
        """Check if user has required role for the establishment."""
        if user.is_superuser:
            return user

        # Find user's membership for this establishment
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None
        )

        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this establishment",
            )

        if membership.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(allowed_roles)}",
            )

        return user

    return check_role


# Convenience dependencies
RequireAdmin = require_role("admin")
RequireOperator = require_role("admin", "operator")
RequireViewer = require_role("admin", "operator", "viewer")


class RoleChecker:
    """Role checker for establishment-level authorization."""

    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    async def __call__(
        self,
        user: Annotated[User, Depends(get_current_user)],
        session: Annotated[AsyncSession, Depends(get_session)],
        establishment_id: str,
    ) -> User:
        """Check if user has required role."""
        if user.is_superuser:
            return user

        result = await session.execute(
            select(EstablishmentUser).where(
                EstablishmentUser.user_id == user.id,
                EstablishmentUser.establishment_id == establishment_id,
            )
        )
        membership = result.scalar_one_or_none()

        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this establishment",
            )

        if membership.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(self.allowed_roles)}",
            )

        return user


async def get_current_user_establishment(
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: str | None = None,
) -> tuple[User, str]:
    """Get the current user and their active establishment ID.
    
    Returns a tuple of (user, establishment_id) for use in routes that need both.
    If establishment_id is not provided, uses the user's first establishment.
    """
    if establishment_id:
        # Verify user has access to this establishment
        membership = next(
            (m for m in user.memberships if str(m.establishment_id) == establishment_id),
            None,
        )
        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this establishment",
            )
        return user, establishment_id
    
    # Use user's first/default establishment
    if not user.memberships:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no establishments",
        )
    
    return user, str(user.memberships[0].establishment_id)
