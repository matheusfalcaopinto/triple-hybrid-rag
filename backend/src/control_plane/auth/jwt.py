"""JWT token handling."""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from control_plane.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Refresh tokens last 7 days by default
        expire = datetime.now(timezone.utc) + timedelta(days=7)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and verify a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None


# Alias for compatibility
verify_token = decode_access_token


class TokenData:
    """Token payload data."""

    def __init__(
        self,
        user_id: str,
        email: str,
        is_superuser: bool = False,
        memberships: list[dict[str, str]] | None = None,
    ):
        self.user_id = user_id
        self.email = email
        self.is_superuser = is_superuser
        self.memberships = memberships or []

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TokenData":
        """Create TokenData from JWT payload."""
        return cls(
            user_id=payload.get("sub", ""),
            email=payload.get("email", ""),
            is_superuser=payload.get("is_superuser", False),
            memberships=payload.get("memberships", []),
        )

    def get_role_for_establishment(self, establishment_id: str) -> str | None:
        """Get user's role for a specific establishment."""
        for membership in self.memberships:
            if membership.get("establishment_id") == establishment_id:
                return membership.get("role")
        return None

    def has_access_to_establishment(self, establishment_id: str) -> bool:
        """Check if user has access to an establishment."""
        if self.is_superuser:
            return True
        return self.get_role_for_establishment(establishment_id) is not None

    def is_admin_for_establishment(self, establishment_id: str) -> bool:
        """Check if user is admin for an establishment."""
        if self.is_superuser:
            return True
        return self.get_role_for_establishment(establishment_id) == "admin"
