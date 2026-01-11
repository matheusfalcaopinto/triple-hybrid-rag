"""Auth schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request."""

    email: EmailStr
    password: str = Field(..., min_length=8)


# Alias for compatibility
UserLogin = LoginRequest


class UserCreate(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=100)


class LoginResponse(BaseModel):
    """Login response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class ForgotPasswordRequest(BaseModel):
    """Forgot password request."""

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Reset password request."""

    token: str
    new_password: str = Field(..., min_length=8)


class UserMembership(BaseModel):
    """User membership in an establishment."""

    establishment_id: str
    establishment_name: str | None = None
    role: str


class UserResponse(BaseModel):
    """User response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    display_name: str
    is_active: bool = True
    default_establishment_id: str | None = None
    created_at: datetime | None = None


class MeResponse(BaseModel):
    """Current user info response."""

    user: UserResponse
    memberships: list[UserMembership]


class UserInviteRequest(BaseModel):
    """User invitation request."""

    email: EmailStr
    role: str = Field(..., pattern="^(admin|operator|viewer)$")


class UserInviteResponse(BaseModel):
    """User invitation response."""

    invitation_id: str
    status: str


class UserUpdate(BaseModel):
    """User update request."""

    display_name: str | None = None
    role: str | None = Field(None, pattern="^(admin|operator|viewer)$")


# Required for forward reference
LoginResponse.model_rebuild()
