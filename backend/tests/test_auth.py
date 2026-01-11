"""Unit tests for authentication module."""

import pytest
from datetime import timedelta

from control_plane.auth.jwt import (
    create_access_token,
    create_refresh_token,
    verify_token,
    verify_password,
    get_password_hash,
)


class TestPasswordHashing:
    """Test password hashing functions."""

    def test_password_hash_and_verify(self):
        """Test that password hashing and verification works."""
        password = "secure_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed)

    def test_wrong_password_fails(self):
        """Test that wrong password fails verification."""
        password = "secure_password_123"
        hashed = get_password_hash(password)
        
        assert not verify_password("wrong_password", hashed)

    def test_different_hashes_for_same_password(self):
        """Test that same password produces different hashes (salting)."""
        password = "secure_password_123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        assert hash1 != hash2
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestTokenCreation:
    """Test JWT token creation."""

    def test_create_access_token(self):
        """Test access token creation."""
        user_id = "test-user-id"
        token = create_access_token(data={"sub": user_id})
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        user_id = "test-user-id"
        token = create_refresh_token(data={"sub": user_id})
        
        assert token is not None
        assert isinstance(token, str)

    def test_access_token_with_custom_expiry(self):
        """Test token with custom expiration."""
        user_id = "test-user-id"
        token = create_access_token(
            data={"sub": user_id},
            expires_delta=timedelta(hours=2),
        )
        
        assert token is not None


class TestTokenVerification:
    """Test JWT token verification."""

    def test_verify_valid_token(self):
        """Test verifying a valid token."""
        user_id = "test-user-id"
        token = create_access_token(data={"sub": user_id})
        
        payload = verify_token(token)
        
        assert payload is not None
        assert payload.get("sub") == user_id

    def test_verify_token_with_additional_claims(self):
        """Test token with additional claims."""
        data = {
            "sub": "test-user-id",
            "email": "test@example.com",
            "role": "admin",
        }
        token = create_access_token(data=data)
        
        payload = verify_token(token)
        
        assert payload.get("sub") == "test-user-id"
        assert payload.get("email") == "test@example.com"
        assert payload.get("role") == "admin"

    def test_verify_invalid_token_returns_none(self):
        """Test that invalid token returns None."""
        payload = verify_token("invalid-token")
        
        assert payload is None

    def test_verify_tampered_token_returns_none(self):
        """Test that tampered token returns None."""
        token = create_access_token(data={"sub": "user-id"})
        
        # Tamper with the token
        tampered = token[:-5] + "xxxxx"
        
        payload = verify_token(tampered)
        
        assert payload is None
