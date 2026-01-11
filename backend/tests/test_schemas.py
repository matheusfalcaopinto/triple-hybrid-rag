"""Unit tests for Pydantic schemas."""

import pytest
from uuid import uuid4

from control_plane.schemas.auth import UserCreate, UserLogin
from control_plane.schemas.establishment import EstablishmentCreate
from control_plane.schemas.agent import AgentCreate


class TestAuthSchemas:
    """Test authentication schemas."""

    def test_user_create_valid(self):
        """Test valid user creation schema."""
        user = UserCreate(
            email="test@example.com",
            password="secure_password_123",
            full_name="Test User",
        )
        
        assert user.email == "test@example.com"
        assert user.password == "secure_password_123"
        assert user.full_name == "Test User"

    def test_user_create_invalid_email(self):
        """Test user creation with invalid email."""
        with pytest.raises(ValueError):
            UserCreate(
                email="not-an-email",
                password="password123",
            )

    def test_user_create_short_password(self):
        """Test user creation with short password."""
        with pytest.raises(ValueError):
            UserCreate(
                email="test@example.com",
                password="short",
            )

    def test_user_login_valid(self):
        """Test valid user login schema."""
        login = UserLogin(
            email="test@example.com",
            password="password123",
        )
        
        assert login.email == "test@example.com"
        assert login.password == "password123"


class TestEstablishmentSchemas:
    """Test establishment schemas."""

    def test_establishment_create_minimal(self):
        """Test establishment creation with minimal data."""
        establishment = EstablishmentCreate(
            name="Test Clinic",
        )
        
        assert establishment.name == "Test Clinic"

    def test_establishment_create_full(self):
        """Test establishment creation with full data."""
        establishment = EstablishmentCreate(
            name="Test Clinic",
            timezone="America/New_York",
            locale="en-US",
        )
        
        assert establishment.name == "Test Clinic"
        assert establishment.timezone == "America/New_York"
        assert establishment.locale == "en-US"


class TestAgentSchemas:
    """Test agent schemas."""

    def test_agent_create_minimal(self):
        """Test agent creation with minimal data."""
        agent = AgentCreate(
            name="Test Agent",
        )
        
        assert agent.name == "Test Agent"

    def test_agent_create_with_type(self):
        """Test agent creation with description."""
        agent = AgentCreate(
            name="Test Agent",
            description="A test agent",
        )
        
        assert agent.name == "Test Agent"
        assert agent.description == "A test agent"
