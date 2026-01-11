"""API endpoint tests."""

import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_healthz(self, client: AsyncClient):
        """Test health endpoint."""
        response = await client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_readyz(self, client: AsyncClient):
        """Test readiness endpoint."""
        response = await client.get("/readyz")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.asyncio
    async def test_info(self, client: AsyncClient):
        """Test info endpoint."""
        response = await client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestAuthEndpoints:
    """Test authentication endpoints."""

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, client: AsyncClient):
        """Test login with invalid credentials."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "wrong_password",
            },
        )
        
        # Should fail with 401 or 400
        assert response.status_code in [400, 401, 422]

    @pytest.mark.asyncio
    async def test_forgot_password(self, client: AsyncClient):
        """Test forgot password endpoint."""
        response = await client.post(
            "/api/v1/auth/forgot-password",
            json={
                "email": "test@example.com",
            },
        )
        
        # Should return success or appropriate error
        assert response.status_code in [200, 404, 422]


class TestProtectedEndpoints:
    """Test protected endpoints require authentication."""

    @pytest.mark.asyncio
    async def test_establishments_requires_auth(self, client: AsyncClient):
        """Test establishments endpoint requires auth."""
        response = await client.get("/api/v1/establishments")
        
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_agents_requires_auth(self, client: AsyncClient):
        """Test agents endpoint requires auth."""
        # Agents are nested under establishments
        response = await client.get("/api/v1/establishments/test-id/agents")
        
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_calls_requires_auth(self, client: AsyncClient):
        """Test calls endpoint requires auth."""
        response = await client.get("/api/v1/calls")
        
        assert response.status_code in [401, 403]
