"""Integration service - manages integration connections and tool proxy."""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.db.models.integration import Integration, IntegrationConnection

logger = logging.getLogger(__name__)


class IntegrationService:
    """Service for managing integrations and tool proxy."""

    async def get_integration_types(
        self, session: AsyncSession
    ) -> list[Integration]:
        """Get all available integration types."""
        result = await session.execute(
            select(Integration).where(Integration.is_active == True)
        )
        return list(result.scalars().all())

    async def get_connections(
        self,
        session: AsyncSession,
        establishment_id: str,
    ) -> list[IntegrationConnection]:
        """Get all integration connections for an establishment."""
        result = await session.execute(
            select(IntegrationConnection)
            .where(IntegrationConnection.establishment_id == establishment_id)
            .order_by(IntegrationConnection.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_connection(
        self,
        session: AsyncSession,
        connection_id: str,
    ) -> IntegrationConnection | None:
        """Get a specific integration connection."""
        result = await session.execute(
            select(IntegrationConnection).where(IntegrationConnection.id == connection_id)
        )
        return result.scalar_one_or_none()

    async def create_connection(
        self,
        session: AsyncSession,
        establishment_id: str,
        integration_type: str,
        display_name: str,
        auth_data: dict[str, Any] | None = None,
    ) -> tuple[IntegrationConnection, str | None]:
        """Create a new integration connection."""
        # Get integration type
        result = await session.execute(
            select(Integration).where(Integration.type == integration_type)
        )
        integration = result.scalar_one_or_none()

        if not integration:
            raise ValueError(f"Unknown integration type: {integration_type}")

        connection = IntegrationConnection(
            id=f"int_{uuid4().hex[:12]}",
            establishment_id=establishment_id,
            integration_id=integration.id,
            display_name=display_name,
            status="pending_auth" if integration.auth_type == "oauth2" else "active",
            auth_data=auth_data,
        )
        session.add(connection)
        await session.flush()

        # Generate OAuth URL if needed
        auth_url = None
        if integration.auth_type == "oauth2" and integration.oauth_config:
            auth_url = self._build_oauth_url(
                integration.oauth_config,
                connection.id,
            )

        return connection, auth_url

    async def update_connection(
        self,
        session: AsyncSession,
        connection_id: str,
        is_enabled: bool | None = None,
        display_name: str | None = None,
    ) -> IntegrationConnection | None:
        """Update an integration connection."""
        connection = await self.get_connection(session, connection_id)
        if not connection:
            return None

        if is_enabled is not None:
            connection.is_enabled = is_enabled
        if display_name is not None:
            connection.display_name = display_name

        await session.flush()
        return connection

    def _build_oauth_url(
        self, oauth_config: dict[str, Any], state: str
    ) -> str:
        """Build OAuth authorization URL."""
        base_url = oauth_config.get("auth_url", "")
        client_id = oauth_config.get("client_id", "")
        redirect_uri = oauth_config.get("redirect_uri", "")
        scope = oauth_config.get("scope", "")

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "response_type": "code",
        }

        from urllib.parse import urlencode

        return f"{base_url}?{urlencode(params)}"

    async def execute_tool(
        self,
        session: AsyncSession,
        establishment_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool via the tool proxy pattern."""
        # Map tool name to integration type
        tool_integration_map = {
            "send_email": "email",
            "create_calendar_event": "google_calendar",
            "send_whatsapp": "whatsapp",
        }

        integration_type = tool_integration_map.get(tool_name)
        if not integration_type:
            return {"error": f"Unknown tool: {tool_name}"}

        # Get active connection for this integration
        result = await session.execute(
            select(IntegrationConnection)
            .join(Integration)
            .where(
                IntegrationConnection.establishment_id == establishment_id,
                Integration.type == integration_type,
                IntegrationConnection.status == "active",
                IntegrationConnection.is_enabled == True,
            )
        )
        connection = result.scalar_one_or_none()

        if not connection:
            return {"error": f"No active {integration_type} integration"}

        # Execute the tool based on type
        try:
            if tool_name == "send_email":
                return await self._send_email(connection, arguments)
            elif tool_name == "create_calendar_event":
                return await self._create_calendar_event(connection, arguments)
            elif tool_name == "send_whatsapp":
                return await self._send_whatsapp(connection, arguments)
            else:
                return {"error": f"Tool {tool_name} not implemented"}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}

    async def _send_email(
        self,
        connection: IntegrationConnection,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Send an email via integration."""
        # This would integrate with the email provider
        # For now, return a stub response
        return {
            "ok": True,
            "message_id": f"msg_{uuid4().hex[:8]}",
            "to": arguments.get("to"),
        }

    async def _create_calendar_event(
        self,
        connection: IntegrationConnection,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a calendar event via integration."""
        # This would integrate with Google Calendar API
        return {
            "ok": True,
            "event_id": f"evt_{uuid4().hex[:8]}",
            "title": arguments.get("title"),
        }

    async def _send_whatsapp(
        self,
        connection: IntegrationConnection,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a WhatsApp message via integration."""
        # This would integrate with WhatsApp Business API
        return {
            "ok": True,
            "message_id": f"wa_{uuid4().hex[:8]}",
            "to": arguments.get("to"),
        }


# Singleton instance
integration_service = IntegrationService()
