"""Twilio router service - webhook routing and TwiML generation."""

import logging
from typing import Any
from urllib.parse import urlencode

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.db.models.agent import Agent
from control_plane.db.models.establishment import Establishment, PhoneNumber
from control_plane.db.models.runtime import Runtime

logger = logging.getLogger(__name__)


class TwilioRouter:
    """Handles Twilio webhook routing and TwiML generation."""

    async def route_inbound_call(
        self,
        session: AsyncSession,
        called_number: str,
        caller_number: str,
        call_sid: str,
    ) -> dict[str, Any]:
        """Route an inbound call to the correct runtime."""
        # Normalize to E.164 format
        to_e164 = self._normalize_e164(called_number)
        from_e164 = self._normalize_e164(caller_number)

        # Look up the phone number to find establishment and agent
        result = await session.execute(
            select(PhoneNumber)
            .where(PhoneNumber.e164 == to_e164)
            .where(PhoneNumber.is_active == True)
        )
        phone_number = result.scalar_one_or_none()

        if not phone_number:
            logger.warning(f"No phone number mapping found for {to_e164}")
            return {
                "found": False,
                "twiml": self._generate_reject_twiml("Number not configured"),
            }

        # Get establishment and runtime
        result = await session.execute(
            select(Establishment)
            .where(Establishment.id == phone_number.establishment_id)
            .where(Establishment.is_active == True)
        )
        establishment = result.scalar_one_or_none()

        if not establishment:
            logger.warning(f"Establishment not found for phone {to_e164}")
            return {
                "found": False,
                "twiml": self._generate_reject_twiml("Service unavailable"),
            }

        # Get runtime for this establishment
        result = await session.execute(
            select(Runtime)
            .where(Runtime.establishment_id == establishment.id)
            .where(Runtime.status == "running")
        )
        runtime = result.scalar_one_or_none()

        if not runtime:
            logger.warning(f"No running runtime for establishment {establishment.id}")
            return {
                "found": False,
                "twiml": self._generate_reject_twiml("Service temporarily unavailable"),
            }

        # Get agent if routed
        agent_id = phone_number.routing_agent_id
        agent = None
        if agent_id:
            result = await session.execute(
                select(Agent).where(Agent.id == agent_id)
            )
            agent = result.scalar_one_or_none()

        # Generate TwiML pointing to the runtime's WebSocket
        twiml = self._generate_connect_twiml(
            runtime_base_url=runtime.base_url,
            call_sid=call_sid,
            from_e164=from_e164,
            to_e164=to_e164,
            establishment_id=establishment.id,
            agent_id=agent.id if agent else None,
            agent_version_id=agent.active_version_id if agent else None,
        )

        return {
            "found": True,
            "establishment_id": establishment.id,
            "runtime_id": runtime.id,
            "agent_id": agent.id if agent else None,
            "twiml": twiml,
        }

    def _normalize_e164(self, number: str) -> str:
        """Normalize phone number to E.164 format."""
        # Remove any non-digit characters except leading +
        if number.startswith("+"):
            return "+" + "".join(c for c in number[1:] if c.isdigit())
        return "+" + "".join(c for c in number if c.isdigit())

    def _generate_connect_twiml(
        self,
        runtime_base_url: str,
        call_sid: str,
        from_e164: str,
        to_e164: str,
        establishment_id: str,
        agent_id: str | None,
        agent_version_id: str | None,
    ) -> str:
        """Generate TwiML to connect call to agent runtime WebSocket."""
        # Build WebSocket URL with query parameters
        ws_base = runtime_base_url.replace("http://", "ws://").replace("https://", "wss://")
        
        params = {
            "call_sid": call_sid,
            "from": from_e164,
            "to": to_e164,
            "establishment_id": establishment_id,
        }
        if agent_id:
            params["agent_id"] = agent_id
        if agent_version_id:
            params["agent_version_id"] = agent_version_id

        ws_url = f"{ws_base}/ws/twilio?{urlencode(params)}"

        # Status callback URL (back to control plane)
        status_callback = "/webhooks/twilio/call-status"

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="establishment_id" value="{establishment_id}" />
            <Parameter name="call_sid" value="{call_sid}" />
        </Stream>
    </Connect>
</Response>"""

    def _generate_reject_twiml(self, reason: str) -> str:
        """Generate TwiML to reject a call."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{reason}</Say>
    <Hangup />
</Response>"""

    def _generate_transfer_twiml(self, transfer_to: str) -> str:
        """Generate TwiML to transfer a call."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>{transfer_to}</Dial>
</Response>"""


# Singleton instance
twilio_router = TwilioRouter()
