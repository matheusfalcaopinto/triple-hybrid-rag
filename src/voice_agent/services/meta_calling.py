"""
Meta WhatsApp Business Calling API Client

Handles communication with Meta's WhatsApp Business Calling API for WebRTC voice calls.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import httpx

from ..config import SETTINGS

logger = logging.getLogger("voice_agent.services.meta_calling")


class CallAction(Enum):
    """Actions that can be performed on a WhatsApp call."""
    ACCEPT = "accept"
    REJECT = "reject"
    TERMINATE = "terminate"


@dataclass
class CallInfo:
    """Information about a WhatsApp call."""
    call_id: str
    from_number: str
    to_number: str
    sdp_offer: Optional[str] = None
    ice_candidates: Optional[list] = None


@dataclass
class CallActionResult:
    """Result of a call action."""
    success: bool
    call_id: Optional[str] = None
    error: Optional[str] = None
    sdp_answer: Optional[str] = None


class MetaCallingClient:
    """Client for Meta WhatsApp Business Calling API."""

    BASE_URL = "https://graph.facebook.com/v19.0"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            if not SETTINGS.meta_access_token:
                raise ValueError("META_ACCESS_TOKEN not configured")
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {SETTINGS.meta_access_token}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def accept_call(
        self,
        call_id: str,
        sdp_answer: str,
    ) -> CallActionResult:
        """
        Accept an incoming WhatsApp call with SDP answer.

        Args:
            call_id: The unique call identifier from Meta
            sdp_answer: WebRTC SDP answer

        Returns:
            CallActionResult with success status
        """
        try:
            client = await self._get_client()
            phone_number_id = SETTINGS.whatsapp_phone_number_id

            url = f"{self.BASE_URL}/{phone_number_id}/calls/{call_id}/accept"
            payload = {
                "sdp": sdp_answer,
            }

            response = await client.post(url, json=payload)
            response.raise_for_status()

            logger.info("Call accepted: call_id=%s", call_id)
            return CallActionResult(success=True, call_id=call_id)

        except httpx.HTTPStatusError as e:
            logger.error("Failed to accept call: %s - %s", e.response.status_code, e.response.text)
            return CallActionResult(success=False, error=str(e))
        except Exception as e:
            logger.exception("Error accepting call: %s", e)
            return CallActionResult(success=False, error=str(e))

    async def reject_call(self, call_id: str, reason: str = "busy") -> CallActionResult:
        """
        Reject an incoming WhatsApp call.

        Args:
            call_id: The unique call identifier
            reason: Rejection reason (busy, declined, etc.)

        Returns:
            CallActionResult with success status
        """
        try:
            client = await self._get_client()
            phone_number_id = SETTINGS.whatsapp_phone_number_id

            url = f"{self.BASE_URL}/{phone_number_id}/calls/{call_id}/reject"
            payload = {"reason": reason}

            response = await client.post(url, json=payload)
            response.raise_for_status()

            logger.info("Call rejected: call_id=%s reason=%s", call_id, reason)
            return CallActionResult(success=True, call_id=call_id)

        except Exception as e:
            logger.exception("Error rejecting call: %s", e)
            return CallActionResult(success=False, error=str(e))

    async def terminate_call(self, call_id: str) -> CallActionResult:
        """
        Terminate an active WhatsApp call.

        Args:
            call_id: The unique call identifier

        Returns:
            CallActionResult with success status
        """
        try:
            client = await self._get_client()
            phone_number_id = SETTINGS.whatsapp_phone_number_id

            url = f"{self.BASE_URL}/{phone_number_id}/calls/{call_id}/terminate"

            response = await client.post(url, json={})
            response.raise_for_status()

            logger.info("Call terminated: call_id=%s", call_id)
            return CallActionResult(success=True, call_id=call_id)

        except Exception as e:
            logger.exception("Error terminating call: %s", e)
            return CallActionResult(success=False, error=str(e))

    async def send_ice_candidate(
        self,
        call_id: str,
        candidate: str,
        sdp_mid: str,
        sdp_mline_index: int,
    ) -> bool:
        """
        Send an ICE candidate to the remote peer.

        Args:
            call_id: The unique call identifier
            candidate: The ICE candidate string
            sdp_mid: The SDP media ID
            sdp_mline_index: The SDP media line index

        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            phone_number_id = SETTINGS.whatsapp_phone_number_id

            url = f"{self.BASE_URL}/{phone_number_id}/calls/{call_id}/ice_candidates"
            payload = {
                "candidate": candidate,
                "sdpMid": sdp_mid,
                "sdpMLineIndex": sdp_mline_index,
            }

            response = await client.post(url, json=payload)
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error("Error sending ICE candidate: %s", e)
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def verify_webhook_signature(payload: bytes, signature: str) -> bool:
        """
        Verify Meta webhook signature.

        Args:
            payload: Raw request body
            signature: X-Hub-Signature-256 header value

        Returns:
            True if signature is valid
        """
        if not SETTINGS.meta_app_secret:
            logger.warning("META_APP_SECRET not configured, skipping signature verification")
            return True

        if not signature or not signature.startswith("sha256="):
            return False

        expected_signature = signature[7:]  # Remove 'sha256=' prefix
        computed_signature = hmac.new(
            SETTINGS.meta_app_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed_signature, expected_signature)


# Global instance
_meta_client: Optional[MetaCallingClient] = None


def get_meta_calling_client() -> MetaCallingClient:
    """Get or create Meta Calling client singleton."""
    global _meta_client
    if _meta_client is None:
        _meta_client = MetaCallingClient()
    return _meta_client
