"""
Outbound Call Service

Handles initiating outbound calls via Twilio with AMD (Answering Machine Detection) support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..config import SETTINGS

logger = logging.getLogger("voice_agent.services.outbound")


class AnsweredBy(Enum):
    """Twilio AMD detection results."""
    HUMAN = "human"
    MACHINE_START = "machine_start"
    MACHINE_END_BEEP = "machine_end_beep"
    MACHINE_END_SILENCE = "machine_end_silence"
    MACHINE_END_OTHER = "machine_end_other"
    FAX = "fax"
    UNKNOWN = "unknown"

    @classmethod
    def from_twilio(cls, value: str) -> "AnsweredBy":
        """Convert Twilio value to enum."""
        clean_value = value.lower().replace("-", "_")
        try:
            return cls(clean_value)
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_voicemail(self) -> bool:
        """Check if this indicates voicemail/machine."""
        return self in {
            AnsweredBy.MACHINE_START,
            AnsweredBy.MACHINE_END_BEEP,
            AnsweredBy.MACHINE_END_SILENCE,
            AnsweredBy.MACHINE_END_OTHER,
        }


@dataclass
class OutboundCallResult:
    """Result of initiating an outbound call."""
    success: bool
    call_sid: Optional[str] = None
    error: Optional[str] = None


class OutboundCallService:
    """Service for making outbound calls with AMD support."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Get or create Twilio client (lazy loading)."""
        if self._client is None:
            if not SETTINGS.twilio_account_sid or not SETTINGS.twilio_auth_token:
                raise ValueError("Twilio credentials not configured")
            try:
                from twilio.rest import Client
                self._client = Client(
                    SETTINGS.twilio_account_sid,
                    SETTINGS.twilio_auth_token,
                )
            except ImportError:
                raise ImportError("twilio package not installed. Run: pip install twilio")
        return self._client

    def initiate_call(
        self,
        to_number: str,
        callback_base_url: str,
        customer_id: Optional[str] = None,
        call_purpose: Optional[str] = None,
    ) -> OutboundCallResult:
        """
        Initiate an outbound call with optional AMD.

        Args:
            to_number: Phone number to call (E.164 format)
            callback_base_url: Base URL for webhooks (e.g., https://example.com)
            customer_id: Optional customer ID for tracking
            call_purpose: Optional purpose for logging

        Returns:
            OutboundCallResult with call_sid or error
        """
        try:
            client = self._get_client()

            # Build callback URLs
            twiml_url = f"{callback_base_url}/outbound-call"
            status_callback = f"{callback_base_url}/call-status"
            
            # Call creation parameters
            call_params = {
                "to": to_number,
                "from_": SETTINGS.twilio_phone_number,
                "url": twiml_url,
                "status_callback": status_callback,
                "status_callback_event": ["initiated", "ringing", "answered", "completed"],
                "status_callback_method": "POST",
            }

            # Add AMD if enabled
            if SETTINGS.voicemail_detection_enabled:
                amd_callback = f"{callback_base_url}/amd-callback"
                call_params.update({
                    "machine_detection": "DetectMessageEnd",
                    "machine_detection_timeout": SETTINGS.voicemail_detection_timeout,
                    "machine_detection_speech_threshold": SETTINGS.voicemail_speech_threshold,
                    "async_amd": True,
                    "async_amd_status_callback": amd_callback,
                    "async_amd_status_callback_method": "POST",
                })
                logger.info("AMD enabled for outbound call to %s", to_number[:6] + "****")

            call = client.calls.create(**call_params)

            logger.info(
                "Outbound call initiated: sid=%s to=%s purpose=%s",
                call.sid, to_number[:6] + "****", call_purpose,
            )

            return OutboundCallResult(success=True, call_sid=call.sid)

        except ImportError as e:
            logger.error("Twilio package error: %s", e)
            return OutboundCallResult(success=False, error=str(e))
        except Exception as e:
            logger.exception("Error initiating outbound call: %s", e)
            return OutboundCallResult(success=False, error=str(e))


# Global instance
_outbound_service: Optional[OutboundCallService] = None


def get_outbound_service() -> OutboundCallService:
    """Get or create outbound call service."""
    global _outbound_service
    if _outbound_service is None:
        _outbound_service = OutboundCallService()
    return _outbound_service
