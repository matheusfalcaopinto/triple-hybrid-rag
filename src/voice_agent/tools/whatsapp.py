"""
WhatsApp Tools for MCP (Multi-Backend Implementation)

Provides WhatsApp messaging integration with support for:
- Evolution API (default): Uses Baileys for WhatsApp Web emulation
- Twilio (fallback): Uses Twilio's WhatsApp API

Backend is selected via WHATSAPP_BACKEND environment variable:
- "evolution" (default): Use Evolution API
- "twilio": Use Twilio WhatsApp API

For Evolution API: Requires Evolution API instance with WhatsApp connected.
For Twilio: Requires Twilio account with WhatsApp enabled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Dict, Optional, Type

from voice_agent.communication import events as comm_events
from voice_agent.communication.storage import CommunicationEvent, record_event_sync
from voice_agent.services.whatsapp_voice import (
    generate_audio_reply,
    media_base_url_configured,
)
from voice_agent.config import SETTINGS

logger = logging.getLogger("voice_agent.whatsapp")

# =============================================================================
# Backend Routing: Evolution API vs Twilio
# =============================================================================

_USE_EVOLUTION = SETTINGS.whatsapp_backend.lower() == "evolution"

if _USE_EVOLUTION:
    logger.info("WhatsApp backend: Evolution API")
    try:
        from voice_agent.tools.whatsapp_evolution import (
            TOOL_DEFINITIONS as EVOLUTION_TOOL_DEFINITIONS,
            send_text_message as _evo_send_text_message,
            send_media_message as _evo_send_media_message,
            send_image as _evo_send_image,
            send_video as _evo_send_video,
            send_document as _evo_send_document,
            send_audio as _evo_send_audio,
            send_location as _evo_send_location,
            send_generated_audio as _evo_send_generated_audio,
            send_template_message as _evo_send_template_message,
            send_text_message_async as _evo_send_text_message_async,
            send_media_message_async as _evo_send_media_message_async,
            send_image_async as _evo_send_image_async,
            send_video_async as _evo_send_video_async,
            send_document_async as _evo_send_document_async,
            send_audio_async as _evo_send_audio_async,
            send_location_async as _evo_send_location_async,
            send_template_message_async as _evo_send_template_message_async,
        )
        EVOLUTION_IMPORTED = True
    except ImportError as e:
        logger.warning("Failed to import Evolution API module: %s. Falling back to Twilio.", e)
        EVOLUTION_IMPORTED = False
        _USE_EVOLUTION = False
else:
    logger.info("WhatsApp backend: Twilio")
    EVOLUTION_IMPORTED = False

# Twilio imports (fallback)

_TwilioRestException: Type[Exception]
Client: Any

try:
    from twilio.base.exceptions import (  # type: ignore[import]
        TwilioRestException as _ImportedTwilioRestException,
    )
    from twilio.rest import Client as _ImportedTwilioClient  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency

    TWILIO_AVAILABLE = False

    class _TwilioRestExceptionFallback(Exception):
        """Fallback Twilio error with optional metadata attributes."""

        def __init__(
            self,
            *args: Any,
            code: str | None = None,
            status: str | None = None,
            msg: str | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.code = code
            self.status = status
            if msg is not None:
                self.msg = msg
            elif args:
                self.msg = str(args[0])
            else:
                self.msg = ""

    class _MissingClient:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - safety
            raise ImportError(
                "Twilio library not available. Install with 'pip install twilio'."
            )

    _TwilioRestException = _TwilioRestExceptionFallback
    Client = _MissingClient()  # type: ignore[assignment]
else:
    TWILIO_AVAILABLE = True
    _TwilioRestException = _ImportedTwilioRestException
    Client = _ImportedTwilioClient

TwilioRestException = _TwilioRestException


def get_twilio_client() -> Any:
    """
    Get authenticated Twilio client for WhatsApp.
    
    Returns:
        Twilio Client instance
        
    Raises:
        ImportError: If twilio library not installed
        ValueError: If credentials not configured
    """
    if not TWILIO_AVAILABLE:
        raise ImportError(
            "Twilio library not available. "
            "Install with: pip install twilio"
        )
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    if not account_sid:
        raise ValueError(
            "TWILIO_ACCOUNT_SID not configured. "
            "Get your credentials from https://www.twilio.com/console"
        )
    
    if not auth_token:
        raise ValueError(
            "TWILIO_AUTH_TOKEN not configured. "
            "Get your auth token from https://www.twilio.com/console"
        )
    
    return Client(account_sid, auth_token)


def _format_whatsapp_number(phone: str) -> str:
    """
    Format phone number for WhatsApp (add whatsapp: prefix if needed).
    
    Args:
        phone: Phone number with country code (e.g., '+5511999990001')
        
    Returns:
        Formatted WhatsApp number (e.g., 'whatsapp:+5511999990001')
    """
    if phone.startswith('whatsapp:'):
        return phone
    if not phone.startswith('+'):
        phone = '+' + phone
    return f'whatsapp:{phone}'


def _status_callback_url(correlation_id: Optional[str]) -> Optional[str]:
    """Build status callback URL using COMMUNICATION_WEBHOOK_BASE."""

    base = SETTINGS.communication_webhook_base.strip()
    if not base or not correlation_id:
        return None
    return f"{base.rstrip('/')}/webhooks/twilio-status?cid={correlation_id}"


def _normalize_template_variables(raw: Any) -> Dict[str, str]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        normalized: Dict[str, str] = {}
        for entry in raw:
            if not isinstance(entry, dict) or "key" not in entry or "value" not in entry:
                raise ValueError("Each template variable must include 'key' and 'value' fields.")
            normalized[str(entry["key"])] = str(entry["value"])
        return normalized
    raise ValueError("content_variables must be a mapping or a list of {key, value} objects.")


def _send_text_message_blocking(
    to: str,
    message: str,
    from_number: Optional[str] = None,
    *,
    correlation_id: Optional[str] = None,
    status_callback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a text message via WhatsApp.
    
    Args:
        to: Recipient phone number with country code (e.g., '+5511999990001')
        message: Text message to send
        from_number: Your Twilio WhatsApp number (defaults to TWILIO_WHATSAPP_FROM env var)
        
    Returns:
        Message send result or error message
        
    Example:
        >>> send_text_message('+5511999990001', 'Hello from Voice Agent!')
        {'success': True, 'message_id': 'SMxxx...', 'status': 'queued'}
    """
    try:
        logger.info("whatsapp.get_twilio_client ref=%r", get_twilio_client)
        client = get_twilio_client()
        logger.info(
            "Resolved Twilio client %s (module=%s)",
            type(client),
            getattr(type(client), "__module__", ""),
        )
        
        if from_number is None:
            from_number = os.getenv("TWILIO_WHATSAPP_FROM")
            if not from_number:
                return {
                    "error": "TWILIO_WHATSAPP_FROM not configured",
                    "setup_required": True,
                    "message": "Set your Twilio WhatsApp number in .env"
                }
        
        params = {
            "from_": _format_whatsapp_number(from_number),
            "body": message,
            "to": _format_whatsapp_number(to),
        }

        if status_callback:
            params["status_callback"] = status_callback

        message_obj = client.messages.create(**params)

        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status=message_obj.status or "unknown",
                payload={
                    "to": to,
                    "message_id": message_obj.sid,
                    "status": message_obj.status,
                    "preview": message[:160],
                },
            )

        result = {
            "success": True,
            "message_id": message_obj.sid,
            "status": message_obj.status,
            "to": to,
            "from": from_number,
            "message": "WhatsApp message sent successfully",
            "correlation_id": correlation_id,
            "channel": "whatsapp",
        }
        return result

    except ImportError as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e)},
            )
        return {"success": False, "error": str(e), "install_required": True}
    except ValueError as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e)},
            )
        return {"success": False, "error": str(e), "setup_required": True}
    except TwilioRestException as e:
        error_msg = getattr(e, "msg", str(e))
        error_code = getattr(e, "code", None)
        error_status = getattr(e, "status", None)
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={
                    "code": error_code,
                    "status": error_status,
                    "message": error_msg,
                },
            )
        return {
            "success": False,
            "error": f"Twilio API error: {error_msg}",
            "code": error_code,
            "status": error_status,
        }
    except Exception as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e)},
            )
        return {"success": False, "error": f"Failed to send message: {str(e)}"}


def _dispatch_async(
    runner: Callable[[], Dict[str, Any]],
    description: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    result = runner()

    if result.get("success"):
        logger.info(
            "%s succeeded: %s",
            description,
            {
                key: result.get(key)
                for key in ("to", "message_id", "status", "template")
                if result.get(key) is not None
            },
        )
    else:
        logger.error("%s failed: %s", description, result)

    return result


def _record_dispatch_event(
    *,
    correlation_id: str,
    channel: str,
    status: str,
    payload: Dict[str, Any],
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        event = CommunicationEvent(
            id=str(uuid.uuid4()),
            correlation_id=correlation_id,
            channel=channel,
            event_type="dispatch",
            status=status,
            payload=payload,
        )
        record_event_sync(event)
    else:
        loop.create_task(
            comm_events.record_dispatch_async(
                correlation_id=correlation_id,
                channel=channel,
                status=status,
                payload=payload,
            )
        )



def send_text_message(
    to: str,
    message: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a WhatsApp text message or dispatch it asynchronously if needed."""
    logger.info("send_text_message invoked from module=%s", __name__)
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={
            "to": to,
            "message_preview": message[:160],
        },
    )

    status_callback = _status_callback_url(correlation_id)

    return _dispatch_async(
        lambda: _send_text_message_blocking(
            to,
            message,
            from_number,
            correlation_id=correlation_id,
            status_callback=status_callback,
        ),
        "WhatsApp text message",
        {"to": to, "correlation_id": correlation_id, "channel": "whatsapp"},
    )


def _send_media_message_blocking(
    to: str,
    media_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
    *,
    correlation_id: Optional[str] = None,
    status_callback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a media message (image, video, document, audio) via WhatsApp.
    
    Args:
        to: Recipient phone number with country code
        media_url: Public URL of the media file
        caption: Optional caption for the media
        from_number: Your Twilio WhatsApp number
        
    Returns:
        Message send result or error message
        
    Note:
        Media must be publicly accessible via HTTPS.
        Supported formats: https://www.twilio.com/docs/sms/accepted-mime-types
    """
    try:
        client = get_twilio_client()
        
        if from_number is None:
            from_number = os.getenv("TWILIO_WHATSAPP_FROM")
            if not from_number:
                return {
                    "error": "TWILIO_WHATSAPP_FROM not configured",
                    "setup_required": True
                }
        
        params = {
            "from_": _format_whatsapp_number(from_number),
            "to": _format_whatsapp_number(to),
            "media_url": [media_url],
        }
        
        if caption:
            params["body"] = caption
        if status_callback:
            params["status_callback"] = status_callback
        
        message_obj = client.messages.create(**params)

        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status=message_obj.status or "unknown",
                payload={
                    "to": to,
                    "media_url": media_url,
                    "message_id": message_obj.sid,
                    "status": message_obj.status,
                    "caption": caption,
                },
            )
        
        return {
            "success": True,
            "message_id": message_obj.sid,
            "status": message_obj.status,
            "media_url": media_url,
            "message": "Media message sent successfully",
            "correlation_id": correlation_id,
            "channel": "whatsapp",
        }
        
    except ImportError as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "media_url": media_url},
            )
        return {"success": False, "error": str(e), "install_required": True}
    except ValueError as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "media_url": media_url},
            )
        return {"success": False, "error": str(e), "setup_required": True}
    except TwilioRestException as e:
        error_msg = getattr(e, "msg", str(e))
        error_code = getattr(e, "code", None)
        error_status = getattr(e, "status", None)
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={
                    "error": error_msg,
                    "code": error_code,
                    "status": error_status,
                    "media_url": media_url,
                },
            )
        return {
            "success": False,
            "error": f"Twilio API error: {error_msg}",
            "code": error_code,
        }
    except Exception as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "media_url": media_url},
            )
        return {"success": False, "error": f"Failed to send media: {str(e)}"}


def send_media_message(
    to: str,
    media_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={
            "to": to,
            "media_url": media_url,
            "caption": caption,
        },
    )

    status_callback = _status_callback_url(correlation_id)

    return _dispatch_async(
        lambda: _send_media_message_blocking(
            to,
            media_url,
            caption,
            from_number,
            correlation_id=correlation_id,
            status_callback=status_callback,
        ),
        "WhatsApp media message",
        {
            "to": to,
            "media_url": media_url,
            "correlation_id": correlation_id,
            "channel": "whatsapp",
        },
    )


def send_image(
    to: str,
    image_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an image via WhatsApp."""
    return send_media_message(to, image_url, caption, from_number)


def send_video(
    to: str,
    video_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a video via WhatsApp."""
    return send_media_message(to, video_url, caption, from_number)


def send_document(
    to: str,
    document_url: str,
    filename: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a document via WhatsApp."""
    caption = filename if filename else None
    return send_media_message(to, document_url, caption, from_number)


def send_audio(
    to: str,
    audio_url: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an audio file via WhatsApp."""
    return send_media_message(to, audio_url, None, from_number)


def send_location(
    to: str,
    latitude: float,
    longitude: float,
    name: Optional[str] = None,
    address: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a location via WhatsApp as a Google Maps link."""
    maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"

    message_parts = []
    if name:
        message_parts.append(f"📍 {name}")
    if address:
        message_parts.append(address)
    message_parts.append(f"Location: {maps_url}")

    message = "\n".join(message_parts)

    return send_text_message(to, message, from_number)



async def send_generated_audio(
    to: str,
    message: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate Cartesia TTS audio and send via WhatsApp.
    """
    if not media_base_url_configured():
        return {
            "success": False,
            "error": "Configure WHATSAPP_MEDIA_BASE_URL to expose generated audio.",
            "setup_required": True,
        }

    audio_path, media_url = await generate_audio_reply(message)
    result = send_audio(to, media_url, from_number=from_number)
    result.update(
        {
            "generated_media_url": media_url,
            "local_path": str(audio_path),
        }
    )
    return result


def _send_template_message_blocking(
    to: str,
    content_sid: str,
    content_variables: Optional[Dict[str, str]] = None,
    from_number: Optional[str] = None,
    *,
    correlation_id: Optional[str] = None,
    status_callback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a pre-approved template message via WhatsApp.
    
    Templates must be created in Twilio Console: Content Templates
    https://www.twilio.com/console/content-editor
    
    Args:
        to: Recipient phone number with country code
        content_sid: Twilio Content SID (e.g., 'HXxxxxx...')
        content_variables: Template variable substitutions
        from_number: Your Twilio WhatsApp number
        
    Returns:
        Message send result or error message
        
    Example:
        >>> send_template_message(
        ...     '+5511999990001',
        ...     'HX1234567890abcdef1234567890abcdef',
        ...     {'1': 'João', '2': 'Tomorrow'}
        ... )
    """
    try:
        client = get_twilio_client()
        
        if from_number is None:
            from_number = os.getenv("TWILIO_WHATSAPP_FROM")
            if not from_number:
                return {
                    "error": "TWILIO_WHATSAPP_FROM not configured",
                    "setup_required": True
                }
        
        params: dict[str, object] = {
            "from_": _format_whatsapp_number(from_number),
            "to": _format_whatsapp_number(to),
            "content_sid": content_sid,
        }

        variables_payload = _normalize_template_variables(content_variables)
        if variables_payload:
            params["content_variables"] = variables_payload
        if status_callback:
            params["status_callback"] = status_callback

        message_obj = client.messages.create(**params)
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status=message_obj.status or "unknown",
                payload={
                    "to": to,
                    "template": content_sid,
                    "message_id": message_obj.sid,
                    "status": message_obj.status,
                    "variables": variables_payload,
                },
            )

        return {
            "success": True,
            "message_id": message_obj.sid,
            "status": message_obj.status,
            "template": content_sid,
            "message": "Template message sent successfully",
            "correlation_id": correlation_id,
            "channel": "whatsapp",
        }
    except ImportError as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "template": content_sid},
            )
        return {"success": False, "error": str(e), "install_required": True}
    except ValueError as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "template": content_sid},
            )
        return {"success": False, "error": str(e), "setup_required": True}
    except TwilioRestException as e:
        error_msg = getattr(e, "msg", str(e))
        error_code = getattr(e, "code", None)
        error_status = getattr(e, "status", None)
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={
                    "error": error_msg,
                    "code": error_code,
                    "status": error_status,
                    "template": content_sid,
                },
            )
        return {
            "success": False,
            "error": f"Twilio API error: {error_msg}",
            "code": error_code,
            "hint": "Check that Content SID exists and is approved",
        }
    except Exception as e:
        if correlation_id:
            comm_events.record_update_sync(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "template": content_sid},
            )
        return {"success": False, "error": f"Failed to send template: {str(e)}"}


def send_template_message(
    to: str,
    content_sid: str,
    content_variables: Optional[Any] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        normalized_variables = _normalize_template_variables(content_variables)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={
            "to": to,
            "template": content_sid,
            "variables": normalized_variables,
        },
    )

    status_callback = _status_callback_url(correlation_id)

    return _dispatch_async(
        lambda: _send_template_message_blocking(
            to,
            content_sid,
            normalized_variables,
            from_number,
            correlation_id=correlation_id,
            status_callback=status_callback,
        ),
        "WhatsApp template message",
        {
            "to": to,
            "template": content_sid,
            "correlation_id": correlation_id,
            "channel": "whatsapp",
        },
    )


async def send_text_message_async(
    to: str,
    message: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_text_message`."""
    return await asyncio.to_thread(send_text_message, to, message, from_number)


async def send_media_message_async(
    to: str,
    media_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_media_message`."""
    return await asyncio.to_thread(send_media_message, to, media_url, caption, from_number)


async def send_image_async(
    to: str,
    image_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_image`."""
    return await asyncio.to_thread(send_image, to, image_url, caption, from_number)


async def send_video_async(
    to: str,
    video_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_video`."""
    return await asyncio.to_thread(send_video, to, video_url, caption, from_number)


async def send_document_async(
    to: str,
    document_url: str,
    filename: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_document`."""
    return await asyncio.to_thread(send_document, to, document_url, filename, from_number)


async def send_audio_async(
    to: str,
    audio_url: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_audio`."""
    return await asyncio.to_thread(send_audio, to, audio_url, from_number)


async def send_location_async(
    to: str,
    latitude: float,
    longitude: float,
    name: Optional[str] = None,
    address: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_location`."""
    return await asyncio.to_thread(
        send_location,
        to,
        latitude,
        longitude,
        name,
        address,
        from_number,
    )


async def send_template_message_async(
    to: str,
    content_sid: str,
    content_variables: Optional[Any] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for `send_template_message`."""
    return await asyncio.to_thread(
        send_template_message,
        to,
        content_sid,
        content_variables,
        from_number,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fire-and-Forget Handlers (Twilio)
# ══════════════════════════════════════════════════════════════════════════════
# These handlers return immediately and process in background.
# The agent can confirm "WhatsApp sendo enviado" without waiting for API response.

async def _fire_and_forget_task(
    coro_func,
    args: tuple,
    description: str,
) -> None:
    """Execute a coroutine in background and log result."""
    try:
        result = await coro_func(*args)
        if result.get("success"):
            logger.info(
                "[Fire-and-Forget] %s succeeded: to=%s",
                description,
                result.get("to", "unknown"),
            )
        else:
            logger.error(
                "[Fire-and-Forget] %s failed: %s",
                description,
                result.get("error", "Unknown error"),
            )
    except Exception as exc:
        logger.exception("[Fire-and-Forget] %s exception: %s", description, exc)


async def send_text_message_fire_and_forget(
    to: str,
    message: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget WhatsApp text message."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_text_message_async,
            (to, message, from_number),
            f"WhatsApp to {to}",
        )
    )
    return {
        "success": True,
        "status": "queued",
        "message": f"WhatsApp para {to} está sendo enviado",
        "to": to,
    }


async def send_image_fire_and_forget(
    to: str,
    image_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget WhatsApp image."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_image_async,
            (to, image_url, caption, from_number),
            f"WhatsApp image to {to}",
        )
    )
    return {
        "success": True,
        "status": "queued",
        "message": f"Imagem WhatsApp para {to} está sendo enviada",
        "to": to,
    }


async def send_video_fire_and_forget(
    to: str,
    video_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget WhatsApp video."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_video_async,
            (to, video_url, caption, from_number),
            f"WhatsApp video to {to}",
        )
    )
    return {
        "success": True,
        "status": "queued",
        "message": f"Vídeo WhatsApp para {to} está sendo enviado",
        "to": to,
    }


async def send_document_fire_and_forget(
    to: str,
    document_url: str,
    filename: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget WhatsApp document."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_document_async,
            (to, document_url, filename, from_number),
            f"WhatsApp document to {to}",
        )
    )
    return {
        "success": True,
        "status": "queued",
        "message": f"Documento WhatsApp para {to} está sendo enviado",
        "to": to,
    }


async def send_audio_fire_and_forget(
    to: str,
    audio_url: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget WhatsApp audio."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_audio_async,
            (to, audio_url, from_number),
            f"WhatsApp audio to {to}",
        )
    )
    return {
        "success": True,
        "status": "queued",
        "message": f"Áudio WhatsApp para {to} está sendo enviado",
        "to": to,
    }


async def send_location_fire_and_forget(
    to: str,
    latitude: float,
    longitude: float,
    name: Optional[str] = None,
    address: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget WhatsApp location."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_location_async,
            (to, latitude, longitude, name, address, from_number),
            f"WhatsApp location to {to}",
        )
    )
    return {
        "success": True,
        "status": "queued",
        "message": f"Localização WhatsApp para {to} está sendo enviada",
        "to": to,
    }


async def send_template_message_fire_and_forget(
    to: str,
    content_sid: str,
    content_variables: Optional[Any] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Fire-and-forget WhatsApp template message."""
    asyncio.create_task(
        _fire_and_forget_task(
            send_template_message_async,
            (to, content_sid, content_variables, from_number),
            f"WhatsApp template to {to}",
        )
    )
    return {
        "success": True,
        "status": "queued",
        "message": f"Template WhatsApp para {to} está sendo enviado",
        "to": to,
        "template": content_sid,
    }


# Tool definitions for MCP server
# NOTE: Using fire-and-forget handlers for instant response to the agent.

TOOL_DEFINITIONS = [
    {
        "name": "send_whatsapp_message",
        "description": "Send a text message via WhatsApp using Twilio. Simple and reliable. Returns immediately while message sends in background.",
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient phone with country code (e.g., '+5511999990001')"
            },
            "message": {
                "type": "string",
                "description": "Text message to send"
            },
            "from_number": {
                "type": "string",
                "description": "Your Twilio WhatsApp number (optional, uses TWILIO_WHATSAPP_FROM)"
            }
        },
        "required": ["to", "message"],
        "handler": send_text_message_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_whatsapp_image",
        "description": "Send an image via WhatsApp. Image must be publicly accessible via HTTPS. Returns immediately while message sends in background.",
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient phone with country code"
            },
            "image_url": {
                "type": "string",
                "description": "Public HTTPS URL of the image (JPEG, PNG)"
            },
            "caption": {
                "type": "string",
                "description": "Optional caption for the image"
            },
            "from_number": {
                "type": "string",
                "description": "Your Twilio WhatsApp number (optional)"
            }
        },
        "required": ["to", "image_url"],
        "handler": send_image_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_whatsapp_video",
        "description": "Send a video via WhatsApp. Video must be publicly accessible via HTTPS. Returns immediately while message sends in background.",
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient phone with country code"
            },
            "video_url": {
                "type": "string",
                "description": "Public HTTPS URL of the video (MP4, 3GP)"
            },
            "caption": {
                "type": "string",
                "description": "Optional caption for the video"
            },
            "from_number": {
                "type": "string",
                "description": "Your Twilio WhatsApp number (optional)"
            }
        },
        "required": ["to", "video_url"],
        "handler": send_video_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_whatsapp_document",
        "description": (
            "Send a document via WhatsApp; the file must be reachable over "
            "HTTPS. Returns immediately while message sends in background."
        ),
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient phone with country code"
            },
            "document_url": {
                "type": "string",
                "description": "Public HTTPS URL of the document (PDF, DOC, etc.)"
            },
            "filename": {
                "type": "string",
                "description": "Optional filename to display to recipient"
            },
            "from_number": {
                "type": "string",
                "description": "Your Twilio WhatsApp number (optional)"
            }
        },
        "required": ["to", "document_url"],
        "handler": send_document_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_whatsapp_audio",
        "description": (
            "Send an audio file via WhatsApp; the media must be accessible over "
            "HTTPS. Returns immediately while message sends in background."
        ),
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient phone with country code"
            },
            "audio_url": {
                "type": "string",
                "description": "Public HTTPS URL of the audio file (MP3, OGG)"
            },
            "from_number": {
                "type": "string",
                "description": "Your Twilio WhatsApp number (optional)"
            }
        },
        "required": ["to", "audio_url"],
        "handler": send_audio_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_whatsapp_generated_audio",
        "description": (
            "Synthesize a voice reply with Cartesia TTS and deliver it as a "
            "WhatsApp audio message."
        ),
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient phone with country code"
            },
            "message": {
                "type": "string",
                "description": "Text content to synthesize into audio"
            },
            "from_number": {
                "type": "string",
                "description": "Your Twilio WhatsApp number (optional)"
            }
        },
        "required": ["to", "message"],
        "handler": send_generated_audio,
    },
    {
        "name": "send_whatsapp_location",
        "description": "Send a location via WhatsApp as a Google Maps link.",
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient phone with country code"
            },
            "latitude": {
                "type": "number",
                "description": "Location latitude (e.g., -23.5505)"
            },
            "longitude": {
                "type": "number",
                "description": "Location longitude (e.g., -46.6333)"
            },
            "name": {
                "type": "string",
                "description": "Optional location name"
            },
            "address": {
                "type": "string",
                "description": "Optional address"
            },
            "from_number": {
                "type": "string",
                "description": "Your Twilio WhatsApp number (optional)"
            }
        },
        "required": ["to", "latitude", "longitude"],
        "handler": send_location_fire_and_forget,  # Fire-and-forget for instant response
    },
    {
        "name": "send_whatsapp_template",
        "description": (
            "Send a pre-approved template message (managed in Twilio Console). "
            "Returns immediately while message sends in background."
        ),
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient phone with country code"
            },
            "content_sid": {
                "type": "string",
                "description": "Twilio Content SID (from Content Editor)"
            },
            "content_variables": {
                "type": "array",
                "description": (
                    "Template variable substitutions as key/value pairs (e.g., "
                    "[{\"key\": \"1\", \"value\": \"Joao\"}])."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Template variable placeholder (e.g., '1').",
                        },
                        "value": {
                            "type": "string",
                            "description": "Value to inject for the placeholder.",
                        },
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                }
            },
            "from_number": {
                "type": "string",
                "description": "Your Twilio WhatsApp number (optional)"
            }
        },
        "required": ["to", "content_sid"],
        "handler": send_template_message_fire_and_forget,  # Fire-and-forget for instant response
    },
]

# =============================================================================
# Backend Routing: Override with Evolution API if enabled
# =============================================================================

# =============================================================================
# Helper Aliases: Expose Twilio implementations for testing regardless of active backend
# =============================================================================
send_text_message_twilio = send_text_message
send_media_message_twilio = send_media_message
send_image_twilio = send_image
send_video_twilio = send_video
send_document_twilio = send_document
send_audio_twilio = send_audio
send_location_twilio = send_location
send_generated_audio_twilio = send_generated_audio
send_template_message_twilio = send_template_message
# Async wrappers
send_text_message_async_twilio = send_text_message_async
send_media_message_async_twilio = send_media_message_async
send_image_async_twilio = send_image_async
send_video_async_twilio = send_video_async
send_document_async_twilio = send_document_async
send_audio_async_twilio = send_audio_async
send_location_async_twilio = send_location_async
send_template_message_async_twilio = send_template_message_async


if _USE_EVOLUTION and EVOLUTION_IMPORTED:
    # Override TOOL_DEFINITIONS with Evolution API versions
    TOOL_DEFINITIONS = EVOLUTION_TOOL_DEFINITIONS
    
    # Re-export Evolution API functions as main exports
    send_text_message = _evo_send_text_message
    send_media_message = _evo_send_media_message
    send_image = _evo_send_image
    send_video = _evo_send_video
    send_document = _evo_send_document
    send_audio = _evo_send_audio
    send_location = _evo_send_location
    send_generated_audio = _evo_send_generated_audio
    send_template_message = _evo_send_template_message
    send_text_message_async = _evo_send_text_message_async
    send_media_message_async = _evo_send_media_message_async
    send_image_async = _evo_send_image_async
    send_video_async = _evo_send_video_async
    send_document_async = _evo_send_document_async
    send_audio_async = _evo_send_audio_async
    send_location_async = _evo_send_location_async
    send_template_message_async = _evo_send_template_message_async
    
    logger.info("WhatsApp tools using Evolution API backend")
else:
    logger.info("WhatsApp tools using Twilio backend")
