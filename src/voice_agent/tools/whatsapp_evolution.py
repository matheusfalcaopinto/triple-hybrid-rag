"""
WhatsApp Tools for MCP (Evolution API Implementation)

Provides WhatsApp messaging integration via Evolution API for:
- Sending text messages
- Sending media (images, videos, documents, audio)
- Sending location (via Google Maps link)
- Sending PTT voice messages (TTS generated audio)

Requires Evolution API instance with WhatsApp connected.
Reference: https://doc.evolution-api.com/v2/
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from voice_agent.communication import events as comm_events
from voice_agent.communication.storage import CommunicationEvent, record_event_sync
from voice_agent.services.evolution_client import (
    EVOLUTION_AVAILABLE,
    EvolutionAPIError,
    EvolutionClient,
    get_evolution_client,
)
from voice_agent.services.whatsapp_voice import (
    generate_audio_reply,
    media_base_url_configured,
)
from voice_agent.config import SETTINGS

logger = logging.getLogger("voice_agent.whatsapp_evolution")


def _record_dispatch_event(
    *,
    correlation_id: str,
    channel: str,
    status: str,
    payload: Dict[str, Any],
) -> None:
    """Record dispatch event for communication tracking."""
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


async def _send_text_message_async(
    to: str,
    message: str,
    from_number: Optional[str] = None,
    *,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a text message via WhatsApp using Evolution API.
    """
    try:
        client = get_evolution_client()
        
        response = await client.send_text(to, message)
        
        message_id = response.get("key", {}).get("id", "")
        status = response.get("status", "sent")
        
        if correlation_id:
            await comm_events.record_update_async(
                correlation_id=correlation_id,
                channel="whatsapp",
                status=status,
                payload={
                    "to": to,
                    "message_id": message_id,
                    "status": status,
                    "preview": message[:160],
                },
            )
        
        return {
            "success": True,
            "message_id": message_id,
            "status": status,
            "to": to,
            "from": SETTINGS.evolution_whatsapp_from or SETTINGS.evolution_instance_name,
            "message": "WhatsApp message sent successfully",
            "correlation_id": correlation_id,
            "channel": "whatsapp",
            "backend": "evolution",
        }
        
    except ValueError as e:
        if correlation_id:
            await comm_events.record_update_async(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e)},
            )
        return {"success": False, "error": str(e), "setup_required": True}
    except EvolutionAPIError as e:
        if correlation_id:
            await comm_events.record_update_async(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={
                    "error": str(e),
                    "status_code": e.status_code,
                    "response": e.response_data,
                },
            )
        return {
            "success": False,
            "error": f"Evolution API error: {str(e)}",
            "status_code": e.status_code,
        }
    except Exception as e:
        logger.exception("Failed to send WhatsApp text: %s", e)
        if correlation_id:
            await comm_events.record_update_async(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e)},
            )
        return {"success": False, "error": f"Failed to send message: {str(e)}"}


def send_text_message(
    to: str,
    message: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a WhatsApp text message (sync wrapper)."""
    logger.info("send_text_message (evolution) invoked from module=%s", __name__)
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={"to": to, "message_preview": message[:160]},
    )
    
    try:
        loop = asyncio.get_running_loop()
        # Use thread to avoid blocking event loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                _send_text_message_async(to, message, from_number, correlation_id=correlation_id)
            )
            return future.result()
    except RuntimeError:
        return asyncio.run(
            _send_text_message_async(to, message, from_number, correlation_id=correlation_id)
        )


async def _send_media_message_async(
    to: str,
    media_url: str,
    media_type: str = "image",
    caption: Optional[str] = None,
    filename: Optional[str] = None,
    from_number: Optional[str] = None,
    *,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a media message via WhatsApp using Evolution API.
    """
    try:
        client = get_evolution_client()
        
        response = await client.send_media(
            to,
            media_type,
            media_url,
            caption=caption,
            filename=filename,
        )
        
        message_id = response.get("key", {}).get("id", "")
        status = response.get("status", "sent")
        
        if correlation_id:
            await comm_events.record_update_async(
                correlation_id=correlation_id,
                channel="whatsapp",
                status=status,
                payload={
                    "to": to,
                    "media_url": media_url,
                    "media_type": media_type,
                    "message_id": message_id,
                    "status": status,
                },
            )
        
        return {
            "success": True,
            "message_id": message_id,
            "status": status,
            "media_url": media_url,
            "message": f"{media_type.capitalize()} sent successfully",
            "correlation_id": correlation_id,
            "channel": "whatsapp",
            "backend": "evolution",
        }
        
    except ValueError as e:
        if correlation_id:
            await comm_events.record_update_async(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "media_url": media_url},
            )
        return {"success": False, "error": str(e), "setup_required": True}
    except EvolutionAPIError as e:
        if correlation_id:
            await comm_events.record_update_async(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "media_url": media_url},
            )
        return {
            "success": False,
            "error": f"Evolution API error: {str(e)}",
            "status_code": e.status_code,
        }
    except Exception as e:
        logger.exception("Failed to send WhatsApp media: %s", e)
        if correlation_id:
            await comm_events.record_update_async(
                correlation_id=correlation_id,
                channel="whatsapp",
                status="error",
                payload={"error": str(e), "media_url": media_url},
            )
        return {"success": False, "error": f"Failed to send media: {str(e)}"}


def _run_async(coro):
    """Helper to run async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


def send_media_message(
    to: str,
    media_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a media message via WhatsApp."""
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={"to": to, "media_url": media_url, "caption": caption},
    )
    return _run_async(
        _send_media_message_async(
            to, media_url, "image", caption, from_number=from_number,
            correlation_id=correlation_id
        )
    )


def send_image(
    to: str,
    image_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an image via WhatsApp."""
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={"to": to, "media_url": image_url, "type": "image"},
    )
    return _run_async(
        _send_media_message_async(
            to, image_url, "image", caption, from_number=from_number,
            correlation_id=correlation_id
        )
    )


def send_video(
    to: str,
    video_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a video via WhatsApp."""
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={"to": to, "media_url": video_url, "type": "video"},
    )
    return _run_async(
        _send_media_message_async(
            to, video_url, "video", caption, from_number=from_number,
            correlation_id=correlation_id
        )
    )


def send_document(
    to: str,
    document_url: str,
    filename: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a document via WhatsApp."""
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={"to": to, "media_url": document_url, "type": "document"},
    )
    return _run_async(
        _send_media_message_async(
            to, document_url, "document", caption=None, filename=filename,
            from_number=from_number, correlation_id=correlation_id
        )
    )


def send_audio(
    to: str,
    audio_url: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an audio file via WhatsApp."""
    correlation_id = str(uuid.uuid4())
    _record_dispatch_event(
        correlation_id=correlation_id,
        channel="whatsapp",
        status="scheduled",
        payload={"to": to, "media_url": audio_url, "type": "audio"},
    )
    return _run_async(
        _send_media_message_async(
            to, audio_url, "audio", from_number=from_number,
            correlation_id=correlation_id
        )
    )


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
    Generate TTS audio and send as WhatsApp PTT voice note.
    """
    try:
        client = get_evolution_client()
        
        # Show "recording..." presence
        try:
            await client.set_presence(to, "recording", delay=3000)
        except Exception as e:
            logger.warning("Failed to set recording presence: %s", e)
        
        # Generate audio using existing TTS pipeline
        mp3_path, media_url = await generate_audio_reply(message)
        
        # Send as audio (Evolution will handle PTT if format is correct)
        result = _run_async(
            _send_media_message_async(
                to, media_url, "audio", from_number=from_number,
                correlation_id=str(uuid.uuid4())
            )
        )
        result.update({
            "generated_media_url": media_url,
            "local_path": str(mp3_path),
        })
        return result
        
    except ValueError as e:
        return {"success": False, "error": str(e), "setup_required": True}
    except Exception as e:
        logger.exception("Failed to send generated audio: %s", e)
        return {"success": False, "error": f"Failed to send audio: {str(e)}"}


def send_template_message(
    to: str,
    content_sid: str,
    content_variables: Optional[Any] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a template message.
    
    Note: Evolution API with Baileys doesn't support official templates.
    Falls back to sending as a regular text message.
    """
    message = f"[Template: {content_sid}]"
    if content_variables:
        if isinstance(content_variables, dict):
            vars_str = ", ".join(f"{k}={v}" for k, v in content_variables.items())
        elif isinstance(content_variables, list):
            vars_str = ", ".join(
                f"{item.get('key')}={item.get('value')}"
                for item in content_variables
                if isinstance(item, dict)
            )
        else:
            vars_str = str(content_variables)
        message = f"{message}\nVariables: {vars_str}"
    
    logger.warning(
        "Template messages not supported with Evolution API Baileys. "
        "Sending as text message instead."
    )
    
    return send_text_message(to, message, from_number)


# =============================================================================
# Async Wrappers
# =============================================================================

async def send_text_message_async(
    to: str,
    message: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of send_text_message."""
    correlation_id = str(uuid.uuid4())
    return await _send_text_message_async(to, message, from_number, correlation_id=correlation_id)


async def send_media_message_async(
    to: str,
    media_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of send_media_message."""
    correlation_id = str(uuid.uuid4())
    return await _send_media_message_async(
        to, media_url, "image", caption, from_number=from_number,
        correlation_id=correlation_id
    )


async def send_image_async(
    to: str,
    image_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of send_image."""
    correlation_id = str(uuid.uuid4())
    return await _send_media_message_async(
        to, image_url, "image", caption, from_number=from_number,
        correlation_id=correlation_id
    )


async def send_video_async(
    to: str,
    video_url: str,
    caption: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of send_video."""
    correlation_id = str(uuid.uuid4())
    return await _send_media_message_async(
        to, video_url, "video", caption, from_number=from_number,
        correlation_id=correlation_id
    )


async def send_document_async(
    to: str,
    document_url: str,
    filename: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of send_document."""
    correlation_id = str(uuid.uuid4())
    return await _send_media_message_async(
        to, document_url, "document", filename=filename, from_number=from_number,
        correlation_id=correlation_id
    )


async def send_audio_async(
    to: str,
    audio_url: str,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of send_audio."""
    correlation_id = str(uuid.uuid4())
    return await _send_media_message_async(
        to, audio_url, "audio", from_number=from_number,
        correlation_id=correlation_id
    )


async def send_location_async(
    to: str,
    latitude: float,
    longitude: float,
    name: Optional[str] = None,
    address: Optional[str] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of send_location."""
    maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"
    
    message_parts = []
    if name:
        message_parts.append(f"📍 {name}")
    if address:
        message_parts.append(address)
    message_parts.append(f"Location: {maps_url}")
    
    message = "\n".join(message_parts)
    return await send_text_message_async(to, message, from_number)


async def send_template_message_async(
    to: str,
    content_sid: str,
    content_variables: Optional[Any] = None,
    from_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of send_template_message."""
    return await asyncio.to_thread(
        send_template_message, to, content_sid, content_variables, from_number
    )


# =============================================================================
# Tool Definitions for MCP Server
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "name": "send_whatsapp_message",
        "description": "Send a text message via WhatsApp using Evolution API.",
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
                "description": "Ignored for Evolution API (uses configured instance)"
            }
        },
        "required": ["to", "message"],
        "handler": send_text_message,
    },
    {
        "name": "send_whatsapp_image",
        "description": "Send an image via WhatsApp. Image must be publicly accessible.",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone with country code"},
            "image_url": {"type": "string", "description": "Public URL of the image"},
            "caption": {"type": "string", "description": "Optional caption"},
            "from_number": {"type": "string", "description": "Ignored for Evolution API"}
        },
        "required": ["to", "image_url"],
        "handler": send_image,
    },
    {
        "name": "send_whatsapp_video",
        "description": "Send a video via WhatsApp.",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone with country code"},
            "video_url": {"type": "string", "description": "Public URL of the video"},
            "caption": {"type": "string", "description": "Optional caption"},
            "from_number": {"type": "string", "description": "Ignored for Evolution API"}
        },
        "required": ["to", "video_url"],
        "handler": send_video,
    },
    {
        "name": "send_whatsapp_document",
        "description": "Send a document via WhatsApp.",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone with country code"},
            "document_url": {"type": "string", "description": "Public URL of the document"},
            "filename": {"type": "string", "description": "Optional filename"},
            "from_number": {"type": "string", "description": "Ignored for Evolution API"}
        },
        "required": ["to", "document_url"],
        "handler": send_document,
    },
    {
        "name": "send_whatsapp_audio",
        "description": "Send an audio file via WhatsApp.",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone with country code"},
            "audio_url": {"type": "string", "description": "Public URL of the audio"},
            "from_number": {"type": "string", "description": "Ignored for Evolution API"}
        },
        "required": ["to", "audio_url"],
        "handler": send_audio,
    },
    {
        "name": "send_whatsapp_generated_audio",
        "description": "Synthesize TTS audio and send as WhatsApp voice note.",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone with country code"},
            "message": {"type": "string", "description": "Text to synthesize"},
            "from_number": {"type": "string", "description": "Ignored for Evolution API"}
        },
        "required": ["to", "message"],
        "handler": send_generated_audio,
    },
    {
        "name": "send_whatsapp_location",
        "description": "Send a location via WhatsApp as a Google Maps link.",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone with country code"},
            "latitude": {"type": "number", "description": "Location latitude"},
            "longitude": {"type": "number", "description": "Location longitude"},
            "name": {"type": "string", "description": "Optional location name"},
            "address": {"type": "string", "description": "Optional address"},
            "from_number": {"type": "string", "description": "Ignored for Evolution API"}
        },
        "required": ["to", "latitude", "longitude"],
        "handler": send_location,
    },
    {
        "name": "send_whatsapp_template",
        "description": "Send a template message (falls back to text with Baileys).",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone with country code"},
            "content_sid": {"type": "string", "description": "Template ID"},
            "content_variables": {
                "type": "array",
                "description": "Template variables",
                "items": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}}
            },
            "from_number": {"type": "string", "description": "Ignored for Evolution API"}
        },
        "required": ["to", "content_sid"],
        "handler": send_template_message,
    },
]
