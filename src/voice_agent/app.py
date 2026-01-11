"""
FastAPI Application for Pipecat Voice Agent

This is the main FastAPI application that handles Twilio webhooks and WebSocket
connections for the Pipecat-based voice agent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from .config import SETTINGS

# Configure logging
logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("voice_agent_pipecat.app")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Voice Agent Pipecat")
    logger.info("OpenAI Model: %s", SETTINGS.openai_model)
    logger.info("Cartesia TTS: %s", SETTINGS.cartesia_tts_model)
    logger.info("Cartesia STT: %s", SETTINGS.cartesia_stt_model)
    logger.info("VAD Threshold: %s", SETTINGS.vad_threshold)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Voice Agent Pipecat")

# Create FastAPI app
app = FastAPI(
    title="Voice Agent Pipecat",
    description="Pipecat-based real-time voice agent",
    version="0.1.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Application State & Session Management
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionInfo:
    """Information about an active call session."""
    
    call_sid: str
    caller_phone: str
    trace_id: str
    start_time: float = field(default_factory=time.monotonic)
    status: str = "active"


class AppState:
    """Application state tracker with session management."""
    
    def __init__(self) -> None:
        self.active_calls: int = 0
        self.is_draining: bool = False
        self.total_calls_handled: int = 0
        self.sessions: Dict[str, SessionInfo] = {}
        self._lock = asyncio.Lock()
        
    _instance: Optional["AppState"] = None
    
    @classmethod
    def get(cls) -> "AppState":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def add_session(
        self,
        call_sid: str,
        caller_phone: str,
        trace_id: str,
    ) -> SessionInfo:
        """Add a new session."""
        async with self._lock:
            session = SessionInfo(
                call_sid=call_sid,
                caller_phone=caller_phone,
                trace_id=trace_id,
            )
            self.sessions[call_sid] = session
            self.active_calls += 1
            self.total_calls_handled += 1
            return session
    
    async def remove_session(self, call_sid: str) -> Optional[SessionInfo]:
        """Remove a session and return its info."""
        async with self._lock:
            session = self.sessions.pop(call_sid, None)
            if session:
                self.active_calls = max(0, self.active_calls - 1)
                session.status = "ended"
            return session
    
    def get_session(self, call_sid: str) -> Optional[SessionInfo]:
        """Get session info by call_sid."""
        return self.sessions.get(call_sid)


# ──────────────────────────────────────────────────────────────────────────────
# Health & Metrics Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> Dict[str, Any]:
    """Root endpoint."""
    return {"ok": True, "message": "voice_agent_pipecat", "framework": "pipecat"}


@app.get("/healthz")
def health() -> JSONResponse:
    """Liveness probe."""
    return JSONResponse({"status": "ok"}, status_code=200)


@app.get("/readyz")
def ready() -> JSONResponse:
    """Readiness probe - checks if all services are configured."""
    issues: list[str] = []
    
    if not SETTINGS.openai_api_key:
        issues.append("openai_api_key_missing")
    if not SETTINGS.cartesia_api_key:
        issues.append("cartesia_api_key_missing")
    if not SETTINGS.cartesia_voice_id:
        issues.append("cartesia_voice_id_missing")
    if not SETTINGS.supabase_url:
        issues.append("supabase_url_missing")
    if not SETTINGS.supabase_service_role_key:
        issues.append("supabase_service_role_key_missing")
    
    if issues:
        return JSONResponse(
            {"status": "not_ready", "issues": issues},
            status_code=503,
        )
    
    return JSONResponse({"status": "ready"}, status_code=200)


@app.get("/info")
def info() -> Dict[str, Any]:
    """Application info endpoint with session details."""
    state = AppState.get()
    return {
        "version": "0.1.0",
        "framework": "pipecat",
        "status": "draining" if state.is_draining else "active",
        "active_calls": state.active_calls,
        "total_calls_handled": state.total_calls_handled,
        "config": {
            "llm_model": SETTINGS.openai_model,
            "tts_model": SETTINGS.cartesia_tts_model,
            "stt_model": SETTINGS.cartesia_stt_model,
        },
    }


@app.get("/sessions")
def list_sessions() -> Dict[str, Any]:
    """List active sessions."""
    state = AppState.get()
    sessions = []
    current_time = time.monotonic()
    
    for call_sid, session in state.sessions.items():
        duration = current_time - session.start_time
        sessions.append({
            "call_sid": call_sid,
            "caller_phone": session.caller_phone[:6] + "****",  # Mask phone
            "duration_seconds": round(duration, 1),
            "status": session.status,
        })
    
    return {
        "active_sessions": len(sessions),
        "sessions": sessions,
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    state = AppState.get()
    
    lines = [
        "# HELP voice_agent_active_calls Number of active calls",
        "# TYPE voice_agent_active_calls gauge",
        f"voice_agent_active_calls {state.active_calls}",
        "",
        "# HELP voice_agent_total_calls_handled Total calls handled",
        "# TYPE voice_agent_total_calls_handled counter",
        f"voice_agent_total_calls_handled {state.total_calls_handled}",
        "",
        "# HELP voice_agent_is_draining Whether the app is draining",
        "# TYPE voice_agent_is_draining gauge",
        f"voice_agent_is_draining {1 if state.is_draining else 0}",
    ]
    
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Control Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/control/drain")
def drain() -> JSONResponse:
    """Set application to draining mode."""
    state = AppState.get()
    state.is_draining = True
    logger.info("Application entering draining mode")
    return JSONResponse({"status": "draining", "active_calls": state.active_calls})


@app.post("/control/resume")
def resume() -> JSONResponse:
    """Resume accepting calls after draining."""
    state = AppState.get()
    state.is_draining = False
    logger.info("Application resumed from draining mode")
    return JSONResponse({"status": "active", "active_calls": state.active_calls})


# ──────────────────────────────────────────────────────────────────────────────
# Twilio Webhook Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/")
async def incoming_call_root(request: Request) -> Response:
    """Handle POST to root (Twilio sometimes omits /incoming-call)."""
    return await incoming_call(request)


@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    """
    Handle incoming Twilio call webhook.
    Returns TwiML to establish a Media Stream connection.
    """
    if AppState.get().is_draining:
        logger.warning("Rejecting incoming call due to draining state")
        return Response(status_code=503)
    
    form_data = await request.form()
    
    # Extract caller info
    from_number = form_data.get("From", "")
    to_number = form_data.get("To", "")
    call_direction = form_data.get("Direction", "inbound")
    call_sid_twilio = form_data.get("CallSid", "")
    
    # Determine customer phone based on call direction
    if call_direction in ("outbound-api", "outbound-dial"):
        caller_phone = to_number
    else:
        caller_phone = from_number
    
    logger.info(
        "Incoming call: From=%s, To=%s, Direction=%s, CallerPhone=%s",
        from_number, to_number, call_direction, caller_phone,
    )
    
    # Build WebSocket URL
    ws_path = SETTINGS.twilio_ws_path
    
    # Check for configured public domain (e.g. ngrok or production domain)
    if SETTINGS.app_public_domain:
        host = SETTINGS.app_public_domain
        # If public domain is set, we default to secure wss unless overridden
        ws_protocol = SETTINGS.app_scheme_override or "wss"
        ws_url = f"{ws_protocol}://{host}{ws_path}"
    else:
        # Fallback: Infer from request (useful for local dev without config)
        ws_protocol = "wss" if request.url.scheme == "https" else "ws"
        host = request.url.hostname or "localhost"
        port = request.url.port
        
        if port and port not in {80, 443}:
            ws_url = f"{ws_protocol}://{host}:{port}{ws_path}"
        else:
            ws_url = f"{ws_protocol}://{host}{ws_path}"
    
    # Add query parameters
    if caller_phone:
        ws_url += f"?caller_phone={quote(caller_phone)}"
    if call_sid_twilio:
        separator = "&" if "?" in ws_url else "?"
        ws_url += f"{separator}call_sid={quote(call_sid_twilio)}"
    
    logger.info("Responding with stream URL: %s", ws_url)
    
    # Build TwiML response
    twiml = _build_stream_twiml(ws_url)
    return Response(content=twiml, media_type="application/xml")


def _build_stream_twiml(ws_url: str) -> str:
    """Build TwiML for Twilio Media Stream connection."""
    escaped_url = escape(ws_url)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{escaped_url}"/></Connect>'
        "</Response>"
    )


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Media Stream Handler
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket(SETTINGS.twilio_ws_path)
async def media_stream(ws: WebSocket) -> None:
    """
    Handle Twilio Media Stream WebSocket connection.
    This is where the Pipecat pipeline runs for each call.
    """
    await ws.accept()
    
    caller_phone = ws.query_params.get("caller_phone", "")
    call_sid = ws.query_params.get("call_sid", "") or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())
    
    logger.info(
        "WebSocket connected: call_sid=%s, caller=%s, trace_id=%s",
        call_sid, caller_phone, trace_id,
    )
    
    state = AppState.get()
    session = await state.add_session(call_sid, caller_phone, trace_id)
    
    try:
        # Import here to avoid circular imports and allow lazy loading
        from .transports.twilio_transport import TwilioWebsocketTransport, TwilioTransportParams
        from .bot import run_bot
        
        # Create transport for this WebSocket
        transport_params = TwilioTransportParams(
            sample_rate=SETTINGS.twilio_sample_rate,
            enable_interruptions=True,
        )
        transport = TwilioWebsocketTransport(
            ws,
            sample_rate=SETTINGS.twilio_sample_rate,
            params=transport_params,
        )
        
        # Run the bot pipeline
        await run_bot(
            transport=transport,
            caller_phone=caller_phone,
            call_sid=call_sid,
        )
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: call_sid=%s", call_sid)
    except asyncio.CancelledError:
        logger.info("WebSocket handler cancelled: call_sid=%s", call_sid)
    except Exception as e:
        logger.exception("WebSocket error for call_sid=%s: %s", call_sid, e)
    finally:
        ended_session = await state.remove_session(call_sid)
        if ended_session:
            duration = time.monotonic() - ended_session.start_time
            logger.info(
                "Call ended: call_sid=%s, duration=%.1fs, active_calls=%d",
                call_sid, duration, state.active_calls,
            )

# ──────────────────────────────────────────────────────────────────────────────
# Outbound Call & Voicemail Detection Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/amd-callback")
async def amd_callback(request: Request) -> JSONResponse:
    """
    Handle Twilio AMD (Answering Machine Detection) callback.
    This is called when Twilio determines if a human or machine answered.
    """
    form_data = await request.form()
    
    call_sid = form_data.get("CallSid", "")
    answered_by = form_data.get("AnsweredBy", "unknown")
    machine_detection_duration = form_data.get("MachineDetectionDuration", "0")
    
    logger.info(
        "AMD result: call_sid=%s answered_by=%s duration=%sms",
        call_sid, answered_by, machine_detection_duration,
    )
    
    # Get session and update with AMD result
    state = AppState.get()
    session = state.get_session(call_sid)
    
    if session:
        # Store AMD result in session (extend SessionInfo if needed)
        # For now, just log it
        from .services.outbound import AnsweredBy as AnsweredByEnum
        answered_by_enum = AnsweredByEnum.from_twilio(answered_by)
        
        if answered_by_enum.is_voicemail:
            logger.info("Voicemail detected for call_sid=%s", call_sid)
            # The bot.py should check this and switch to voicemail mode
    
    return JSONResponse({"status": "ok", "answered_by": answered_by})


@app.post("/outbound-call")
async def outbound_call_webhook(request: Request) -> Response:
    """
    TwiML webhook for outbound calls.
    Returns TwiML to connect the call to the media stream.
    """
    form_data = await request.form()
    
    call_sid = form_data.get("CallSid", "")
    to_number = form_data.get("To", "")
    from_number = form_data.get("From", "")
    
    logger.info(
        "Outbound call webhook: call_sid=%s from=%s to=%s",
        call_sid, from_number, to_number,
    )
    
    # Build WebSocket URL for outbound call
    ws_path = SETTINGS.twilio_ws_path
    
    if SETTINGS.app_public_domain:
        host = SETTINGS.app_public_domain
        ws_protocol = SETTINGS.app_scheme_override or "wss"
        ws_url = f"{ws_protocol}://{host}{ws_path}"
    else:
        # Need to have a public domain configured for outbound calls
        logger.error("APP_PUBLIC_DOMAIN not configured for outbound calls")
        return Response(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Configuration error</Say><Hangup/></Response>',
            media_type="application/xml",
        )
    
    # Add query parameters
    ws_url += f"?caller_phone={quote(to_number)}&call_sid={quote(call_sid)}&direction=outbound"
    
    logger.info("Outbound call: connecting to stream URL: %s", ws_url)
    
    twiml = _build_stream_twiml(ws_url)
    return Response(content=twiml, media_type="application/xml")


@app.post("/call-status")
async def call_status_webhook(request: Request) -> JSONResponse:
    """
    Handle Twilio call status callback.
    This is called when the call status changes (initiated, ringing, answered, completed).
    """
    form_data = await request.form()
    
    call_sid = form_data.get("CallSid", "")
    call_status = form_data.get("CallStatus", "")
    call_duration = form_data.get("CallDuration", "0")
    
    logger.info(
        "Call status update: call_sid=%s status=%s duration=%ss",
        call_sid, call_status, call_duration,
    )
    
    return JSONResponse({
        "status": "ok",
        "call_sid": call_sid,
        "call_status": call_status,
    })


@app.post("/api/outbound-call")
async def initiate_outbound_call(request: Request) -> JSONResponse:
    """
    API endpoint to initiate an outbound call.
    
    Request Body:
        - to_number: Phone number to call (E.164 format)
        - customer_id: Optional customer ID for tracking
        - purpose: Optional call purpose
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body"},
            status_code=400,
        )
    
    to_number = data.get("to_number")
    if not to_number:
        return JSONResponse(
            {"error": "to_number is required"},
            status_code=400,
        )
    
    customer_id = data.get("customer_id")
    purpose = data.get("purpose", "general")
    
    # Get callback base URL
    if SETTINGS.app_public_domain:
        scheme = SETTINGS.app_scheme_override or "https"
        callback_base_url = f"{scheme}://{SETTINGS.app_public_domain}"
    else:
        return JSONResponse(
            {"error": "APP_PUBLIC_DOMAIN not configured"},
            status_code=500,
        )
    
    # Initiate call
    from .services.outbound import get_outbound_service
    
    service = get_outbound_service()
    result = service.initiate_call(
        to_number=to_number,
        callback_base_url=callback_base_url,
        customer_id=customer_id,
        call_purpose=purpose,
    )
    
    if result.success:
        return JSONResponse({
            "status": "initiated",
            "call_sid": result.call_sid,
        })
    else:
        return JSONResponse(
            {"error": result.error},
            status_code=500,
        )


# ──────────────────────────────────────────────────────────────────────────────
# WhatsApp Calling Webhooks (Meta Business API)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/whatsapp/call/webhook")
async def whatsapp_call_webhook_verify(request: Request) -> Response:
    """
    Verify Meta webhook subscription.
    Meta sends a GET request to verify the webhook URL.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if mode == "subscribe" and token == SETTINGS.whatsapp_calling_webhook_verify_token:
        logger.info("WhatsApp calling webhook verified")
        return Response(content=challenge, media_type="text/plain")
    
    logger.warning("WhatsApp webhook verification failed")
    return Response(status_code=403)


@app.post("/whatsapp/call/webhook")
async def whatsapp_call_webhook(request: Request) -> JSONResponse:
    """
    Handle WhatsApp calling webhook events.
    
    Events include:
    - incoming_call: New incoming call
    - call_accepted: Call was accepted
    - call_ended: Call ended
    - ice_candidate: ICE candidate from remote peer
    """
    # Verify signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    body = await request.body()
    
    from .services.meta_calling import MetaCallingClient
    if not MetaCallingClient.verify_webhook_signature(body, signature):
        logger.warning("Invalid webhook signature")
        return JSONResponse({"error": "Invalid signature"}, status_code=403)
    
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    # Process webhook entries
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") == "calls":
                await _handle_whatsapp_call_event(change.get("value", {}))
    
    return JSONResponse({"status": "ok"})


async def _handle_whatsapp_call_event(event: Dict[str, Any]) -> None:
    """
    Handle individual WhatsApp call events.
    
    Args:
        event: Call event data from Meta webhook
    """
    event_type = event.get("event_type")
    call_id = event.get("call_id", "")
    from_number = event.get("from", "")
    
    logger.info("WhatsApp call event: type=%s call_id=%s from=%s", 
                event_type, call_id, from_number[:6] + "****" if from_number else "N/A")
    
    if event_type == "incoming_call":
        await _handle_whatsapp_incoming_call(event)
    elif event_type == "call_ended":
        await _handle_whatsapp_call_ended(event)
    elif event_type == "ice_candidate":
        await _handle_whatsapp_ice_candidate(event)


async def _handle_whatsapp_incoming_call(event: Dict[str, Any]) -> None:
    """Handle incoming WhatsApp call."""
    if not SETTINGS.whatsapp_calling_enabled:
        logger.info("WhatsApp calling disabled, ignoring incoming call")
        from .services.meta_calling import get_meta_calling_client
        client = get_meta_calling_client()
        await client.reject_call(event.get("call_id", ""), reason="declined")
        return
    
    call_id = event.get("call_id", "")
    from_number = event.get("from", "")
    sdp_offer = event.get("sdp", "")
    
    if not sdp_offer:
        logger.error("No SDP offer in incoming call event")
        return
    
    # Create WebRTC transport
    from .transports.whatsapp_webrtc import (
        WhatsAppWebRTCTransport,
        register_transport,
    )
    
    transport = WhatsAppWebRTCTransport(
        call_id=call_id,
        sample_rate=16000,
    )
    
    # Initialize WebRTC and get SDP answer
    sdp_answer = await transport.initialize(sdp_offer)
    
    if not sdp_answer:
        logger.error("Failed to create SDP answer for call_id=%s", call_id)
        from .services.meta_calling import get_meta_calling_client
        client = get_meta_calling_client()
        await client.reject_call(call_id, reason="error")
        return
    
    # Accept the call with SDP answer
    from .services.meta_calling import get_meta_calling_client
    client = get_meta_calling_client()
    result = await client.accept_call(call_id, sdp_answer)
    
    if not result.success:
        logger.error("Failed to accept call: %s", result.error)
        await transport.close()
        return
    
    # Register transport and start bot
    register_transport(call_id, transport)
    
    # Start the bot pipeline in background
    asyncio.create_task(_run_whatsapp_bot(transport, call_id, from_number))
    
    logger.info("WhatsApp call accepted: call_id=%s from=%s", call_id, from_number[:6] + "****")


async def _run_whatsapp_bot(
    transport: Any,
    call_id: str,
    caller_phone: str,
) -> None:
    """Run the bot pipeline for a WhatsApp call."""
    from .transports.whatsapp_webrtc import unregister_transport
    
    state = AppState.get()
    trace_id = str(uuid.uuid4())
    
    session = await state.add_session(call_id, caller_phone, trace_id)
    
    try:
        from .bot import run_bot
        
        await run_bot(
            transport=transport,
            caller_phone=caller_phone,
            call_sid=call_id,
        )
    except Exception as e:
        logger.exception("WhatsApp bot error for call_id=%s: %s", call_id, e)
    finally:
        await transport.close()
        unregister_transport(call_id)
        ended_session = await state.remove_session(call_id)
        if ended_session:
            duration = time.monotonic() - ended_session.start_time
            logger.info("WhatsApp call ended: call_id=%s duration=%.1fs", call_id, duration)


async def _handle_whatsapp_call_ended(event: Dict[str, Any]) -> None:
    """Handle WhatsApp call ended event."""
    call_id = event.get("call_id", "")
    reason = event.get("reason", "unknown")
    
    logger.info("WhatsApp call ended: call_id=%s reason=%s", call_id, reason)
    
    # Close transport if exists
    from .transports.whatsapp_webrtc import get_transport, unregister_transport
    
    transport = get_transport(call_id)
    if transport:
        await transport.close()
        unregister_transport(call_id)


async def _handle_whatsapp_ice_candidate(event: Dict[str, Any]) -> None:
    """Handle ICE candidate from remote peer."""
    call_id = event.get("call_id", "")
    candidate = event.get("candidate", "")
    sdp_mid = event.get("sdpMid", "")
    sdp_mline_index = event.get("sdpMLineIndex", 0)
    
    from .transports.whatsapp_webrtc import get_transport
    
    transport = get_transport(call_id)
    if transport:
        await transport.add_ice_candidate(candidate, sdp_mid, sdp_mline_index)


# ──────────────────────────────────────────────────────────────────────────────
# Startup Events
# ──────────────────────────────────────────────────────────────────────────────
