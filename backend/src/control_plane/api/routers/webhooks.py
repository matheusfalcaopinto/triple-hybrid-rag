"""Webhook routes (Twilio, etc.)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.db.session import get_session
from control_plane.services.call_service import call_service
from control_plane.services.twilio_router import twilio_router

router = APIRouter()


@router.post("/twilio/incoming-call")
async def twilio_incoming_call(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Handle Twilio incoming call webhook."""
    form = await request.form()
    
    called = form.get("Called", "")
    caller = form.get("From", "")
    call_sid = form.get("CallSid", "")

    routing_result = await twilio_router.route_inbound_call(
        session,
        called_number=str(called),
        caller_number=str(caller),
        call_sid=str(call_sid),
    )

    # Create call record if routing succeeded
    if routing_result.get("found"):
        await call_service.create_call(
            session,
            establishment_id=routing_result["establishment_id"],
            provider="twilio",
            provider_call_sid=str(call_sid),
            direction="inbound",
            from_e164=str(caller),
            to_e164=str(called),
            agent_id=routing_result.get("agent_id"),
            runtime_id=routing_result.get("runtime_id"),
        )
        await session.commit()

    return Response(
        content=routing_result["twiml"],
        media_type="application/xml",
    )


@router.post("/twilio/call-status")
async def twilio_call_status(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Handle Twilio call status webhook."""
    form = await request.form()
    
    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    duration = form.get("CallDuration", "0")

    # Find and update call
    call = await call_service.get_call_by_sid(session, str(call_sid))
    if call:
        status_map = {
            "queued": "queued",
            "ringing": "ringing",
            "in-progress": "in_progress",
            "completed": "completed",
            "busy": "busy",
            "failed": "failed",
            "no-answer": "no_answer",
            "canceled": "cancelled",
        }
        new_status = status_map.get(str(call_status).lower(), str(call_status))
        
        await call_service.update_call_status(
            session,
            call.id,
            status=new_status,
            duration_seconds=int(duration) if duration else None,
        )
        await session.commit()

    return Response(content="", media_type="text/plain")


@router.post("/twilio/amd-callback")
async def twilio_amd_callback(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Handle Twilio Answering Machine Detection callback."""
    form = await request.form()
    
    call_sid = form.get("CallSid", "")
    answered_by = form.get("AnsweredBy", "")

    # Find call and record AMD result
    call = await call_service.get_call_by_sid(session, str(call_sid))
    if call:
        from datetime import datetime, timezone

        await call_service.add_call_event(
            session,
            call_id=call.id,
            event_type="amd_result",
            occurred_at=datetime.now(timezone.utc),
            idempotency_key=f"{call_sid}:amd:{answered_by}",
            payload={"answered_by": str(answered_by)},
        )
        await session.commit()

    return Response(content="", media_type="text/plain")
