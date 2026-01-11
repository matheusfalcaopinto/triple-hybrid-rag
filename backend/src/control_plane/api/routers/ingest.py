"""Ingest routes (runtime -> backend events)."""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.db.session import get_session
from control_plane.schemas.ingest import IngestCallEventRequest, IngestResponse
from control_plane.services.call_service import call_service

router = APIRouter()


@router.post("/call-events", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_call_event(
    request: IngestCallEventRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Ingest call events from agent runtime."""
    # Get or create call
    call = await call_service.get_call_by_sid(
        session, request.call.provider_call_sid
    )

    if not call:
        # Create call if it doesn't exist
        call = await call_service.create_call(
            session,
            establishment_id=request.establishment_id,
            provider=request.call.provider,
            provider_call_sid=request.call.provider_call_sid,
            direction=request.call.direction,
            from_e164=request.call.from_e164,
            to_e164=request.call.to_e164,
            agent_id=request.agent_id,
            agent_version_id=request.agent_version_id,
        )

    event = request.event
    payload = event.payload

    # Process event based on type
    if event.type == "call_started":
        await call_service.update_call_status(
            session,
            call.id,
            status="in_progress",
            started_at=payload.get("started_at"),
        )

    elif event.type == "call_ended":
        await call_service.update_call_status(
            session,
            call.id,
            status="completed",
            ended_at=payload.get("ended_at"),
            duration_seconds=payload.get("duration_seconds"),
            final_status=payload.get("final_status"),
            hangup_by=payload.get("hangup_by"),
        )

    elif event.type == "transcript_segment":
        started_at = payload.get("started_at")
        ended_at = payload.get("ended_at")
        
        # Parse datetime strings if needed
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))

        await call_service.add_transcript_segment(
            session,
            call_id=call.id,
            segment_id=payload.get("segment_id", event.id),
            speaker=payload.get("speaker", "unknown"),
            text=payload.get("text", ""),
            started_at=started_at or datetime.now(timezone.utc),
            ended_at=ended_at or datetime.now(timezone.utc),
            confidence=payload.get("confidence"),
            idempotency_key=event.idempotency_key,
        )

    elif event.type == "tool_called":
        await call_service.add_tool_log(
            session,
            call_id=call.id,
            tool_name=payload.get("tool_name", "unknown"),
            occurred_at=event.occurred_at,
            idempotency_key=event.idempotency_key,
            arguments=payload.get("arguments"),
        )

    elif event.type == "tool_result":
        await call_service.add_tool_log(
            session,
            call_id=call.id,
            tool_name=payload.get("tool_name", "unknown"),
            occurred_at=event.occurred_at,
            idempotency_key=event.idempotency_key,
            result=payload.get("result"),
            error=payload.get("error"),
            duration_ms=payload.get("duration_ms"),
        )

    elif event.type == "recording_available":
        await call_service.add_recording(
            session,
            call_id=call.id,
            recording_url=payload.get("recording_url", ""),
            storage_provider=payload.get("storage_provider", "s3"),
            content_type=payload.get("content_type", "audio/wav"),
            duration_seconds=payload.get("duration_seconds"),
        )

    elif event.type == "runtime_error":
        await call_service.add_call_event(
            session,
            call_id=call.id,
            event_type="runtime_error",
            occurred_at=event.occurred_at,
            idempotency_key=event.idempotency_key,
            payload=payload,
        )

    # Always store raw event
    await call_service.add_call_event(
        session,
        call_id=call.id,
        event_type=event.type,
        occurred_at=event.occurred_at,
        idempotency_key=event.idempotency_key,
        payload=payload,
    )

    await session.commit()

    return IngestResponse(status="accepted")
