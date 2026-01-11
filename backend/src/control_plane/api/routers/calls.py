"""Call routes."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import RoleChecker, get_current_user
from control_plane.db.models.call import Call, CallRecording
from control_plane.db.models.establishment import EstablishmentTelephonyPolicy
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.call import (
    ActiveCallsResponse,
    CallHandoffRequest,
    CallHandoffResponse,
    CallResponse,
    CallsListResponse,
    OutboundCallRequest,
    SentimentInfo,
    TranscriptResponse,
    TranscriptSegmentResponse,
)
from control_plane.services.call_service import call_service

router = APIRouter()

# Role checkers
require_operator = RoleChecker(["admin", "operator"])
require_viewer = RoleChecker(["admin", "operator", "viewer"])


def _call_to_response(call: Call) -> CallResponse:
    """Convert Call model to response."""
    sentiment = None
    if call.sentiment_label:
        sentiment = SentimentInfo(
            label=call.sentiment_label,
            score=call.sentiment_score,
            computed_at=call.sentiment_computed_at,
        )
    
    return CallResponse(
        call_id=call.id,
        provider=call.provider,
        provider_call_sid=call.provider_call_sid,
        establishment_id=call.establishment_id,
        agent_id=call.agent_id,
        agent_version_id=call.agent_version_id,
        status=call.status,
        direction=call.direction,
        from_e164=call.from_e164,
        to_e164=call.to_e164,
        started_at=call.started_at,
        answered_at=call.answered_at,
        ended_at=call.ended_at,
        duration_seconds=call.duration_seconds,
        sentiment=sentiment,
        hangup_by=call.hangup_by,
        final_status=call.final_status,
    )


@router.get("/active", response_model=ActiveCallsResponse)
async def get_active_calls(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str | None, Query()] = None,
):
    """Get all active calls."""
    # Filter by user's establishments if not superuser
    if not user.is_superuser and establishment_id:
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None,
        )
        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this establishment",
            )

    calls = await call_service.get_active_calls(session, establishment_id)
    
    return ActiveCallsResponse(
        items=[_call_to_response(c) for c in calls]
    )


@router.get("", response_model=CallsListResponse)
async def list_calls(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
    agent_id: Annotated[str | None, Query()] = None,
    call_status: Annotated[str | None, Query(alias="status")] = None,
    direction: Annotated[str | None, Query()] = None,
    from_date: Annotated[datetime | None, Query()] = None,
    to_date: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """List calls with filters."""
    # Check access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None,
        )
        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this establishment",
            )

    calls, total = await call_service.get_calls(
        session,
        establishment_id=establishment_id,
        agent_id=agent_id,
        status=call_status,
        direction=direction,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=offset,
    )

    return CallsListResponse(
        items=[_call_to_response(c) for c in calls],
        total=total,
    )


@router.get("/{call_id}", response_model=CallResponse)
async def get_call(
    call_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Get call details."""
    call = await call_service.get_call(session, call_id)
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found",
        )

    # Check access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == call.establishment_id),
            None,
        )
        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this call",
            )

    return _call_to_response(call)


@router.get("/{call_id}/transcript", response_model=TranscriptResponse)
async def get_call_transcript(
    call_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
):
    """Get call transcript with cursor pagination."""
    call = await call_service.get_call(session, call_id)
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found",
        )

    segments, next_cursor, is_complete = await call_service.get_transcript(
        session, call_id, cursor, limit
    )

    return TranscriptResponse(
        call_id=call_id,
        items=[
            TranscriptSegmentResponse(
                segment_id=s.segment_id,
                speaker=s.speaker,
                text=s.text,
                started_at=s.started_at,
                ended_at=s.ended_at,
                confidence=s.confidence,
            )
            for s in segments
        ],
        cursor=next_cursor,
        is_complete=is_complete,
    )


@router.get("/{call_id}/recording")
async def get_call_recording(
    call_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Get call recording URL."""
    recording = await call_service.get_recording(session, call_id)
    
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found",
        )

    return {
        "call_id": call_id,
        "recording_url": recording.recording_url,
        "content_type": recording.content_type,
        "duration_seconds": recording.duration_seconds,
    }


@router.get("/{call_id}/tool-log")
async def get_call_tool_log(
    call_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Get tool call log for a call."""
    logs = await call_service.get_tool_logs(session, call_id)
    
    return {
        "call_id": call_id,
        "items": [
            {
                "id": log.id,
                "tool_name": log.tool_name,
                "arguments": log.arguments,
                "result": log.result,
                "error": log.error,
                "occurred_at": log.occurred_at,
                "duration_ms": log.duration_ms,
            }
            for log in logs
        ],
    }


@router.post("/outbound", response_model=CallResponse, status_code=status.HTTP_201_CREATED)
async def initiate_outbound_call(
    request: OutboundCallRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
):
    """Initiate an outbound call."""
    # Check operator access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None,
        )
        if not membership or membership.role not in ["admin", "operator"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator access required",
            )

    # TODO: Integrate with Twilio to actually place the call
    # For now, create a call record in queued status
    
    from uuid import uuid4
    
    call = await call_service.create_call(
        session,
        establishment_id=establishment_id,
        provider="twilio",
        provider_call_sid=f"CA{uuid4().hex}",
        direction="outbound",
        from_e164=request.from_e164 or "+15550000000",
        to_e164=request.to_e164,
        agent_id=request.agent_id,
        agent_version_id=request.agent_version_id,
        campaign_id=request.campaign_id,
        lead_id=request.lead_id,
    )

    return _call_to_response(call)


@router.post("/{call_id}/handoff", response_model=CallHandoffResponse)
async def handoff_call(
    call_id: Annotated[str, Path()],
    request: CallHandoffRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Transfer call to handoff number."""
    call = await call_service.get_call(session, call_id)
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found",
        )

    if call.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only handoff active calls",
        )

    # Get handoff number from telephony policy
    result = await session.execute(
        select(EstablishmentTelephonyPolicy).where(
            EstablishmentTelephonyPolicy.establishment_id == call.establishment_id
        )
    )
    policy = result.scalar_one_or_none()

    if not policy or not policy.handoff_e164:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No handoff number configured",
        )

    # TODO: Integrate with Twilio to transfer the call
    # For now, just update the status

    call.status = "handoff"
    await session.flush()

    return CallHandoffResponse(
        call_id=call_id,
        status="handoff_initiated",
        handoff_e164=policy.handoff_e164,
    )
