"""Call service - call management and artifacts."""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.db.models.call import (
    Call,
    CallEvent,
    CallRecording,
    CallSummary,
    CallToolLog,
    CallTranscriptSegment,
)

logger = logging.getLogger(__name__)


class CallService:
    """Service for managing calls and their artifacts."""

    async def create_call(
        self,
        session: AsyncSession,
        establishment_id: str,
        provider: str,
        provider_call_sid: str,
        direction: str,
        from_e164: str,
        to_e164: str,
        agent_id: str | None = None,
        agent_version_id: str | None = None,
        runtime_id: str | None = None,
        campaign_id: str | None = None,
        lead_id: str | None = None,
    ) -> Call:
        """Create a new call record."""
        call = Call(
            id=f"call_{uuid4().hex[:12]}",
            establishment_id=establishment_id,
            provider=provider,
            provider_call_sid=provider_call_sid,
            direction=direction,
            from_e164=from_e164,
            to_e164=to_e164,
            agent_id=agent_id,
            agent_version_id=agent_version_id,
            runtime_id=runtime_id,
            campaign_id=campaign_id,
            lead_id=lead_id,
            status="queued",
        )
        session.add(call)
        await session.flush()
        return call

    async def get_call_by_sid(
        self, session: AsyncSession, provider_call_sid: str
    ) -> Call | None:
        """Get a call by provider call SID."""
        result = await session.execute(
            select(Call).where(Call.provider_call_sid == provider_call_sid)
        )
        return result.scalar_one_or_none()

    async def get_call(self, session: AsyncSession, call_id: str) -> Call | None:
        """Get a call by ID."""
        result = await session.execute(
            select(Call).where(Call.id == call_id)
        )
        return result.scalar_one_or_none()

    async def update_call_status(
        self,
        session: AsyncSession,
        call_id: str,
        status: str,
        **kwargs: Any,
    ) -> Call | None:
        """Update call status and optional fields."""
        call = await self.get_call(session, call_id)
        if not call:
            return None

        call.status = status
        for key, value in kwargs.items():
            if hasattr(call, key):
                setattr(call, key, value)

        await session.flush()
        return call

    async def get_active_calls(
        self,
        session: AsyncSession,
        establishment_id: str | None = None,
    ) -> list[Call]:
        """Get all active calls, optionally filtered by establishment."""
        query = select(Call).where(
            Call.status.in_(["queued", "ringing", "in_progress"])
        )
        if establishment_id:
            query = query.where(Call.establishment_id == establishment_id)
        query = query.order_by(desc(Call.started_at))

        result = await session.execute(query)
        return list(result.scalars().all())

    async def get_calls(
        self,
        session: AsyncSession,
        establishment_id: str,
        agent_id: str | None = None,
        status: str | None = None,
        direction: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Call], int]:
        """Get calls with filters and pagination."""
        query = select(Call).where(Call.establishment_id == establishment_id)

        if agent_id:
            query = query.where(Call.agent_id == agent_id)
        if status:
            query = query.where(Call.status == status)
        if direction:
            query = query.where(Call.direction == direction)
        if from_date:
            query = query.where(Call.started_at >= from_date)
        if to_date:
            query = query.where(Call.started_at <= to_date)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = (await session.execute(count_query)).scalar() or 0

        # Get paginated results
        query = query.order_by(desc(Call.started_at)).offset(offset).limit(limit)
        result = await session.execute(query)

        return list(result.scalars().all()), total

    async def add_call_event(
        self,
        session: AsyncSession,
        call_id: str,
        event_type: str,
        occurred_at: datetime,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
    ) -> CallEvent | None:
        """Add a call event (idempotent)."""
        # Check for existing event with same idempotency key
        result = await session.execute(
            select(CallEvent).where(CallEvent.idempotency_key == idempotency_key)
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.debug(f"Duplicate event ignored: {idempotency_key}")
            return existing

        event = CallEvent(
            id=f"evt_{uuid4().hex[:12]}",
            call_id=call_id,
            event_type=event_type,
            occurred_at=occurred_at,
            idempotency_key=idempotency_key,
            payload=payload,
        )
        session.add(event)
        await session.flush()
        return event

    async def add_transcript_segment(
        self,
        session: AsyncSession,
        call_id: str,
        segment_id: str,
        speaker: str,
        text: str,
        started_at: datetime,
        ended_at: datetime,
        idempotency_key: str,
        confidence: float | None = None,
    ) -> CallTranscriptSegment | None:
        """Add a transcript segment (idempotent)."""
        # Check for existing segment
        result = await session.execute(
            select(CallTranscriptSegment).where(
                CallTranscriptSegment.idempotency_key == idempotency_key
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            return existing

        segment = CallTranscriptSegment(
            id=f"seg_{uuid4().hex[:12]}",
            call_id=call_id,
            segment_id=segment_id,
            speaker=speaker,
            text=text,
            started_at=started_at,
            ended_at=ended_at,
            confidence=confidence,
            idempotency_key=idempotency_key,
        )
        session.add(segment)
        await session.flush()
        return segment

    async def get_transcript(
        self,
        session: AsyncSession,
        call_id: str,
        cursor: str | None = None,
        limit: int = 100,
    ) -> tuple[list[CallTranscriptSegment], str | None, bool]:
        """Get transcript segments with cursor pagination."""
        query = select(CallTranscriptSegment).where(
            CallTranscriptSegment.call_id == call_id
        )

        if cursor:
            # Cursor is the segment_id to start after
            query = query.where(CallTranscriptSegment.segment_id > cursor)

        query = query.order_by(CallTranscriptSegment.started_at).limit(limit + 1)
        result = await session.execute(query)
        segments = list(result.scalars().all())

        # Check if there are more results
        has_more = len(segments) > limit
        if has_more:
            segments = segments[:limit]

        # Get next cursor
        next_cursor = segments[-1].segment_id if segments and has_more else None

        # Check if call is complete
        call = await self.get_call(session, call_id)
        is_complete = call is not None and call.status in [
            "completed",
            "failed",
            "busy",
            "no_answer",
            "cancelled",
        ]

        return segments, next_cursor, is_complete

    async def add_tool_log(
        self,
        session: AsyncSession,
        call_id: str,
        tool_name: str,
        occurred_at: datetime,
        idempotency_key: str,
        arguments: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> CallToolLog | None:
        """Add a tool call log (idempotent)."""
        # Check for existing
        existing = await session.execute(
            select(CallToolLog).where(CallToolLog.idempotency_key == idempotency_key)
        )
        if existing.scalar_one_or_none():
            return None

        log = CallToolLog(
            id=f"tl_{uuid4().hex[:12]}",
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            error=error,
            occurred_at=occurred_at,
            duration_ms=duration_ms,
            idempotency_key=idempotency_key,
        )
        session.add(log)
        await session.flush()
        return log

    async def get_tool_logs(
        self, session: AsyncSession, call_id: str
    ) -> list[CallToolLog]:
        """Get all tool logs for a call."""
        result = await session.execute(
            select(CallToolLog)
            .where(CallToolLog.call_id == call_id)
            .order_by(CallToolLog.occurred_at)
        )
        return list(result.scalars().all())

    async def add_recording(
        self,
        session: AsyncSession,
        call_id: str,
        recording_url: str,
        storage_provider: str,
        content_type: str = "audio/wav",
        duration_seconds: int | None = None,
        file_size_bytes: int | None = None,
    ) -> CallRecording:
        """Add recording reference."""
        recording = CallRecording(
            id=f"rec_{uuid4().hex[:12]}",
            call_id=call_id,
            recording_url=recording_url,
            storage_provider=storage_provider,
            content_type=content_type,
            duration_seconds=duration_seconds,
            file_size_bytes=file_size_bytes,
        )
        session.add(recording)
        await session.flush()
        return recording

    async def get_recording(
        self, session: AsyncSession, call_id: str
    ) -> CallRecording | None:
        """Get recording for a call."""
        result = await session.execute(
            select(CallRecording).where(CallRecording.call_id == call_id)
        )
        return result.scalar_one_or_none()

    async def add_summary(
        self,
        session: AsyncSession,
        call_id: str,
        summary_text: str,
        key_points: list[str] | None = None,
        action_items: list[str] | None = None,
        customer_intent: str | None = None,
        resolution_status: str | None = None,
        model_used: str | None = None,
    ) -> CallSummary:
        """Add call summary."""
        summary = CallSummary(
            id=f"sum_{uuid4().hex[:12]}",
            call_id=call_id,
            summary_text=summary_text,
            key_points=key_points,
            action_items=action_items,
            customer_intent=customer_intent,
            resolution_status=resolution_status,
            model_used=model_used,
        )
        session.add(summary)
        await session.flush()
        return summary

    async def get_summary(
        self, session: AsyncSession, call_id: str
    ) -> CallSummary | None:
        """Get summary for a call."""
        result = await session.execute(
            select(CallSummary).where(CallSummary.call_id == call_id)
        )
        return result.scalar_one_or_none()


# Singleton instance
call_service = CallService()
