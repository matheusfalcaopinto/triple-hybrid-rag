"""
Communication Events Storage

Refactored to use Supabase for storing communication events.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from voice_agent.utils.db import get_supabase_client

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class CommunicationEvent:
    """Normalized representation of a communication event."""

    id: str
    correlation_id: str
    channel: str
    event_type: str  # dispatch | update | incoming
    status: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """
    Ensure the database tables exist.
    For Supabase, this is handled by migrations/schema.sql.
    This function is kept for compatibility but does nothing.
    """
    pass


def _record_event_sync(event: CommunicationEvent) -> None:
    """Synchronously insert an event (used by background threads)."""
    try:
        supabase = get_supabase_client()
        
        data = {
            "id": event.id,
            "correlation_id": event.correlation_id,
            "channel": event.channel,
            "event_type": event.event_type,
            "status": event.status,
            "payload": event.payload,
            "created_at": event.created_at.isoformat(),
        }
        
        supabase.table("communication_events").insert(data).execute()
        
    except Exception as e:
        logger.error(f"Failed to record communication event: {e}")


async def record_event(event: CommunicationEvent) -> None:
    """Asynchronously insert an event by delegating to a thread executor."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _record_event_sync, event)


def record_event_sync(event: CommunicationEvent) -> None:
    """Public synchronous helper for background threads."""
    _record_event_sync(event)


def fetch_events_sync(correlation_id: str, limit: int = 20) -> List[CommunicationEvent]:
    """Fetch recent events synchronously."""
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("communication_events").select(
            "id, correlation_id, channel, event_type, status, payload, created_at"
        ).eq("correlation_id", correlation_id).order("created_at", desc=True).limit(limit).execute()
        
        events = []
        for row in response.data:
            events.append(CommunicationEvent(
                id=row["id"],
                correlation_id=row["correlation_id"],
                channel=row["channel"],
                event_type=row["event_type"],
                status=row["status"],
                payload=row["payload"] or {},
                created_at=datetime.fromisoformat(row["created_at"]),
            ))
        return events
        
    except Exception as e:
        logger.error(f"Failed to fetch communication events: {e}")
        return []


async def fetch_events(correlation_id: str, limit: int = 20) -> List[CommunicationEvent]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_events_sync, correlation_id, limit)


async def get_latest_event(correlation_id: str) -> Optional[CommunicationEvent]:
    events = await fetch_events(correlation_id, limit=1)
    return events[0] if events else None
