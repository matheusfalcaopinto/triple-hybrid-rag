from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional, Set

from .storage import (
    CommunicationEvent,
    get_latest_event,
    record_event,
    record_event_sync,
)


async def record_dispatch_async(
    *,
    correlation_id: str,
    channel: str,
    status: str,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a dispatch event asynchronously."""
    event = CommunicationEvent(
        id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        channel=channel,
        event_type="dispatch",
        status=status,
        payload=payload or {},
    )
    await record_event(event)
    return event.id


def record_update_sync(
    *,
    correlation_id: str,
    channel: str,
    status: str,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist an update event synchronously (usable from background threads)."""
    event = CommunicationEvent(
        id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        channel=channel,
        event_type="update",
        status=status,
        payload=payload or {},
    )
    record_event_sync(event)
    return event.id


async def record_update_async(
    *,
    correlation_id: str,
    channel: str,
    status: str,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist an update event asynchronously."""
    event = CommunicationEvent(
        id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        channel=channel,
        event_type="update",
        status=status,
        payload=payload or {},
    )
    await record_event(event)
    return event.id


async def record_incoming_async(
    *,
    correlation_id: str,
    channel: str,
    status: str,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist an incoming-message event asynchronously."""
    event = CommunicationEvent(
        id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        channel=channel,
        event_type="incoming",
        status=status,
        payload=payload or {},
    )
    await record_event(event)
    return event.id


async def wait_for_final_status(
    correlation_id: str,
    *,
    timeout: float = 3.0,
    poll_interval: float = 0.25,
    pending_statuses: Optional[Set[str]] = None,
) -> Optional[CommunicationEvent]:
    """Poll for a non-dispatch status event within a timeout window."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    pending = {
        status.lower()
        for status in (pending_statuses or {"scheduled", "pending", "queued"})
    }
    last_event: Optional[CommunicationEvent] = None

    while True:
        event = await get_latest_event(correlation_id)
        if event:
            last_event = event
            normalized_status = (event.status or "").lower()
            if event.event_type != "dispatch" and normalized_status not in pending:
                return event

        if loop.time() >= deadline:
            return last_event

        await asyncio.sleep(poll_interval)
