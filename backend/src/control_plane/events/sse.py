"""Server-Sent Events router for real-time updates."""

import asyncio
import logging
from typing import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from control_plane.auth.deps import get_current_user_establishment
from control_plane.events.broker import event_broker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


async def event_generator(
    establishment_id: UUID,
    channels: list[str],
) -> AsyncGenerator[str, None]:
    """Generate SSE events for the client.
    
    Args:
        establishment_id: The establishment to listen for
        channels: List of channels to subscribe to
        
    Yields:
        SSE formatted event strings
    """
    queues: list[tuple[str, asyncio.Queue]] = []
    
    try:
        # Subscribe to all requested channels
        for channel in channels:
            queue = await event_broker.subscribe(establishment_id, channel)
            queues.append((channel, queue))
        
        # Send initial connection event
        yield f"event: connected\ndata: {{}}\n\n"
        
        # Send heartbeat every 30 seconds to keep connection alive
        heartbeat_interval = 30
        last_heartbeat = asyncio.get_event_loop().time()
        
        while True:
            # Check all queues for events
            for channel, queue in queues:
                try:
                    event = queue.get_nowait()
                    event_type = event.get("type", "message")
                    
                    # Format as SSE
                    import json
                    data = json.dumps(event)
                    yield f"event: {event_type}\ndata: {data}\n\n"
                    
                except asyncio.QueueEmpty:
                    pass
            
            # Send heartbeat if needed
            current_time = asyncio.get_event_loop().time()
            if current_time - last_heartbeat >= heartbeat_interval:
                yield f"event: heartbeat\ndata: {{}}\n\n"
                last_heartbeat = current_time
            
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.1)
            
    except asyncio.CancelledError:
        logger.debug(f"SSE connection cancelled for {establishment_id}")
    finally:
        # Unsubscribe from all channels
        for channel, queue in queues:
            await event_broker.unsubscribe(establishment_id, channel, queue)


@router.get("/stream")
async def stream_events(
    channels: list[str] = Query(
        default=["all"],
        description="Channels to subscribe to: calls, dashboard, agents, all",
    ),
    current: tuple = Depends(get_current_user_establishment),
):
    """Stream real-time events via Server-Sent Events.
    
    Clients can subscribe to specific channels:
    - `calls`: Call lifecycle and transcript events
    - `dashboard`: Metric updates
    - `agents`: Agent status changes
    - `all`: All events
    
    Example usage with JavaScript:
    ```javascript
    const eventSource = new EventSource('/api/v1/events/stream?channels=calls');
    eventSource.addEventListener('call.started', (event) => {
        console.log(JSON.parse(event.data));
    });
    ```
    """
    user, establishment = current
    
    return StreamingResponse(
        event_generator(establishment.id, channels),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/status")
async def get_event_status(
    current: tuple = Depends(get_current_user_establishment),
):
    """Get event system status."""
    user, establishment = current
    
    return {
        "subscribers": {
            "calls": event_broker.get_subscriber_count(establishment.id, "calls"),
            "dashboard": event_broker.get_subscriber_count(establishment.id, "dashboard"),
            "agents": event_broker.get_subscriber_count(establishment.id, "agents"),
            "all": event_broker.get_subscriber_count(establishment.id, "all"),
            "total": event_broker.get_subscriber_count(establishment.id),
        }
    }


# Alias for convenience
sse_router = router
