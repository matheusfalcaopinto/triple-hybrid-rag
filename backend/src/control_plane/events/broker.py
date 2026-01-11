"""Event broker for real-time event distribution."""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class EventBroker:
    """In-memory event broker for real-time updates.
    
    This broker distributes events to connected clients via SSE/WebSocket.
    For production scale, consider Redis pub/sub.
    """

    def __init__(self):
        # Mapping of (establishment_id, channel) -> set of queues
        self._subscriptions: dict[tuple[UUID, str], set[asyncio.Queue]] = defaultdict(set)
        # Lock for thread-safe subscription management
        self._lock = asyncio.Lock()

    async def subscribe(
        self, 
        establishment_id: UUID, 
        channel: str,
    ) -> asyncio.Queue:
        """Subscribe to events for a channel.
        
        Args:
            establishment_id: The establishment to subscribe to
            channel: Event channel (e.g., "calls", "dashboard", "agents")
            
        Returns:
            Queue that will receive events
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        async with self._lock:
            key = (establishment_id, channel)
            self._subscriptions[key].add(queue)
            logger.debug(f"New subscription for {key}, total: {len(self._subscriptions[key])}")
            
        return queue

    async def unsubscribe(
        self,
        establishment_id: UUID,
        channel: str,
        queue: asyncio.Queue,
    ) -> None:
        """Unsubscribe from events.
        
        Args:
            establishment_id: The establishment
            channel: Event channel
            queue: The queue to remove
        """
        async with self._lock:
            key = (establishment_id, channel)
            self._subscriptions[key].discard(queue)
            
            # Clean up empty sets
            if not self._subscriptions[key]:
                del self._subscriptions[key]
                
            logger.debug(f"Removed subscription for {key}")

    async def publish(
        self,
        establishment_id: UUID,
        channel: str,
        event_type: str,
        data: dict[str, Any],
    ) -> int:
        """Publish an event to all subscribers.
        
        Args:
            establishment_id: Target establishment
            channel: Event channel
            event_type: Type of event (e.g., "call.started")
            data: Event payload
            
        Returns:
            Number of subscribers notified
        """
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        async with self._lock:
            key = (establishment_id, channel)
            subscribers = self._subscriptions.get(key, set())
            
            # Also publish to "all" channel for this establishment
            all_key = (establishment_id, "all")
            all_subscribers = self._subscriptions.get(all_key, set())
            
            combined = subscribers | all_subscribers
            
        count = 0
        for queue in combined:
            try:
                queue.put_nowait(event)
                count += 1
            except asyncio.QueueFull:
                # Skip slow consumers
                logger.warning("Event queue full, dropping event")
                
        if count > 0:
            logger.debug(f"Published {event_type} to {count} subscribers")
            
        return count

    async def publish_call_event(
        self,
        establishment_id: UUID,
        call_id: UUID,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> int:
        """Convenience method for call events."""
        payload = {
            "call_id": str(call_id),
            **(data or {}),
        }
        return await self.publish(
            establishment_id=establishment_id,
            channel="calls",
            event_type=f"call.{event_type}",
            data=payload,
        )

    async def publish_transcript_segment(
        self,
        establishment_id: UUID,
        call_id: UUID,
        segment: dict[str, Any],
    ) -> int:
        """Publish a real-time transcript segment."""
        return await self.publish(
            establishment_id=establishment_id,
            channel="calls",
            event_type="call.transcript",
            data={
                "call_id": str(call_id),
                "segment": segment,
            },
        )

    async def publish_dashboard_update(
        self,
        establishment_id: UUID,
        metric: str,
        value: Any,
    ) -> int:
        """Publish dashboard metric update."""
        return await self.publish(
            establishment_id=establishment_id,
            channel="dashboard",
            event_type="metric.updated",
            data={
                "metric": metric,
                "value": value,
            },
        )

    def get_subscriber_count(
        self,
        establishment_id: UUID | None = None,
        channel: str | None = None,
    ) -> int:
        """Get number of active subscribers."""
        count = 0
        for (est_id, ch), queues in self._subscriptions.items():
            if establishment_id and est_id != establishment_id:
                continue
            if channel and ch != channel:
                continue
            count += len(queues)
        return count


# Singleton instance
event_broker = EventBroker()
