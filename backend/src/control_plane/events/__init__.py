"""Events package - real-time updates via WebSocket/SSE."""

from control_plane.events.broker import event_broker
from control_plane.events.sse import sse_router

__all__ = ["event_broker", "sse_router"]
