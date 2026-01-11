"""Communication status lookup tool."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from voice_agent.communication import storage


async def get_communication_status(
    correlation_id: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """Return the latest recorded events for a correlation ID."""

    if not correlation_id:
        return {
            "correlation_id": correlation_id,
            "status": "unknown",
            "events": [],
            "message": "No correlation_id provided.",
        }

    events = await storage.fetch_events(correlation_id, limit)
    if not events:
        return {
            "correlation_id": correlation_id,
            "status": "unknown",
            "events": [],
            "message": "No events found for correlation id.",
        }

    latest = events[0]
    return {
        "correlation_id": correlation_id,
        "status": latest.status,
        "latest_event": asdict(latest),
        "events": [asdict(event) for event in events],
    }


TOOL_DEFINITION = {
    "name": "get_communication_status",
    "description": "Retrieve recorded delivery events for a communication correlation ID.",
    "parameters": {
        "correlation_id": {
            "type": "string",
            "description": "Correlation identifier returned by send_email/send_whatsapp tools.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of events to return (default 5).",
        },
    },
    "required": ["correlation_id"],
    "handler": get_communication_status,
}
