"""Summary service - generates call summaries using LLM."""

import logging
from datetime import datetime, timezone

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.config import settings
from control_plane.db.models.call import Call, CallSummary, CallTranscriptSegment
from control_plane.services.call_service import call_service

logger = logging.getLogger(__name__)


class SummaryService:
    """Service for generating call summaries."""

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get OpenAI client."""
        if self._client is None:
            api_key = settings.openai_api_key
            if api_key:
                self._client = AsyncOpenAI(api_key=api_key.get_secret_value())
            else:
                raise ValueError("OpenAI API key not configured")
        return self._client

    async def generate_summary(
        self,
        session: AsyncSession,
        call_id: str,
    ) -> CallSummary | None:
        """Generate a summary for a completed call."""
        # Get the call
        call = await call_service.get_call(session, call_id)
        if not call:
            logger.warning(f"Call {call_id} not found")
            return None

        # Check if summary already exists
        existing = await call_service.get_summary(session, call_id)
        if existing:
            logger.info(f"Summary already exists for call {call_id}")
            return existing

        # Get transcript
        result = await session.execute(
            select(CallTranscriptSegment)
            .where(CallTranscriptSegment.call_id == call_id)
            .order_by(CallTranscriptSegment.started_at)
        )
        segments = result.scalars().all()

        if not segments:
            logger.warning(f"No transcript segments for call {call_id}")
            return None

        # Format transcript
        transcript_text = self._format_transcript(segments)

        # Generate summary using LLM
        try:
            summary_data = await self._call_llm(transcript_text)
        except Exception as e:
            logger.error(f"Failed to generate summary for call {call_id}: {e}")
            return None

        # Store summary
        summary = await call_service.add_summary(
            session,
            call_id=call_id,
            summary_text=summary_data.get("summary", ""),
            key_points=summary_data.get("key_points"),
            action_items=summary_data.get("action_items"),
            customer_intent=summary_data.get("customer_intent"),
            resolution_status=summary_data.get("resolution_status"),
            model_used=settings.openai_model,
        )

        return summary

    def _format_transcript(self, segments: list[CallTranscriptSegment]) -> str:
        """Format transcript segments into text."""
        lines = []
        for seg in segments:
            speaker = seg.speaker.upper()
            lines.append(f"{speaker}: {seg.text}")
        return "\n".join(lines)

    async def _call_llm(self, transcript: str) -> dict:
        """Call LLM to generate summary."""
        prompt = f"""Analyze the following call transcript and provide a structured summary.

Transcript:
{transcript}

Provide the response as a JSON object with the following fields:
- summary: A brief 2-3 sentence summary of the call
- key_points: List of 3-5 key points discussed
- action_items: List of any action items or follow-ups needed
- customer_intent: The main reason for the call (1-2 words)
- resolution_status: One of "resolved", "pending", "escalated", "unresolved"

JSON response:"""

        response = await self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a call analysis assistant. Provide structured summaries of customer calls.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        import json

        content = response.choices[0].message.content or "{}"
        return json.loads(content)


# Singleton instance
summary_service = SummaryService()
