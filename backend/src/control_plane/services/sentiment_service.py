"""Sentiment service - computes sentiment from transcripts."""

import logging
from datetime import datetime, timezone

from openai import AsyncOpenAI
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.config import settings
from control_plane.db.models.call import Call, CallTranscriptSegment

logger = logging.getLogger(__name__)


class SentimentService:
    """Service for computing call sentiment."""

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

    async def compute_sentiment(
        self,
        session: AsyncSession,
        call_id: str,
    ) -> dict | None:
        """Compute sentiment for a call based on transcript."""
        # Get transcript segments
        result = await session.execute(
            select(CallTranscriptSegment)
            .where(CallTranscriptSegment.call_id == call_id)
            .order_by(CallTranscriptSegment.started_at)
        )
        segments = list(result.scalars().all())

        if not segments:
            logger.debug(f"No transcript segments for call {call_id}")
            return None

        # Get only customer segments for sentiment
        customer_text = " ".join(
            seg.text for seg in segments if seg.speaker == "customer"
        )

        if not customer_text.strip():
            return None

        try:
            sentiment = await self._analyze_sentiment(customer_text)
        except Exception as e:
            logger.error(f"Failed to compute sentiment for call {call_id}: {e}")
            return None

        # Update call with sentiment
        await session.execute(
            update(Call)
            .where(Call.id == call_id)
            .values(
                sentiment_label=sentiment["label"],
                sentiment_score=sentiment["score"],
                sentiment_computed_at=datetime.now(timezone.utc),
            )
        )

        return sentiment

    async def _analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment using LLM."""
        prompt = f"""Analyze the sentiment of the following customer text from a phone call.

Text: {text[:2000]}

Respond with a JSON object containing:
- label: "positive", "negative", or "neutral"
- score: A confidence score between 0 and 1

JSON response:"""

        response = await self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis assistant. Analyze customer sentiment accurately.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=50,
            response_format={"type": "json_object"},
        )

        import json

        content = response.choices[0].message.content or '{"label": "neutral", "score": 0.5}'
        result = json.loads(content)

        return {
            "label": result.get("label", "neutral"),
            "score": float(result.get("score", 0.5)),
        }


# Singleton instance
sentiment_service = SentimentService()
