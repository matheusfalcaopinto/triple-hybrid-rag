"""
Post-Call Processing Service

Handles automatic post-call actions:
- Save call summary to CRM
- Update call transcript
- Detect outcome and sentiment
- Generate AI-powered summaries
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import SETTINGS

logger = logging.getLogger("voice_agent.services.post_call")


@dataclass
class PostCallData:
    """Data collected for post-call processing."""
    
    call_sid: str
    caller_phone: str
    customer_id: Optional[str] = None
    duration_seconds: float = 0.0
    messages: List[Dict[str, str]] = None
    transcript: str = ""
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []


def extract_transcript_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Extract readable transcript from LLM context messages.
    
    Args:
        messages: List of conversation messages from LLM context
        
    Returns:
        Human-readable transcript string
    """
    lines = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Skip system messages and empty content
        if role == "system" or not content:
            continue
        
        # Skip tool calls and results
        if role in ("tool", "function"):
            continue
            
        # Format based on role
        if role == "user":
            lines.append(f"Cliente: {content}")
        elif role == "assistant":
            lines.append(f"Agente: {content}")
    
    return "\n".join(lines)


async def generate_ai_summary(transcript: str) -> str:
    """
    Generate concise summary from transcript using LLM.
    
    Args:
        transcript: Full conversation transcript
        
    Returns:
        Concise summary (1-3 sentences)
    """
    if not transcript or len(transcript) < 50:
        return "Chamada breve - sem conteúdo significativo"
    
    if not SETTINGS.use_ai_summary_generation:
        # Simple truncation fallback
        return transcript[:200] + "..." if len(transcript) > 200 else transcript
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            base_url=SETTINGS.openai_base_url,
            api_key=SETTINGS.openai_api_key,
        )
        
        response = await client.chat.completions.create(
            model=SETTINGS.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Resuma esta conversa telefônica em 1-2 frases em português. "
                        "Foque em: necessidades do cliente, informações fornecidas, próximos passos. "
                        "Seja conciso e objetivo."
                    )
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ],
            max_tokens=150,
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error("Failed to generate AI summary: %s", e)
        # Fallback to simple truncation
        return transcript[:200] + "..." if len(transcript) > 200 else transcript


def analyze_conversation_outcome(transcript: str) -> tuple[str, str]:
    """
    Analyze conversation to determine outcome and sentiment.
    
    Args:
        transcript: Full conversation transcript
        
    Returns:
        Tuple of (outcome, sentiment)
        - outcome: interested, not_interested, callback, demo_scheduled, issue_resolved, completed
        - sentiment: positive, neutral, negative
    """
    transcript_lower = transcript.lower()
    
    # Detect outcome - Portuguese and English keywords
    if any(word in transcript_lower for word in [
        "interessado", "interested", "sounds good", "parece bom",
        "vamos fazer", "let's do it", "quero", "i want",
        "me inscreva", "sign me up", "fechado", "agreed"
    ]):
        outcome = "interested"
    elif any(word in transcript_lower for word in [
        "não interessado", "not interested", "não obrigado", "no thank you",
        "não agora", "not now", "sem interesse", "no interest"
    ]):
        outcome = "not_interested"
    elif any(word in transcript_lower for word in [
        "ligar depois", "call back", "retornar", "follow up",
        "semana que vem", "next week", "outro dia", "another day",
        "me liga", "call me"
    ]):
        outcome = "callback"
    elif any(word in transcript_lower for word in [
        "agendar", "schedule", "marcar", "book", "appointment",
        "reunião", "meeting", "demo", "demonstração"
    ]):
        outcome = "demo_scheduled"
    elif any(word in transcript_lower for word in [
        "resolvido", "resolved", "solucionado", "solved",
        "ajudou", "helped", "funcionou", "worked",
        "consertado", "fixed"
    ]):
        outcome = "issue_resolved"
    else:
        outcome = "completed"
    
    # Detect sentiment - Portuguese and English keywords
    positive_words = [
        "obrigado", "thank you", "thanks", "ótimo", "great",
        "excelente", "excellent", "perfeito", "perfect",
        "maravilhoso", "wonderful", "ajudou muito", "very helpful",
        "agradeço", "appreciate", "bom", "good"
    ]
    negative_words = [
        "frustrado", "frustrated", "decepcionado", "disappointed",
        "irritado", "angry", "infeliz", "unhappy", "péssimo", "terrible",
        "horrível", "horrible", "ruim", "bad", "problema", "problem"
    ]
    
    positive_count = sum(1 for word in positive_words if word in transcript_lower)
    negative_count = sum(1 for word in negative_words if word in transcript_lower)
    
    if positive_count > negative_count + 1:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    logger.debug(
        "Conversation analysis: outcome=%s, sentiment=%s (pos=%d, neg=%d)",
        outcome, sentiment, positive_count, negative_count
    )
    
    return outcome, sentiment


async def process_post_call(
    call_sid: str,
    caller_phone: str,
    duration: float,
    messages: List[Dict[str, str]],
    customer_id: Optional[str] = None,
) -> Optional[str]:
    """
    Process post-call data and save to CRM.
    
    This is the main entry point for post-call processing.
    It extracts the transcript, generates a summary, analyzes
    outcome/sentiment, and saves everything to the database.
    
    Args:
        call_sid: Unique call identifier
        caller_phone: Caller's phone number
        duration: Call duration in seconds
        messages: List of conversation messages from LLM context
        customer_id: Known customer ID (if prefetched)
        
    Returns:
        call_id if saved successfully, None otherwise
    """
    if not SETTINGS.auto_save_call_summary:
        logger.debug("Auto-save call summary disabled, skipping post-call processing")
        return None
    
    if duration < SETTINGS.min_call_duration_to_save:
        logger.info(
            "Call too short to save: %.1fs < %.1fs minimum",
            duration, SETTINGS.min_call_duration_to_save
        )
        return None
    
    logger.info(
        "Starting post-call processing for call_sid=%s, duration=%.1fs",
        call_sid, duration
    )
    
    try:
        from voice_agent.tools import get_mcp_server
        
        server = get_mcp_server()
        
        # Get or find customer ID
        if not customer_id and caller_phone:
            result = await server.call_tool_async(
                "get_customer_by_phone",
                {"phone": caller_phone}
            )
            if result.get("success") and result.get("result"):
                customer_data = result["result"]
                customer_id = customer_data.get("customer_id") or customer_data.get("id")
                logger.info("Found customer ID for post-call: %s", customer_id)
        
        if not customer_id:
            logger.warning(
                "No customer ID for post-call save, caller=%s",
                caller_phone[:6] + "****" if len(caller_phone) > 6 else caller_phone
            )
            # We could create a customer here, but for now we'll skip
            return None
        
        # Extract transcript from messages
        transcript = extract_transcript_from_messages(messages)
        
        if not transcript:
            logger.info("No transcript to save for call_sid=%s", call_sid)
            return None
        
        logger.debug("Extracted transcript: %d characters", len(transcript))
        
        # Generate AI summary
        summary = await generate_ai_summary(transcript)
        logger.debug("Generated summary: %s", summary[:100])
        
        # Analyze outcome and sentiment
        outcome, sentiment = analyze_conversation_outcome(transcript)
        
        # Save call summary
        result = await server.call_tool_async(
            "save_call_summary",
            {
                "customer_id": customer_id,
                "summary": summary,
                "outcome": outcome,
                "call_type": "inbound_support",  # Could be improved to detect
                "duration_seconds": int(duration),
                "sentiment": sentiment,
            }
        )
        
        if result.get("success"):
            call_result = result.get("result", {})
            call_id = call_result.get("call_id")
            logger.info(
                "Post-call summary saved: call_id=%s, outcome=%s, sentiment=%s",
                call_id, outcome, sentiment
            )
            
            # Update transcript if we have a call_id
            if call_id and transcript:
                transcript_result = await server.call_tool_async(
                    "update_call_transcript",
                    {"call_id": call_id, "transcript": transcript}
                )
                if transcript_result.get("success"):
                    logger.info("Transcript saved for call_id=%s", call_id)
                else:
                    logger.warning(
                        "Failed to save transcript: %s",
                        transcript_result.get("error")
                    )
            
            return call_id
        else:
            logger.error("Failed to save call summary: %s", result.get("error"))
            return None
            
    except ImportError as e:
        logger.warning("Could not import tools for post-call: %s", e)
        return None
    except Exception as e:
        logger.exception("Error in post-call processing: %s", e)
        return None


# Singleton pattern
_post_call_service: Optional["PostCallService"] = None


class PostCallService:
    """Service for managing post-call processing."""
    
    def __init__(self):
        logger.info("Post-call processing service initialized")
    
    async def process(
        self,
        call_sid: str,
        caller_phone: str,
        duration: float,
        messages: List[Dict[str, str]],
        customer_id: Optional[str] = None,
    ) -> Optional[str]:
        """Process post-call data."""
        return await process_post_call(
            call_sid=call_sid,
            caller_phone=caller_phone,
            duration=duration,
            messages=messages,
            customer_id=customer_id,
        )


def get_post_call_service() -> PostCallService:
    """Get or create the post-call service singleton."""
    global _post_call_service
    if _post_call_service is None:
        _post_call_service = PostCallService()
    return _post_call_service
