"""
Context Management for Pipecat Voice Agent

This module handles:
- Customer context prefetch based on phone number
- Conversation history management
- Session state tracking
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("voice_agent_pipecat.context")


@dataclass
class CustomerContext:
    """Customer information from CRM prefetch."""
    
    customer_id: Optional[str] = None
    phone: str = ""
    name: str = ""
    status: str = ""
    facts: List[Dict[str, Any]] = field(default_factory=list)
    last_call_summary: Optional[str] = None
    pending_tasks: List[Dict[str, Any]] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_known(self) -> bool:
        """Check if this is a known customer."""
        return self.customer_id is not None
    
    def to_context_string(self) -> str:
        """Convert to a string for LLM context injection."""
        lines = ["[CUSTOMER CONTEXT]"]
        
        if not self.is_known:
            lines.append("Customer: Unknown/New (not in CRM)")
            lines.append(f"Phone: {self.phone}")
            return "\n".join(lines)
        
        lines.append(f"Customer ID: {self.customer_id}")
        lines.append(f"Name: {self.name or 'Unknown'}")
        lines.append(f"Phone: {self.phone}")
        lines.append(f"Status: {self.status or 'Active'}")
        
        if self.facts:
            lines.append("\nKnown Facts:")
            for fact in self.facts[:5]:  # Limit to 5 facts
                fact_type = fact.get("fact_type", "")
                fact_value = fact.get("fact_value", "")
                lines.append(f"  - {fact_type}: {fact_value}")
        
        if self.last_call_summary:
            lines.append(f"\nLast Call Summary: {self.last_call_summary[:200]}")
        
        if self.pending_tasks:
            lines.append("\nPending Tasks:")
            for task in self.pending_tasks[:3]:
                lines.append(f"  - {task.get('title', 'Task')}")
        
        return "\n".join(lines)


@dataclass
class SessionContext:
    """Session state for a voice call."""
    
    call_sid: str = ""
    trace_id: str = ""
    caller_phone: str = ""
    customer: Optional[CustomerContext] = None
    
    # Conversation state
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_topic: str = ""
    
    # Tool state
    last_tool_call: Optional[str] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    call_start_time: float = 0.0
    last_activity_time: float = 0.0


async def prefetch_customer_context(phone: str) -> CustomerContext:
    """
    Prefetch customer context from CRM based on phone number.
    
    Args:
        phone: Caller's phone number
        
    Returns:
        CustomerContext with CRM data
    """
    context = CustomerContext(phone=phone)
    
    if not phone:
        logger.warning("No phone number provided for prefetch")
        return context
    
    try:
        # Import tools from the adapter
        from voice_agent.tools import get_tool_handlers
        
        handlers = get_tool_handlers()
        
        # Get customer by phone
        if "get_customer_by_phone" in handlers:
            handler = handlers["get_customer_by_phone"]
            result = await handler(phone=phone)
            
            if result and not result.get("error"):
                context.customer_id = result.get("customer_id") or result.get("id")
                context.name = result.get("name", "")
                context.status = result.get("status", "")
                context.raw_data = result
                logger.info(
                    "Prefetched customer: id=%s, name=%s",
                    context.customer_id, context.name,
                )
        
        # Get customer facts if we have an ID
        if context.customer_id and "get_customer_facts" in handlers:
            handler = handlers["get_customer_facts"]
            result = await handler(customer_id=context.customer_id)
            
            if result and isinstance(result, list):
                context.facts = result[:10]  # Limit facts
            elif result and isinstance(result, dict):
                context.facts = result.get("facts", [])[:10]
        
        # Get last call summary
        if context.customer_id and "get_last_call" in handlers:
            handler = handlers["get_last_call"]
            result = await handler(customer_id=context.customer_id)
            
            if result and not result.get("error"):
                context.last_call_summary = result.get("summary", "")
        
        # Get pending tasks
        if context.customer_id and "get_pending_tasks" in handlers:
            handler = handlers["get_pending_tasks"]
            result = await handler(customer_id=context.customer_id)
            
            if result and isinstance(result, list):
                context.pending_tasks = result[:5]
            elif result and isinstance(result, dict):
                context.pending_tasks = result.get("tasks", [])[:5]
        
    except ImportError as e:
        logger.warning("Could not import tools for prefetch: %s", e)
    except Exception as e:
        logger.exception("Error during customer prefetch: %s", e)
    
    return context


def build_system_prompt_with_context(
    base_prompt: str,
    customer_context: Optional[CustomerContext] = None,
    caller_phone: str = "",
) -> str:
    """
    Build the system prompt with injected customer context.
    
    Args:
        base_prompt: Base system prompt
        customer_context: Prefetched customer context
        caller_phone: Caller's phone number
        
    Returns:
        Enhanced system prompt with context
    """
    parts = [base_prompt]
    
    # Add customer context if available
    if customer_context:
        parts.append("\n\n" + customer_context.to_context_string())
    elif caller_phone:
        parts.append(f"\n\n[CALLER CONTEXT]\nCaller phone: {caller_phone}")
    
    return "".join(parts)


__all__ = [
    "CustomerContext",
    "SessionContext",
    "prefetch_customer_context",
    "build_system_prompt_with_context",
]
