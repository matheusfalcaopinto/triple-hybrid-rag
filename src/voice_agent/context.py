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
    
    This function calls the MCP tools server directly (not via handlers)
    to avoid FunctionCallParams signature issues during prefetch.
    
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
        # Import MCP server directly (bypasses handler layer)
        from voice_agent.tools import get_mcp_server
        
        server = get_mcp_server()
        
        # 1. Get customer by phone
        result = await server.call_tool_async("get_customer_by_phone", {"phone": phone})
        
        if result.get("success") and result.get("result"):
            customer_data = result["result"]
            context.customer_id = customer_data.get("customer_id") or customer_data.get("id")
            context.name = customer_data.get("name", "")
            context.status = customer_data.get("status", "")
            context.raw_data = customer_data
            logger.info(
                "Prefetched customer: id=%s, name=%s",
                context.customer_id, context.name,
            )
        elif result.get("error"):
            logger.warning("Failed to get customer by phone: %s", result.get("error"))
        
        # 2. Get customer facts (if we have an ID)
        if context.customer_id:
            result = await server.call_tool_async("get_customer_facts", {"customer_id": context.customer_id})
            
            if result.get("success") and result.get("result"):
                facts_data = result["result"]
                if isinstance(facts_data, list):
                    context.facts = facts_data[:10]  # Limit to 10 facts
                elif isinstance(facts_data, dict):
                    context.facts = facts_data.get("facts", [])[:10]
        
        # 3. Get last call summary
        if context.customer_id:
            result = await server.call_tool_async("get_last_call", {"customer_id": context.customer_id})
            
            if result.get("success") and result.get("result"):
                call_data = result["result"]
                context.last_call_summary = call_data.get("summary", "")
        
        # 4. Get pending tasks
        if context.customer_id:
            result = await server.call_tool_async("get_pending_tasks", {"customer_id": context.customer_id})
            
            if result.get("success") and result.get("result"):
                tasks_data = result["result"]
                if isinstance(tasks_data, list):
                    context.pending_tasks = tasks_data[:5]  # Limit to 5 tasks
                elif isinstance(tasks_data, dict):
                    context.pending_tasks = tasks_data.get("tasks", [])[:5]
        
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
