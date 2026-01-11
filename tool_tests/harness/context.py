"""
Pipecat Test Harness - Context and Types

This module provides the context and type definitions for the Pipecat-compatible
test harness. It mirrors the original tool_tests/harness/context.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from voice_agent.tools import PipecatToolsAdapter


class OutcomeStatus(Enum):
    SUCCESS = "success"
    EXPECTED_FAILURE = "expected_failure"
    UNEXPECTED_FAILURE = "unexpected_failure"
    SKIPPED = "skipped"


# Type aliases for callbacks
ValidationCallable = Callable[["HarnessContext", "ToolExecutionResult"], Awaitable[None] | None]
SetupCallable = Callable[["HarnessContext"], Awaitable[None] | None]
TeardownCallable = Callable[["HarnessContext", "ToolExecutionResult"], Awaitable[None] | None]
ArgumentsFactory = Callable[["HarnessContext"], Awaitable[Dict[str, Any]] | Dict[str, Any]]


@dataclass(slots=True)
class ToolTestCase:
    """
    Declarative description of an automated tool test.

    Attributes:
        tool_name: MCP tool name to invoke.
        description: Human-friendly summary for logs.
        arguments: JSON-serializable payload passed to the tool.
        expect_success: Whether the tool should produce a successful response.
        allow_error_substrings: Optional list of substrings that, when present in
            the error payload, downgrade failures to EXPECTED_FAILURE.
        setup: Optional async/sync callable executed before the tool runs.
        teardown: Optional async/sync callable executed after the tool runs.
        validator: Optional callable that performs assertions on the result.
        arguments_factory: Factory function to generate arguments dynamically.
    """

    tool_name: str
    description: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    expect_success: bool = True
    allow_error_substrings: tuple[str, ...] = ()
    setup: Optional[SetupCallable] = None
    teardown: Optional[TeardownCallable] = None
    validator: Optional[ValidationCallable] = None
    arguments_factory: Optional[ArgumentsFactory] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionResult:
    """Result of executing a single tool test."""
    
    tool_name: str
    arguments: Dict[str, Any]
    raw_result: Dict[str, Any]
    success: bool
    status: OutcomeStatus
    message: str
    duration_s: float
    error: Optional[str] = None


@dataclass(slots=True)  
class HarnessContext:
    """
    Runtime context shared between setup/validator/teardown functions.
    
    Uses the Pipecat tools adapter for tool invocation.
    """

    handlers: Dict[str, Callable]
    temp_root: Path
    resources: Dict[str, Any] = field(default_factory=dict)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool using the Pipecat handlers.
        """
        if name not in self.handlers:
            return {"success": False, "error": f"Unknown tool: {name}"}
        
        handler = self.handlers[name]
        try:
            result = await handler(**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def store(self, key: str, value: Any) -> None:
        """Store a value in the context resources."""
        self.resources[key] = value

    def fetch(self, key: str, default: Any = None) -> Any:
        """Fetch a value from the context resources."""
        return self.resources.get(key, default)

    async def maybe_await(self, callable_or_coroutine: Any) -> Any:
        """Await the object if it is awaitable, otherwise return directly."""
        if asyncio.iscoroutine(callable_or_coroutine):
            return await callable_or_coroutine
        return callable_or_coroutine


@dataclass
class SampleData:
    """Sample data for test scenarios."""
    
    primary_customer_id: str = ""
    primary_customer_phone: str = ""
    unique_phone: str = ""
    knowledge_category: str = ""
    knowledge_id: str = ""
    existing_call_type: str = ""
    objection_key: str = ""
    existing_call_id: str = ""
    existing_call_customer_id: str = ""
    
    @classmethod
    def load_defaults(cls) -> "SampleData":
        """Load default sample data."""
        import uuid
        # Use valid UUID format for customer ID (test with fixed prefix for reproducibility)
        return cls(
            primary_customer_id="00000000-0000-0000-0000-000000000001",
            primary_customer_phone="+5511999990001",
            unique_phone=f"+551199999{uuid.uuid4().hex[:4]}",
            knowledge_category="general",
            existing_call_type="outbound_sales",
            objection_key="price_too_high",
        )


__all__ = [
    "OutcomeStatus",
    "ToolTestCase", 
    "ToolExecutionResult",
    "HarnessContext",
    "SampleData",
    "ValidationCallable",
    "SetupCallable",
    "TeardownCallable",
    "ArgumentsFactory",
]
