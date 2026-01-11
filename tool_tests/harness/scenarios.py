"""
Pipecat Test Harness - Predefined Test Scenarios

Contains test cases matching the original tool_tests/harness/scenarios.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from .context import HarnessContext, ToolExecutionResult, ToolTestCase, SampleData


def _payload_from_raw(raw: Dict) -> object:
    if isinstance(raw, dict):
        return raw.get("result")
    return None


def _payload_as_dict(payload: object) -> Dict:
    return payload if isinstance(payload, dict) else {}


def _payload_from_execution(result: ToolExecutionResult) -> object:
    return _payload_from_raw(result.raw_result)


# ──────────────────────────────────────────────────────────────────────────────
# Validators
# ──────────────────────────────────────────────────────────────────────────────

async def _validator_calculate(_: HarnessContext, result: ToolExecutionResult) -> None:
    payload = _payload_from_execution(result)
    if isinstance(payload, dict):
        value = payload.get("result")
    else:
        value = payload
    assert value == 20, f"Expected 12 + 8 to equal 20, got {value}"


async def _validator_format_date(_: HarnessContext, result: ToolExecutionResult) -> None:
    payload = _payload_from_execution(result)
    assert "2025" in str(payload), "Formatted date should reference 2025"


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_tool_test_cases(
    sample_data: SampleData,
    *,
    include: Optional[Set[str]] = None,
) -> List[ToolTestCase]:
    """Build the list of tool test cases."""
    
    cases: List[ToolTestCase] = []

    def should_include(name: str) -> bool:
        return include is None or name in include

    def add(case: ToolTestCase) -> None:
        if should_include(case.tool_name):
            cases.append(case)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    # Setup function to resolve customer_id from phone and store it
    async def _setup_resolve_customer_id(ctx: HarnessContext) -> None:
        """Resolve customer_id via phone lookup and store for subsequent tests."""
        sample = ctx.fetch("sample_data")
        result = await ctx.call_tool("get_customer_by_phone", {"phone": sample.primary_customer_phone})
        if result.get("success") and result.get("result", {}).get("id"):
            ctx.store("resolved_customer_id", result["result"]["id"])

    # ─────────────────────────────────────────────────────────────────────────
    # Core Utilities
    # ─────────────────────────────────────────────────────────────────────────
    
    add(ToolTestCase(
        tool_name="get_current_time",
        description="Returns current time without parameters",
    ))

    add(ToolTestCase(
        tool_name="calculate",
        description="Basic arithmetic: 12 + 8 = 20",
        arguments={"operation": "add", "a": 12, "b": 8},
        validator=_validator_calculate,
    ))

    add(ToolTestCase(
        tool_name="get_system_info",
        description="Report basic system metadata",
    ))

    add(ToolTestCase(
        tool_name="format_date",
        description="Formats ISO date into Brazilian representation",
        arguments={"date_string": "2025-11-04", "format": "brazilian"},
        validator=_validator_format_date,
    ))

    add(ToolTestCase(
        tool_name="get_weather",
        description="Retrieves mock weather data for São Paulo",
        arguments={"location": "São Paulo"},
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Customers
    # ─────────────────────────────────────────────────────────────────────────
    
    add(ToolTestCase(
        tool_name="get_customer_by_phone",
        description="Lookup customer by phone",
        arguments={"phone": sample_data.primary_customer_phone},
    ))

    add(ToolTestCase(
        tool_name="create_customer",
        description="Create a synthetic customer record",
        arguments_factory=lambda ctx: {
            "phone": ctx.fetch("sample_data").unique_phone,
            "name": f"Test Customer {uuid.uuid4().hex[:6]}",
            "company": "Test Labs",
        },
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Facts
    # ─────────────────────────────────────────────────────────────────────────
    
    add(ToolTestCase(
        tool_name="get_customer_facts",
        description="Retrieve facts for customer",
        arguments={"phone": sample_data.primary_customer_phone},
    ))

    add(ToolTestCase(
        tool_name="add_customer_fact",
        description="Add a new fact",
        arguments_factory=lambda ctx: {
            "phone": ctx.fetch("sample_data").primary_customer_phone,
            "fact_type": "preference",
            "content": f"Harness test {uuid.uuid4().hex[:6]}",
        },
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Email (expected configuration errors)
    # ─────────────────────────────────────────────────────────────────────────
    
    email_error_tokens = ("smtp", "not configured", "stub")
    email_error_tokens = ("smtp", "not configured", "stub", "template")
    crm_error_tokens = (
        "Database error", "not found", "connection", "No task found",
        "No knowledge base entry found", "No active script found", "No handler found",
        "No active call context", "Context access failed"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Customers (Extended)
    # ─────────────────────────────────────────────────────────────────────────

    add(ToolTestCase(
        tool_name="get_customer_by_id",
        description="Lookup customer by ID",
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or str(uuid.uuid4())
        },
        setup=_setup_resolve_customer_id,
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="update_customer_status",
        description="Update customer status",
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or str(uuid.uuid4()),
            "status": "qualified"
        },
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="update_customer_info",
        description="Update customer info",
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or str(uuid.uuid4()),
            "company": "Updated Corp"
        },
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="search_customers",
        description="Search customers",
        arguments={"query": "Test"},
        allow_error_substrings=crm_error_tokens,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Facts (Extended)
    # ─────────────────────────────────────────────────────────────────────────
    
    add(ToolTestCase(
        tool_name="get_facts_by_type",
        description="Get facts by type",
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or str(uuid.uuid4()),
            "fact_type": "preference"
        },
        allow_error_substrings=crm_error_tokens,
    ))
    
    add(ToolTestCase(
        tool_name="search_customers_by_fact",
        description="Search customers by fact content",
        arguments={"query": "harness"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="delete_customer_fact",
        description="Delete a fact",
        arguments={"fact_id": str(uuid.uuid4())},
        allow_error_substrings=crm_error_tokens,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Calls (Extended)
    # ─────────────────────────────────────────────────────────────────────────

    add(ToolTestCase(
        tool_name="get_last_call",
        description="Retrieve last call",
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or str(uuid.uuid4())
        },
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="update_call_transcript",
        description="Update call transcript",
        arguments={
            "call_id": str(uuid.uuid4()),
            "transcript": "Harness test transcript"
        },
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_calls_by_outcome",
        description="Get calls by outcome",
        arguments={"outcome": "interested"},
        allow_error_substrings=crm_error_tokens,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Email (expected configuration errors)
    # ─────────────────────────────────────────────────────────────────────────
    add(ToolTestCase(
        tool_name="send_email",
        description="Attempt to send email (should use stub)",
        arguments={"to": "test@example.com", "subject": "Test", "body": "Hello"},
        allow_error_substrings=email_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_bulk_email",
        description="Attempt bulk email (should use stub)",
        arguments={"recipients": "test@example.com", "subject": "Bulk", "body": "Hello"},
        allow_error_substrings=email_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_html_email",
        description="Attempt HTML email",
        arguments={"to": "test@example.com", "subject": "HTML", "html_body": "<h1>Hi</h1>"},
        allow_error_substrings=email_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_email_template",
        description="Attempt email template",
        arguments={
            "to": "test@example.com",
            "template": "welcome",
            "variables": '{"name": "Tester", "company": "Acme"}'
        },
        allow_error_substrings=email_error_tokens,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # WhatsApp (expected missing credentials)
    # ─────────────────────────────────────────────────────────────────────────
    
    whatsapp_tokens = ("twilio", "not configured", "evolution", "stub", "setup_required")
    default_to = "+5511999999999"
    
    add(ToolTestCase(
        tool_name="send_whatsapp_message",
        description="Attempt WhatsApp message (should use stub)",
        arguments={"to": default_to, "message": "Test message"},
        allow_error_substrings=whatsapp_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_whatsapp_image",
        description="Attempt WhatsApp image (should use stub)",
        arguments={
            "to": default_to,
            "image_url": "https://example.com/image.jpg",
            "caption": "Test",
        },
        allow_error_substrings=whatsapp_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_whatsapp_video",
        description="Attempt WhatsApp video",
        arguments={
            "to": default_to,
            "video_url": "https://example.com/video.mp4",
            "caption": "Test Video",
        },
        allow_error_substrings=whatsapp_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_whatsapp_document",
        description="Attempt WhatsApp document",
        arguments={
            "to": default_to,
            "document_url": "https://example.com/doc.pdf",
            "filename": "test.pdf",
        },
        allow_error_substrings=whatsapp_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_whatsapp_audio",
        description="Attempt WhatsApp audio",
        arguments={
            "to": default_to,
            "audio_url": "https://example.com/audio.mp3",
        },
        allow_error_substrings=whatsapp_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_whatsapp_location",
        description="Attempt WhatsApp location",
        arguments={
            "to": default_to,
            "latitude": -23.5505,
            "longitude": -46.6333,
            "name": "São Paulo",
        },
        allow_error_substrings=whatsapp_tokens,
    ))

    add(ToolTestCase(
        tool_name="send_whatsapp_template",
        description="Attempt WhatsApp template",
        arguments={
            "to": default_to,
            "content_sid": "HX123456",
            "content_variables": {"1": "Val"},
        },
        allow_error_substrings=whatsapp_tokens,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Call Control & Status
    # ─────────────────────────────────────────────────────────────────────────

    add(ToolTestCase(
        tool_name="end_call",
        description="End call signal",
        arguments={"reason": "test_completed"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_communication_status",
        description="Check status",
        arguments={"correlation_id": str(uuid.uuid4())},
        allow_error_substrings=crm_error_tokens + ("No events found", "No correlation_id"),
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Google Calendar (expected missing OAuth)
    # ─────────────────────────────────────────────────────────────────────────
    
    calendar_tokens = ("google", "credentials", "not available", "stub")
    
    add(ToolTestCase(
        tool_name="create_calendar_event",
        description="Attempt to create calendar event (should use stub)",
        arguments={
            "summary": "Test Meeting",
            "start_time": (datetime.utcnow() + timedelta(days=1)).isoformat(),
            "end_time": (datetime.utcnow() + timedelta(days=1, hours=1)).isoformat(),
        },
        allow_error_substrings=calendar_tokens,
    ))

    add(ToolTestCase(
        tool_name="list_upcoming_calendar_events",
        description="List calendar events (should use stub)",
        allow_error_substrings=calendar_tokens,
    ))

    add(ToolTestCase(
        tool_name="check_calendar_availability",
        description="Check availability (should use stub)",
        arguments={
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
        },
        allow_error_substrings=calendar_tokens,
    ))

    add(ToolTestCase(
        tool_name="list_calendar_attendees",
        description="List available attendees (should use stub)",
        allow_error_substrings=calendar_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_calendar_availability_for_attendee",
        description="Check specific attendee availability (should use stub)",
        arguments={"attendee_name": "Dr. Smith"},
        allow_error_substrings=calendar_tokens + ("not found",),
    ))

    add(ToolTestCase(
        tool_name="book_appointment",
        description="Book appointment (should use stub)",
        arguments={
            "attendee_name": "Dr. Smith",
            "customer_phone": "+5511999999999",
            "start_time": (datetime.utcnow() + timedelta(days=2)).isoformat(),
        },
        allow_error_substrings=calendar_tokens + ("not found",),
    ))

    add(ToolTestCase(
        tool_name="cancel_calendar_event",
        description="Cancel event (should use stub)",
        arguments={"event_id": "test_event_id"},
        allow_error_substrings=calendar_tokens + ("object has no attribute",),
    ))

    add(ToolTestCase(
        tool_name="update_calendar_event",
        description="Update event (should use stub)",
        arguments={"event_id": "test_event_id", "summary": "Updated Title"},
        allow_error_substrings=calendar_tokens + ("object has no attribute",),
    ))

    add(ToolTestCase(
        tool_name="search_calendar_events",
        description="Search events (should use stub)",
        arguments={"query": "meeting"},
        allow_error_substrings=calendar_tokens,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Calls
    # ─────────────────────────────────────────────────────────────────────────
    


    add(ToolTestCase(
        tool_name="get_call_history",
        description="Retrieve call history for customer",
        setup=_setup_resolve_customer_id,
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or ctx.fetch("sample_data").primary_customer_id,
            "limit": 3,
        },
    ))

    add(ToolTestCase(
        tool_name="save_call_summary",
        description="Insert synthetic call summary",
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or ctx.fetch("sample_data").primary_customer_id,
            "summary": "Harness test summary",
            "outcome": "interested",
            "call_type": "outbound_followup",
            "duration_seconds": 180,
        },
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Action Items (Tasks)
    # ─────────────────────────────────────────────────────────────────────────

    add(ToolTestCase(
        tool_name="get_pending_tasks",
        description="Get pending tasks for customer",
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or str(uuid.uuid4())
        },
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_all_pending_tasks",
        description="Get all pending tasks",
        arguments={"assigned_to": "agent_007"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="create_task",
        description="Create a task",
        arguments_factory=lambda ctx: {
            "customer_id": ctx.fetch("resolved_customer_id") or str(uuid.uuid4()),
            "description": "Follow up with client",
            "due_date": (datetime.now() + timedelta(days=2)).isoformat(),
        },
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="complete_task",
        description="Complete a task",
        arguments={"task_id": str(uuid.uuid4())},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="update_task",
        description="Update a task",
        arguments={"task_id": str(uuid.uuid4()), "priority": "high"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_overdue_tasks",
        description="Get overdue tasks",
        arguments={"assigned_to": "agent_007"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="delete_task",
        description="Delete a task",
        arguments={"task_id": str(uuid.uuid4())},
        allow_error_substrings=crm_error_tokens,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Knowledge Base
    # ─────────────────────────────────────────────────────────────────────────

    add(ToolTestCase(
        tool_name="search_knowledge_base",
        description="Search knowledge base",
        arguments={"query": "pricing"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_knowledge_by_category",
        description="Get knowledge by category",
        arguments={"category": "pricing"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_knowledge_by_id",
        description="Get knowledge by ID",
        arguments={"kb_id": str(uuid.uuid4())},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="add_knowledge_item",
        description="Add knowledge item",
        arguments={
            "category": "faq",
            "title": "Test Item",
            "content": "Test content",
        },
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="update_knowledge_item",
        description="Update knowledge item",
        arguments={"kb_id": str(uuid.uuid4()), "title": "Updated Title"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="delete_knowledge_item",
        description="Delete knowledge item",
        arguments={"kb_id": str(uuid.uuid4())},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="list_categories",
        description="List knowledge categories",
        allow_error_substrings=crm_error_tokens,
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # CRM Scripts
    # ─────────────────────────────────────────────────────────────────────────

    add(ToolTestCase(
        tool_name="get_call_script",
        description="Get call script",
        arguments={"call_type": "outbound_cold_call"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_all_call_scripts",
        description="Get all scripts",
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="get_objection_handler",
        description="Get objection handler",
        arguments={"call_type": "outbound_cold_call", "objection_key": "price"},
        allow_error_substrings=crm_error_tokens,
    ))

    add(ToolTestCase(
        tool_name="list_objections",
        description="List objections",
        arguments={"call_type": "outbound_cold_call"},
        allow_error_substrings=crm_error_tokens,
    ))

    return cases


def get_utility_cases() -> List[ToolTestCase]:
    """Get just the utility tool test cases."""
    sample = SampleData.load_defaults()
    return [
        c for c in build_tool_test_cases(sample)
        if c.tool_name in {"get_current_time", "calculate", "get_system_info",
                           "format_date", "get_weather", "end_call"}
    ]


def get_crm_cases() -> List[ToolTestCase]:
    """Get CRM-related test cases."""
    sample = SampleData.load_defaults()
    crm_tools = {
        "get_customer_by_phone", "create_customer", "get_customer_facts",
        "add_customer_fact", "get_call_history", "save_call_summary",
        "get_customer_by_id", "update_customer_status", "update_customer_info", "search_customers",
        "get_facts_by_type", "search_customers_by_fact", "delete_customer_fact",
        "get_last_call", "update_call_transcript", "get_calls_by_outcome",
        "get_pending_tasks", "get_all_pending_tasks", "create_task", "complete_task",
        "update_task", "get_overdue_tasks", "delete_task",
        "search_knowledge_base", "get_knowledge_by_category", "get_knowledge_by_id",
        "add_knowledge_item", "update_knowledge_item", "delete_knowledge_item", "list_categories",
        "get_call_script", "get_all_call_scripts", "get_objection_handler", "list_objections",
    }
    return [c for c in build_tool_test_cases(sample) if c.tool_name in crm_tools]


def get_communication_cases() -> List[ToolTestCase]:
    """Get communication tool test cases (email, WhatsApp)."""
    sample = SampleData.load_defaults()
    comm_tools = {
        "send_email", "send_bulk_email", "send_html_email", "send_email_template",
        "send_whatsapp_message", "send_whatsapp_image", "send_whatsapp_video",
        "send_whatsapp_document", "send_whatsapp_audio", "send_whatsapp_location",
        "send_whatsapp_template", "get_communication_status",
    }
    return [c for c in build_tool_test_cases(sample) if c.tool_name in comm_tools]


def get_calendar_cases() -> List[ToolTestCase]:
    """Get calendar tool test cases."""
    sample = SampleData.load_defaults()
    cal_tools = {
        "create_calendar_event", "list_upcoming_calendar_events", "check_calendar_availability",
        "list_calendar_attendees", "get_calendar_availability_for_attendee", "book_appointment",
        "cancel_calendar_event", "update_calendar_event", "search_calendar_events",
    }
    return [c for c in build_tool_test_cases(sample) if c.tool_name in cal_tools]


def get_communication_twilio_cases() -> List[ToolTestCase]:
    """Get WhatsApp tool test cases explicitly targeting the Twilio backend."""
    cases = []
    default_to = "+5511999999999"
    
    # Text Message (Success Expected via Mock)
    cases.append(ToolTestCase(
        tool_name="send_whatsapp_message",
        description="Send WhatsApp message (Twilio Backend)",
        arguments={"to": default_to, "message": "Test message from Twilio Harness"},
        expect_success=True,
        metadata={"backend": "twilio"},
    ))

    # Image Message (Success Expected via Mock)
    cases.append(ToolTestCase(
        tool_name="send_whatsapp_image",
        description="Send WhatsApp image (Twilio Backend)",
        arguments={
            "to": default_to,
            "image_url": "https://example.com/image.jpg",
            "caption": "Test Twilio Image",
        },
        expect_success=True,
        metadata={"backend": "twilio"},
    ))

    return cases


__all__ = [
    "build_tool_test_cases",
    "get_utility_cases",
    "get_crm_cases",
    "get_communication_cases",
    "get_communication_twilio_cases",
    "get_calendar_cases",
]
