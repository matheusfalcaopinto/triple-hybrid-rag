"""
Tests for Google Calendar MCP tool
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_agent.tools.google_calendar import (
    TOOL_DEFINITIONS,
    cancel_event,
    check_availability,
    create_event,
    get_event,
    list_upcoming_events,
    parse_datetime,
    search_events,
    update_event,
    get_freebusy,
    get_attendee_availability,
    resolve_attendee_by_name,
    list_attendees,
    book_appointment,
)


@pytest.fixture
def calendar_service(monkeypatch):
    class _ServiceHandle:
        def __init__(self) -> None:
            self.mock = MagicMock()
            self.side_effect: Exception | None = None

        def get(self):
            if self.side_effect:
                raise self.side_effect
            return self.mock

    handle = _ServiceHandle()

    def _factory():
        return handle.get()

    import importlib

    default_module = importlib.import_module("voice_agent.tools.google_calendar")
    voice_module = importlib.import_module("voice_agent.tools.google_calendar")

    monkeypatch.setattr(default_module, "get_calendar_service", _factory)
    monkeypatch.setattr(voice_module, "get_calendar_service", _factory)
    return handle


class TestGoogleCalendarTools:
    """Test Google Calendar tool implementations"""

    def test_parse_datetime_iso_format(self):
        """Test parsing ISO format datetime"""
        dt_string = "2025-10-07T14:00:00"
        result = parse_datetime(dt_string)
        assert "2025-10-07" in result
        assert "14:00:00" in result

    def test_parse_datetime_with_timezone(self):
        """Test parsing datetime with timezone"""
        dt_string = "2025-10-07T14:00:00Z"
        result = parse_datetime(dt_string)
        assert "2025-10-07" in result

    def test_create_event_success(self, calendar_service):
        """Test successful event creation"""
        mock_service = calendar_service.mock

        mock_event = {
            "id": "event123",
            "summary": "Team Meeting",
            "start": {"dateTime": "2025-10-07T14:00:00"},
            "end": {"dateTime": "2025-10-07T15:00:00"},
            "htmlLink": "https://calendar.google.com/event?eid=event123",
        }
        mock_service.events().insert().execute.return_value = mock_event

        result = create_event(
            summary="Team Meeting",
            start_time="2025-10-07T14:00:00",
            end_time="2025-10-07T15:00:00",
            description="Discuss project updates",
            location="Conference Room A",
        )
        assert result["success"] is True
        assert result["event_id"] == "event123"
        assert result["summary"] == "Team Meeting"
        assert "event123" in result["link"]

    def test_create_event_with_attendees(self, calendar_service):
        """Test event creation with attendees"""
        mock_service = calendar_service.mock

        mock_event = {
            "id": "event123",
            "summary": "Client Call",
            "start": {"dateTime": "2025-10-07T14:00:00"},
            "end": {"dateTime": "2025-10-07T15:00:00"},
            "htmlLink": "https://calendar.google.com/event?eid=event123",
        }
        mock_service.events().insert().execute.return_value = mock_event

        result = create_event(
            summary="Client Call",
            start_time="2025-10-07T14:00:00",
            end_time="2025-10-07T15:00:00",
            attendees="client@example.com, manager@example.com",
        )

        assert result["success"] is True
        assert mock_service.events().insert.call_count >= 1

    def test_create_event_import_error(self, calendar_service):
        """Test create event when Google API not installed"""
        calendar_service.side_effect = ImportError("google-api-python-client not installed")

        result = create_event(
            summary="Meeting",
            start_time="2025-10-07T14:00:00",
            end_time="2025-10-07T15:00:00",
        )

        assert "error" in result
        assert "install_required" in result

    def test_list_upcoming_events_success(self, calendar_service):
        """Test listing upcoming events"""
        mock_service = calendar_service.mock

        mock_events = {
            "items": [
                {
                    "id": "event1",
                    "summary": "Meeting 1",
                    "start": {"dateTime": "2025-10-07T10:00:00"},
                    "end": {"dateTime": "2025-10-07T11:00:00"},
                    "location": "Room A",
                    "htmlLink": "https://calendar.google.com/event1",
                },
                {
                    "id": "event2",
                    "summary": "Meeting 2",
                    "start": {"dateTime": "2025-10-08T14:00:00"},
                    "end": {"dateTime": "2025-10-08T15:00:00"},
                    "htmlLink": "https://calendar.google.com/event2",
                },
            ]
        }
        mock_service.events().list().execute.return_value = mock_events

        result = list_upcoming_events(max_results=10, days_ahead=7)

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["events"]) == 2
        assert result["events"][0]["summary"] == "Meeting 1"

    def test_list_upcoming_events_empty(self, calendar_service):
        """Test listing events when none exist"""
        mock_service = calendar_service.mock

        mock_events = {"items": []}
        mock_service.events().list().execute.return_value = mock_events

        result = list_upcoming_events()

        assert result["success"] is True
        assert result["count"] == 0
        assert result["events"] == []

    def test_search_events_success(self, calendar_service):
        """Test searching events"""
        mock_service = calendar_service.mock

        mock_events = {
            "items": [
                {
                    "id": "event1",
                    "summary": "Team Standup",
                    "start": {"dateTime": "2025-10-07T09:00:00"},
                    "htmlLink": "https://calendar.google.com/event1",
                }
            ]
        }
        mock_service.events().list().execute.return_value = mock_events

        result = search_events(query="standup", max_results=10)

        assert result["success"] is True
        assert result["count"] == 1
        assert result["query"] == "standup"
        assert result["events"][0]["summary"] == "Team Standup"

    def test_get_event_success(self, calendar_service):
        """Test getting specific event details"""
        mock_service = calendar_service.mock

        mock_event = {
            "id": "event123",
            "summary": "Important Meeting",
            "start": {"dateTime": "2025-10-07T14:00:00"},
            "end": {"dateTime": "2025-10-07T15:00:00"},
            "location": "Conference Room",
            "description": "Quarterly review",
            "attendees": [
                {"email": "user1@example.com", "responseStatus": "accepted"},
                {"email": "user2@example.com", "responseStatus": "needsAction"},
            ],
            "htmlLink": "https://calendar.google.com/event123",
            "status": "confirmed",
        }
        mock_service.events().get().execute.return_value = mock_event

        result = get_event(event_id="event123")

        assert result["success"] is True
        assert result["event_id"] == "event123"
        assert result["summary"] == "Important Meeting"
        assert len(result["attendees"]) == 2

    def test_get_event_not_found(self, calendar_service):
        """Test getting non-existent event"""
        mock_service = calendar_service.mock

        # Simulate 404 error without importing googleapiclient
        mock_service.events().get().execute.side_effect = Exception("Event not found: 404")

        result = get_event(event_id="nonexistent")

        assert "error" in result

    def test_update_event_success(self, calendar_service):
        """Test updating event"""
        mock_service = calendar_service.mock

        existing_event = {
            "id": "event123",
            "summary": "Old Title",
            "start": {"dateTime": "2025-10-07T14:00:00"},
            "end": {"dateTime": "2025-10-07T15:00:00"},
        }
        mock_service.events().get().execute.return_value = existing_event

        updated_event = {
            "id": "event123",
            "summary": "New Title",
            "start": {"dateTime": "2025-10-07T14:00:00"},
            "end": {"dateTime": "2025-10-07T15:00:00"},
            "htmlLink": "https://calendar.google.com/event123",
        }
        mock_service.events().update().execute.return_value = updated_event

        result = update_event(event_id="event123", summary="New Title")

        assert result["success"] is True
        assert result["summary"] == "New Title"

    @patch("voice_agent.tools.google_calendar.send_text_message")
    def test_cancel_event_success(self, mock_send_text, calendar_service):
        """Test canceling event with notification"""
        mock_service = calendar_service.mock

        mock_event = {
            "id": "event123",
            "summary": "Meeting to Cancel",
            "description": "Customer: John Doe (+5511999999999)",
            "start": {"dateTime": "2025-10-07T14:00:00Z"},
        }
        mock_service.events().get().execute.return_value = mock_event
        mock_service.events().delete().execute.return_value = None
        mock_send_text.return_value = {"success": True}

        result = cancel_event(event_id="event123")

        assert result["success"] is True
        assert result["event_id"] == "event123"
        assert "cancelled" in result["message"].lower()
        assert result["notification_sent"] is True
        
        # Verify notification logic
        mock_send_text.assert_called_once()
        call_args = mock_send_text.call_args[1]
        assert call_args["to"] == "5511999999999"
        assert "John Doe" in call_args["message"]

    @patch("voice_agent.tools.google_calendar.send_text_message")
    def test_update_event_reschedule(self, mock_send_text, calendar_service):
        """Test rescheduling event sends notification"""
        mock_service = calendar_service.mock

        # Existing event
        mock_event = {
            "id": "event123",
            "summary": "Checkup",
            "description": "Phone: +5511988888888\nName: Alice",
            "start": {"dateTime": "2025-10-07T14:00:00Z"},
            "end": {"dateTime": "2025-10-07T15:00:00Z"},
        }
        mock_service.events().get().execute.return_value = mock_event
        
        # Mock update response (new time)
        updated_event_resp = mock_event.copy()
        updated_event_resp["start"] = {"dateTime": "2025-10-08T10:00:00Z"} # Changed date
        updated_event_resp["end"] = {"dateTime": "2025-10-08T11:00:00Z"}
        updated_event_resp["htmlLink"] = "http://meet.google.com/abc"
        
        mock_service.events().update().execute.return_value = updated_event_resp
        mock_send_text.return_value = {"success": True}

        # Update with new start_time
        result = update_event(
            event_id="event123", 
            start_time="2025-10-08T10:00:00Z",
            end_time="2025-10-08T11:00:00Z"
        )

        assert result["success"] is True
        assert result["notification_sent"] is True
        
        # Verify notification
        mock_send_text.assert_called_once()
        call_args = mock_send_text.call_args[1]
        assert call_args["to"] == "5511988888888"
        assert "Alice" in call_args["message"]
        assert "08/10/2025" in call_args["message"] # New date


    def test_check_availability_available(self, calendar_service):
        """Test checking availability when slot is free"""
        mock_service = calendar_service.mock

        mock_events = {"items": []}
        mock_service.events().list().execute.return_value = mock_events

        result = check_availability(
            start_time="2025-10-07T14:00:00",
            end_time="2025-10-07T15:00:00",
        )

        assert result["success"] is True
        assert result["available"] is True

    def test_check_availability_conflict(self, calendar_service):
        """Test checking availability with conflicts"""
        mock_service = calendar_service.mock

        mock_events = {
            "items": [
                {
                    "summary": "Existing Meeting",
                    "start": {"dateTime": "2025-10-07T14:30:00"},
                    "end": {"dateTime": "2025-10-07T15:30:00"},
                }
            ]
        }
        mock_service.events().list().execute.return_value = mock_events

        result = check_availability(
            start_time="2025-10-07T14:00:00",
            end_time="2025-10-07T16:00:00",
        )

        assert result["success"] is True
        assert result["available"] is False
        assert len(result["conflicts"]) == 1

    def test_tool_definitions_format(self):
        """Test that tool definitions follow MCP format"""
        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) == 10

        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "handler" in tool
            assert callable(tool["handler"])

            if "required" in tool:
                assert isinstance(tool["required"], list)

    def test_tool_names_unique(self):
        """Test that all tool names are unique"""
        tool_names = [tool["name"] for tool in TOOL_DEFINITIONS]
        assert len(tool_names) == len(set(tool_names))

    def test_tool_handlers_valid(self):
        """Test that all tool handlers are callable"""
        handlers = [
            create_event,
            list_upcoming_events,
            search_events,
            get_event,
            update_event,
            cancel_event,
            check_availability,
        ]

        for handler in handlers:
            assert callable(handler)


class TestGoogleCalendarIntegration:
    """Integration tests for Google Calendar (require mock service)"""

    @patch("voice_agent.tools.google_calendar.GOOGLE_AVAILABLE", False)
    def test_library_not_installed(self, *_patches, calendar_service):
        """Test behavior when Google libraries not installed"""
        calendar_service.side_effect = ImportError(
            "Google Calendar API not available"
        )

        result = create_event(
            summary="Test",
            start_time="2025-10-07T14:00:00",
            end_time="2025-10-07T15:00:00",
        )

        assert "error" in result
        assert result.get("install_required") is True

    def test_credentials_not_found(self, calendar_service):
        """Test behavior when credentials file not found"""
        calendar_service.side_effect = FileNotFoundError(
            "Google Calendar credentials not found"
        )

        result = create_event(
            summary="Test",
            start_time="2025-10-07T14:00:00",
            end_time="2025-10-07T15:00:00",
        )

        assert "error" in result
        assert "error" in result
        assert result.get("setup_required") is True


class TestAvailabilityLogic:
    """Test availability and FreeBusy logic"""
    
    @patch("voice_agent.tools.google_calendar.resolve_attendee_by_name")
    @patch("voice_agent.tools.google_calendar.get_freebusy")
    def test_get_attendee_availability_success(self, mock_freebusy, mock_resolve):
        """Test getting availability with working hours and busy slots"""
        # Mock Attendee
        mock_resolve.return_value = {
            "attendee_name": "Dr. Smith",
            "attendee_email": "dr.smith@example.com",
            "calendar_id": "dr.smith@example.com",
            "working_hours": {
                "monday": {"start": "09:00", "end": "12:00"},
                "tuesday": {"start": "09:00", "end": "18:00"},
                "wednesday": {"start": "09:00", "end": "18:00"},
                "thursday": {"start": "09:00", "end": "18:00"},
                "friday": {"start": "09:00", "end": "18:00"}
            }
        }
        
        # Mock FreeBusy Response
        # Lets say Monday is queried (e.g. 2030-10-07 is a Monday)
        mock_freebusy.return_value = {
            "calendars": {
                "dr.smith@example.com": {
                    "busy": [
                        # Busy from 10:00 to 11:00 on Monday
                         {"start": "2030-10-07T10:00:00Z", "end": "2030-10-07T11:00:00Z"}
                    ]
                }
            }
        }
        
        # We need to control "now" inside the function or pass a future start date
        # Passing start_date="2030-10-07" (Monday)
        
        result = get_attendee_availability(
            attendee_name="Dr. Smith",
            start_date="2030-10-07", # Monday
            days=1
        )
        
        assert result["success"] is True
        assert result["attendee"] == "Dr. Smith"
        
        # Working hours: 09-12 (3 hours)
        # Busy: 10-11
        # Expected slots: 09-10, 11-12
        slots = result["available_slots"]
        
        # Helper to extract hour from ISO string
        def get_hour(iso):
            return int(iso.split("T")[1].split(":")[0])
            
        start_hours = [get_hour(s["start"]) for s in slots]
        

class TestBookingLogic:
    """Test book_appointment logic"""

    @patch("voice_agent.tools.google_calendar.resolve_attendee_by_name")
    @patch("voice_agent.tools.google_calendar.get_calendar_service")
    @patch("voice_agent.tools.google_calendar._get_customer_by_phone")
    @patch("voice_agent.tools.google_calendar.send_text_message")
    def test_book_appointment_success(self, mock_send_whatsapp, mock_get_customer, mock_service_factory, mock_resolve):
        """Test booking appointment with customer lookup and WhatsApp notification"""
        # 1. Mock Attendee
        mock_resolve.return_value = {
            "attendee_name": "Dr. Smith",
            "attendee_email": "dr.smith@example.com",
            "calendar_id": "dr.smith@example.com",
        }
        
        # 2. Mock Customer
        mock_get_customer.return_value = {
            "name": "Maria Customer",
            "phone": "+5511999999999",
            "email": "maria@example.com",
            "google_calendar_email": None,
            "whatsapp_number": "+5511999999999"  # Has WhatsApp
        }
        
        # 3. Mock Calendar Service
        mock_service = MagicMock()
        mock_service_factory.return_value = mock_service
        
        mock_event_resp = {
            "id": "evt123",
            "htmlLink": "http://cal/evt123",
            "conferenceData": {"entryPoints": [{"uri": "http://meet/abc"}]}
        }
        mock_service.events().insert().execute.return_value = mock_event_resp
        
        # 4. Mock WhatsApp
        mock_send_whatsapp.return_value = {"success": True}
        
        result = book_appointment(
            attendee_name="Dr. Smith",
            customer_phone="+5511999999999",
            start_time="2025-10-10T14:00:00"
        )
        
        assert result["success"] is True
        assert result["event_id"] == "evt123"
        # Assert Notifications
        # NOTE: Email is False now because service accounts without Domain-Wide Delegation
        # cannot invite attendees. We rely on WhatsApp notifications instead.
        assert result["notifications_generated"]["email"] is False
        assert result["notifications_generated"]["whatsapp"] is True
        
        # Verify WhatsApp Call
        mock_send_whatsapp.assert_called_once()
        args, kwargs = mock_send_whatsapp.call_args
        assert kwargs["to"] == "5511999999999" # Clean number
        assert "14:00" in kwargs["message"]

        # Verify API Call
        args, kwargs = mock_service.events().insert.call_args
        body = kwargs["body"]
        assert body["summary"] == "Consulta - Maria Customer"
        # Attendees are NOT added for service accounts without Domain-Wide Delegation
        assert "attendees" not in body or len(body.get("attendees", [])) == 0
        # Check sendUpdates (should be "none" since no attendees)
        assert kwargs["sendUpdates"] == "none"

    @patch("voice_agent.tools.google_calendar.resolve_attendee_by_name")
    def test_book_appointment_no_customer(self, mock_resolve):
        """Test booking when customer not found"""
        mock_resolve.return_value = {
            "attendee_name": "Dr. Smith",
            "calendar_id": "dr.smith@example.com"
        }
        
        # We need to mock _get_customer_by_phone too, but lets do it via context manager or just verify it fails 
        # Actually easier to use the decorator pattern on class or test
        
        with patch("voice_agent.tools.google_calendar._get_customer_by_phone", return_value=None):
            result = book_appointment(
                attendee_name="Dr. Smith",
                customer_phone="+55000000000",
                start_time="2025-10-10T10:00:00"
            )
            assert result["success"] is False
            assert "Customer with phone" in result.get("error", "")
