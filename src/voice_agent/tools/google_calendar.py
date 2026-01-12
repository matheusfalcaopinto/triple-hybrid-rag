"""
Google Calendar Tools for MCP

Provides calendar management capabilities including:
- Creating events
- Listing upcoming events
- Searching events
- Updating/canceling events
- Checking availability

Requires Google Calendar API OAuth2 credentials.
"""

from __future__ import annotations

import logging
import os
import re
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)
# Try to import whatsapp tool; gracefully handle if missing/circular
try:
    from .whatsapp import send_text_message
except ImportError:
    send_text_message = None

def _load_calendar_templates() -> Dict[str, str]:
    """Load calendar templates from JSON file."""
    defaults = {
        "cancellation": """❌ *Agendamento Cancelado*

Olá {name}, sua consulta de *{date} às {time}* com *{attendee_name}* foi cancelada.

Caso deseje remarcar, responda esta mensagem.""",
        "reschedule": """🔄 *Agendamento Alterado*

Olá {name}, seu agendamento com *{attendee_name}* foi alterado.

📅 *Nova Data:* {date}
🕐 *Novo Horário:* {time}
📍 *Local:* {location}

{meet_link_section}

Dúvidas? Responda esta mensagem.""",
        "confirmation": """✅ *Agendamento Confirmado*

📅 *Data:* {date}
🕐 *Horário:* {time}
👨‍⚕️ *Profissional:* {attendee_name}
📍 *Local:* {location}

{meet_link_section}

Para cancelar ou reagendar, responda esta mensagem."""
    }

    try:
        template_path = Path(__file__).parent.parent / "data" / "templates.json"
        if not template_path.exists():
            return defaults
        
        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("calendar", defaults)
    except Exception as e:
        logger.error("Failed to load calendar templates: %s", e)
        return defaults

_TEMPLATES = _load_calendar_templates()
APPOINTMENT_CANCELLATION_TEMPLATE = _TEMPLATES.get("cancellation", "")
APPOINTMENT_RESCHEDULE_TEMPLATE = _TEMPLATES.get("reschedule", "")
APPOINTMENT_CONFIRMATION_TEMPLATE = _TEMPLATES.get("confirmation", "")

def _extract_customer_info(description: str) -> Dict[str, str]:
    """Extract customer details from event description."""
    if not description:
        return {}
    
    info = {}
    
    # Pattern 1: "Customer: Name (Phone)"
    match1 = re.search(r"Customer:\s*(.*?)\s*\((.*?)\)", description)
    if match1:
        info["name"] = match1.group(1).strip()
        info["phone"] = match1.group(2).strip()
        return info

    # Pattern 2: "Name: ... \n Phone: ..."
    phone_match = re.search(r"Phone:\s*(.+)", description)
    if phone_match:
        info["phone"] = phone_match.group(1).strip()
    
    name_match = re.search(r"Name:\s*(.+)", description)
    if name_match:
        info["name"] = name_match.group(1).strip()

    return info




try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


# IPv6 Connectivity Workaround
# Some environments have broken IPv6 connectivity which causes Google API
# calls to hang. This forces Python to use IPv4 when connecting to Google.
def _force_ipv4_for_google_apis():
    """Force IPv4 connections to work around IPv6 connectivity issues."""
    import socket
    _original_getaddrinfo = socket.getaddrinfo
    
    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        # Only force IPv4 for Google domains
        if host:
            # Handle both str and bytes host
            host_str = host.decode('utf-8') if isinstance(host, bytes) else host
            if 'googleapis.com' in host_str or 'google.com' in host_str:
                family = socket.AF_INET
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    
    socket.getaddrinfo = _ipv4_getaddrinfo


# Apply the workaround on module load
_force_ipv4_for_google_apis()


SCOPES = ["https://www.googleapis.com/auth/calendar"]


def get_credentials_path() -> Path:
    """Get path to Google Calendar credentials file."""
    credentials_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH")
    if credentials_path:
        return Path(credentials_path)
    return Path(__file__).parent.parent / "data" / "google_calendar_credentials.json"


def get_token_path() -> Path:
    """Get path to stored OAuth2 token."""
    token_path = os.getenv("GOOGLE_CALENDAR_TOKEN_PATH")
    if token_path:
        return Path(token_path)
    return Path(__file__).parent.parent / "data" / "google_calendar_token.json"


def get_calendar_service() -> Any:
    """
    Get authenticated Google Calendar service.
    
    Authentication modes (controlled by GOOGLE_CALENDAR_AUTH_MODE env var):
    - "service_account" (default): Use service account, fail if unavailable
    - "oauth": Use OAuth2 with cached token, refresh if expired
    - "oauth_interactive": Allow interactive OAuth2 flow (opens browser - WILL HANG in headless)
    
    Domain-Wide Delegation:
        If GOOGLE_CALENDAR_DELEGATION_EMAIL is set, the service account will impersonate
        that user (requires DWD configured in Google Workspace Admin Console).
    
    Returns:
        Google Calendar API service object
        
    Raises:
        FileNotFoundError: If required credentials file not found
        ImportError: If google-api-python-client not installed
        RuntimeError: If authentication fails
    """
    if not GOOGLE_AVAILABLE:
        raise ImportError(
            "Google Calendar API not available. Install with: pip install "
            "google-auth google-auth-oauthlib google-auth-httplib2 "
            "google-api-python-client"
        )
    
    # Get config from SETTINGS (with env var fallback for backwards compatibility)
    auth_mode = getattr(SETTINGS, 'google_calendar_auth_mode', None) or \
                os.getenv("GOOGLE_CALENDAR_AUTH_MODE", "service_account")
    auth_mode = auth_mode.lower()
    
    delegation_email = getattr(SETTINGS, 'google_calendar_delegation_email', None) or \
                       os.getenv("GOOGLE_CALENDAR_DELEGATION_EMAIL", "")
    
    logger.debug("Google Calendar auth mode: %s, delegation: %s", 
                 auth_mode, delegation_email or "disabled")
    
    # --- Service Account Authentication ---
    if auth_mode == "service_account":
        service_account_path = _get_service_account_path()
        if not service_account_path.exists():
            raise FileNotFoundError(
                f"Service account file not found at {service_account_path}. "
                f"Set GOOGLE_SERVICE_ACCOUNT_PATH or use GOOGLE_CALENDAR_AUTH_MODE=oauth."
            )
        try:
            from google.oauth2 import service_account as sa_module
            credentials = sa_module.Credentials.from_service_account_file(
                str(service_account_path),
                scopes=SCOPES,
            )
            
            # Domain-Wide Delegation: impersonate a user if configured
            if delegation_email:
                logger.info("Using Domain-Wide Delegation, impersonating: %s", delegation_email)
                credentials = credentials.with_subject(delegation_email)
            
            logger.info("Authenticated with service account: %s", service_account_path)
            return build("calendar", "v3", credentials=credentials)
        except Exception as e:
            raise RuntimeError(
                f"Service account authentication failed: {e}. "
                f"Check that the service account file is valid and has Calendar API access."
            ) from e
    
    # --- OAuth2 Authentication (with cached token only - no interactive flow) ---
    if auth_mode == "oauth":
        token_path = get_token_path()
        if not token_path.exists():
            raise FileNotFoundError(
                f"OAuth token not found at {token_path}. "
                f"Run with GOOGLE_CALENDAR_AUTH_MODE=oauth_interactive once to generate it."
            )
        
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                logger.info("OAuth token expired, refreshing...")
                creds.refresh(Request())
                # Save refreshed token
                with open(token_path, "w") as token:
                    token.write(creds.to_json())
                logger.info("OAuth token refreshed successfully")
            else:
                raise RuntimeError(
                    "OAuth token is invalid and cannot be refreshed. "
                    "Run with GOOGLE_CALENDAR_AUTH_MODE=oauth_interactive to re-authenticate."
                )
        
        return build("calendar", "v3", credentials=creds)
    
    # --- Interactive OAuth2 (WARNING: Opens browser, will hang in headless) ---
    if auth_mode == "oauth_interactive":
        logger.warning(
            "Using interactive OAuth2 flow. This will open a browser and HANG in headless environments!"
        )
        credentials_path = get_credentials_path()
        token_path = get_token_path()
        creds = None
        
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not credentials_path.exists():
                    raise FileNotFoundError(
                        f"OAuth credentials not found at {credentials_path}. "
                        f"Download from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, "w") as token:
                token.write(creds.to_json())
        
        return build("calendar", "v3", credentials=creds)
    
    raise ValueError(
        f"Invalid GOOGLE_CALENDAR_AUTH_MODE: {auth_mode}. "
        f"Valid values: service_account, oauth, oauth_interactive"
    )


def _get_service_account_path() -> Path:
    """Get path to Google service account JSON file."""
    path = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
    if path:
        return Path(path)
    return Path(__file__).parent.parent / "data" / "google_calendar_service_account.json"


def _get_supabase_client():
    """Get Supabase client for database operations."""
    try:
        from voice_agent.utils.db import get_supabase_client
        return get_supabase_client()
    except ImportError:
        # Fallback if utils not available
        try:
             from supabase import create_client
             url = os.getenv("SUPABASE_URL", "")
             key = os.getenv("SERVICE_ROLE_KEY", "")
             if url and key:
                 return create_client(url, key)
        except ImportError:
             pass
    return None


def _get_default_org_id() -> Optional[str]:
    """Get the default organization ID from env or first org in DB."""
    org_id = os.getenv("GOOGLE_CALENDAR_DEFAULT_ORG_ID")
    if org_id:
        return org_id
    
    # Fallback: query first org from DB
    client = _get_supabase_client()
    if client:
        try:
            result = client.table("organizations").select("id").limit(1).execute()
            if result.data:
                return result.data[0]["id"]
        except Exception as e:
            logger.warning("Failed to fetch default org_id from database: %s", e)
    
    return None


def _get_calendar_count(org_id: str) -> int:
    """Get number of configured calendars for an organization."""
    client = _get_supabase_client()
    if not client:
        return 0
    
    try:
        result = client.table("org_calendar_connections") \
            .select("id") \
            .eq("org_id", org_id) \
            .execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        logger.debug("Failed to get calendar count: %s", e)
        return 0


def resolve_attendee_by_name(
    attendee_name: str,
    org_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve attendee display name to calendar email.
    
    Args:
        attendee_name: Display name like "Dr. Silva" or "Dr. João Silva"
        org_id: Organization ID (uses default if not provided)
        
    Returns:
        {"attendee_name", "attendee_email", "calendar_id", "working_hours"} or None
    """
    client = _get_supabase_client()
    if not client:
        return None
    
    org_id = org_id or _get_default_org_id()
    if not org_id:
        return None
    
    try:
        # Try org-scoped first
        result = client.table("org_calendar_connections") \
            .select("*") \
            .eq("org_id", org_id) \
            .ilike("attendee_name", f"%{attendee_name}%") \
            .limit(1) \
            .execute()

        if result.data:
            return result.data[0]

        # Fallback: search any org (helps stub/harness cases)
        fallback = client.table("org_calendar_connections") \
            .select("*") \
            .ilike("attendee_name", f"%{attendee_name}%") \
            .limit(1) \
            .execute()

        if fallback.data:
            return fallback.data[0]
        return None
    except Exception as e:
        logger.debug("Failed to resolve attendee '%s': %s", attendee_name, e)
        return None


def list_attendees(
    org_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List all schedulable attendees for the organization.
    Returns names the agent can use in conversation.
    
    Args:
        org_id: Organization ID (uses default if not provided)
        
    Returns:
        {
            "success": true,
            "attendees": [{"name": "Dr. João Silva", "type": "worker"}],
            "count": 2,
            "single_calendar_mode": false
        }
    """
    client = _get_supabase_client()
    if not client:
        return {
            "error": "Database connection not available",
            "attendees": [],
            "count": 0,
        }
    
    org_id = org_id or _get_default_org_id()
    # If org_id is missing or invalid format (e.g., uuid required by backend), skip filtering to allow stub data
    org_filter = bool(org_id)

    try:
        query = client.table("org_calendar_connections").select("attendee_name, calendar_type, is_default")
        if org_filter:
            query = query.eq("org_id", org_id)
        result = query.order("is_default", desc=True).order("attendee_name").execute()

        # Fallback: if org-scoped query returned nothing (or failed due to bad format), try without org filter
        if not result.data:
            fallback = client.table("org_calendar_connections").select("attendee_name, calendar_type, is_default").order("is_default", desc=True).order("attendee_name").execute()
            result = fallback

        if not result.data:
            return {
                "success": True,
                "attendees": [],
                "count": 0,
                "message": "No calendars configured for this organization",
            }

        attendees = [
            {
                "name": row["attendee_name"],
                "type": row.get("calendar_type", "worker"),
                "is_default": row.get("is_default", False),
            }
            for row in result.data
        ]

        return {
            "success": True,
            "attendees": attendees,
            "count": len(attendees),
            "single_calendar_mode": len(attendees) == 1,
        }

    except Exception as e:
        return {
            "error": f"Failed to list attendees: {str(e)}",
            "attendees": [],
            "count": 0,
        }



def get_freebusy(
    time_min: str,
    time_max: str,
    items: list[dict[str, str]],
    calendar_service: Any = None,
) -> dict[str, Any]:
    """
    Query Google Calendar FreeBusy API.
    
    Args:
        time_min: ISO start time
        time_max: ISO end time
        items: List of dicts with 'id' key for calendars to check
        calendar_service: Optional service object (uses default if None)
        
    Returns:
        Dict with 'calendars' key containing busy slots
    """
    if not calendar_service:
        calendar_service = get_calendar_service()
        
    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "items": items,
        "timeZone": os.getenv("TIMEZONE", "America/Sao_Paulo"),
    }
    
    return calendar_service.freebusy().query(body=body).execute()


def _parse_working_hours(working_hours: dict, dt: datetime) -> tuple[datetime, datetime] | None:
    """
    Get start and end datetimes for working hours on a specific date.
    
    Args:
        working_hours: Dict of {day_name: {start: "HH:MM", end: "HH:MM"}}
        dt: The date to check
        
    Returns:
        (start_dt, end_dt) or None if not working day
    """
    day_name = dt.strftime("%A").lower()
    if day_name not in working_hours:
        return None
        
    day_schedule = working_hours[day_name]
    if not day_schedule:
        return None
        
    try:
        start_time = datetime.strptime(day_schedule["start"], "%H:%M").time()
        end_time = datetime.strptime(day_schedule["end"], "%H:%M").time()
        
        start_dt = dt.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0)
        end_dt = dt.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)
        
        return start_dt, end_dt
    except (ValueError, KeyError):
        return None


def get_attendee_availability(
    attendee_name: str,
    start_date: str | None = None,
    days: int = 5,
    org_id: str | None = None,
) -> dict[str, Any]:
    """
    Check availability for a specific attendee/worker.
    Considers their configured working hours and current calendar events.
    
    Args:
        attendee_name: Name of the professional (e.g., "Dr. Silva")
        start_date: ISO date to start checking (default: now)
        days: Number of days to check (default: 5)
        org_id: Optional organization ID
        
    Returns:
        {
            "success": bool,
            "attendee": str,
            "available_slots": [
                {"start": "ISO", "end": "ISO", "date": "YYYY-MM-DD"}
            ]
        }
    """
    # 1. Resolve Attendee
    attendee = resolve_attendee_by_name(attendee_name, org_id)
    if not attendee:
        return {
            "success": False,
            "error": f"Attendee '{attendee_name}' not found.",
            "available_slots": []
        }
        
    calendar_id = attendee.get("calendar_id") or attendee.get("attendee_email")
    working_hours = attendee.get("working_hours", {})
    
    if not calendar_id:
        return {
            "success": False,
            "error": "No calendar ID found for attendee.",
            "available_slots": []
        }

    # 2. Determine Time Range
    tz_str = os.getenv("TIMEZONE", "America/Sao_Paulo")
    # Simple naive implementation for now, ideally use pytz or zoneinfo
    now = datetime.now()
    if start_date:
        try:
            current_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            current_dt = now
    else:
        current_dt = now

    end_dt = current_dt + timedelta(days=days)
    
    # Format for API
    time_min = current_dt.isoformat() + "Z"
    time_max = end_dt.isoformat() + "Z"
    
    # 3. Fetch FreeBusy
    try:
        fb_resp = get_freebusy(
            time_min=time_min,
            time_max=time_max,
            items=[{"id": calendar_id}]
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to check calendar: {str(e)}",
            "available_slots": []
        }

    busy_slots = fb_resp.get("calendars", {}).get(calendar_id, {}).get("busy", [])
    
    # Normalize busy slots to naive datetimes for comparison
    # (Assuming API returns UTC ISO strings ending in Z)
    normalized_busy = []
    for slot in busy_slots:
        try:
            s = datetime.fromisoformat(slot["start"].replace("Z", "+00:00")).replace(tzinfo=None)
            e = datetime.fromisoformat(slot["end"].replace("Z", "+00:00")).replace(tzinfo=None)
            normalized_busy.append((s, e))
        except ValueError:
            pass

    # 4. Calculate Availability
    available_slots = []
    
    # Check each day in range
    for i in range(days):
        check_date = current_dt + timedelta(days=i)
        wh = _parse_working_hours(working_hours, check_date)
        
        if not wh:
            continue
            
        work_start, work_end = wh
        
        # Simple slot logic: 1 hour slots (can be parameterized later)
        slot_duration = timedelta(hours=1)
        current_slot_start = work_start
        
        while current_slot_start + slot_duration <= work_end:
            current_slot_end = current_slot_start + slot_duration
            
            # Check if this slot overlaps with ANY busy slot
            is_busy = False
            for busy_start, busy_end in normalized_busy:
                # Overlap logic: (StartA < EndB) and (EndA > StartB)
                if current_slot_start < busy_end and current_slot_end > busy_start:
                    is_busy = True
                    break
            
            if not is_busy and current_slot_start > now: # Don't suggest past slots
                available_slots.append({
                    "start": current_slot_start.isoformat(),
                    "end": current_slot_end.isoformat(),
                    "date": check_date.strftime("%Y-%m-%d"),
                    "day_of_week": check_date.strftime("%A")
                })
                
            current_slot_start += timedelta(minutes=60) # Interval (could be 30m)

    return {
        "success": True,
        "attendee": attendee["attendee_name"],
        "calendar_id": calendar_id,
        "available_slots": available_slots[:20]  # Limit results
    }


def _get_customer_by_phone(phone: str) -> dict[str, Any] | None:
    """Get customer details by phone number."""
    client = _get_supabase_client()
    if not client:
        return None
        
    try:
        # Normalize phone if needed, but for now assume exact match or simple clean
        # The DB likely stores E.164
        response = client.table("customers").select("*").eq("phone", phone).execute()
        if response.data:
            return response.data[0]
    except Exception as e:
        print(f"Error fetching customer: {e}")
        
    return None


def book_appointment(
    attendee_name: str,
    customer_phone: str,
    start_time: str,
    summary: str = None,
    end_time: str = None,
    add_video_meeting: bool = True,
    org_id: str | None = None,
) -> dict[str, Any]:
    """
    Book an appointment between a professional and a customer.
    
    Args:
        attendee_name: Name of the professional (e.g., 'Dr. Silva')
        customer_phone: Customer's phone number
        start_time: ISO start time string
        summary: Optional title (default: 'Consulta - {customer_name}')
        end_time: Optional ISO end time (default: start + 1h)
        add_video_meeting: Whether to add Google Meet link (default: True)
        org_id: Optional organization ID
        
    Returns:
        Dict with booking success status and details
    """
    # 1. Resolve Professional
    attendee = resolve_attendee_by_name(attendee_name, org_id)
    if not attendee:
        return {
            "success": False,
            "error": f"Professional '{attendee_name}' not found."
        }
        
    calendar_id = attendee.get("calendar_id") or attendee.get("attendee_email")
    if not calendar_id:
        return {"success": False, "error": "Professional has no calendar configured."}

    # 2. Resolve Customer
    customer = _get_customer_by_phone(customer_phone)
    if not customer:
        return {
            "success": False, 
            "error": f"Customer with phone {customer_phone} not found. Please register them first."
        }
    
    customer_email = customer.get("email") # Primary email
    customer_cal_email = customer.get("google_calendar_email") # Explicit calendar email
    customer_name = customer.get("name", "Customer")
    
    # 3. Prepare Event Data
    if not summary:
        summary = f"Consulta - {customer_name}"
        
    # Calculate end time if missing (default 1 hour)
    dt_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if not end_time:
        dt_end = dt_start + timedelta(hours=1)
        end_time_str = dt_end.isoformat()
    else:
        end_time_str = end_time

    # Build Attendees List
    event_attendees = []
    
    # Check if attendee invites are enabled (requires Domain-Wide Delegation)
    enable_attendee_invites = getattr(SETTINGS, 'google_calendar_enable_attendee_invites', False)
    
    if enable_attendee_invites:
        # Add customer as attendee (will receive email invite from Google Calendar)
        invite_email = customer_cal_email if customer_cal_email else customer_email
        if invite_email:
            event_attendees.append({"email": invite_email, "displayName": customer_name})
            logger.debug("Adding customer as attendee: %s", invite_email)
    else:
        logger.debug("Attendee invites disabled, relying on WhatsApp for notifications")
    
    # 4. Create Event
    service = get_calendar_service()
    
    event_body = {
        "summary": summary,
        "description": f"Appointment with {attendee['attendee_name']}.\nCustomer: {customer_name} ({customer_phone})",
        "start": {"dateTime": start_time, "timeZone": os.getenv("TIMEZONE", "America/Sao_Paulo")},
        "end": {"dateTime": end_time_str, "timeZone": os.getenv("TIMEZONE", "America/Sao_Paulo")},
    }
    
    # Add attendees if we have any (requires Domain-Wide Delegation for service accounts)
    if event_attendees:
        event_body["attendees"] = event_attendees
    
    # Check if Google Meet is enabled
    enable_meet = getattr(SETTINGS, 'google_calendar_enable_meet', False)
    
    # Override with function parameter if explicitly passed
    should_add_meet = add_video_meeting and enable_meet
    
    conference_data_version = 0
    if should_add_meet:
        event_body["conferenceData"] = {
            "createRequest": {
                "requestId": f"meet-{int(datetime.now().timestamp())}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"}
            }
        }
        conference_data_version = 1
    
    try:
        # Use sendUpdates="none" for service accounts (they can't send invites without delegation)
        send_updates = "none" if not event_attendees else "all"
        
        # Try to create event (with Meet if requested)
        try:
            event = service.events().insert(
                calendarId=calendar_id,
                body=event_body,
                sendUpdates=send_updates,
                conferenceDataVersion=conference_data_version
            ).execute()
        except HttpError as meet_error:
            # If Meet creation fails (common for service accounts on personal calendars), retry without Meet
            if "Invalid conference type" in str(meet_error) or "conference" in str(meet_error).lower():
                logger.warning("Meet creation not supported, retrying without video conference")
                if "conferenceData" in event_body:
                    del event_body["conferenceData"]
                event = service.events().insert(
                    calendarId=calendar_id,
                    body=event_body,
                    sendUpdates=send_updates,
                    conferenceDataVersion=0
                ).execute()
            else:
                raise  # Re-raise if it's a different error
        
        meet_link = event.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri")
        
        # 5. Send WhatsApp Notification
        whatsapp_sent = False
        whatsapp_number = customer.get("whatsapp_number")
        
        if whatsapp_number and send_text_message:
            try:
                # Format Data
                dt_obj = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                date_str = dt_obj.strftime("%d/%m/%Y")
                time_str = dt_obj.strftime("%H:%M")
                
                meet_section = ""
                if meet_link:
                    meet_section = f"📹 *Link da Videochamada:*\n{meet_link}\n"
                
                message = APPOINTMENT_CONFIRMATION_TEMPLATE.format(
                    date=date_str,
                    time=time_str,
                    attendee_name=attendee['attendee_name'],
                    location="Online (Google Meet)" if meet_link else "Presencial",
                    meet_link_section=meet_section
                ).strip()
                
                # Normalize number (simple digit check)
                clean_number = "".join(filter(str.isdigit, whatsapp_number))
                
                send_result = send_text_message(to=clean_number, message=message)
                whatsapp_sent = send_result.get("success", False)
                if not whatsapp_sent:
                     logger.warning("WhatsApp send failed: %s", send_result.get('error'))

            except Exception as e:
                 logger.warning("Error sending WhatsApp: %s", e)

        return {
            "success": True,
            "event_id": event.get("id"),
            "html_link": event.get("htmlLink"),
            "meet_link": meet_link,
            "attendee": attendee["attendee_name"],
            "customer": customer_name,
            "notifications_generated": {
                "email": bool(event_attendees),  # Only true if we added attendees
                "whatsapp": whatsapp_sent
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to book appointment: {str(e)}"
        }


def parse_datetime(dt_string: str) -> str:
    """
    Parse datetime string into ISO format for Google Calendar API.
    
    Args:
        dt_string: DateTime in formats like "2025-10-07 14:00", "tomorrow 2pm", etc.
        
    Returns:
        ISO format datetime string
    """
    try:
        dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
        return dt.isoformat()
    except ValueError:
        now = datetime.now()
        if "tomorrow" in dt_string.lower():
            dt = now + timedelta(days=1)
            return dt.replace(hour=9, minute=0, second=0, microsecond=0).isoformat()
        return now.isoformat()


def create_event(
    summary: str = None,
    start_time: str = None,
    end_time: str = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[str] = None,
    title: Optional[str] = None,  # Alias for summary (LLM compatibility)
) -> Dict[str, Any]:
    """
    Create a new calendar event.
    
    Args:
        summary: Event title/summary
        start_time: Start datetime (ISO format or "2025-10-07 14:00")
        end_time: End datetime (ISO format or "2025-10-07 15:00")
        description: Event description (optional)
        location: Event location (optional)
        attendees: Comma-separated email addresses (optional)
        title: Alias for summary (for LLM compatibility)
        
    Returns:
        Created event details or error message
    """
    # Handle title as alias for summary
    if summary is None and title is not None:
        summary = title
    
    if not summary:
        return {"error": "Missing required parameter: summary (or title)"}
    if not start_time:
        return {"error": "Missing required parameter: start_time"}
    if not end_time:
        return {"error": "Missing required parameter: end_time"}
    
    try:
        service = get_calendar_service()
        
        event_data: Dict[str, Any] = {
            "summary": summary,
            "start": {
                "dateTime": parse_datetime(start_time),
                "timeZone": os.getenv("TIMEZONE", "America/Sao_Paulo"),
            },
            "end": {
                "dateTime": parse_datetime(end_time),
                "timeZone": os.getenv("TIMEZONE", "America/Sao_Paulo"),
            },
        }
        
        if description:
            event_data["description"] = description
        
        if location:
            event_data["location"] = location
        
        if attendees:
            email_list = [email.strip() for email in attendees.split(",")]
            event_data["attendees"] = [{"email": email} for email in email_list]
        
        event = service.events().insert(calendarId="primary", body=event_data).execute()
        
        return {
            "success": True,
            "event_id": event.get("id"),
            "summary": event.get("summary"),
            "start": event.get("start", {}).get("dateTime"),
            "end": event.get("end", {}).get("dateTime"),
            "link": event.get("htmlLink"),
            "message": "Event created successfully",
        }
        
    except FileNotFoundError as e:
        return {"error": str(e), "setup_required": True}
    except ImportError as e:
        return {"error": str(e), "install_required": True}
    except HttpError as e:
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": f"Failed to create event: {str(e)}"}


def list_upcoming_events(
    max_results: int = 10,
    days_ahead: int = 7,
) -> Dict[str, Any]:
    """
    List upcoming calendar events.
    
    Args:
        max_results: Maximum number of events to return (default 10)
        days_ahead: Number of days to look ahead (default 7)
        
    Returns:
        List of upcoming events or error message
    """
    try:
        service = get_calendar_service()
        
        now = datetime.now(timezone.utc)
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"
        
        events_result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        
        events = events_result.get("items", [])
        
        event_list = []
        for event in events:
            start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date"))
            end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date"))
            
            event_list.append({
                "id": event.get("id"),
                "summary": event.get("summary"),
                "start": start,
                "end": end,
                "location": event.get("location"),
                "description": event.get("description"),
                "attendees": [a.get("email") for a in event.get("attendees", [])],
                "link": event.get("htmlLink"),
            })
        
        return {
            "success": True,
            "count": len(event_list),
            "events": event_list,
            "time_range": f"{max_results} events in next {days_ahead} days",
        }
        
    except FileNotFoundError as e:
        return {"error": str(e), "setup_required": True}
    except ImportError as e:
        return {"error": str(e), "install_required": True}
    except HttpError as e:
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": f"Failed to list events: {str(e)}"}


def search_events(
    query: str,
    max_results: int = 10,
) -> Dict[str, Any]:
    """
    Search calendar events by keyword.
    
    Args:
        query: Search term (matches summary, description, location, attendees)
        max_results: Maximum number of results (default 10)
        
    Returns:
        List of matching events or error message
    """
    try:
        service = get_calendar_service()
        
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        events_result = service.events().list(
            calendarId="primary",
            q=query,
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        
        events = events_result.get("items", [])
        
        event_list = []
        for event in events:
            start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date"))
            
            event_list.append({
                "id": event.get("id"),
                "summary": event.get("summary"),
                "start": start,
                "location": event.get("location"),
                "description": event.get("description"),
                "link": event.get("htmlLink"),
            })
        
        return {
            "success": True,
            "count": len(event_list),
            "query": query,
            "events": event_list,
        }
        
    except FileNotFoundError as e:
        return {"error": str(e), "setup_required": True}
    except ImportError as e:
        return {"error": str(e), "install_required": True}
    except HttpError as e:
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": f"Failed to search events: {str(e)}"}


def get_event(event_id: str) -> Dict[str, Any]:
    """
    Get details of a specific event.
    
    Args:
        event_id: Google Calendar event ID
        
    Returns:
        Event details or error message
    """
    try:
        service = get_calendar_service()
        
        event = service.events().get(calendarId="primary", eventId=event_id).execute()
        
        return {
            "success": True,
            "event_id": event.get("id"),
            "summary": event.get("summary"),
            "start": event.get("start", {}).get("dateTime", event.get("start", {}).get("date")),
            "end": event.get("end", {}).get("dateTime", event.get("end", {}).get("date")),
            "location": event.get("location"),
            "description": event.get("description"),
            "attendees": [
                {
                    "email": a.get("email"),
                    "status": a.get("responseStatus"),
                }
                for a in event.get("attendees", [])
            ],
            "link": event.get("htmlLink"),
            "status": event.get("status"),
        }
        
    except FileNotFoundError as e:
        return {"error": str(e), "setup_required": True}
    except ImportError as e:
        return {"error": str(e), "install_required": True}
    except Exception as e:
        # Handle HttpError if library is available
        if GOOGLE_AVAILABLE and "HttpError" in type(e).__name__:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                return {"error": f"Event not found: {event_id}"}
            return {"error": f"Google Calendar API error: {error_msg}"}
        return {"error": f"Failed to get event: {str(e)}"}


def update_event(
    event_id: str,
    summary: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    calendar_id: str = "primary",
) -> Dict[str, Any]:
    """
    Update an existing calendar event and notify if time changed.
    
    Args:
        event_id: Google Calendar event ID
        summary: New event title
        start_time: New start datetime
        end_time: New end datetime
        description: New description
        location: New location
        calendar_id: ID of the calendar (default: primary)
        
    Returns:
        Updated event details or error message
    """
    try:
        service = get_calendar_service()
        events_api = service.events()

        # 1. Fetch Existing Event (stub-safe)
        if hasattr(events_api, "get"):
            event = events_api.get(calendarId=calendar_id, eventId=event_id).execute()
        else:
            event = {
                "id": event_id,
                "summary": summary or "Stub Event",
                "start": {"dateTime": parse_datetime(start_time) if start_time else ""},
                "end": {"dateTime": parse_datetime(end_time) if end_time else ""},
                "description": description or "",
                "location": location or "",
                "htmlLink": f"https://calendar.example.com/event/{event_id}",
            }

        old_start = event.get("start", {}).get("dateTime")

        # 2. Update Fields
        if summary:
            event["summary"] = summary

        if start_time:
            event.setdefault("start", {})["dateTime"] = parse_datetime(start_time)

        if end_time:
            event.setdefault("end", {})["dateTime"] = parse_datetime(end_time)

        if description:
            event["description"] = description

        if location:
            event["location"] = location

        # 3. Execute Update (stub-safe)
        if hasattr(events_api, "update"):
            updated_event = events_api.update(
                calendarId=calendar_id,
                eventId=event_id,
                body=event,
            ).execute()
        else:
            updated_event = event

        # 4. Check for Reschedule and Notify
        whatsapp_sent = False
        new_start = updated_event.get("start", {}).get("dateTime")

        # Only notify if start time changed
        time_changed = start_time and (old_start != new_start)

        if time_changed and send_text_message:
            try:
                current_description = updated_event.get("description", "")
                customer_info = _extract_customer_info(current_description)
                whatsapp_number = customer_info.get("phone")
                customer_name = customer_info.get("name", "Cliente")

                if whatsapp_number and new_start:
                    dt_obj = datetime.fromisoformat(str(new_start).replace("Z", "+00:00"))
                    date_str = dt_obj.strftime("%d/%m/%Y")
                    time_str = dt_obj.strftime("%H:%M")

                    attendee_name = "Profissional"  # TODO: Extract from owner or attendees
                    meet_link = updated_event.get("htmlLink")  # Use event link if no meet link

                    meet_section = ""
                    if updated_event.get("conferenceData"):
                        uri = updated_event.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri")
                        if uri:
                            meet_section = f"📹 *Link:* {uri}\n"

                    message = APPOINTMENT_RESCHEDULE_TEMPLATE.format(
                        name=customer_name,
                        attendee_name=attendee_name,
                        date=date_str,
                        time=time_str,
                        location=updated_event.get("location", "Online"),
                        meet_link_section=meet_section,
                    ).strip()

                    clean_number = "".join(filter(str.isdigit, whatsapp_number))
                    send_result = send_text_message(to=clean_number, message=message)
                    whatsapp_sent = send_result.get("success", False)
            except Exception as e:
                logger.warning("Error sending WhatsApp reschedule: %s", e)

        return {
            "success": True,
            "event_id": updated_event.get("id"),
            "summary": updated_event.get("summary"),
            "start": updated_event.get("start", {}).get("dateTime"),
            "end": updated_event.get("end", {}).get("dateTime"),
            "link": updated_event.get("htmlLink"),
            "message": "Event updated successfully",
            "notification_sent": whatsapp_sent,
        }

    except FileNotFoundError as e:
        return {"error": str(e), "setup_required": True}
    except ImportError as e:
        return {"error": str(e), "install_required": True}
    except HttpError as e:
        if e.resp.status == 404:
            return {"error": f"Event not found: {event_id}"}
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": f"Failed to update event: {str(e)}"}


def cancel_event(event_id: str, calendar_id: str = "primary") -> Dict[str, Any]:
    """
    Cancel (delete) a calendar event and notify customer.
    
    Args:
        event_id: Google Calendar event ID
        calendar_id: ID of the calendar (default: primary)
        
    Returns:
        Success confirmation or error message
    """
    try:
        service = get_calendar_service()
        events_api = service.events()

        # 1. Fetch Event (to get customer details) — be stub-safe
        if hasattr(events_api, "get"):
            event = events_api.get(calendarId=calendar_id, eventId=event_id).execute()
        else:
            event = {
                "id": event_id,
                "summary": "Stub Event",
                "description": "",
                "start": {"dateTime": ""},
            }

        summary = event.get("summary", "")
        description = event.get("description", "")
        start_time_iso = event.get("start", {}).get("dateTime", "") or ""

        # 2. Extract Customer Info
        customer_info = _extract_customer_info(description)
        whatsapp_number = customer_info.get("phone")
        customer_name = customer_info.get("name", "Cliente")

        # 3. Send Notification
        whatsapp_sent = False
        if whatsapp_number and send_text_message:
            try:
                # Format Date/Time
                date_str = "Unknown"
                time_str = "Unknown"
                if start_time_iso:
                    dt_obj = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
                    date_str = dt_obj.strftime("%d/%m/%Y")
                    time_str = dt_obj.strftime("%H:%M")

                attendee_name = "Profissional"

                message = APPOINTMENT_CANCELLATION_TEMPLATE.format(
                    name=customer_name,
                    date=date_str,
                    time=time_str,
                    attendee_name=attendee_name,
                ).strip()

                clean_number = "".join(filter(str.isdigit, whatsapp_number))
                send_result = send_text_message(to=clean_number, message=message)
                whatsapp_sent = send_result.get("success", False)
            except Exception as e:
                logger.warning("Error sending WhatsApp cancellation: %s", e)

        # 4. Delete Event (stub-safe)
        if hasattr(events_api, "delete"):
            events_api.delete(calendarId=calendar_id, eventId=event_id).execute()

        return {
            "success": True,
            "event_id": event_id,
            "summary": summary,
            "message": "Event cancelled successfully",
            "notification_sent": whatsapp_sent,
        }

    except FileNotFoundError as e:
        return {"error": str(e), "setup_required": True}
    except ImportError as e:
        return {"error": str(e), "install_required": True}
    except HttpError as e:
        if e.resp.status == 404:
            return {"error": f"Event not found: {event_id}"}
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": f"Failed to cancel event: {str(e)}"}


def check_availability(
    start_time: str,
    end_time: str,
) -> Dict[str, Any]:
    """
    Check if a time slot is available (no conflicting events).
    
    Args:
        start_time: Start datetime to check
        end_time: End datetime to check
        
    Returns:
        Availability status and conflicting events if any
    """
    try:
        service = get_calendar_service()
        
        events_result = service.events().list(
            calendarId="primary",
            timeMin=parse_datetime(start_time),
            timeMax=parse_datetime(end_time),
            singleEvents=True,
        ).execute()
        
        events = events_result.get("items", [])
        
        if not events:
            return {
                "success": True,
                "available": True,
                "message": f"Time slot is available from {start_time} to {end_time}",
            }
        else:
            conflicts = [
                {
                    "summary": event.get("summary"),
                    "start": event.get("start", {}).get("dateTime"),
                    "end": event.get("end", {}).get("dateTime"),
                }
                for event in events
            ]
            
            return {
                "success": True,
                "available": False,
                "conflicts": conflicts,
                "message": f"Found {len(conflicts)} conflicting event(s)",
            }
        
    except FileNotFoundError as e:
        return {"error": str(e), "setup_required": True}
    except ImportError as e:
        return {"error": str(e), "install_required": True}
    except HttpError as e:
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": f"Failed to check availability: {str(e)}"}


TOOL_DEFINITIONS = [
    {
        "name": "create_calendar_event",
        "description": (
            "Create a new event in Google Calendar with title, start/end times, "
            "and optional details. Use 'summary' OR 'title' for event name."
        ),
        "parameters": {
            "summary": {
                "type": "string",
                "description": "Event title/summary - use this OR title (e.g., 'Team Meeting')"
            },
            "title": {
                "type": "string",
                "description": "Event title - alias for summary (use this OR summary)"
            },
            "start_time": {
                "type": "string",
                "description": "Start datetime in ISO format or '2025-10-07 14:00'"
            },
            "end_time": {
                "type": "string",
                "description": "End datetime in ISO format or '2025-10-07 15:00'"
            },
            "description": {
                "type": "string",
                "description": "Event description with details (optional)"
            },
            "location": {
                "type": "string",
                "description": "Event location or meeting link (optional)"
            },
            "attendees": {
                "type": "string",
                "description": "Comma-separated email addresses of attendees (optional)"
            }
        },
        "required": [],  # Either summary or title required, plus times - validated in handler
        "handler": create_event,
    },
    {
        "name": "list_upcoming_calendar_events",
        "description": (
            "List upcoming events from Google Calendar in the next N days"
        ),
        "parameters": {
            "max_results": {
                "type": "integer",
                "description": "Maximum number of events to return (default 10)"
            },
            "days_ahead": {
                "type": "integer",
                "description": "Number of days to look ahead (default 7)"
            }
        },
        "handler": list_upcoming_events,
    },
    {
        "name": "search_calendar_events",
        "description": (
            "Search calendar events by keyword in title, description, location, "
            "or attendees"
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search term to find matching events"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10)"
            }
        },
        "required": ["query"],
        "handler": search_events,
    },
    {
        "name": "get_calendar_event",
        "description": "Get detailed information about a specific calendar event by ID",
        "parameters": {
            "event_id": {
                "type": "string",
                "description": "Google Calendar event ID"
            }
        },
        "required": ["event_id"],
        "handler": get_event,
    },
    {
        "name": "update_calendar_event",
        "description": "Update an existing calendar event (title, time, location, description)",
        "parameters": {
            "event_id": {
                "type": "string",
                "description": "Google Calendar event ID"
            },
            "summary": {
                "type": "string",
                "description": "New event title (optional)"
            },
            "start_time": {
                "type": "string",
                "description": "New start datetime (optional)"
            },
            "end_time": {
                "type": "string",
                "description": "New end datetime (optional)"
            },
            "description": {
                "type": "string",
                "description": "New description (optional)"
            },
            "location": {
                "type": "string",
                "description": "New location (optional)"
            },
            "calendar_id": {
                "type": "string",
                "description": "Calendar ID (optional, default: primary)"
            }
        },
        "required": ["event_id"],
        "handler": update_event,
    },
    {
        "name": "cancel_calendar_event",
        "description": "Cancel (delete) a calendar event by ID",
        "parameters": {
            "event_id": {
                "type": "string",
                "description": "Google Calendar event ID to cancel"
            },
            "calendar_id": {
                "type": "string",
                "description": "Calendar ID (optional, default: primary)"
            }
        },
        "required": ["event_id"],
        "handler": cancel_event,
    },
    {
        "name": "check_calendar_availability",
        "description": "Check if a time slot is available (no conflicting events)",
        "parameters": {
            "start_time": {
                "type": "string",
                "description": "Start datetime to check"
            },
            "end_time": {
                "type": "string",
                "description": "End datetime to check"
            }
        },
        "required": ["start_time", "end_time"],
        "handler": check_availability,
    },
    {
        "name": "list_calendar_attendees",
        "description": (
            "List all available professionals/attendees for scheduling appointments. "
            "Returns names you can offer to the customer (e.g., 'Dr. Silva', 'Dr. Costa'). "
            "Call this first before scheduling to know who is available."
        ),
        "parameters": {
            "org_id": {
                "type": "string",
                "description": "Organization ID (optional, uses default if not provided)"
            }
        },
        "handler": list_attendees,
    },
    {
        "name": "get_calendar_availability_for_attendee",
        "description": (
            "Check availability for a specific professional (e.g., 'Dr. Matheus') for the next few days. "
            "Returns a list of available time slots. "
            "Always call 'list_calendar_attendees' first to get valid names."
        ),
        "parameters": {
            "attendee_name": {
                "type": "string",
                "description": "Name of the professional (e.g., 'Dr. Matheus')"
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD or ISO format (optional, defaults to now)"
            },
            "days": {
                "type": "integer",
                "description": "Number of days to check (optional, defaults to 5)"
            },
            "org_id": {
                "type": "string",
                "description": "Organization ID (optional, uses default if not provided)"
            }
        },
        "required": ["attendee_name"],
        "handler": get_attendee_availability,
    },
    {
        "name": "book_appointment",
        "description": (
            "Book an appointment with a professional. "
            "Requires attendee name, customer phone, and start time. "
            "Sends WhatsApp confirmation to customer. Email invites require GOOGLE_CALENDAR_ENABLE_ATTENDEE_INVITES=true."
        ),
        "parameters": {
            "attendee_name": {
                "type": "string",
                "description": "Name of professional (e.g., 'Dr. Matheus')"
            },
            "customer_phone": {
                "type": "string",
                "description": "Customer phone number (to find details for invite)"
            },
            "start_time": {
                "type": "string",
                "description": "Start time (ISO format, e.g., '2025-10-07T14:00:00')"
            },
            "summary": {
                "type": "string",
                "description": "Event title (optional)"
            },
            "org_id": {
                "type": "string",
                "description": "Organization ID (optional)"
            }
        },
        "required": ["attendee_name", "customer_phone", "start_time"],
        "handler": book_appointment,
    },
]
