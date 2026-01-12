import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

try:  # Optional dependency: real Supabase client
    from supabase import Client, create_client
except Exception:  # pragma: no cover - allow stub-only usage
    Client = object  # type: ignore[assignment]

    def create_client(url: str, key: str):  # type: ignore[misc]
        raise ImportError("supabase client not installed")

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)

# Optional runtime cache; keep loose typing because Client may be stub
_supabase_client: Any = None



class _StubResponse:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data


class _StubTable:
    def __init__(self, name: str, store: Dict[str, List[Dict[str, Any]]]):
        self.name = name
        self.store = store
        self._filters: List[Any] = []
        self._or_filters: List[List[Any]] = []
        self._limit: Optional[int] = None
        self._order: Optional[tuple[str, bool]] = None
        self._pending_update: Optional[Dict[str, Any]] = None
        self._pending_delete: bool = False
        self._insert_data: Optional[List[Dict[str, Any]]] = None

    # Query builders -----------------------------------------------------
    def select(self, *args, **kwargs):  # type: ignore[override]
        return self

    def eq(self, field: str, value: Any):
        self._filters.append((field, value))
        return self

    def or_(self, expr: str):
        parts = [p for p in expr.split(",") if ".eq." in p]
        or_list = []
        for p in parts:
            tokens = p.split(".eq.")
            if len(tokens) == 2:
                field, val = tokens
                field = field.split(".")[-1]
                or_list.append((field, val))
        if or_list:
            self._or_filters.append(or_list)
        return self

    def ilike(self, field: str, pattern: str):
        needle = pattern.replace("%", "").lower()
        self._filters.append((field, ("ilike", needle)))
        return self

    def order(self, field: str, desc: bool = False):  # type: ignore[override]
        self._order = (field, desc)
        return self

    def limit(self, n: int):
        self._limit = n
        return self

    def update(self, data: Dict[str, Any]):
        self._pending_update = data
        return self

    def insert(self, data: Any):
        if isinstance(data, dict):
            self._insert_data = [data]
        elif isinstance(data, list):
            self._insert_data = data
        else:
            self._insert_data = []
        return self

    def delete(self):
        self._pending_delete = True
        return self

    # Execution ----------------------------------------------------------
    def execute(self):
        records = deepcopy(self.store.get(self.name, []))

        def match_filters(row: Dict[str, Any]) -> bool:
            for field, val in self._filters:
                if isinstance(val, tuple) and val[0] == "ilike":
                    needle = val[1]
                    if needle not in str(row.get(field, "")).lower():
                        return False
                else:
                    if row.get(field) != val:
                        return False
            if self._or_filters:
                ok = False
                for or_list in self._or_filters:
                    for field, val in or_list:
                        if row.get(field) == val:
                            ok = True
                            break
                    if ok:
                        break
                if not ok:
                    return False
            return True

        matched = [r for r in records if match_filters(r)]

        if self._pending_delete:
            remaining = [r for r in self.store.get(self.name, []) if not match_filters(r)]
            self.store[self.name] = remaining
            return _StubResponse(matched)

        if self._pending_update is not None:
            updated = []
            for r in self.store.get(self.name, []):
                if match_filters(r):
                    r.update(self._pending_update)
                    updated.append(deepcopy(r))
            return _StubResponse(updated)

        if self._insert_data is not None:
            self.store.setdefault(self.name, []).extend(deepcopy(self._insert_data))
            return _StubResponse(self._insert_data)

        if self._order:
            field, desc = self._order
            matched.sort(key=lambda x: (x.get(field) is None, x.get(field)), reverse=desc)

        if self._limit is not None:
            matched = matched[: self._limit]

        return _StubResponse(matched)


class _StubSupabaseClient:
    def __init__(self) -> None:
        self.store: Dict[str, List[Dict[str, Any]]] = {}
        self._seed()

    def _seed(self) -> None:
        primary_customer_id = "00000000-0000-0000-0000-000000000001"
        primary_phone = "+5511999990001"
        org_id = "00000000-0000-0000-0000-000000000000"

        self.store["organizations"] = [{"id": org_id, "name": "Stub Org"}]

        self.store["customers"] = [
            {
                "id": primary_customer_id,
                "org_id": org_id,
                "phone": primary_phone,
                "name": "Maria Gomez",
                "company": "Acme Health",
                "status": "active",
                "updated_at": "2025-01-01T10:00:00Z",
            }
        ]

        self.store["customer_facts"] = [
            {
                "id": "fact-1",
                "org_id": org_id,
                "customer_id": primary_customer_id,
                "fact_type": "preference",
                "content": "Prefers morning calls",
                "confidence": 0.9,
                "learned_from_call": None,
                "created_at": "2025-01-02T12:00:00Z",
            }
        ]

        self.store["calls"] = [
            {
                "id": "call-1",
                "customer_id": primary_customer_id,
                "call_date": "2025-01-05T15:30:00Z",
                "duration_seconds": 300,
                "call_type": "outbound_followup",
                "outcome": "interested",
                "summary": "Discussed product demo",
                "sentiment": "positive",
                "next_action": "Send proposal",
                "next_action_date": "2025-01-07",
                "transcript": "",
            }
        ]

        self.store["action_items"] = [
            {
                "id": "task-1",
                "org_id": org_id,
                "customer_id": primary_customer_id,
                "created_in_call": "call-1",
                "task_type": "follow_up",
                "description": "Send proposal",
                "due_date": "2025-01-08",
                "status": "pending",
                "priority": "high",
                "assigned_to": "agent_007",
                "completed_at": None,
            }
        ]

        self.store["knowledge_base"] = [
            {
                "id": "kb-1",
                "org_id": org_id,
                "category": "general",
                "title": "Pricing overview",
                "content": "Base plan starts at $99",
                "created_at": "2025-01-03T09:00:00Z",
            }
        ]

        self.store["call_scripts"] = [
            {
                "id": "script-1",
                "name": "Cold call v1",
                "call_type": "outbound_cold_call",
                "industry": None,
                "greeting": "Olá, aqui é da Acme!",
                "qualification_questions": ["Você já usa solução similar?"],
                "objection_handlers": {"price": "Entendo o ponto. Posso mostrar o valor."},
                "call_goals": ["Agendar demo"],
                "closing_statements": "Obrigado pelo tempo!",
                "active": True,
                "created_at": "2025-01-04T10:00:00Z",
            }
        ]

        # Calendar connections (for harness availability/booking tests)
        working_hours = {
            "monday": {"start": "09:00", "end": "17:00"},
            "tuesday": {"start": "09:00", "end": "17:00"},
            "wednesday": {"start": "09:00", "end": "17:00"},
            "thursday": {"start": "09:00", "end": "17:00"},
            "friday": {"start": "09:00", "end": "17:00"},
        }
        self.store["org_calendar_connections"] = [
            {
                "id": "cal-1",
                "org_id": org_id,
                "attendee_name": "Dr. Smith",
                "attendee_email": "dr.smith@example.com",
                "calendar_id": "primary",
                "calendar_type": "doctor",
                "is_default": True,
                "working_hours": working_hours,
            },
            {
                "id": "cal-2",
                "org_id": org_id,
                "attendee_name": "Dr. Costa",
                "attendee_email": "dr.costa@example.com",
                "calendar_id": "calendar-dr-costa",
                "calendar_type": "doctor",
                "is_default": False,
                "working_hours": working_hours,
            },
        ]

        # Extra seed data for harness cases (ids referenced in tests)
        self.store["customer_facts"].append(
            {
                "id": "4eba7430-ca34-4274-a116-9e7a6385e65b",
                "org_id": org_id,
                "customer_id": primary_customer_id,
                "fact_type": "preference",
                "content": "Existing fact for deletion",
                "confidence": 0.8,
                "learned_from_call": None,
                "created_at": "2025-01-03T08:00:00Z",
            }
        )

        self.store["calls"].append(
            {
                "id": "4da187ed-db61-48df-b4e1-5ea49048b132",
                "customer_id": primary_customer_id,
                "call_date": "2025-01-06T10:00:00Z",
                "duration_seconds": 200,
                "call_type": "outbound_followup",
                "outcome": "pending",
                "summary": "Awaiting transcript update",
                "sentiment": "neutral",
                "next_action": None,
                "next_action_date": None,
                "transcript": "",
            }
        )

        self.store["action_items"].extend(
            [
                {
                    "id": "e0e6beb0-a0a1-4103-bcb6-2cec3543bd90",
                    "org_id": org_id,
                    "customer_id": primary_customer_id,
                    "created_in_call": "call-1",
                    "task_type": "follow_up",
                    "description": "Existing task for completion",
                    "due_date": "2025-01-09",
                    "status": "pending",
                    "priority": "medium",
                    "assigned_to": "agent_007",
                    "completed_at": None,
                },
                {
                    "id": "8f9c6cd6-9aec-43b5-935f-fb82344be3d8",
                    "org_id": org_id,
                    "customer_id": primary_customer_id,
                    "created_in_call": "call-1",
                    "task_type": "follow_up",
                    "description": "Existing task for update",
                    "due_date": "2025-01-10",
                    "status": "pending",
                    "priority": "high",
                    "assigned_to": "agent_008",
                    "completed_at": None,
                },
                {
                    "id": "40dae190-241b-4243-8c88-ffef1556c341",
                    "org_id": org_id,
                    "customer_id": primary_customer_id,
                    "created_in_call": "call-1",
                    "task_type": "follow_up",
                    "description": "Existing task for deletion",
                    "due_date": "2025-01-11",
                    "status": "pending",
                    "priority": "low",
                    "assigned_to": "agent_009",
                    "completed_at": None,
                },
            ]
        )

        self.store["knowledge_base"].extend(
            [
                {
                    "id": "88cb9ded-2108-4a12-8444-308b5e3d314e",
                    "org_id": org_id,
                    "category": "product",
                    "title": "Legacy entry for update",
                    "content": "Old content",
                    "created_at": "2025-01-04T09:00:00Z",
                },
                {
                    "id": "0dbd2be6-419f-49de-a3d3-db02eb04c0e8",
                    "org_id": org_id,
                    "category": "product",
                    "title": "Legacy entry for deletion",
                    "content": "To be deleted",
                    "created_at": "2025-01-04T09:30:00Z",
                },
            ]
        )

    def table(self, name: str) -> _StubTable:
        return _StubTable(name, self.store)


def _should_use_stub(url: str, key: str) -> bool:
    if not url or not key:
        return True
    return os.getenv("SUPABASE_USE_STUB", "false").lower() == "true"


def get_supabase_client() -> Client:
    """
    Get or initialize the Supabase client. Falls back to an in-memory stub when
    Supabase credentials are missing (useful for local tests/harness).
    """
    global _supabase_client

    url = SETTINGS.supabase_url
    key = SETTINGS.supabase_service_role_key

    if _should_use_stub(url, key):
        logger.info("Using stub Supabase client (in-memory)")
        _supabase_client = _StubSupabaseClient()  # type: ignore[assignment]
        return _supabase_client

    if _supabase_client is None:
        try:
            _supabase_client = create_client(url, key)
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    return _supabase_client
