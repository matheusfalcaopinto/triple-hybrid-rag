# MVP/PRD — Control-Plane Backend Layer (Frontend Power + Agent Runtime Management)

Date: 2026-01-10  
Scope: **New control-plane backend** that powers the admin frontend and manages agent runtime instances (Docker).  
Related: `frontend/CORRELATION_REPORT.md`

---

## 1) Executive summary

We will build a **multi-tenant SaaS control plane** (“backend layer”) that:

- Powers the admin frontend for **Establishments**, **Agents**, **Calls**, **Leads**, **Reports**, **Settings/Integrations**, and **Auth/Roles**.
- Manages **agent runtime instances** as Docker containers on a single host (MVP).
- Orchestrates calling via **Twilio** and **WhatsApp Calling**, and supports operational integrations (**WhatsApp messaging**, **Google Calendar**, **Email**) in a secure multi-tenant manner.
- Uses **polling** for frontend live updates (MVP).
- Stores call artifacts: **transcripts**, **summaries**, **recordings**, **tool-call logs**.

Core design choice:
- **Dedicated runtime per establishment** (MVP) with **routing by phone number** (Twilio number → agent).
- **Telephony policy per establishment**, while **agent “brain” configuration is versioned**.
- **Hybrid secret strategy**: only telephony secrets may live on runtimes; all business integrations are backend-managed (tool proxy).

---

## 2) Goals (MVP)

### Product goals
1. Multi-tenant admin UI works end-to-end for:
   - Establishments (tenant) management
   - Agent creation/config/versioning + rollback
   - Calling: active calls + call history + call details
   - Leads: import + automated campaign dialer
   - Reports: basic performance analytics
   - Integrations: connect/manage supported integrations

2. Dedicated Docker runtime per establishment:
   - Create, start, stop, restart
   - Health checks and safe draining

3. Call artifacts reliably stored:
   - Transcript (per call)
   - Summary (post-call)
   - Recording link/object
   - Tool call log

4. MVP realtime:
   - Polling endpoints for dashboard/calls views

### Engineering goals
- Clear contracts between:
  - Frontend ↔ control plane (REST + polling)
  - Control plane ↔ agent runtime (HTTP/webhooks + event ingestion)
  - Agent runtime ↔ integrations (mostly via backend tool proxy)

- Idempotent ingestion for call events and artifacts.
- Basic operator role separation (admin/operator/viewer).

---

## 3) Non-goals (MVP)

- True “human takeover” audio intervention (barge-in) — out of scope.
- Pooled runtimes shared across tenants — out of scope.
- Kubernetes orchestration — out of scope.
- Real-time UI via WS/SSE — out of scope (polling only).
- Fine-grained RBAC with custom policies — out of scope.

---

## 4) Personas & permissions

### Personas
- **Admin**: manages establishments, agents, integrations, routing numbers, campaigns.
- **Operator**: views calls, initiates test calls/campaign calls, monitors outcomes.
- **Viewer**: read-only insight (dashboard/reports/calls).

### MVP roles
- `admin`, `operator`, `viewer`

---

## 5) System architecture

### High-level data flow
1. Frontend calls control-plane REST APIs.
2. Control plane stores all state in DB (system of record).
3. Each establishment has one agent runtime container.
4. Incoming calls route by **Twilio number**:
   - Twilio hits control plane webhook
   - Control plane selects establishment/agent
   - Control plane responds with TwiML pointing the media stream to the selected runtime
5. Runtime runs the call pipeline and emits events/artifacts:
   - Minimal: runtime POSTs events to control-plane ingestion endpoints
   - Control plane persists transcripts/tool logs, generates summary (async)

### Runtime management (Docker)
MVP implements runtime lifecycle via local Docker engine:
- create container with environment and image
- start/stop/restart
- health check by polling `GET /readyz`
- drain/resume via runtime control endpoints

---

## 6) Key product decisions (locked)

- Multi-tenant SaaS: **Yes**
- Users: **Multiple users per org** with basic roles
- Establishments: **Keep full module** (CRUD + integrations ownership + routing numbers)
- Billing: **Stub only** (UI visible, no real billing logic)
- Leads: **Automated campaign dialer** (MVP)
- Reports: **Agent performance metrics** (MVP)
- Agents per establishment: **Multiple**
- Versioning: **Versioning + rollback**
- Orchestration: **Docker**, backend can start/stop/restart
- Runtime model: **1 runtime per establishment**
- Deployment trigger: **backend runs docker compose / Docker API locally**
- Call routing: **by phone number**
- Config granularity: **telephony policy per establishment; agent version = brain only**
- Calls MVP: **live transcript + sentiment** (sentiment derived async in backend)
- Handoff: **Twilio transfer to fixed number per establishment**
- Artifacts: transcript, summary, recording, tool-log stored
- Integrations/secrets: **Hybrid (D)**
- UI live updates: **Polling**

---

## 7) API surface (control-plane)

This is the MVP REST contract the frontend will use.

### 7.1 Auth & users
- `POST /auth/login`
- `POST /auth/logout`
- `POST /auth/forgot-password`
- `POST /auth/reset-password`
- `GET /me`

- `GET /establishments/{establishmentId}/users`
- `POST /establishments/{establishmentId}/users` (invite)
- `PATCH /users/{userId}` (role)

### 7.2 Establishments
- `GET /establishments`
- `POST /establishments`
- `GET /establishments/{id}`
- `PATCH /establishments/{id}`

Routing/telephony policy:
- `GET /establishments/{id}/telephony`
- `PATCH /establishments/{id}/telephony`
  - business hours
  - max call duration
  - blacklist
  - transfer/handoff number

### 7.3 Agents (resource + versioning)
- `GET /establishments/{id}/agents`
- `POST /establishments/{id}/agents`
- `GET /agents/{agentId}`
- `PATCH /agents/{agentId}`

Versions:
- `GET /agents/{agentId}/versions`
- `POST /agents/{agentId}/versions` (create new draft)
- `POST /agents/{agentId}/versions/{versionId}/publish`
- `POST /agents/{agentId}/versions/{versionId}/rollback`

Version payload (“brain only”):
- prompt/system prompt template
- model params (temperature, max tokens)
- enabled tools (flags)
- policies that affect LLM/tooling (not telephony)

### 7.4 Runtime instances (1 per establishment)
- `GET /establishments/{id}/runtime`
- `POST /establishments/{id}/runtime` (provision/create)
- `POST /runtimes/{runtimeId}/start`
- `POST /runtimes/{runtimeId}/stop`
- `POST /runtimes/{runtimeId}/restart`
- `POST /runtimes/{runtimeId}/drain`
- `POST /runtimes/{runtimeId}/resume`
- `GET /runtimes/{runtimeId}/status`

### 7.5 Calls
Inbound webhooks:
- `POST /webhooks/twilio/incoming-call` → returns TwiML pointing to the correct runtime
- `POST /webhooks/twilio/call-status` (fan-in from runtime or directly from Twilio)
- `POST /webhooks/twilio/amd-callback`

Outbound initiation:
- `POST /calls/outbound` (operator action or campaign engine)

Query for UI:
- `GET /calls/active` (polling)
- `GET /calls?filters...`
- `GET /calls/{callId}`
- `GET /calls/{callId}/transcript`
- `GET /calls/{callId}/recording`
- `GET /calls/{callId}/tool-log`

Handoff:
- `POST /calls/{callId}/handoff` (transfer to establishment’s configured number)

### 7.6 Leads & campaigns
- `POST /leads/import` (CSV)
- `GET /leads?filters...`
- `PATCH /leads/{leadId}`

Campaigns:
- `POST /campaigns` (create)
- `POST /campaigns/{campaignId}/start`
- `POST /campaigns/{campaignId}/pause`
- `GET /campaigns/{campaignId}`

### 7.7 Reports
- `GET /dashboard/kpis`
- `GET /reports/agent-performance?range=...`
- `GET /reports/calls?range=...`

### 7.8 Integrations
Establishment-owned connections:
- `GET /establishments/{id}/integrations`
- `POST /establishments/{id}/integrations` (connect)
- `PATCH /integrations/{integrationId}` (enable/disable)

Integration operations (tool proxy surfaces) — internal to agent/backend:
- `POST /tool-proxy/email/send`
- `POST /tool-proxy/calendar/create_event`
- `POST /tool-proxy/whatsapp/send_message`
- `POST /tool-proxy/crm/*`

---

## 8) Data model (MVP)

Suggested DB tables (names illustrative).

### Tenancy & auth
- `establishments`
- `users`
- `establishment_users` (role)

### Agents
- `agents` (belongs to establishment)
- `agent_versions` (belongs to agent)
  - status: `draft|published|archived`
  - prompt, model params, enabled tools, metadata
- `agent_deployments` (which version is active)

### Runtimes
- `runtimes`
  - establishment_id
  - docker container name/id
  - image tag
  - ports
  - last_status
  - last_readyz

### Telephony
- `establishment_telephony_policies`
- `phone_numbers`
  - provider
  - number
  - establishment_id
  - mapped_agent_id (routing by phone number)

### Calls + artifacts
- `calls`
- `call_events` (timeline)
- `call_transcript_segments`
- `call_summaries`
- `call_recordings`
- `call_tool_logs`

### Leads & campaigns
- `leads`
- `campaigns`
- `campaign_enrollments`
- `campaign_runs`

### Integrations
- `integrations`
- `integration_connections` (per establishment)
- `integration_tokens` (encrypted)

---

## 9) Agent runtime compatibility requirements

The runtime (this repo) must support the following behaviors so the control plane can power the UI:

### 9.1 Event ingestion
Runtime must emit structured events to the control plane:
- `call_started`, `call_connected`, `call_ended`
- `transcript_segment` (speaker, text, timestamps)
- `tool_called`, `tool_result`
- `recording_available` (location)

MVP approach:
- runtime `POST /ingest/call-events` on the control plane

### 9.2 Artifact generation
- Transcript segments stored continuously
- Recording stored (or uploaded) and linked
- Summary generated asynchronously (backend) from transcript

### 9.3 Telephony hooks
Since routing is by phone number:
- Control plane must be able to generate TwiML that targets the correct runtime endpoint.

### 9.4 Secrets strategy (hybrid)
- Telephony secrets may exist on runtime env (Twilio keys if needed locally)
- Business integrations are executed via **tool proxy** endpoints on the control plane

---

## 10) Sentiment (MVP definition)

Locked choice: **Sentiment derived asynchronously in backend from transcript**.

Implementation:
- Background job reads transcript segments
- Produces sentiment score/label per call and/or per time window
- Exposed on `GET /calls/{id}` and aggregates for reports

---

## 11) Handoff (MVP definition)

Locked choice: **Twilio transfer to fixed number per establishment**.

Implementation idea:
- Establishment telephony policy stores `handoff_phone_number`
- `POST /calls/{id}/handoff` triggers provider action

Notes:
- True shared ring groups/queues can be a later enhancement.

---

## 12) Milestones & delivery plan

### Milestone M0 — Skeleton backend + auth + tenancy
- Establishments CRUD
- Users + roles
- Basic integration connection data model (no OAuth yet)

### Milestone M1 — Runtime management (Docker)
- Provision/start/stop/restart
- Health aggregation
- Drain/resume proxy

### Milestone M2 — Calling orchestration + call history
- Twilio inbound webhook routing by phone number
- Outbound call endpoint
- Calls table + lifecycle events
- Active calls polling

### Milestone M3 — Transcripts + artifacts
- Ingestion endpoint + transcript storage
- Recording linking
- Summary job
- Tool-call log ingestion

### Milestone M4 — Agents + versioning + deployment
- Agents CRUD
- Agent versioning + publish + rollback
- Deployment mapping (which version active)

### Milestone M5 — Leads + campaign dialer + reports
- Lead import
- Campaign scheduler
- Agent performance reports

---

## 13) Testing plan

### Backend tests
- Unit tests: routing rules, role checks, schema validation
- Contract tests: control-plane ↔ runtime endpoints
- Idempotency tests: event ingestion replays

### End-to-end tests (automation)
1. Establishment created → runtime provisioned → ready status OK
2. Twilio inbound webhook → TwiML points to correct runtime
3. Outbound call initiated → call status updates → call appears in history
4. Runtime sends transcript segments → UI polling sees transcript grow
5. After call: summary + recording links available

---

## 14) Open items (confirm later)

- Provider for auth (Supabase Auth vs custom vs external IdP)
- Storage target for recordings (S3 vs Supabase storage)
- How the backend talks to Docker (CLI vs docker socket API)
- Campaign pacing/compliance (time windows, retries, opt-outs)

---

## Appendix A — Canonical API schemas (MVP)

This appendix defines concrete JSON contracts for the control-plane backend.

Notes:

- Unless otherwise stated, timestamps are ISO8601 (`YYYY-MM-DDTHH:mm:ssZ`).
- IDs are opaque strings.
- All endpoints require auth except webhook endpoints.
- Errors use a consistent shape:
   - `{"error": {"code": "...", "message": "...", "details": {...}}}`

### A.1 Auth & users

#### `POST /auth/login`

Request:
```json
{
   "email": "admin@example.com",
   "password": "********"
}
```

Response (200):
```json
{
   "access_token": "...",
   "token_type": "bearer",
   "expires_in": 3600,
   "user": {
      "id": "usr_...",
      "email": "admin@example.com",
      "display_name": "Admin",
      "default_establishment_id": "est_..."
   }
}
```

#### `GET /me`

Response (200):
```json
{
   "user": {
      "id": "usr_...",
      "email": "admin@example.com",
      "display_name": "Admin"
   },
   "memberships": [
      {
         "establishment_id": "est_...",
         "role": "admin"
      }
   ]
}
```

#### `POST /establishments/{establishmentId}/users` (invite)

Request:
```json
{
   "email": "operator@example.com",
   "role": "operator"
}
```

Response (201):
```json
{
   "invitation_id": "inv_...",
   "status": "sent"
}
```

---

### A.2 Establishments

#### `POST /establishments`

Request:
```json
{
   "name": "Helios Energy HQ",
   "timezone": "America/Sao_Paulo",
   "locale": "pt-BR"
}
```

Response (201):
```json
{
   "id": "est_...",
   "name": "Helios Energy HQ",
   "timezone": "America/Sao_Paulo",
   "locale": "pt-BR",
   "created_at": "2026-01-10T00:00:00Z"
}
```

#### `GET /establishments`

Response (200):
```json
{
   "items": [
      {
         "id": "est_...",
         "name": "Helios Energy HQ",
         "timezone": "America/Sao_Paulo",
         "locale": "pt-BR"
      }
   ]
}
```

#### `GET /establishments/{id}/telephony`

Response (200):
```json
{
   "establishment_id": "est_...",
   "inbound_numbers": [
      {
         "provider": "twilio",
         "e164": "+15553012211",
         "routing_agent_id": "agt_..."
      }
   ],
   "handoff_e164": "+15551234567",
   "max_call_duration_seconds": 900,
   "business_hours": [
      {
         "days": ["mon", "tue", "wed", "thu", "fri"],
         "start": "08:00",
         "end": "22:00"
      }
   ],
   "blacklist_e164": ["+15550001111"]
}
```

#### `PATCH /establishments/{id}/telephony`

Request (partial allowed):
```json
{
   "handoff_e164": "+15551234567",
   "max_call_duration_seconds": 900
}
```

Response (200): same shape as GET.

---

### A.3 Agents + versions

#### `POST /establishments/{id}/agents`

Request:
```json
{
   "name": "Nova GPT",
   "description": "Concierge and support agent"
}
```

Response (201):
```json
{
   "id": "agt_...",
   "establishment_id": "est_...",
   "name": "Nova GPT",
   "description": "Concierge and support agent",
   "active_version_id": null,
   "created_at": "2026-01-10T00:00:00Z"
}
```

#### `POST /agents/{agentId}/versions`

Creates a draft.

Request:
```json
{
   "name": "v1",
   "prompt": "You are a helpful assistant...",
   "model": {
      "provider": "openai",
      "model": "gpt-4.1-mini",
      "temperature": 0.7,
      "max_tokens": 1024
   },
   "tools": {
      "email": {"enabled": true},
      "calendar": {"enabled": true},
      "whatsapp_messaging": {"enabled": false}
   }
}
```

Response (201):
```json
{
   "id": "agv_...",
   "agent_id": "agt_...",
   "status": "draft",
   "name": "v1"
}
```

#### `POST /agents/{agentId}/versions/{versionId}/publish`

Response (200):
```json
{
   "agent_id": "agt_...",
   "active_version_id": "agv_...",
   "previous_version_id": "agv_prev_..."
}
```

---

### A.4 Runtimes (Docker)

#### `POST /establishments/{id}/runtime`

Provision a dedicated runtime instance.

Request:
```json
{
   "image": "voice-agent-runtime:latest",
   "env": {
      "APP_PUBLIC_DOMAIN": "agent-heliose-hq.example.com",
      "SUPABASE_URL": "...",
      "SUPABASE_SERVICE_ROLE_KEY": "..."
   }
}
```

Response (201):
```json
{
   "id": "rt_...",
   "establishment_id": "est_...",
   "status": "created",
   "container_name": "agent_rt_est_...",
   "base_url": "http://127.0.0.1:18080"
}
```

#### `GET /runtimes/{runtimeId}/status`

Response (200):
```json
{
   "id": "rt_...",
   "status": "running",
   "ready": true,
   "ready_issues": [],
   "last_checked_at": "2026-01-10T00:00:00Z",
   "agent_info": {
      "active_calls": 1,
      "total_calls_handled": 42
   }
}
```

---

### A.5 Calls

#### `GET /calls/active`

Response (200):
```json
{
   "items": [
      {
         "call_id": "call_...",
         "provider": "twilio",
         "provider_call_sid": "CA...",
         "establishment_id": "est_...",
         "agent_id": "agt_...",
         "status": "in_progress",
         "direction": "inbound",
         "from_e164": "+15550102244",
         "to_e164": "+15553012211",
         "started_at": "2026-01-10T00:00:00Z",
         "duration_seconds": 202,
         "sentiment": {
            "label": "positive",
            "score": 0.72,
            "computed_at": "2026-01-10T00:03:22Z"
         }
      }
   ]
}
```

#### `GET /calls/{callId}/transcript`

Response (200):
```json
{
   "call_id": "call_...",
   "items": [
      {
         "segment_id": "seg_...",
         "speaker": "customer",
         "text": "I need to update my service plan.",
         "started_at": "2026-01-10T00:00:03Z",
         "ended_at": "2026-01-10T00:00:05Z",
         "confidence": 0.91
      }
   ],
   "cursor": "..."
}
```

#### `POST /calls/{callId}/handoff`

Request:
```json
{
   "reason": "customer_requested_human"
}
```

Response (200):
```json
{
   "call_id": "call_...",
   "status": "handoff_initiated",
   "handoff_e164": "+15551234567"
}
```

---

### A.6 Leads + campaigns

#### `POST /leads/import`

Request: `multipart/form-data` with CSV file.

Response (202):
```json
{
   "job_id": "job_...",
   "status": "accepted"
}
```

#### `POST /campaigns`

Request:
```json
{
   "establishment_id": "est_...",
   "name": "January winback",
   "agent_id": "agt_...",
   "agent_version_id": "agv_...",
   "lead_filter": {"status": "new"},
   "pace": {"calls_per_minute": 6},
   "schedule": {"timezone": "America/Sao_Paulo", "days": ["mon","tue","wed","thu","fri"], "start": "09:00", "end": "18:00"}
}
```

Response (201):
```json
{
   "id": "cmp_...",
   "status": "created"
}
```

---

### A.7 Integrations

#### `POST /establishments/{id}/integrations`

Request:
```json
{
   "type": "google_calendar",
   "display_name": "Main calendar",
   "auth": {
      "mode": "oauth",
      "redirect_url": "https://app.example.com/settings/integrations"
   }
}
```

Response (201):
```json
{
   "integration_id": "int_...",
   "type": "google_calendar",
   "status": "pending_auth",
   "auth_url": "https://accounts.google.com/o/oauth2/v2/auth?..."
}
```

---

## Appendix B — Runtime → Backend event schemas

The runtime emits events to the control plane via an ingestion endpoint.

### B.1 Ingestion endpoint

`POST /ingest/call-events`

Request:
```json
{
   "runtime_id": "rt_...",
   "establishment_id": "est_...",
   "agent_id": "agt_...",
   "agent_version_id": "agv_...",
   "call": {
      "provider": "twilio",
      "provider_call_sid": "CA...",
      "direction": "inbound",
      "from_e164": "+15550102244",
      "to_e164": "+15553012211"
   },
   "event": {
      "id": "evt_...",
      "type": "transcript_segment",
      "occurred_at": "2026-01-10T00:00:05Z",
      "idempotency_key": "rt_...:CA...:seg:000123",
      "payload": {
         "speaker": "customer",
         "text": "Hello",
         "started_at": "2026-01-10T00:00:03Z",
         "ended_at": "2026-01-10T00:00:05Z",
         "confidence": 0.91
      }
   }
}
```

Response (202):
```json
{
   "status": "accepted"
}
```

### B.2 Canonical event types

#### `call_started`
Payload:
```json
{
   "started_at": "2026-01-10T00:00:00Z"
}
```

#### `call_ended`
Payload:
```json
{
   "ended_at": "2026-01-10T00:10:00Z",
   "duration_seconds": 600,
   "final_status": "completed",
   "hangup_by": "customer"
}
```

#### `transcript_segment`
Payload:
```json
{
   "segment_id": "seg_...",
   "speaker": "customer",
   "text": "...",
   "started_at": "...",
   "ended_at": "...",
   "confidence": 0.0
}
```

#### `tool_called`
Payload:
```json
{
   "tool_name": "send_email",
   "arguments": {"to": "..."}
}
```

#### `tool_result`
Payload:
```json
{
   "tool_name": "send_email",
   "result": {"ok": true, "correlation_id": "..."},
   "error": null
}
```

#### `recording_available`
Payload:
```json
{
   "recording_url": "https://...",
   "storage_provider": "s3",
   "content_type": "audio/wav"
}
```

#### `runtime_error`
Payload:
```json
{
   "component": "stt|tts|llm|transport",
   "message": "...",
   "fatal": false
}
```

### B.3 Idempotency rules

- Backend must treat `event.idempotency_key` as unique per event; duplicates are ignored.
- Runtime should generate deterministic keys (call + segment counter) to support retries.

---

## Appendix C — Acceptance criteria per milestone

### M0 — Skeleton backend + auth + tenancy
- Admin can create an establishment and invite an operator.
- Operator can log in and view the establishment.
- Role enforcement: viewer cannot create resources.

### M1 — Runtime management (Docker)
- Admin provisions runtime for establishment.
- Backend can start/stop/restart runtime and reflects status.
- Backend polls runtime `/readyz` and exposes ready issues.

### M2 — Calling orchestration + call history
- Twilio inbound webhook returns TwiML routing by called number.
- Outbound call initiation creates a call record.
- Calls appear in history with status progression.

### M3 — Transcripts + artifacts
- Runtime can ingest transcript segments during a call.
- UI polling shows transcript growing within 2 polling cycles.
- Call summary appears within agreed SLA (e.g., < 2 minutes after call end).
- Recording link is attached to the call record (if recording enabled).

### M4 — Agents + versioning + deployment
- Admin creates an agent and version v1 and publishes it.
- Admin creates v2 and can rollback to v1.
- Incoming calls route to the currently active version for the mapped agent.

### M5 — Leads + campaign dialer + reports
- CSV import creates leads and validates phone formats.
- Campaign start initiates outbound calls respecting schedule and pacing.
- Reports show agent performance metrics by date range.

---

## Appendix D — Implementation notes (pragmatic)

This appendix provides concrete guidance for building the backend layer with minimal churn later.

### D.1 Database indexing & partitioning

Recommended indexes (Postgres/Supabase):

- `calls`
   - `(establishment_id, started_at DESC)` for UI history
   - `(provider, provider_call_sid)` unique to prevent duplicates
   - `(agent_id, started_at DESC)` for performance reports

- `call_transcript_segments`
   - `(call_id, started_at ASC, segment_id)` for ordered retrieval
   - `(call_id, created_at ASC)` if using ingestion time

- `call_tool_logs`
   - `(call_id, created_at ASC)`
   - `(establishment_id, created_at DESC)` for audits

- `leads`
   - `(establishment_id, status, created_at DESC)` for queue
   - `(establishment_id, phone_e164)` unique (or at least indexed) for dedupe

Partitioning is not required in MVP, but consider monthly partitioning for `call_transcript_segments` once volume grows.

### D.2 Transcript pagination & polling protocol

For `GET /calls/{callId}/transcript` support cursor-based incremental fetch.

Suggested behavior:

- Client passes `cursor` (opaque) or `since_segment_id`.
- Server returns:
   - `items`: new segments
   - `cursor`: next cursor
   - `is_complete`: boolean when call ended and transcript finalized

This avoids re-downloading the full transcript on each poll.

### D.3 Polling cadence recommendations (MVP)

- `/calls/active`: every 3–5 seconds
- `/calls/{id}/transcript`: every 1–2 seconds while viewing a live call
- Dashboard KPIs: every 15–30 seconds

Add server-side caching for dashboard endpoints to protect DB.

### D.4 Background jobs required (MVP)

Minimum job set:

1) **Sentiment computation** (locked choice = backend async)
    - Trigger: transcript segment ingested OR call ended
    - Output: call-level sentiment + optionally windowed sentiment

2) **Call summary generation**
    - Trigger: call ended
    - Output: `call_summaries` record

3) **Campaign scheduler/dialer**
    - Trigger: campaign start
    - Enforces schedule, pacing, retry policy, max concurrency per establishment

4) **Runtime health refresher**
    - Polls runtime `/readyz` + `/info`
    - Stores latest in `runtimes` table

These can be implemented as:
- a single worker process + cron-like loop in MVP
- migrate later to a proper queue (RQ/Celery/Temporal) if needed

### D.5 Docker runtime port allocation & naming

Since MVP runs Docker locally:

- Assign each runtime a stable container name, e.g. `agent_rt_{establishment_id}`.
- Allocate ports deterministically or via a registry table:
   - Example: store `host_port` in `runtimes` table
   - Avoid collisions by enforcing a unique constraint on `host_port`

Runtime base URL should be derived from stored host port.

### D.6 Webhook routing strategy (Twilio)

Routing by phone number requires:

- Store mapping `to_e164` → `establishment_id` and `routing_agent_id`.
- Ensure Twilio sends `To`/called number in webhook.
- Control plane returns TwiML with stream URL pointing to the correct runtime.

Operational note:
- If you later move to multiple hosts, the stream URL must be public and point to the correct host; keep the runtime registry flexible.

### D.7 Recording storage strategy

The PRD leaves storage open. Practical MVP options:

- **Supabase Storage**: simplest if you’re already using Supabase.
- **S3-compatible bucket**: best long-term.

In both cases:
- store only references in DB (`recording_url`, `provider`, `content_type`, `duration_seconds`)
- require authz to access (signed URLs recommended)

### D.8 Security notes (multi-tenant)

- Never allow the frontend to call runtimes directly.
- Runtimes should authenticate to ingestion endpoints (shared secret per runtime or mTLS later).
- Integration tokens must be encrypted at rest.
- Tool proxy endpoints must enforce establishment scoping and tool allow-lists from agent version.

---

## Appendix E — Backend repository skeleton checklist

This appendix turns the PRD into a concrete implementation skeleton, to reduce architecture churn during development.

### E.1 Recommended baseline stack

- Language/runtime: Python 3.11+
- Web framework: FastAPI
- DB: Postgres (Supabase-hosted OK)
- Migrations: Alembic
- Background jobs: single worker loop in MVP (upgrade later)
- Auth: JWT middleware (e.g., Supabase Auth / external IdP)
- Docker control: Docker Engine (local host MVP)

### E.2 Suggested module layout

Create a new backend package (example name: `control_plane`) with a domain-first structure:

- `src/control_plane/app.py`
   - FastAPI application, router inclusion, middleware

- `src/control_plane/config.py`
   - environment parsing and settings

- `src/control_plane/auth/`
   - `deps.py` (get_current_user, require_role)
   - `jwt.py` or `supabase_auth.py`
   - `models.py` (claims/session models)

- `src/control_plane/db/`
   - `session.py` (DB engine/session)
   - `models/` (SQLAlchemy models)
   - `migrations/` (Alembic)

- `src/control_plane/api/routers/`
   - `auth.py`
   - `establishments.py`
   - `users.py`
   - `agents.py`
   - `runtimes.py`
   - `calls.py`
   - `leads.py`
   - `campaigns.py`
   - `reports.py`
   - `integrations.py`
   - `ingest.py` (runtime → backend events)

- `src/control_plane/schemas/`
   - Pydantic request/response models matching Appendix A/B

- `src/control_plane/services/`
   - `runtime_manager.py` (Docker CRUD + health polling + drain/resume proxy)
   - `twilio_router.py` (incoming-call webhook routing + TwiML)
   - `call_service.py` (calls + events + artifacts)
   - `transcript_service.py` (cursor paging)
   - `summary_service.py` (post-call summaries)
   - `sentiment_service.py` (async sentiment)
   - `campaign_service.py` (dialer scheduler/pacing)
   - `integration_service.py` (token store + tool-proxy dispatcher)

- `src/control_plane/workers/`
   - `worker.py` (periodic jobs: runtime health refresh, campaign tick, summary/sentiment)

- `tests/`
   - `test_auth_roles.py`
   - `test_twilio_routing.py`
   - `test_ingest_idempotency.py`
   - `test_transcript_cursor.py`
   - `test_runtime_lifecycle.py`
   - `test_campaign_pacing.py`

### E.3 Implementation order (Milestone-driven)

This order aligns to M0→M2 and provides an early end-to-end loop.

1) M0 Tenancy/Auth: establishments + memberships + role checks
2) M1 Runtime management: provision/start/stop/restart + status polling
3) M2 Calling: inbound webhook router + calls table + active calls list
4) M3 Artifacts: transcript ingestion + recording link + summary job + tool log

### E.4 Contracts that should be “frozen” early

To minimize UI/runtime churn, treat these as stable contracts from the start:

1) `POST /ingest/call-events` event envelope + idempotency semantics
2) `GET /calls/{callId}/transcript` cursor pagination shape
3) phone-number routing mapping (`to_e164 → establishment_id + agent_id + active_version_id`)

### E.5 Quality gates (from day 1)

- Lint: ruff
- Typecheck: mypy (recommended)
- Tests: pytest
- Add contract tests for Appendix A/B JSON shapes (snapshot tests are fine)



