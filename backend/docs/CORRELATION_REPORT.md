# Correlation Report — Frontend Mock ↔ Control-Plane Backend ↔ Agent Runtime

Date: 2026-01-10  
Workspace: `voice_agent_pipecat_standalone`

This report consolidates the assessment performed so far in both directions:

1) **Mock frontend ⇒ required backend/agent capabilities** (what the UI expects)  
2) **Agent runtime ⇒ usable frontend features** (what’s already implemented by default)

It’s intended to be the reference for the **control-plane (“stack layer”) backend roadmap**, plus agent-compatibility requirements, test planning, and documentation.

## PRD link

The MVP/PRD for the control-plane backend layer (decisions + API surface + data model + milestones) is in:

- `frontend/BACKEND_LAYER_MVP_PRD.md`

---

## 0) Architecture framing (the 3 blocks)

### Block A — Frontend (admin console)
The folder `frontend/frontend-mookup/` is a Vite/React mock UI with these route groups:

- Auth: `/login`, `/forgot-password`
- Dashboard: `/dashboard`
- Establishments: `/establishments/*`
- Agents: `/agents/*`
- Calls: `/calls/*`
- Leads: `/leads/*`
- Reports: `/reports/*`
- Settings: `/settings/*`
- Help: `/help/*`

The mock currently renders mostly static data/preview overlays. It implies a full SaaS admin backend.

### Block B — Control-plane backend (“stack layer”)
This is the planned central backend between UI and agent runtime.

**Owns**: auth, tenants, agents as resources, config/versioning, call orchestration, integrations/tokens, analytics/reporting, API keys/RBAC, data aggregation, event fan-in/fan-out to UI.

### Block C — Agent runtime (this repo)
This repo already runs a real-time voice agent (Pipecat pipeline), handling:

- Twilio inbound/outbound call flows (webhooks + media WebSocket)
- WhatsApp calling webhooks + WebRTC transport
- LLM tool calling (function calling)
- Basic health/metrics/session endpoints
- Partial persistence of “communication events” in Supabase

**Important:** The agent runtime is *not* a multi-tenant SaaS backend; it’s a low-latency runtime.

---

## 1) Verified agent runtime surfaces (already implemented)

### 1.1 Health / readiness / info
In `src/voice_agent/app.py`:

- `GET /healthz` — liveness
- `GET /readyz` — readiness (checks key configuration presence)
- `GET /info` — runtime info (status, active calls, models)

Useful frontend widgets:
- Runtime status indicator (online / not-ready + reasons)
- Display active model IDs

### 1.2 Sessions & metrics
In `src/voice_agent/app.py`:

- `GET /sessions` — list active sessions (masked phone, durations)
- `GET /metrics` — Prometheus-style counters/gauges

Useful frontend widgets:
- “Live calls” list
- Basic KPIs (active calls, total calls handled)

### 1.3 Drain/resume controls
In `src/voice_agent/app.py`:

- `POST /control/drain`
- `POST /control/resume`

Useful frontend controls:
- “Pause accepting new calls” per agent instance (safe deploy workflow)

### 1.4 Twilio inbound/outbound call runtime
In `src/voice_agent/app.py`:

- `POST /incoming-call` (and `POST /`) — returns TwiML to connect Twilio → WebSocket
- `WS <SETTINGS.twilio_ws_path>` — media stream handler running Pipecat pipeline
- `POST /amd-callback` — voicemail / AMD callback
- `POST /outbound-call` — TwiML webhook for outbound calls
- `POST /call-status` — call status callback
- `POST /api/outbound-call` — JSON API to initiate outbound call

Useful frontend features:
- “Place test call” / “Outbound call” UI (if protected/mediated by backend)
- “Telephony setup” screen showing required callback URLs

### 1.5 WhatsApp calling runtime (Meta)
In `src/voice_agent/app.py`:

- `GET /whatsapp/call/webhook` — verification
- `POST /whatsapp/call/webhook` — call events + signature verification

Useful frontend features:
- WhatsApp calling enabled/disabled status
- Webhook verification status

### 1.6 Tooling and context prefetch
In `src/voice_agent/bot.py` and `src/voice_agent/context.py`:

- LLM supports tool/function calling (OpenAI function model via Pipecat)
- Customer context prefetch calls tool handlers:
  - `get_customer_by_phone`
  - `get_customer_facts`
  - `get_last_call`
  - `get_pending_tasks`

In `src/voice_agent/tools/server.py`:

- MCP Tools Server that loads tools from `tools/*.py` and optionally external MCP servers

Implication:
- “Tools” are an agent feature by default; the *control-plane/backend* decides which tools are enabled and how secrets are resolved.

### 1.7 Persistence primitives
In `src/voice_agent/communication/storage.py`:

- Communication events are stored in Supabase table `communication_events`.

This is a good building block for:
- message delivery statuses
- call artifact audit trails

---

## 2) Frontend mock feature inventory (what UI expects)

From `frontend/frontend-mookup/src/App.tsx` routes:

- Dashboard
- Establishments
- Agents
- Calls (Active/History/Details)
- Leads
- Reports
- Settings (Profile/Integrations/API Keys/MCP)
- Auth

Representative feature expectations discovered in key pages:

- `CallsActive.tsx`: live calls grid, “Intervene”, live transcript, sentiment/customer card
- `CallsHistory.tsx`: call history table + export CSV + costs
- `AgentConfig.tsx`: prompt, LLM params, call settings, Twilio number
- `AgentTools.tsx`: integrations per agent + “MCP Custom builder”
- `Dashboard.tsx`: KPIs, chart ranges, agent status, alerts, recent calls
- `SettingsIntegrations.tsx`: connect/manage integrations
- `SettingsApiKeys.tsx`: manage API keys

---

## 3) Correlation matrix (grouped by complexity)

This section cross-checks UI features with:

- **Control-plane backend requirements** (what the stack layer must implement)
- **Agent compatibility requirements** (what the runtime must emit/support)

### Legend
- **Preserve**: Keep the feature concept and UI structure.
- **Reinterpret**: Keep UI but change meaning/scope to match feasible runtime.
- **Defer**: Do later.
- **Cut**: Remove if not aligned.

Complexity categories below assume: **control-plane backend is the system of record**.

---

## 3A) LOW complexity (good early wins)

### A1) Runtime status widgets
**Frontend locations:** Dashboard, Settings/Integrations (status cards)  
**Agent already has:** `/healthz`, `/readyz`, `/info`  
**Backend should provide:**
- `GET /runtimes` (list agent runtime instances)
- `GET /runtimes/{id}/status` (aggregated / cached)

**Agent compatibility:** none beyond current endpoints.

**Preserve / choice:** Preserve.

**Testing:**
- Backend contract tests for status aggregation
- Agent smoke test: `/readyz` returns not-ready when keys missing

---

### A2) Drain/resume (maintenance mode)
**Frontend locations:** Agents list/status, runtime controls  
**Agent already has:** `POST /control/drain`, `POST /control/resume`  
**Backend should provide:**
- `POST /runtimes/{id}/drain`
- `POST /runtimes/{id}/resume`

**Agent compatibility:** none beyond current endpoints.

**Preserve / choice:** Preserve.

**Testing:**
- Ensure calls are rejected when draining

---

### A3) Active calls list (basic)
**Frontend:** `/calls` (Active calls page)  
**Agent already has:** `GET /sessions`  
**Backend should provide:**
- `GET /calls/active` (aggregated across runtimes)

**Agent compatibility:** none beyond current `/sessions`.

**Preserve / choice:** Preserve.

---

### A4) Outbound call (basic “place test call”)
**Frontend:** Agent test / calls tooling  
**Agent already has:** `POST /api/outbound-call`  
**Backend should provide:**
- `POST /calls/outbound` (authz + rate limiting + auditing)

**Agent compatibility:** none beyond current outbound call endpoint.

**Preserve / choice:** Preserve.

---

## 3B) MEDIUM complexity (requires backend data model + event ingestion)

### B1) Call history + call details (without advanced analytics)
**Frontend:** `/calls/history`, `/calls/:id`  
**Agent status:** has callbacks (`/call-status`) but no unified calls DB exposed.

**Backend should implement:**
- Data model: `calls` table (call_sid, runtime_id, agent_id, establishment_id, direction, from/to, status, start/end, duration, external provider IDs)
- `GET /calls?filters`
- `GET /calls/{id}`
- `GET /calls/{id}/events` (timeline)
- `POST /calls/export` (CSV)

**Agent compatibility needed:**
- Emit/store call lifecycle events at least at start/end.

**Preserve / choice:** Preserve.

---

### B2) Dashboard KPIs (basic)
**Frontend:** `/dashboard`  
**Agent status:** `/metrics` provides a small set of counters.

**Backend should implement:**
- `GET /dashboard/kpis` (computed from calls DB)
- periodic scrape from agent `/metrics` or compute from ingested events

**Agent compatibility:**
- Minimal: either scrape existing metrics OR emit events that backend aggregates.

**Preserve:** Preserve.

---

### B3) Integrations UI (environment-level / single-tenant)
**Frontend:** `/settings/integrations`, `/agents/:id/tools`  
**Agent status:** supports tools; some tools/services exist.

**Backend should implement:**
- `GET/POST /integrations`
- token storage (encrypted)
- `PATCH /integrations/{id}` enable/disable

**Agent compatibility:**
- Prefer **tool proxy** pattern: agent calls backend for integration actions.

**Preserve / choice:** Preserve (start environment-level), expand later.

---

## 3C) HIGH complexity (multi-tenant platform features)

### C1) Auth + RBAC + API keys
**Frontend:** `/login`, `/forgot-password`, `/settings/api-keys`  
**Agent status:** none; should not own auth.

**Backend should implement:**
- Auth provider (recommended: Supabase Auth / external IdP)
- JWT verification middleware
- API key issuance + scopes + rate limits + last used

**Agent compatibility:** none.

**Preserve / choice:** Preserve, but build in backend first.

---

### C2) Establishments + billing
**Frontend:** `/establishments/*` + billing page  
**Backend should implement:**
- multi-tenant data model
- billing integration (Stripe) if needed

**Agent compatibility:**
- pass `establishment_id` metadata downstream for logging/policies

**Preserve / choice:** Preserve if SaaS is core; otherwise defer.

---

### C3) Agents (CRUD + versioning + deployment)
**Frontend:** `/agents/*`  
**Agent status:** runtime takes config from env/settings; no built-in multi-agent routing.

**Backend should implement:**
- `agents` resource + versions
- `GET/POST /agents`
- `GET/POST /agents/{id}/versions`
- `POST /agents/{id}/deploy` (assign version to runtime)

**Agent compatibility needed:**
- Ability to load config dynamically (not only env)
- Accept “agent_version_id” at session start

**Preserve / choice:** Preserve, but treat as control-plane-owned.

---

## 3D) VERY HIGH complexity (agent-runtime feature work or major infra)

### D1) Live transcript streaming to UI
**Frontend:** Calls Active “Live transcript”  
**Agent status:** STT runs inside pipeline, but transcript isn’t emitted to UI today.

**Backend should implement:**
- Realtime channel to UI: `WS /events` or `SSE /events`
- transcript persistence: `call_transcripts` table

**Agent compatibility needed:**
- Emit `transcript_segment` events from pipeline

**Preserve / choice:** Preserve, but implement incrementally:
1) persist segments (batch)
2) realtime streaming

---

### D2) Human intervention / takeover
**Frontend:** Calls Active “Intervene”  
**Agent status:** supports interruptions for STT/LLM flow; no operator channel.

**Backend should implement:**
- command API: `POST /calls/{id}/inject-text`, `POST /calls/{id}/handoff`
- (later) injected audio / operator bridge

**Agent compatibility:**
- Needs a command channel into call pipeline

**Preserve / choice:** **Reinterpret for MVP** as:
- “Inject text” and/or “handoff to human” (transfer)
Defer true takeover.

---

### D3) Sentiment / advanced analytics
**Frontend:** sentiment card, “alerts”, analytics charts  

**Backend should implement:**
- store turn-level features (sentiment/intents) if desired

**Agent compatibility:**
- either agent emits sentiment (extra model) OR backend derives sentiment from transcript asynchronously

**Preserve / choice:** Preserve, but start asynchronous in backend.

---

## 4) Agent-default capabilities → frontend setup parameters

This section lists what the frontend can expose/configure today because the agent runtime already uses them.

### Telephony / runtime settings (observability + setup screens)
- Public domain/scheme for stream URLs: `APP_PUBLIC_DOMAIN`, `APP_SCHEME_OVERRIDE`
- WebSocket path: `TWILIO_WS_PATH`
- Sample rate: `TWILIO_SAMPLE_RATE`
- Draining state (ops)

### AI service keys and models (readiness + config screen)
- `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL`
- `CARTESIA_API_KEY`, `CARTESIA_TTS_MODEL`, `CARTESIA_STT_MODEL`, `CARTESIA_VOICE_ID`
- VAD tuning: `VAD_THRESHOLD`, `VAD_MIN_SILENCE_MS`

### Storage / persistence
- Supabase connection: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`

### Recording
- Recording enabled flags and sample rate (present in pipeline; depends on environment + Pipecat AudioBuffer availability)

### WhatsApp calling
- webhook verify token + enable flag (from settings)

---

## 5) Design choices & recommended options

### Choice 1 — Source of truth
**Recommended:** Control-plane backend is the source of truth for:
- calls history
- transcripts
- tool call logs
- integrations/tokens

Agent runtime should **emit events** or call backend ingestion endpoints.

### Choice 2 — Tool execution model
Two viable options:

1) **Tool proxy (recommended for multi-tenant):**
   - Agent calls backend endpoints for “send email”, “create calendar event”, “CRM update”, etc.
   - Backend resolves credentials per tenant/agent.

2) **Token vending:**
   - Agent requests short-lived tokens for direct provider calls.
   - Backend remains secret holder.

### Choice 3 — Realtime channel to UI
- Start with polling (fast)
- Then move to `WS /events` or `SSE /events`

---

## 6) Roadmap (phased)

### Phase 0 — Observability MVP (fastest)
- Backend proxies agent status/metrics/sessions
- UI shows runtime status + active calls list
- Add protected “test outbound call” (backend → agent)

### Phase 1 — Calls as a first-class resource
- Backend introduces `calls` table
- Ingest call lifecycle events
- Call history + details UI becomes real

### Phase 2 — Transcript + recording artifacts
- Agent emits transcript segments
- Backend stores transcript; UI renders live transcript + history view

### Phase 3 — Agent management
- Agents CRUD + versioned configs
- Deployments (assign versions to runtimes)

### Phase 4 — Integrations + policies
- OAuth/token storage
- Tool proxy endpoints
- Per-tenant/per-agent enablement

### Phase 5 — Advanced features
- Intervention (inject-text → handoff → audio takeover)
- Reports/costs analytics
- Leads/campaign dialer

---

## 7) Testing strategy (what to test where)

### Agent runtime tests
- Smoke tests for `/readyz`, `/info`, `/sessions`
- Webhook signature verification tests (WhatsApp)
- Outbound call request validation tests

### Backend control-plane tests
- Contract tests asserting:
  - status aggregation behavior
  - call ingestion correctness
  - transcript ordering + idempotency
- Authz/RBAC tests for drain/outbound initiation

### End-to-end tests
- “Outbound call initiated → call status updates → call appears in history”
- “Transcript segments emitted → UI receives events + renders”

---

## 8) Notes / known hard parts

- **Intervention** is expensive: start with inject-text/handoff.
- Multi-tenant integrations are best solved by **tool proxy**.
- Don’t make the agent runtime a general CRUD API; keep it a runtime.

---

## Appendix — Agent endpoints (verified)

From `src/voice_agent/app.py`:

- `GET /`
- `GET /healthz`
- `GET /readyz`
- `GET /info`
- `GET /sessions`
- `GET /metrics`
- `POST /control/drain`
- `POST /control/resume`
- `POST /` (alias of incoming call)
- `POST /incoming-call`
- `WS <SETTINGS.twilio_ws_path>`
- `POST /amd-callback`
- `POST /outbound-call`
- `POST /call-status`
- `POST /api/outbound-call`
- `GET /whatsapp/call/webhook`
- `POST /whatsapp/call/webhook`
