# Control-Plane Backend

Multi-tenant SaaS control plane that powers the admin frontend and manages agent runtime instances.

## Features

- **Multi-tenant Management**: Establishments, users, roles (admin/operator/viewer)
- **Agent Management**: CRUD, versioning, publish/rollback
- **Runtime Management**: Docker-based agent runtime lifecycle (create/start/stop/restart/drain/resume)
- **Call Orchestration**: Twilio inbound/outbound, WhatsApp calling
- **Artifacts**: Transcripts, summaries, recordings, tool-call logs
- **Integrations**: Email, Calendar, WhatsApp messaging, CRM via tool proxy
- **Reports**: Agent performance metrics, KPIs
- **Real-time Events**: Server-Sent Events (SSE) for live updates

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (or Supabase)
- Docker (for runtime management)
- Redis (optional, for caching)

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Run database migrations
alembic upgrade head

# Start the server
uvicorn src.control_plane.app:app --reload --port 8000
```

### Development

```bash
# Run tests
pytest

# Run tests with verbose output
pytest -v

# Run linter
ruff check .

# Run type checker
mypy src
```

## Project Structure

```
control_plane/
├── src/control_plane/
│   ├── api/
│   │   └── routers/          # API route handlers
│   │       ├── agents.py     # Agent CRUD and versioning
│   │       ├── auth.py       # Authentication (login, logout, password reset)
│   │       ├── calls.py      # Call management
│   │       ├── campaigns.py  # Automated dialing campaigns
│   │       ├── dashboard.py  # Dashboard metrics
│   │       ├── establishments.py  # Multi-tenant management
│   │       ├── integrations.py   # Third-party integrations
│   │       ├── ingest.py     # Webhook ingestion
│   │       ├── leads.py      # Lead management
│   │       ├── reports.py    # Analytics and reporting
│   │       ├── runtimes.py   # Docker runtime lifecycle
│   │       ├── users.py      # User management
│   │       └── webhooks.py   # Outbound webhooks
│   ├── auth/
│   │   ├── deps.py           # FastAPI dependencies for auth
│   │   └── jwt.py            # JWT token handling
│   ├── db/
│   │   ├── models/           # SQLAlchemy ORM models
│   │   └── session.py        # Database session management
│   ├── events/
│   │   ├── broker.py         # In-memory event broker
│   │   └── sse.py            # Server-Sent Events router
│   ├── schemas/              # Pydantic v2 request/response schemas
│   ├── services/             # Business logic services
│   ├── workers/              # Background workers
│   ├── app.py                # FastAPI application entry point
│   └── config.py             # Configuration via pydantic-settings
├── tests/                    # Unit and integration tests
├── migrations/               # Alembic database migrations
└── pyproject.toml           # Project configuration
```

## API Routes

All API routes are prefixed with `/api/v1`:

| Endpoint | Description |
|----------|-------------|
| `/api/v1/auth/*` | Authentication (login, logout, password reset) |
| `/api/v1/establishments/*` | Multi-tenant establishment management |
| `/api/v1/establishments/{id}/agents/*` | Agent CRUD within establishments |
| `/api/v1/calls/*` | Call history and active calls |
| `/api/v1/leads/*` | Lead management and import |
| `/api/v1/campaigns/*` | Automated dialing campaigns |
| `/api/v1/dashboard/*` | Dashboard metrics |
| `/api/v1/reports/*` | Analytics and reporting |
| `/api/v1/integrations/*` | Third-party integrations |
| `/api/v1/runtimes/*` | Docker runtime management |
| `/api/v1/events/*` | Server-Sent Events for real-time updates |

### Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `/healthz` | Kubernetes-style liveness probe |
| `/readyz` | Kubernetes-style readiness probe |
| `/info` | Application info (name, version) |
| `/health` | Simple health check |

## Technology Stack

- **Framework**: FastAPI with async support
- **ORM**: SQLAlchemy 2.0+ with async PostgreSQL driver
- **Validation**: Pydantic v2 with `ConfigDict` (no deprecated `class Config`)
- **Authentication**: JWT tokens via `python-jose`
- **Password Hashing**: `passlib` with bcrypt
- **Testing**: pytest with pytest-asyncio (SQLite in-memory for tests)
- **Database Compatibility**: PostgreSQL for production, SQLite for testing (custom JSONB type decorator)

## Environment Variables

See `.env.example` for required configuration.

Key variables:
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET_KEY`: Secret key for JWT signing
- `DOCKER_HOST`: Docker socket for runtime management
- `TWILIO_*`: Twilio credentials for telephony

## Testing

The test suite uses SQLite in-memory database with custom type decorators that automatically fall back from PostgreSQL-specific types (JSONB, ARRAY) to SQLite-compatible types (JSON).

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/control_plane

# Run specific test file
pytest tests/test_auth.py -v
```

## License

MIT
