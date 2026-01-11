# Changelog

All notable changes to the Control Plane Backend will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-10

### Added
- Initial implementation of the Control Plane API
- Multi-tenant establishment management
- Agent CRUD with versioning and publish/rollback
- Runtime management for Docker-based agent containers
- Call orchestration with Twilio integration
- Lead management and import
- Campaign system for automated dialing
- Real-time Server-Sent Events (SSE) for live updates
- Dashboard metrics and analytics
- Integration framework for Email, Calendar, WhatsApp

### API Endpoints
- `/api/v1/auth/*` - Authentication (login, logout, password reset)
- `/api/v1/establishments/*` - Multi-tenant management
- `/api/v1/establishments/{id}/agents/*` - Agent management
- `/api/v1/calls/*` - Call history and management
- `/api/v1/leads/*` - Lead management
- `/api/v1/campaigns/*` - Campaign management
- `/api/v1/dashboard/*` - Dashboard metrics
- `/api/v1/reports/*` - Analytics reporting
- `/api/v1/integrations/*` - Third-party integrations
- `/api/v1/runtimes/*` - Docker runtime management
- `/api/v1/events/*` - SSE real-time events
- `/healthz`, `/readyz`, `/info` - Health check endpoints

### Technical
- FastAPI with async support
- SQLAlchemy 2.0+ with async PostgreSQL driver
- Pydantic v2 with `ConfigDict` (migrated from deprecated `class Config`)
- JWT authentication with `python-jose`
- Custom JSONB/ARRAY type decorators for SQLite test compatibility
- pytest-asyncio for async test support
- 33 unit tests passing

### Fixed
- Migrated all Pydantic schemas from deprecated `class Config` to `model_config = ConfigDict(...)`
- Added SQLite-compatible type decorators for JSONB and ARRAY types (enables testing without PostgreSQL)
- Fixed pytest-asyncio fixture scope issues for proper async test execution
