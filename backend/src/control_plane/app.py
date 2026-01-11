"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

from control_plane.api.routers import (
    agents,
    auth,
    calls,
    campaigns,
    dashboard,
    establishments,
    ingest,
    integrations,
    leads,
    reports,
    runtimes,
    users,
    webhooks,
)
from control_plane.config import settings
from control_plane.db.session import close_db, init_db
from control_plane.events.sse import sse_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("Starting Control Plane API...")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down Control Plane API...")
    await close_db()
    logger.info("Database closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Control Plane API",
        description="Multi-tenant SaaS backend for voice agent platform",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else ["https://app.example.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoints at root level
    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict:
        """Kubernetes-style health check endpoint."""
        return {"status": "healthy", "version": "0.1.0"}

    @app.get("/readyz", tags=["health"])
    async def readyz() -> dict:
        """Kubernetes-style readiness check endpoint."""
        return {"status": "ready"}

    @app.get("/info", tags=["health"])
    async def info() -> dict:
        """Application info endpoint."""
        return {"name": "control-plane", "version": "0.1.0"}

    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.1.0"}

    # Create API v1 router
    api_v1_router = APIRouter(prefix="/api/v1")
    
    # Register routers under /api/v1
    api_v1_router.include_router(auth.router, prefix="/auth", tags=["auth"])
    api_v1_router.include_router(users.router, tags=["users"])
    api_v1_router.include_router(establishments.router, prefix="/establishments", tags=["establishments"])
    api_v1_router.include_router(agents.router, tags=["agents"])
    api_v1_router.include_router(runtimes.router, tags=["runtimes"])
    api_v1_router.include_router(calls.router, prefix="/calls", tags=["calls"])
    api_v1_router.include_router(leads.router, prefix="/leads", tags=["leads"])
    api_v1_router.include_router(campaigns.router, prefix="/campaigns", tags=["campaigns"])
    api_v1_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
    api_v1_router.include_router(reports.router, prefix="/reports", tags=["reports"])
    api_v1_router.include_router(integrations.router, tags=["integrations"])
    api_v1_router.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])
    api_v1_router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
    api_v1_router.include_router(sse_router, prefix="/events", tags=["events"])
    
    # Include the API v1 router
    app.include_router(api_v1_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "control_plane.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
    )
