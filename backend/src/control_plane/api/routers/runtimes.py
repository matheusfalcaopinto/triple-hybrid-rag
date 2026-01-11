"""Runtime routes."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import RoleChecker, get_current_user
from control_plane.db.models.runtime import Runtime
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.runtime import (
    RuntimeAgentInfo,
    RuntimeCreateRequest,
    RuntimeResponse,
    RuntimeStatusResponse,
)
from control_plane.services.runtime_manager import runtime_manager

router = APIRouter()

# Role checkers
require_admin = RoleChecker(["admin"])
require_operator = RoleChecker(["admin", "operator"])
require_viewer = RoleChecker(["admin", "operator", "viewer"])


@router.get(
    "/establishments/{establishment_id}/runtime",
    response_model=RuntimeResponse | None,
)
async def get_establishment_runtime(
    establishment_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_viewer)],
):
    """Get runtime for an establishment."""
    result = await session.execute(
        select(Runtime).where(Runtime.establishment_id == establishment_id)
    )
    runtime = result.scalar_one_or_none()

    if not runtime:
        return None

    return RuntimeResponse(
        id=runtime.id,
        establishment_id=runtime.establishment_id,
        status=runtime.status,
        container_name=runtime.container_name,
        container_id=runtime.container_id,
        image_tag=runtime.image_tag,
        host_port=runtime.host_port,
        base_url=runtime.base_url,
        created_at=runtime.created_at,
    )


@router.post(
    "/establishments/{establishment_id}/runtime",
    response_model=RuntimeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_runtime(
    establishment_id: Annotated[str, Path()],
    request: RuntimeCreateRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_admin)],
):
    """Create a runtime for an establishment."""
    # Check if runtime already exists
    result = await session.execute(
        select(Runtime).where(Runtime.establishment_id == establishment_id)
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Runtime already exists for this establishment",
        )

    # Create Docker container
    try:
        container_info = await runtime_manager.create_runtime(
            establishment_id=establishment_id,
            image=request.image,
            env=request.env,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create runtime: {e}",
        )

    # Store runtime record
    runtime = Runtime(
        id=container_info["id"],
        establishment_id=establishment_id,
        container_name=container_info["container_name"],
        container_id=container_info["container_id"],
        image_tag=container_info["image_tag"],
        host_port=container_info["host_port"],
        base_url=container_info["base_url"],
        status="created",
        env_config=request.env,
    )
    session.add(runtime)
    await session.flush()

    return RuntimeResponse(
        id=runtime.id,
        establishment_id=runtime.establishment_id,
        status=runtime.status,
        container_name=runtime.container_name,
        container_id=runtime.container_id,
        image_tag=runtime.image_tag,
        host_port=runtime.host_port,
        base_url=runtime.base_url,
        created_at=runtime.created_at,
    )


@router.post("/runtimes/{runtime_id}/start", response_model=RuntimeStatusResponse)
async def start_runtime(
    runtime_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Start a runtime."""
    result = await session.execute(
        select(Runtime).where(Runtime.id == runtime_id)
    )
    runtime = result.scalar_one_or_none()

    if not runtime:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Runtime not found",
        )

    try:
        await runtime_manager.start_runtime(runtime.container_name)
        runtime.status = "starting"
        await session.flush()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start runtime: {e}",
        )

    return RuntimeStatusResponse(
        id=runtime.id,
        status=runtime.status,
        ready=runtime.is_ready,
        ready_issues=runtime.ready_issues,
        last_checked_at=runtime.last_health_check_at,
    )


@router.post("/runtimes/{runtime_id}/stop", response_model=RuntimeStatusResponse)
async def stop_runtime(
    runtime_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Stop a runtime."""
    result = await session.execute(
        select(Runtime).where(Runtime.id == runtime_id)
    )
    runtime = result.scalar_one_or_none()

    if not runtime:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Runtime not found",
        )

    try:
        await runtime_manager.stop_runtime(runtime.container_name)
        runtime.status = "stopping"
        runtime.is_ready = False
        await session.flush()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop runtime: {e}",
        )

    return RuntimeStatusResponse(
        id=runtime.id,
        status=runtime.status,
        ready=runtime.is_ready,
        ready_issues=runtime.ready_issues,
        last_checked_at=runtime.last_health_check_at,
    )


@router.post("/runtimes/{runtime_id}/restart", response_model=RuntimeStatusResponse)
async def restart_runtime(
    runtime_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Restart a runtime."""
    result = await session.execute(
        select(Runtime).where(Runtime.id == runtime_id)
    )
    runtime = result.scalar_one_or_none()

    if not runtime:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Runtime not found",
        )

    try:
        await runtime_manager.restart_runtime(runtime.container_name)
        runtime.status = "starting"
        runtime.is_ready = False
        await session.flush()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart runtime: {e}",
        )

    return RuntimeStatusResponse(
        id=runtime.id,
        status=runtime.status,
        ready=runtime.is_ready,
        ready_issues=runtime.ready_issues,
        last_checked_at=runtime.last_health_check_at,
    )


@router.post("/runtimes/{runtime_id}/drain", response_model=RuntimeStatusResponse)
async def drain_runtime(
    runtime_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Drain a runtime (stop accepting new calls)."""
    result = await session.execute(
        select(Runtime).where(Runtime.id == runtime_id)
    )
    runtime = result.scalar_one_or_none()

    if not runtime:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Runtime not found",
        )

    try:
        await runtime_manager.drain_runtime(runtime.base_url)
        runtime.status = "draining"
        await session.flush()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to drain runtime: {e}",
        )

    return RuntimeStatusResponse(
        id=runtime.id,
        status=runtime.status,
        ready=runtime.is_ready,
        ready_issues=runtime.ready_issues,
        last_checked_at=runtime.last_health_check_at,
    )


@router.post("/runtimes/{runtime_id}/resume", response_model=RuntimeStatusResponse)
async def resume_runtime(
    runtime_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Resume a drained runtime."""
    result = await session.execute(
        select(Runtime).where(Runtime.id == runtime_id)
    )
    runtime = result.scalar_one_or_none()

    if not runtime:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Runtime not found",
        )

    try:
        await runtime_manager.resume_runtime(runtime.base_url)
        runtime.status = "running"
        await session.flush()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume runtime: {e}",
        )

    return RuntimeStatusResponse(
        id=runtime.id,
        status=runtime.status,
        ready=runtime.is_ready,
        ready_issues=runtime.ready_issues,
        last_checked_at=runtime.last_health_check_at,
    )


@router.get("/runtimes/{runtime_id}/status", response_model=RuntimeStatusResponse)
async def get_runtime_status(
    runtime_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Get runtime status."""
    result = await session.execute(
        select(Runtime).where(Runtime.id == runtime_id)
    )
    runtime = result.scalar_one_or_none()

    if not runtime:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Runtime not found",
        )

    # Get fresh status from container
    container_status = await runtime_manager.get_container_status(runtime.container_name)

    # Get health check
    health = await runtime_manager.check_runtime_health(runtime.base_url)

    # Get runtime info
    info = await runtime_manager.get_runtime_info(runtime.base_url)

    return RuntimeStatusResponse(
        id=runtime.id,
        status=container_status.get("status", runtime.status),
        ready=health.get("ready", False),
        ready_issues=health.get("issues"),
        last_checked_at=runtime.last_health_check_at,
        agent_info=RuntimeAgentInfo(**info) if info else None,
    )
