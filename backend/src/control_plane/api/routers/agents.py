"""Agent routes."""

from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import RoleChecker, get_current_user
from control_plane.db.models.agent import Agent, AgentVersion
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.agent import (
    AgentCreate,
    AgentResponse,
    AgentsListResponse,
    AgentUpdate,
    AgentVersionCreate,
    AgentVersionPublish,
    AgentVersionResponse,
    ModelConfig,
    ToolsConfig,
)

router = APIRouter()

# Role checkers
require_admin = RoleChecker(["admin"])
require_operator = RoleChecker(["admin", "operator"])
require_viewer = RoleChecker(["admin", "operator", "viewer"])


@router.get(
    "/establishments/{establishment_id}/agents",
    response_model=AgentsListResponse,
)
async def list_agents(
    establishment_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_viewer)],
):
    """List agents for an establishment."""
    result = await session.execute(
        select(Agent)
        .where(Agent.establishment_id == establishment_id)
        .order_by(Agent.created_at.desc())
    )
    agents = result.scalars().all()

    return AgentsListResponse(
        items=[
            AgentResponse(
                id=a.id,
                establishment_id=a.establishment_id,
                name=a.name,
                description=a.description,
                is_active=a.is_active,
                active_version_id=a.active_version_id,
                created_at=a.created_at,
            )
            for a in agents
        ]
    )


@router.post(
    "/establishments/{establishment_id}/agents",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent(
    establishment_id: Annotated[str, Path()],
    request: AgentCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_admin)],
):
    """Create a new agent."""
    agent = Agent(
        id=f"agt_{uuid4().hex[:12]}",
        establishment_id=establishment_id,
        name=request.name,
        description=request.description,
    )
    session.add(agent)
    await session.flush()

    return AgentResponse(
        id=agent.id,
        establishment_id=agent.establishment_id,
        name=agent.name,
        description=agent.description,
        is_active=agent.is_active,
        active_version_id=agent.active_version_id,
        created_at=agent.created_at,
    )


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Get agent details."""
    result = await session.execute(
        select(Agent).where(Agent.id == agent_id)
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # Check access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == agent.establishment_id),
            None,
        )
        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this agent",
            )

    return AgentResponse(
        id=agent.id,
        establishment_id=agent.establishment_id,
        name=agent.name,
        description=agent.description,
        is_active=agent.is_active,
        active_version_id=agent.active_version_id,
        created_at=agent.created_at,
    )


@router.patch("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: Annotated[str, Path()],
    request: AgentUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Update agent details."""
    result = await session.execute(
        select(Agent).where(Agent.id == agent_id)
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # Check admin access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == agent.establishment_id),
            None,
        )
        if not membership or membership.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )

    if request.name is not None:
        agent.name = request.name
    if request.description is not None:
        agent.description = request.description
    if request.is_active is not None:
        agent.is_active = request.is_active

    await session.flush()

    return AgentResponse(
        id=agent.id,
        establishment_id=agent.establishment_id,
        name=agent.name,
        description=agent.description,
        is_active=agent.is_active,
        active_version_id=agent.active_version_id,
        created_at=agent.created_at,
    )


@router.get("/agents/{agent_id}/versions", response_model=list[AgentVersionResponse])
async def list_agent_versions(
    agent_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """List all versions of an agent."""
    result = await session.execute(
        select(AgentVersion)
        .where(AgentVersion.agent_id == agent_id)
        .order_by(AgentVersion.created_at.desc())
    )
    versions = result.scalars().all()

    return [
        AgentVersionResponse(
            id=v.id,
            agent_id=v.agent_id,
            name=v.name,
            status=v.status,
            prompt=v.prompt,
            model_config=ModelConfig(**v.model_config),
            tools_config=ToolsConfig(**v.tools_config),
            published_at=v.published_at,
            created_at=v.created_at,
        )
        for v in versions
    ]


@router.post(
    "/agents/{agent_id}/versions",
    response_model=AgentVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent_version(
    agent_id: Annotated[str, Path()],
    request: AgentVersionCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Create a new agent version (draft)."""
    # Verify agent exists
    result = await session.execute(
        select(Agent).where(Agent.id == agent_id)
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    version = AgentVersion(
        id=f"agv_{uuid4().hex[:12]}",
        agent_id=agent_id,
        name=request.name,
        status="draft",
        prompt=request.prompt,
        model_config=request.model.model_dump(),
        tools_config=request.tools.model_dump(),
    )
    session.add(version)
    await session.flush()

    return AgentVersionResponse(
        id=version.id,
        agent_id=version.agent_id,
        name=version.name,
        status=version.status,
        prompt=version.prompt,
        model_config=ModelConfig(**version.model_config),
        tools_config=ToolsConfig(**version.tools_config),
        published_at=version.published_at,
        created_at=version.created_at,
    )


@router.post(
    "/agents/{agent_id}/versions/{version_id}/publish",
    response_model=AgentVersionPublish,
)
async def publish_agent_version(
    agent_id: Annotated[str, Path()],
    version_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Publish an agent version."""
    # Get agent and version
    result = await session.execute(
        select(Agent).where(Agent.id == agent_id)
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    result = await session.execute(
        select(AgentVersion).where(
            AgentVersion.id == version_id,
            AgentVersion.agent_id == agent_id,
        )
    )
    version = result.scalar_one_or_none()

    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Version not found",
        )

    if version.status != "draft":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot publish version in status {version.status}",
        )

    # Archive current active version
    previous_version_id = agent.active_version_id
    if previous_version_id:
        result = await session.execute(
            select(AgentVersion).where(AgentVersion.id == previous_version_id)
        )
        prev_version = result.scalar_one_or_none()
        if prev_version:
            prev_version.status = "archived"

    # Publish new version
    version.status = "published"
    version.published_at = datetime.now(timezone.utc)
    version.published_by_user_id = user.id
    agent.active_version_id = version.id

    await session.flush()

    return AgentVersionPublish(
        agent_id=agent_id,
        active_version_id=version.id,
        previous_version_id=previous_version_id,
    )


@router.post(
    "/agents/{agent_id}/versions/{version_id}/rollback",
    response_model=AgentVersionPublish,
)
async def rollback_agent_version(
    agent_id: Annotated[str, Path()],
    version_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Rollback to a previous agent version."""
    # Get agent and version
    result = await session.execute(
        select(Agent).where(Agent.id == agent_id)
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    result = await session.execute(
        select(AgentVersion).where(
            AgentVersion.id == version_id,
            AgentVersion.agent_id == agent_id,
        )
    )
    version = result.scalar_one_or_none()

    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Version not found",
        )

    if version.status not in ["published", "archived"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only rollback to published or archived versions",
        )

    # Archive current active version
    previous_version_id = agent.active_version_id

    # Restore selected version
    version.status = "published"
    agent.active_version_id = version.id

    await session.flush()

    return AgentVersionPublish(
        agent_id=agent_id,
        active_version_id=version.id,
        previous_version_id=previous_version_id,
    )
