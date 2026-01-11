"""Integration routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import RoleChecker, get_current_user
from control_plane.db.models.integration import Integration
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.integration import (
    IntegrationConnectionResponse,
    IntegrationConnectRequest,
    IntegrationConnectResponse,
    IntegrationConnectionUpdate,
    IntegrationResponse,
)
from control_plane.services.integration_service import integration_service

router = APIRouter()

# Role checkers
require_admin = RoleChecker(["admin"])
require_viewer = RoleChecker(["admin", "operator", "viewer"])


@router.get("/integrations", response_model=list[IntegrationResponse])
async def list_integration_types(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """List available integration types."""
    integrations = await integration_service.get_integration_types(session)
    
    return [
        IntegrationResponse(
            id=i.id,
            type=i.type,
            name=i.name,
            description=i.description,
            icon_url=i.icon_url,
            category=i.category,
            auth_type=i.auth_type,
            is_active=i.is_active,
        )
        for i in integrations
    ]


@router.get(
    "/establishments/{establishment_id}/integrations",
    response_model=list[IntegrationConnectionResponse],
)
async def list_integration_connections(
    establishment_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_viewer)],
):
    """List integration connections for an establishment."""
    connections = await integration_service.get_connections(session, establishment_id)
    
    return [
        IntegrationConnectionResponse(
            id=c.id,
            establishment_id=c.establishment_id,
            integration_id=c.integration_id,
            display_name=c.display_name,
            status=c.status,
            is_enabled=c.is_enabled,
            last_used_at=c.last_used_at,
            last_error=c.last_error,
            created_at=c.created_at,
        )
        for c in connections
    ]


@router.post(
    "/establishments/{establishment_id}/integrations",
    response_model=IntegrationConnectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def connect_integration(
    establishment_id: Annotated[str, Path()],
    request: IntegrationConnectRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(require_admin)],
):
    """Connect an integration to an establishment."""
    try:
        connection, auth_url = await integration_service.create_connection(
            session,
            establishment_id=establishment_id,
            integration_type=request.type,
            display_name=request.display_name,
            auth_data=request.auth,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return IntegrationConnectResponse(
        integration_id=connection.id,
        type=request.type,
        status=connection.status,
        auth_url=auth_url,
    )


@router.patch(
    "/integrations/{integration_id}",
    response_model=IntegrationConnectionResponse,
)
async def update_integration_connection(
    integration_id: Annotated[str, Path()],
    request: IntegrationConnectionUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Update an integration connection."""
    connection = await integration_service.get_connection(session, integration_id)
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration connection not found",
        )

    # Check admin access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == connection.establishment_id),
            None,
        )
        if not membership or membership.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )

    connection = await integration_service.update_connection(
        session,
        integration_id,
        is_enabled=request.is_enabled,
        display_name=request.display_name,
    )

    return IntegrationConnectionResponse(
        id=connection.id,
        establishment_id=connection.establishment_id,
        integration_id=connection.integration_id,
        display_name=connection.display_name,
        status=connection.status,
        is_enabled=connection.is_enabled,
        last_used_at=connection.last_used_at,
        last_error=connection.last_error,
        created_at=connection.created_at,
    )
