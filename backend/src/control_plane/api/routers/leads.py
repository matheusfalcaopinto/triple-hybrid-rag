"""Lead routes."""

import csv
import io
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import RoleChecker, get_current_user
from control_plane.db.models.lead import Lead
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.lead import (
    LeadCreate,
    LeadImportResponse,
    LeadResponse,
    LeadsListResponse,
    LeadUpdate,
)

router = APIRouter()

# Role checkers
require_operator = RoleChecker(["admin", "operator"])
require_viewer = RoleChecker(["admin", "operator", "viewer"])


@router.get("", response_model=LeadsListResponse)
async def list_leads(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
    lead_status: Annotated[str | None, Query(alias="status")] = None,
    source: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """List leads with filters."""
    # Check access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None,
        )
        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this establishment",
            )

    query = select(Lead).where(Lead.establishment_id == establishment_id)
    
    if lead_status:
        query = query.where(Lead.status == lead_status)
    if source:
        query = query.where(Lead.source == source)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_query)).scalar() or 0

    # Get paginated results
    query = query.order_by(Lead.created_at.desc()).offset(offset).limit(limit)
    result = await session.execute(query)
    leads = result.scalars().all()

    return LeadsListResponse(
        items=[
            LeadResponse(
                id=lead.id,
                establishment_id=lead.establishment_id,
                phone_e164=lead.phone_e164,
                name=lead.name,
                email=lead.email,
                status=lead.status,
                source=lead.source,
                custom_fields=lead.custom_fields,
                notes=lead.notes,
                last_contacted_at=lead.last_contacted_at,
                total_calls=lead.total_calls,
                created_at=lead.created_at,
            )
            for lead in leads
        ],
        total=total,
    )


@router.post("", response_model=LeadResponse, status_code=status.HTTP_201_CREATED)
async def create_lead(
    request: LeadCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
):
    """Create a new lead."""
    # Check operator access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None,
        )
        if not membership or membership.role not in ["admin", "operator"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator access required",
            )

    lead = Lead(
        id=f"lead_{uuid4().hex[:12]}",
        establishment_id=establishment_id,
        phone_e164=request.phone_e164,
        name=request.name,
        email=request.email,
        source=request.source,
        custom_fields=request.custom_fields,
        notes=request.notes,
        status="new",
    )
    session.add(lead)
    await session.flush()

    return LeadResponse(
        id=lead.id,
        establishment_id=lead.establishment_id,
        phone_e164=lead.phone_e164,
        name=lead.name,
        email=lead.email,
        status=lead.status,
        source=lead.source,
        custom_fields=lead.custom_fields,
        notes=lead.notes,
        last_contacted_at=lead.last_contacted_at,
        total_calls=lead.total_calls,
        created_at=lead.created_at,
    )


@router.post("/import", response_model=LeadImportResponse, status_code=status.HTTP_202_ACCEPTED)
async def import_leads(
    file: Annotated[UploadFile, File()],
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
):
    """Import leads from CSV file."""
    # Check operator access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == establishment_id),
            None,
        )
        if not membership or membership.role not in ["admin", "operator"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator access required",
            )

    # Validate file type
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV",
        )

    # Read and parse CSV
    content = await file.read()
    decoded = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))

    job_id = f"job_{uuid4().hex[:12]}"
    imported_count = 0

    for row in reader:
        phone = row.get("phone") or row.get("phone_e164") or row.get("Phone")
        if not phone:
            continue

        # Normalize phone number
        if not phone.startswith("+"):
            phone = f"+{phone}"

        lead = Lead(
            id=f"lead_{uuid4().hex[:12]}",
            establishment_id=establishment_id,
            phone_e164=phone,
            name=row.get("name") or row.get("Name"),
            email=row.get("email") or row.get("Email"),
            source="csv_import",
            import_job_id=job_id,
            status="new",
            custom_fields={
                k: v for k, v in row.items()
                if k.lower() not in ["phone", "phone_e164", "name", "email"]
            },
        )
        session.add(lead)
        imported_count += 1

    await session.flush()

    return LeadImportResponse(
        job_id=job_id,
        status="accepted",
    )


@router.patch("/{lead_id}", response_model=LeadResponse)
async def update_lead(
    lead_id: Annotated[str, Path()],
    request: LeadUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    """Update a lead."""
    result = await session.execute(
        select(Lead).where(Lead.id == lead_id)
    )
    lead = result.scalar_one_or_none()

    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found",
        )

    # Check access
    if not user.is_superuser:
        membership = next(
            (m for m in user.memberships if m.establishment_id == lead.establishment_id),
            None,
        )
        if not membership or membership.role not in ["admin", "operator"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator access required",
            )

    if request.name is not None:
        lead.name = request.name
    if request.email is not None:
        lead.email = request.email
    if request.status is not None:
        lead.status = request.status
    if request.custom_fields is not None:
        lead.custom_fields = request.custom_fields
    if request.notes is not None:
        lead.notes = request.notes

    await session.flush()

    return LeadResponse(
        id=lead.id,
        establishment_id=lead.establishment_id,
        phone_e164=lead.phone_e164,
        name=lead.name,
        email=lead.email,
        status=lead.status,
        source=lead.source,
        custom_fields=lead.custom_fields,
        notes=lead.notes,
        last_contacted_at=lead.last_contacted_at,
        total_calls=lead.total_calls,
        created_at=lead.created_at,
    )
