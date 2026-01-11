"""Dashboard routes."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import get_current_user
from control_plane.db.models.call import Call
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.report import DashboardKPIs, KPIValue

router = APIRouter()


@router.get("/kpis", response_model=DashboardKPIs)
async def get_dashboard_kpis(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
    days: Annotated[int, Query(ge=1, le=90)] = 7,
):
    """Get dashboard KPIs."""
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

    now = datetime.now(timezone.utc)
    period_start = now - timedelta(days=days)
    previous_period_start = period_start - timedelta(days=days)

    # Current period calls
    current_result = await session.execute(
        select(
            func.count(Call.id).label("total"),
            func.avg(Call.duration_seconds).label("avg_duration"),
            func.avg(Call.sentiment_score).label("avg_sentiment"),
        )
        .where(
            Call.establishment_id == establishment_id,
            Call.started_at >= period_start,
        )
    )
    current = current_result.one()

    # Previous period for comparison
    previous_result = await session.execute(
        select(func.count(Call.id).label("total"))
        .where(
            Call.establishment_id == establishment_id,
            Call.started_at >= previous_period_start,
            Call.started_at < period_start,
        )
    )
    previous = previous_result.one()

    # Active calls
    active_result = await session.execute(
        select(func.count(Call.id))
        .where(
            Call.establishment_id == establishment_id,
            Call.status.in_(["queued", "ringing", "in_progress"]),
        )
    )
    active_calls = active_result.scalar() or 0

    # Calculate changes
    total_calls = current.total or 0
    prev_total = previous.total or 0
    change = ((total_calls - prev_total) / prev_total * 100) if prev_total > 0 else 0

    return DashboardKPIs(
        total_calls=KPIValue(
            value=total_calls,
            change=round(change, 1),
            trend="up" if change > 0 else "down" if change < 0 else "stable",
        ),
        active_calls=KPIValue(value=active_calls),
        avg_call_duration=KPIValue(
            value=round(current.avg_duration or 0, 1),
        ),
        calls_per_day=KPIValue(
            value=round(total_calls / days, 1) if days > 0 else 0,
        ),
        sentiment_score=KPIValue(
            value=round(current.avg_sentiment or 0.5, 2),
        ) if current.avg_sentiment else None,
        period_start=period_start,
        period_end=now,
    )
