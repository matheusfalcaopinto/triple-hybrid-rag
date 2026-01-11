"""Reports routes."""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.auth.deps import get_current_user
from control_plane.db.models.agent import Agent
from control_plane.db.models.call import Call
from control_plane.db.models.user import User
from control_plane.db.session import get_session
from control_plane.schemas.report import AgentMetrics, AgentPerformanceReport

router = APIRouter()


@router.get("/agent-performance", response_model=AgentPerformanceReport)
async def get_agent_performance_report(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
    days: Annotated[int, Query(ge=1, le=90)] = 30,
):
    """Get agent performance report."""
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

    # Get agents for this establishment
    agents_result = await session.execute(
        select(Agent).where(Agent.establishment_id == establishment_id)
    )
    agents = {a.id: a for a in agents_result.scalars().all()}

    # Get call metrics per agent
    metrics_result = await session.execute(
        select(
            Call.agent_id,
            func.count(Call.id).label("total_calls"),
            func.count(
                case((Call.status == "completed", 1))
            ).label("answered_calls"),
            func.avg(Call.duration_seconds).label("avg_duration"),
            func.avg(Call.sentiment_score).label("avg_sentiment"),
        )
        .where(
            Call.establishment_id == establishment_id,
            Call.started_at >= period_start,
            Call.agent_id.isnot(None),
        )
        .group_by(Call.agent_id)
    )
    metrics = metrics_result.all()

    agent_metrics = []
    totals = {
        "total_calls": 0,
        "answered_calls": 0,
        "avg_duration": 0,
        "avg_sentiment": 0,
    }

    for m in metrics:
        agent = agents.get(m.agent_id)
        if not agent:
            continue

        agent_metrics.append(
            AgentMetrics(
                agent_id=m.agent_id,
                agent_name=agent.name,
                total_calls=m.total_calls or 0,
                answered_calls=m.answered_calls or 0,
                avg_duration_seconds=float(m.avg_duration or 0),
                avg_sentiment_score=float(m.avg_sentiment) if m.avg_sentiment else None,
            )
        )
        totals["total_calls"] += m.total_calls or 0
        totals["answered_calls"] += m.answered_calls or 0

    if agent_metrics:
        totals["avg_duration"] = sum(
            a.avg_duration_seconds for a in agent_metrics
        ) / len(agent_metrics)

    return AgentPerformanceReport(
        period_start=period_start,
        period_end=now,
        agents=agent_metrics,
        totals=totals,
    )


@router.get("/calls")
async def get_calls_report(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    establishment_id: Annotated[str, Query()],
    days: Annotated[int, Query(ge=1, le=90)] = 30,
) -> dict[str, Any]:
    """Get calls report."""
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

    # Overall stats
    stats_result = await session.execute(
        select(
            func.count(Call.id).label("total"),
            func.count(case((Call.direction == "inbound", 1))).label("inbound"),
            func.count(case((Call.direction == "outbound", 1))).label("outbound"),
            func.count(case((Call.status == "completed", 1))).label("answered"),
            func.count(
                case((Call.status.in_(["no_answer", "busy", "failed"]), 1))
            ).label("missed"),
            func.avg(Call.duration_seconds).label("avg_duration"),
        )
        .where(
            Call.establishment_id == establishment_id,
            Call.started_at >= period_start,
        )
    )
    stats = stats_result.one()

    # By status
    status_result = await session.execute(
        select(Call.status, func.count(Call.id))
        .where(
            Call.establishment_id == establishment_id,
            Call.started_at >= period_start,
        )
        .group_by(Call.status)
    )
    by_status = {row[0]: row[1] for row in status_result.all()}

    return {
        "period_start": period_start.isoformat(),
        "period_end": now.isoformat(),
        "total_calls": stats.total or 0,
        "inbound_calls": stats.inbound or 0,
        "outbound_calls": stats.outbound or 0,
        "answered_calls": stats.answered or 0,
        "missed_calls": stats.missed or 0,
        "avg_duration_seconds": float(stats.avg_duration or 0),
        "by_status": by_status,
    }
