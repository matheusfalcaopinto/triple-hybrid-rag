"""Report and dashboard schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class KPIValue(BaseModel):
    """KPI value with optional trend."""

    value: float | int
    change: float | None = None  # Percentage change from previous period
    trend: str | None = None  # up, down, stable


class DashboardKPIs(BaseModel):
    """Dashboard KPIs response."""

    total_calls: KPIValue
    active_calls: KPIValue
    avg_call_duration: KPIValue
    calls_per_day: KPIValue
    conversion_rate: KPIValue | None = None
    sentiment_score: KPIValue | None = None
    period_start: datetime
    period_end: datetime


class AgentMetrics(BaseModel):
    """Agent performance metrics."""

    agent_id: str
    agent_name: str
    total_calls: int
    answered_calls: int
    avg_duration_seconds: float
    avg_sentiment_score: float | None = None
    conversion_rate: float | None = None


class AgentPerformanceReport(BaseModel):
    """Agent performance report."""

    period_start: datetime
    period_end: datetime
    agents: list[AgentMetrics]
    totals: dict[str, Any]


class CallsReport(BaseModel):
    """Calls report."""

    period_start: datetime
    period_end: datetime
    total_calls: int
    inbound_calls: int
    outbound_calls: int
    answered_calls: int
    missed_calls: int
    avg_duration_seconds: float
    by_status: dict[str, int]
    by_day: list[dict[str, Any]]
