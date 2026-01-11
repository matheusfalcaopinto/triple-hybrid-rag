"""Campaign service - manages automated dialing campaigns."""

import logging
from datetime import datetime, time, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from control_plane.db.models.campaign import Campaign, CampaignEnrollment, CampaignRun
from control_plane.db.models.lead import Lead

logger = logging.getLogger(__name__)


class CampaignService:
    """Service for managing campaigns and dialing."""

    async def create_campaign(
        self,
        session: AsyncSession,
        establishment_id: str,
        name: str,
        agent_id: str,
        agent_version_id: str | None = None,
        lead_filter: dict[str, Any] | None = None,
        pace_config: dict[str, Any] | None = None,
        schedule: dict[str, Any] | None = None,
    ) -> Campaign:
        """Create a new campaign."""
        campaign = Campaign(
            id=f"cmp_{uuid4().hex[:12]}",
            establishment_id=establishment_id,
            name=name,
            agent_id=agent_id,
            agent_version_id=agent_version_id,
            lead_filter=lead_filter,
            pace_config=pace_config or {"calls_per_minute": 6, "max_concurrent": 10},
            schedule=schedule
            or {
                "timezone": "UTC",
                "days": ["mon", "tue", "wed", "thu", "fri"],
                "start": "09:00",
                "end": "18:00",
            },
            status="created",
        )
        session.add(campaign)
        await session.flush()
        return campaign

    async def get_campaign(
        self, session: AsyncSession, campaign_id: str
    ) -> Campaign | None:
        """Get a campaign by ID."""
        result = await session.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        return result.scalar_one_or_none()

    async def get_campaigns(
        self,
        session: AsyncSession,
        establishment_id: str,
        status: str | None = None,
    ) -> list[Campaign]:
        """Get campaigns for an establishment."""
        query = select(Campaign).where(Campaign.establishment_id == establishment_id)
        if status:
            query = query.where(Campaign.status == status)
        query = query.order_by(Campaign.created_at.desc())
        result = await session.execute(query)
        return list(result.scalars().all())

    async def enroll_leads(
        self,
        session: AsyncSession,
        campaign_id: str,
        lead_filter: dict[str, Any] | None = None,
    ) -> int:
        """Enroll leads into a campaign based on filter."""
        campaign = await self.get_campaign(session, campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Build query based on filter
        query = select(Lead).where(
            Lead.establishment_id == campaign.establishment_id,
            Lead.status.in_(["new", "contacted"]),
        )

        filter_criteria = lead_filter or campaign.lead_filter or {}
        if filter_criteria.get("status"):
            query = query.where(Lead.status == filter_criteria["status"])
        if filter_criteria.get("source"):
            query = query.where(Lead.source == filter_criteria["source"])

        result = await session.execute(query)
        leads = result.scalars().all()

        # Create enrollments
        enrolled = 0
        for lead in leads:
            # Check if already enrolled
            existing = await session.execute(
                select(CampaignEnrollment).where(
                    CampaignEnrollment.campaign_id == campaign_id,
                    CampaignEnrollment.lead_id == lead.id,
                )
            )
            if existing.scalar_one_or_none():
                continue

            enrollment = CampaignEnrollment(
                id=f"enr_{uuid4().hex[:12]}",
                campaign_id=campaign_id,
                lead_id=lead.id,
                status="pending",
            )
            session.add(enrollment)
            enrolled += 1

        # Update campaign total leads
        await session.execute(
            update(Campaign)
            .where(Campaign.id == campaign_id)
            .values(total_leads=Campaign.total_leads + enrolled)
        )

        await session.flush()
        return enrolled

    async def start_campaign(
        self, session: AsyncSession, campaign_id: str
    ) -> Campaign | None:
        """Start a campaign."""
        campaign = await self.get_campaign(session, campaign_id)
        if not campaign:
            return None

        if campaign.status not in ["created", "paused"]:
            raise ValueError(f"Cannot start campaign in status {campaign.status}")

        campaign.status = "running"
        campaign.started_at = datetime.now(timezone.utc)

        # Create a new run
        run = CampaignRun(
            id=f"run_{uuid4().hex[:12]}",
            campaign_id=campaign_id,
            started_at=datetime.now(timezone.utc),
            status="running",
        )
        session.add(run)

        await session.flush()
        return campaign

    async def pause_campaign(
        self, session: AsyncSession, campaign_id: str
    ) -> Campaign | None:
        """Pause a campaign."""
        campaign = await self.get_campaign(session, campaign_id)
        if not campaign:
            return None

        if campaign.status != "running":
            raise ValueError(f"Cannot pause campaign in status {campaign.status}")

        campaign.status = "paused"
        await session.flush()
        return campaign

    async def get_next_leads_to_call(
        self,
        session: AsyncSession,
        campaign_id: str,
        limit: int = 10,
    ) -> list[Lead]:
        """Get next leads to call for a campaign."""
        # Get pending enrollments
        result = await session.execute(
            select(CampaignEnrollment)
            .where(
                CampaignEnrollment.campaign_id == campaign_id,
                CampaignEnrollment.status == "pending",
            )
            .limit(limit)
        )
        enrollments = result.scalars().all()

        # Get leads
        lead_ids = [e.lead_id for e in enrollments]
        if not lead_ids:
            return []

        result = await session.execute(
            select(Lead).where(Lead.id.in_(lead_ids))
        )
        return list(result.scalars().all())

    def is_within_schedule(
        self, schedule: dict[str, Any], now: datetime | None = None
    ) -> bool:
        """Check if current time is within campaign schedule."""
        now = now or datetime.now(timezone.utc)

        # Get day of week
        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_day = day_names[now.weekday()]

        if current_day not in schedule.get("days", []):
            return False

        # Parse times
        try:
            start_time = time.fromisoformat(schedule.get("start", "00:00"))
            end_time = time.fromisoformat(schedule.get("end", "23:59"))
            current_time = now.time()

            return start_time <= current_time <= end_time
        except ValueError:
            return True  # Default to allowing if time parsing fails


# Singleton instance
campaign_service = CampaignService()
