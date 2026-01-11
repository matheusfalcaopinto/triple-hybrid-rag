"""Background worker for periodic tasks."""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from control_plane.config import settings
from control_plane.db.models.call import Call
from control_plane.db.models.campaign import Campaign
from control_plane.db.models.runtime import Runtime
from control_plane.db.session import AsyncSessionLocal
from control_plane.services.campaign_service import campaign_service
from control_plane.services.runtime_manager import runtime_manager
from control_plane.services.sentiment_service import sentiment_service
from control_plane.services.summary_service import summary_service

logger = logging.getLogger(__name__)


class Worker:
    """Background worker for scheduled tasks."""

    def __init__(self):
        self.running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        """Start background tasks."""
        self.running = True
        logger.info("Starting background worker")

        # Start all task loops
        self._tasks = [
            asyncio.create_task(self._runtime_health_loop()),
            asyncio.create_task(self._summary_loop()),
            asyncio.create_task(self._sentiment_loop()),
            asyncio.create_task(self._campaign_loop()),
        ]

    async def stop(self):
        """Stop background tasks."""
        self.running = False
        logger.info("Stopping background worker")

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _runtime_health_loop(self):
        """Check runtime health periodically."""
        interval = 30  # seconds
        
        while self.running:
            try:
                async with AsyncSessionLocal() as session:
                    # Get all runtimes
                    result = await session.execute(select(Runtime))
                    runtimes = result.scalars().all()

                    for runtime in runtimes:
                        if runtime.status not in ["running", "starting", "draining"]:
                            continue

                        try:
                            # Check container status
                            container_status = await runtime_manager.get_container_status(
                                runtime.container_name
                            )
                            
                            # Check health endpoint
                            health = await runtime_manager.check_runtime_health(
                                runtime.base_url
                            )

                            # Update runtime record
                            runtime.last_health_check_at = datetime.now(timezone.utc)
                            runtime.is_ready = health.get("ready", False)
                            runtime.ready_issues = health.get("issues")

                            if container_status.get("status") == "running":
                                if runtime.status == "starting" and runtime.is_ready:
                                    runtime.status = "running"
                            elif container_status.get("status") == "exited":
                                runtime.status = "stopped"
                                runtime.is_ready = False

                        except Exception as e:
                            logger.error(f"Health check failed for {runtime.id}: {e}")
                            runtime.is_ready = False
                            runtime.ready_issues = [str(e)]

                    await session.commit()

            except Exception as e:
                logger.error(f"Runtime health loop error: {e}")

            await asyncio.sleep(interval)

    async def _summary_loop(self):
        """Generate summaries for completed calls."""
        interval = 30  # seconds
        
        while self.running:
            try:
                async with AsyncSessionLocal() as session:
                    # Find recently completed calls without summaries
                    result = await session.execute(
                        select(Call)
                        .where(
                            Call.status == "completed",
                            Call.ended_at.isnot(None),
                        )
                        .limit(10)
                    )
                    calls = result.scalars().all()

                    for call in calls:
                        # Check if summary exists
                        existing = await summary_service.summary_exists(session, call.id)
                        if existing:
                            continue

                        try:
                            await summary_service.generate_summary(session, call.id)
                            await session.commit()
                            logger.info(f"Generated summary for call {call.id}")
                        except Exception as e:
                            logger.error(f"Summary generation failed for {call.id}: {e}")

            except Exception as e:
                logger.error(f"Summary loop error: {e}")

            await asyncio.sleep(interval)

    async def _sentiment_loop(self):
        """Compute sentiment for calls."""
        interval = 15  # seconds
        
        while self.running:
            try:
                async with AsyncSessionLocal() as session:
                    # Find calls needing sentiment computation
                    result = await session.execute(
                        select(Call)
                        .where(
                            Call.sentiment_label.is_(None),
                            Call.status.in_(["in_progress", "completed"]),
                        )
                        .limit(10)
                    )
                    calls = result.scalars().all()

                    for call in calls:
                        try:
                            await sentiment_service.compute_sentiment(session, call.id)
                            await session.commit()
                        except Exception as e:
                            logger.debug(f"Sentiment computation for {call.id}: {e}")

            except Exception as e:
                logger.error(f"Sentiment loop error: {e}")

            await asyncio.sleep(interval)

    async def _campaign_loop(self):
        """Process campaign dialing."""
        interval = 10  # seconds
        
        while self.running:
            try:
                async with AsyncSessionLocal() as session:
                    # Get running campaigns
                    campaigns = await campaign_service.get_campaigns(
                        session, None, status="running"
                    )

                    for campaign in campaigns:
                        # Check if within schedule
                        if not campaign_service.is_within_schedule(campaign.schedule):
                            continue

                        # Get next leads to call
                        leads = await campaign_service.get_next_leads_to_call(
                            session, campaign.id, limit=5
                        )

                        for lead in leads:
                            # TODO: Actually initiate outbound call
                            # This would integrate with the calls router
                            logger.info(
                                f"Campaign {campaign.id}: would call lead {lead.id}"
                            )

                    await session.commit()

            except Exception as e:
                logger.error(f"Campaign loop error: {e}")

            await asyncio.sleep(interval)


# Singleton instance
worker = Worker()


async def run_worker():
    """Run the worker."""
    await worker.start()
    
    try:
        while worker.running:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await worker.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
