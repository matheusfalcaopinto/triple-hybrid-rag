"""Tests for event broker."""

import asyncio
from uuid import uuid4

import pytest

from control_plane.events.broker import EventBroker


class TestEventBroker:
    """Test event broker functionality."""

    @pytest.fixture
    def broker(self):
        """Create a fresh event broker."""
        return EventBroker()

    @pytest.fixture
    def establishment_id(self):
        """Generate establishment ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self, broker, establishment_id):
        """Test subscribing and receiving events."""
        queue = await broker.subscribe(establishment_id, "calls")
        
        # Publish event
        count = await broker.publish(
            establishment_id,
            "calls",
            "call.started",
            {"call_id": "123"},
        )
        
        assert count == 1
        
        # Check queue
        event = queue.get_nowait()
        assert event["type"] == "call.started"
        assert event["data"]["call_id"] == "123"

    @pytest.mark.asyncio
    async def test_unsubscribe(self, broker, establishment_id):
        """Test unsubscribing from events."""
        queue = await broker.subscribe(establishment_id, "calls")
        await broker.unsubscribe(establishment_id, "calls", queue)
        
        # Publish should not reach anyone
        count = await broker.publish(
            establishment_id,
            "calls",
            "call.started",
            {},
        )
        
        assert count == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, broker, establishment_id):
        """Test multiple subscribers receive events."""
        queue1 = await broker.subscribe(establishment_id, "calls")
        queue2 = await broker.subscribe(establishment_id, "calls")
        
        count = await broker.publish(
            establishment_id,
            "calls",
            "call.started",
            {},
        )
        
        assert count == 2
        assert not queue1.empty()
        assert not queue2.empty()

    @pytest.mark.asyncio
    async def test_channel_isolation(self, broker, establishment_id):
        """Test events are isolated by channel."""
        calls_queue = await broker.subscribe(establishment_id, "calls")
        dashboard_queue = await broker.subscribe(establishment_id, "dashboard")
        
        await broker.publish(
            establishment_id,
            "calls",
            "call.started",
            {},
        )
        
        assert not calls_queue.empty()
        assert dashboard_queue.empty()

    @pytest.mark.asyncio
    async def test_establishment_isolation(self, broker):
        """Test events are isolated by establishment."""
        est1 = uuid4()
        est2 = uuid4()
        
        queue1 = await broker.subscribe(est1, "calls")
        queue2 = await broker.subscribe(est2, "calls")
        
        await broker.publish(est1, "calls", "call.started", {})
        
        assert not queue1.empty()
        assert queue2.empty()

    @pytest.mark.asyncio
    async def test_all_channel_receives_everything(self, broker, establishment_id):
        """Test 'all' channel receives all events."""
        all_queue = await broker.subscribe(establishment_id, "all")
        calls_queue = await broker.subscribe(establishment_id, "calls")
        
        await broker.publish(
            establishment_id,
            "calls",
            "call.started",
            {},
        )
        
        # Both should receive the event
        assert not all_queue.empty()
        assert not calls_queue.empty()

    @pytest.mark.asyncio
    async def test_subscriber_count(self, broker, establishment_id):
        """Test subscriber counting."""
        assert broker.get_subscriber_count(establishment_id, "calls") == 0
        
        queue = await broker.subscribe(establishment_id, "calls")
        assert broker.get_subscriber_count(establishment_id, "calls") == 1
        
        await broker.unsubscribe(establishment_id, "calls", queue)
        assert broker.get_subscriber_count(establishment_id, "calls") == 0
