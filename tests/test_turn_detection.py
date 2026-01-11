"""Tests for turn detection and idle handling."""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestIdleHandlerProcessor:
    """Test idle handler processor."""

    @pytest.fixture
    def mock_pipecat_frames(self):
        """Mock pipecat frames module."""
        # Create mock frame classes
        mock_frame = MagicMock()
        mock_user_started = MagicMock()
        mock_user_stopped = MagicMock()
        mock_end_frame = MagicMock()
        mock_text_frame = MagicMock()
        
        return {
            "Frame": mock_frame,
            "UserStartedSpeakingFrame": mock_user_started,
            "UserStoppedSpeakingFrame": mock_user_stopped, 
            "EndFrame": mock_end_frame,
            "TextFrame": mock_text_frame,
        }

    def test_init_defaults(self):
        """Test initialization with defaults."""
        from voice_agent.processors.idle_handler import IdleHandlerProcessor
        
        handler = IdleHandlerProcessor()
        
        assert handler._warning_seconds == 8.0
        assert handler._max_warnings == 2
        assert handler._warning_count == 0
        assert handler._user_speaking is False
        assert handler._call_active is True
        assert handler._bot_has_spoken is False

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        from voice_agent.processors.idle_handler import IdleHandlerProcessor
        
        handler = IdleHandlerProcessor(
            warning_seconds=5.0,
            timeout_seconds=10.0,
            max_warnings=3,
            warning_message="Custom warning",
            goodbye_message="Custom goodbye",
        )
        
        assert handler._warning_seconds == 5.0
        assert handler._timeout_seconds == 10.0
        assert handler._max_warnings == 3
        assert handler._warning_message == "Custom warning"
        assert handler._goodbye_message == "Custom goodbye"

    def test_reset_idle_timer_clears_warnings(self):
        """Test that reset clears warning count."""
        from voice_agent.processors.idle_handler import IdleHandlerProcessor
        
        handler = IdleHandlerProcessor(warning_seconds=1.0)
        handler._warning_count = 5
        
        mock_task = MagicMock()
        mock_task.done.return_value = False
        handler._idle_task = mock_task
        
        handler._reset_idle_timer()
        
        assert handler._warning_count == 0
        mock_task.cancel.assert_called_once()

    def test_cancel_idle_timer(self):
        """Test cancel idle timer."""
        from voice_agent.processors.idle_handler import IdleHandlerProcessor
        
        handler = IdleHandlerProcessor(warning_seconds=1.0)
        
        mock_task = MagicMock()
        mock_task.done.return_value = False
        handler._idle_task = mock_task
        
        handler._cancel_idle_timer()
        
        mock_task.cancel.assert_called_once()
        assert handler._idle_task is None

    def test_cancel_idle_timer_no_task(self):
        """Test cancel when no task exists."""
        from voice_agent.processors.idle_handler import IdleHandlerProcessor
        
        handler = IdleHandlerProcessor(warning_seconds=1.0)
        handler._idle_task = None
        
        # Should not raise
        handler._cancel_idle_timer()
        
        assert handler._idle_task is None

    def test_start_idle_timer_in_event_loop(self):
        """Test starting idle timer creates task inside event loop."""
        from voice_agent.processors.idle_handler import IdleHandlerProcessor
        
        async def run_test():
            handler = IdleHandlerProcessor(warning_seconds=10.0)
            handler._start_idle_timer()
            assert handler._idle_task is not None
            handler._cancel_idle_timer()
        
        asyncio.run(run_test())


class TestIdleMonitorBehavior:
    """Test the idle monitoring logic."""

    def test_idle_monitor_increments_warnings(self):
        """Test that idle monitor increments warning count."""
        from voice_agent.processors.idle_handler import IdleHandlerProcessor
        
        async def run_test():
            handler = IdleHandlerProcessor(
                warning_seconds=0.01,  # Very short for testing
                max_warnings=3,
            )
            handler._call_active = True
            handler._user_speaking = False
            handler._bot_has_spoken = True
            handler.push_frame = AsyncMock()
            
            # Start timer
            handler._start_idle_timer()
            
            # Wait for warning to trigger
            await asyncio.sleep(0.05)
            
            # Should have warned once
            assert handler._warning_count >= 1
            
            # Clean up
            handler._call_active = False
            handler._cancel_idle_timer()
        
        asyncio.run(run_test())


class TestConfigIntegration:
    """Test config integration."""

    def test_config_has_turn_detection_settings(self):
        """Test that config has all turn detection settings."""
        from voice_agent.config import Settings
        
        with patch.dict("os.environ", {}, clear=False):
            settings = Settings()
            
            assert hasattr(settings, "user_idle_timeout_seconds")
            assert hasattr(settings, "user_idle_warning_seconds")
            assert hasattr(settings, "user_idle_max_warnings")
            assert hasattr(settings, "mute_during_function_call")
            assert hasattr(settings, "mute_until_first_bot_speech")
            
            # Check defaults
            assert settings.user_idle_timeout_seconds == 15.0
            assert settings.user_idle_warning_seconds == 8.0
            assert settings.user_idle_max_warnings == 2

    def test_config_custom_values(self):
        """Test config with custom values."""
        from voice_agent.config import Settings
        
        with patch.dict("os.environ", {
            "USER_IDLE_TIMEOUT_SECONDS": "30.0",
            "USER_IDLE_WARNING_SECONDS": "10.0",
            "USER_IDLE_MAX_WARNINGS": "5",
        }, clear=False):
            settings = Settings()
            
            assert settings.user_idle_timeout_seconds == 30.0
            assert settings.user_idle_warning_seconds == 10.0
            assert settings.user_idle_max_warnings == 5
