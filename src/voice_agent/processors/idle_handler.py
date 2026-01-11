"""
Idle Handler Processor

Handles user inactivity with configurable warnings and graceful call termination.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..config import SETTINGS

logger = logging.getLogger("voice_agent.processors.idle_handler")


class IdleHandlerProcessor(FrameProcessor):
    """
    Monitors user activity and handles prolonged silence.
    
    Behavior:
    1. After `warning_seconds` of silence → push warning message
    2. After `max_warnings` warnings → push goodbye message and end call
    3. Any user speech resets the timer and warning counter
    """

    def __init__(
        self,
        warning_seconds: float = 8.0,
        timeout_seconds: float = 15.0,
        max_warnings: int = 2,
        warning_message: str = "Você ainda está aí? Posso ajudar em algo mais?",
        goodbye_message: str = "Como não recebi resposta, vou encerrar a ligação. Até logo!",
        on_timeout: Optional[Callable] = None,
    ):
        super().__init__()
        self._warning_seconds = warning_seconds
        self._timeout_seconds = timeout_seconds
        self._max_warnings = max_warnings
        self._warning_message = warning_message
        self._goodbye_message = goodbye_message
        self._on_timeout = on_timeout

        self._warning_count = 0
        self._idle_task: Optional[asyncio.Task] = None
        self._user_speaking = False
        self._call_active = True
        self._bot_has_spoken = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and track user activity."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            self._reset_idle_timer()
            logger.debug("User started speaking, reset idle timer")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            # Only start idle timer after bot has had a chance to speak
            if self._bot_has_spoken:
                self._start_idle_timer()
                logger.debug("User stopped speaking, started idle timer")
            else:
                logger.debug("User stopped speaking, waiting for first bot response")

        elif isinstance(frame, EndFrame):
            self._call_active = False
            self._cancel_idle_timer()
            logger.debug("Call ended, cancelled idle timer")

        # Track when bot speaks (TextFrame going to TTS)
        elif isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._bot_has_spoken = True
            # Reset timer when bot speaks - user might need time to process
            self._reset_idle_timer()

        await self.push_frame(frame, direction)

    def _reset_idle_timer(self):
        """Reset idle timer and warning count."""
        self._cancel_idle_timer()
        self._warning_count = 0

    def _cancel_idle_timer(self):
        """Cancel any running idle timer."""
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None

    def _start_idle_timer(self):
        """Start the idle monitoring timer."""
        self._cancel_idle_timer()
        self._idle_task = asyncio.create_task(self._idle_monitor())

    async def _idle_monitor(self):
        """Monitor for user inactivity."""
        try:
            # Wait for warning threshold
            await asyncio.sleep(self._warning_seconds)

            if not self._call_active or self._user_speaking:
                return

            # Issue warning
            self._warning_count += 1
            logger.info(
                "User idle warning %d/%d",
                self._warning_count, self._max_warnings
            )

            if self._warning_count >= self._max_warnings:
                # Max warnings reached, end call
                logger.info("Max idle warnings reached, ending call")
                await self.push_frame(TextFrame(text=self._goodbye_message))
                
                if self._on_timeout:
                    try:
                        result = self._on_timeout()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error("Error in timeout callback: %s", e)
                
                # Give TTS time to speak goodbye before ending
                await asyncio.sleep(3.0)
                await self.push_frame(EndFrame())
            else:
                # Push warning message to LLM to prompt the user
                await self.push_frame(TextFrame(text=self._warning_message))
                # Restart timer for next warning
                self._idle_task = asyncio.create_task(self._idle_monitor())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Error in idle monitor: %s", e)

    async def cleanup(self):
        """Clean up resources."""
        self._cancel_idle_timer()
        await super().cleanup()
