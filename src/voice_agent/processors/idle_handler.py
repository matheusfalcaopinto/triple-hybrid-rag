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
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
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
        self._tool_in_progress = False  # Suspend idle timer during tool execution
        self._active_tool_calls: int = 0  # Track concurrent tool calls

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and track user activity."""
        await super().process_frame(frame, direction)

        # Track tool call start - suspend idle timer during tool execution
        if isinstance(frame, FunctionCallInProgressFrame):
            self._active_tool_calls += 1
            self._tool_in_progress = True
            self._cancel_idle_timer()
            logger.debug(
                "Tool call started (active=%d), suspended idle timer",
                self._active_tool_calls
            )
        
        # Track tool call completion - resume idle timer when all tools complete
        elif isinstance(frame, FunctionCallResultFrame):
            self._active_tool_calls = max(0, self._active_tool_calls - 1)
            if self._active_tool_calls == 0:
                self._tool_in_progress = False
                # Restart idle timer now that tool is done
                if self._bot_has_spoken and not self._user_speaking:
                    self._start_idle_timer()
                logger.debug("All tool calls complete, resumed idle timer")
            else:
                logger.debug(
                    "Tool call complete (active=%d remaining)",
                    self._active_tool_calls
                )

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            self._reset_idle_timer()
            logger.debug("User started speaking, reset idle timer")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            # Only start idle timer after bot has had a chance to speak
            # and when no tool is in progress
            if self._bot_has_spoken and not self._tool_in_progress:
                self._start_idle_timer()
                logger.debug("User stopped speaking, started idle timer")
            elif self._tool_in_progress:
                logger.debug("User stopped speaking, but tool in progress - timer suspended")
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

            # Don't issue warnings if call ended, user is speaking, or tool is in progress
            if not self._call_active or self._user_speaking or self._tool_in_progress:
                if self._tool_in_progress:
                    logger.debug("Idle timer expired but tool in progress, skipping warning")
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
