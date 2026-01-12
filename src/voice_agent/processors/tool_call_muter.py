"""
Tool Call Muter Processor

Mutes user audio input briefly after tool calls start to prevent:
- Out-of-order responses
- Confused LLM context
- Race conditions between tool results and user speech

The processor drops UserStartedSpeakingFrame and transcription frames
for a configurable duration after a tool call starts. This allows:
1. TTS to finish announcing the action
2. Brief protection against accidental barge-in
3. User can still speak after the mute window (even if tool still running)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Set

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger("voice_agent.processors.tool_call_muter")


class ToolCallMuterProcessor(FrameProcessor):
    """
    Briefly mutes user input after tool calls start.
    
    This prevents race conditions where user speech arrives before
    the tool result, which would confuse the LLM context.
    
    Behavior:
    1. When FunctionCallInProgressFrame arrives → Start muting
    2. Drop transcription/speaking frames while muted
    3. Auto-unmute after `mute_duration` seconds (even if tool still running)
    4. OR unmute immediately when FunctionCallResultFrame arrives
    
    Args:
        mute_duration: How long to mute after tool call starts (default: 3.0s)
        enabled: Whether muting is enabled (default: True)
    """

    def __init__(
        self,
        mute_duration: float = 3.0,
        enabled: bool = True,
    ):
        super().__init__()
        self._mute_duration = mute_duration
        self._enabled = enabled
        self._active_tool_calls: Set[str] = set()
        self._muted = False
        self._frames_dropped = 0
        self._unmute_task: Optional[asyncio.Task] = None

    @property
    def is_muted(self) -> bool:
        """Check if currently muting user input."""
        return self._muted

    def _cancel_unmute_task(self):
        """Cancel any pending auto-unmute task."""
        if self._unmute_task and not self._unmute_task.done():
            self._unmute_task.cancel()
            self._unmute_task = None

    async def _auto_unmute(self):
        """Auto-unmute after the configured duration."""
        try:
            await asyncio.sleep(self._mute_duration)
            if self._muted:
                self._muted = False
                logger.debug(
                    "Auto-unmuted after %.1fs (tool may still be running)",
                    self._mute_duration
                )
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and mute briefly after tool calls."""
        await super().process_frame(frame, direction)

        # If disabled, pass everything through
        if not self._enabled:
            await self.push_frame(frame, direction)
            return

        # Track tool call start - start mute window
        if isinstance(frame, FunctionCallInProgressFrame):
            tool_id = getattr(frame, 'tool_call_id', None) or str(id(frame))
            tool_name = getattr(frame, 'function_name', 'unknown')
            self._active_tool_calls.add(tool_id)
            
            # Start muting and schedule auto-unmute
            if not self._muted:
                self._muted = True
                self._frames_dropped = 0
                self._cancel_unmute_task()
                self._unmute_task = asyncio.create_task(self._auto_unmute())
                logger.debug(
                    "Tool call [%s] started, muting for %.1fs",
                    tool_name,
                    self._mute_duration
                )
            
            await self.push_frame(frame, direction)
            return

        # Track tool call completion - unmute immediately
        if isinstance(frame, FunctionCallResultFrame):
            tool_id = getattr(frame, 'tool_call_id', None) or str(id(frame))
            self._active_tool_calls.discard(tool_id)
            
            # Unmute when all tools complete
            if not self._active_tool_calls and self._muted:
                self._muted = False
                self._cancel_unmute_task()
                if self._frames_dropped > 0:
                    logger.info(
                        "Tool complete, unmuted (dropped %d frames during mute)",
                        self._frames_dropped
                    )
                else:
                    logger.debug("Tool complete, unmuted")
                self._frames_dropped = 0
            
            await self.push_frame(frame, direction)
            return

        # If muted, drop user speech/transcription frames
        if self._muted:
            if isinstance(frame, (
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
                TranscriptionFrame,
                InterimTranscriptionFrame,
            )):
                self._frames_dropped += 1
                if self._frames_dropped == 1:
                    logger.debug(
                        "Dropping user frame during mute window: %s",
                        type(frame).__name__
                    )
                # Don't push - effectively mutes the user
                return

        # Pass through all other frames
        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up on shutdown."""
        self._cancel_unmute_task()
        self._active_tool_calls.clear()
        self._muted = False
        await super().cleanup()
