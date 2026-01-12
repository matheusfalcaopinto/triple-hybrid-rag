"""
FrameProcessor adapter that wraps a Pipecat VADAnalyzer (e.g., SileroVADAnalyzer)
so it can run inside the pipeline and emit speaking state frames.
"""

from __future__ import annotations

from typing import Optional

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    AudioRawFrame,
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class VADFrameProcessor(FrameProcessor):
    """Adapt a VAD analyzer to the pipeline frame processor API."""

    def __init__(
        self,
        analyzer: VADAnalyzer,
        sample_rate: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._forced_sample_rate = sample_rate
        self._is_speaking = False

    async def process_frame(self, frame, direction: FrameDirection):
        # Only analyze downstream audio frames
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (InputAudioRawFrame, AudioRawFrame)
        ):
            if self._analyzer.sample_rate == 0:
                if self._forced_sample_rate:
                    self._analyzer.set_sample_rate(self._forced_sample_rate)
                elif getattr(frame, "sample_rate", None):
                    self._analyzer.set_sample_rate(frame.sample_rate)

            state = await self._analyzer.analyze_audio(frame.audio)
            speaking = state in (VADState.STARTING, VADState.SPEAKING)

            if speaking and not self._is_speaking:
                self._is_speaking = True
                await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            elif not speaking and self._is_speaking:
                self._is_speaking = False
                await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        await self.push_frame(frame, direction)
