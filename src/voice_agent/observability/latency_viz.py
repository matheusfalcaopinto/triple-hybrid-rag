from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("voice_agent_v4.latency")

_BAR_WIDTH = 35


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class TurnTiming:
    call_sid: str
    turn_id: int
    start_ms: float = field(default_factory=_now_ms)

    vad_to_stt_ms: Optional[float] = None
    stt_to_llm_ms: Optional[float] = None
    llm_streaming_ms: Optional[float] = None
    llm_to_tts_ms: Optional[float] = None
    tts_first_chunk_ms: Optional[float] = None

    stt_final_ts: Optional[float] = None
    first_token_ts: Optional[float] = None
    llm_done_ts: Optional[float] = None
    tts_start_ts: Optional[float] = None
    tts_first_chunk_ts: Optional[float] = None

    def stages(self) -> List[Tuple[str, float]]:
        entries: List[Tuple[str, Optional[float]]] = [
            ("User Speech → STT", self.vad_to_stt_ms),
            ("STT Final → First Token", self.stt_to_llm_ms),
            ("LLM Streaming", self.llm_streaming_ms),
            ("LLM → TTS Start", self.llm_to_tts_ms),
            ("TTS First Chunk", self.tts_first_chunk_ms),
        ]
        return [(label, value) for label, value in entries if value is not None and value >= 0]


class LatencyVisualizer:
    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_LATENCY_VIZ", "false").lower() == "true"
        self._turns: Dict[str, TurnTiming] = {}

    def _key(self, call_sid: str, turn_id: int) -> str:
        return f"{call_sid}_{turn_id}"

    def on_turn_begin(self, call_sid: str, turn_id: int) -> None:
        if not self.enabled:
            return
        self._turns[self._key(call_sid, turn_id)] = TurnTiming(call_sid=call_sid, turn_id=turn_id)

    def on_stt_final(self, call_sid: str, turn_id: int) -> None:
        if not self.enabled:
            return
        key = self._key(call_sid, turn_id)
        timing = self._turns.get(key)
        if not timing:
            return
        timing.stt_final_ts = _now_ms()
        timing.vad_to_stt_ms = timing.stt_final_ts - timing.start_ms

    def on_first_token(self, call_sid: str, turn_id: int) -> None:
        if not self.enabled:
            return
        key = self._key(call_sid, turn_id)
        timing = self._turns.get(key)
        if not timing:
            return
        timing.first_token_ts = _now_ms()
        if timing.stt_final_ts is not None:
            timing.stt_to_llm_ms = timing.first_token_ts - timing.stt_final_ts

    def on_llm_done(self, call_sid: str, turn_id: int) -> None:
        if not self.enabled:
            return
        key = self._key(call_sid, turn_id)
        timing = self._turns.get(key)
        if not timing:
            return
        timing.llm_done_ts = _now_ms()
        if timing.first_token_ts is not None:
            timing.llm_streaming_ms = timing.llm_done_ts - timing.first_token_ts

    def on_tts_start(self, call_sid: str, turn_id: int) -> None:
        if not self.enabled:
            return
        key = self._key(call_sid, turn_id)
        timing = self._turns.get(key)
        if not timing:
            return
        timing.tts_start_ts = _now_ms()
        if timing.llm_done_ts is not None:
            timing.llm_to_tts_ms = timing.tts_start_ts - timing.llm_done_ts

    def on_tts_first_chunk(self, call_sid: str, turn_id: int) -> None:
        if not self.enabled:
            return
        key = self._key(call_sid, turn_id)
        timing = self._turns.get(key)
        if not timing:
            return
        timing.tts_first_chunk_ts = _now_ms()
        if timing.tts_start_ts is not None:
            timing.tts_first_chunk_ms = timing.tts_first_chunk_ts - timing.tts_start_ts
        self._finalize(key)

    def complete_without_tts(self, call_sid: str, turn_id: int) -> None:
        if not self.enabled:
            return
        key = self._key(call_sid, turn_id)
        if key in self._turns:
            self._finalize(key)

    def abandon_turn(self, call_sid: str, turn_id: int) -> None:
        if not self.enabled:
            return
        self._turns.pop(self._key(call_sid, turn_id), None)

    def _finalize(self, key: str) -> None:
        timing = self._turns.pop(key, None)
        if not timing:
            return
        stages = timing.stages()
        if not stages:
            return
        total = sum(value for _, value in stages)
        if total <= 0:
            return

        logger.info("=" * 70)
        logger.info("📊 TURN LATENCY SUMMARY – call=%s turn=%s", timing.call_sid, timing.turn_id)
        logger.info("=" * 70)
        for label, value in stages:
            percentage = (value / total * 100) if total else 0.0
            bar_length = int(round((percentage / 100) * _BAR_WIDTH))
            bar = ("█" * bar_length).ljust(_BAR_WIDTH, "░")
            logger.info("%-30s %8.2fms (%5.1f%%) |%s|", label, value, percentage, bar)
        logger.info("-" * 70)
        logger.info("%-30s %8.2fms (100.0%%)", "TOTAL", total)
        logger.info("=" * 70)


LATENCY_VIZ = LatencyVisualizer()


def on_turn_begin(call_sid: str, turn_id: int) -> None:
    LATENCY_VIZ.on_turn_begin(call_sid, turn_id)


def on_stt_final(call_sid: str, turn_id: int) -> None:
    LATENCY_VIZ.on_stt_final(call_sid, turn_id)


def on_first_token(call_sid: str, turn_id: int) -> None:
    LATENCY_VIZ.on_first_token(call_sid, turn_id)


def on_llm_done(call_sid: str, turn_id: int) -> None:
    LATENCY_VIZ.on_llm_done(call_sid, turn_id)


def on_tts_start(call_sid: str, turn_id: int) -> None:
    LATENCY_VIZ.on_tts_start(call_sid, turn_id)


def on_tts_first_chunk(call_sid: str, turn_id: int) -> None:
    LATENCY_VIZ.on_tts_first_chunk(call_sid, turn_id)


def on_turn_complete(call_sid: str, turn_id: int) -> None:
    LATENCY_VIZ.complete_without_tts(call_sid, turn_id)
