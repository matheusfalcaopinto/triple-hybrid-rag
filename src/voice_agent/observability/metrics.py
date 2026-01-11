from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Optional

logger = logging.getLogger("voice_agent_v4")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

@dataclass
class MetricSample:
    value: float
    trace_id: Optional[str]


_METRICS: DefaultDict[str, List[MetricSample]] = defaultdict(list)


def record(metric: str, value: float, *, trace_id: Optional[str] = None) -> None:
    _METRICS[metric].append(MetricSample(value=value, trace_id=trace_id))


def increment(metric: str, *, trace_id: Optional[str] = None) -> None:
    _METRICS[metric].append(MetricSample(value=1.0, trace_id=trace_id))


def snapshot() -> dict[str, List[MetricSample]]:
    return {key: list(values) for key, values in _METRICS.items()}


def reset() -> None:
    _METRICS.clear()


def export_prometheus() -> str:
    """Export metrics in Prometheus text format.

    Returns a string in Prometheus exposition format with:
    - HELP lines describing each metric
    - TYPE lines declaring metric types (counter or histogram)
    - Metric values with optional trace_id labels

    Example output:
        # HELP cartesia_tts_error_total Cartesia TTS errors
        # TYPE cartesia_tts_error_total counter
        cartesia_tts_error_total 5.0
    """
    lines: list[str] = []
    metrics_snapshot = snapshot()

    # Define metric metadata (help text and type)
    metric_metadata = {
        # Provider errors
        "cartesia_tts_error": ("Cartesia TTS errors", "counter"),
        "cartesia_tts_rate_limit": ("Cartesia TTS rate limit hits", "counter"),
        "cartesia_tts_timeout": ("Cartesia TTS timeouts", "counter"),
        "cartesia_tts_timeout_or_closed": ("Cartesia TTS timeout or connection closed", "counter"),
        "cartesia_stt_error": ("Cartesia STT errors", "counter"),
        "cartesia_stt_backpressure": ("Cartesia STT backpressure events", "counter"),
        "cartesia_stt_flush_requested": ("Cartesia STT flush requests", "counter"),
        "cartesia_stt_flush_timeout": ("Cartesia STT flush timeouts", "counter"),
        "cartesia_stt_flush_error": ("Cartesia STT flush errors", "counter"),
        "openai_stream_error": ("OpenAI stream errors", "counter"),
        "openai_stream_timeout": ("OpenAI stream timeouts", "counter"),
        "mcp_tool_invocation": ("MCP tool invocations", "counter"),
        "mcp_tool_success": ("MCP tool successes", "counter"),
        "mcp_tool_error": ("MCP tool errors", "counter"),

        # Application metrics
        "late_event_dropped": ("Events dropped due to late arrival", "counter"),
        "duplicate_final_dropped": ("Duplicate STT final events dropped", "counter"),

        # Latency metrics (recorded as values, not counters)
        "time_to_first_audio_ms": ("Time to first audio chunk in milliseconds", "histogram"),
        "turn_latency_ms": ("Turn processing latency in milliseconds", "histogram"),
        "mcp_tool_latency_ms": ("Latency of MCP tool executions in milliseconds", "histogram"),
    }

    # Sort metrics by name for consistent output
    for metric_name in sorted(metrics_snapshot.keys()):
        samples = metrics_snapshot[metric_name]
        if not samples:
            continue

        # Get metadata or use defaults
        help_text, metric_type = metric_metadata.get(
            metric_name, (f"Metric: {metric_name}", "counter")
        )

        # Add HELP and TYPE lines
        metric_full_name = f"{metric_name}_total" if metric_type == "counter" else metric_name
        lines.append(f"# HELP {metric_full_name} {help_text}")
        lines.append(f"# TYPE {metric_full_name} {metric_type}")

        # Aggregate metric values
        if metric_type == "counter":
            # Sum all counter values
            total = sum(sample.value for sample in samples)
            lines.append(f"{metric_full_name} {total}")
        else:
            # For histograms, export count, sum, and buckets
            values = [sample.value for sample in samples]
            count = len(values)
            total_val = sum(values)

            if count > 0:
                # Export histogram buckets (using fixed buckets for latency)
                buckets = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
                for bucket in buckets:
                    bucket_count = sum(1 for v in values if v <= bucket)
                    lines.append(f'{metric_full_name}_bucket{{le="{bucket}"}} {bucket_count}')

                # +Inf bucket
                lines.append(f'{metric_full_name}_bucket{{le="+Inf"}} {count}')
                lines.append(f"{metric_full_name}_count {count}")
                lines.append(f"{metric_full_name}_sum {total_val}")

    return "\n".join(lines) + "\n" if lines else ""


@dataclass
class Trace:
    call_sid: str
    turn_id: int
    event: str
    trace_id: str
    seq: int = 0
    extra: dict | None = None

    def log(self) -> None:
        payload = {
            "event": self.event,
            "call_sid": self.call_sid,
            "turn_id": self.turn_id,
            "seq": self.seq,
            "trace_id": self.trace_id,
        }
        if self.extra:
            for key, value in self.extra.items():
                if isinstance(value, str) and len(value) > 120:
                    payload[key] = value[:117] + "..."
                else:
                    payload[key] = value
        logger.info(self.event, extra={"trace_payload": payload})


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)
