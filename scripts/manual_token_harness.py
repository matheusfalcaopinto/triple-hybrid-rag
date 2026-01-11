"""Interactive harness for exercising SessionActor without Twilio/Cartesia.

This script starts a SessionActor instance using a console-based sink and a
dummy TTS session that streams raw text chunks instead of μ-law audio. It lets
developers type manual utterances and observe the agent's token stream and
tool behaviour without routing through Twilio Media Streams or Cartesia.

Usage:
    PYTHONPATH=.. venv/bin/python scripts/manual_token_harness.py

Optional flags:
    --caller-phone  E.164 number to seed CRM lookups
    --call-sid      Custom call SID (defaults to random)
    --show-marks    Print mark events such as filler-start/assistant-end
    --log-level     Python log level (default INFO)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional, TextIO

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency guard
    load_dotenv = None  # type: ignore[assignment]


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    parent = repo_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    return repo_root


def _load_env(repo_root: Path) -> None:
    if load_dotenv is None:  # pragma: no cover - fallback if python-dotenv missing
        return
    env_path = repo_root / ".env"
    load_dotenv(dotenv_path=str(env_path), override=False)


_ensure_repo_on_path()

# Imports after path/env setup
from dataclasses import asdict  # noqa: E402

from voice_agent_v4.actor import SessionActor, SinkProtocol  # noqa: E402
from voice_agent_v4.config import SETTINGS, Settings  # noqa: E402
from voice_agent_v4.core import tts  # noqa: E402
from voice_agent_v4.events import SttPartial  # noqa: E402
from voice_agent_v4.providers.cartesia_adapter import (  # noqa: E402
    TTSSessionProtocol,
)


def _manual_mode_settings(base: Settings) -> Settings:
    """Return a Settings copy tuned for the CLI harness."""

    snapshot = base.model_dump()
    snapshot["cartesia_live_enabled"] = False
    snapshot["cartesia_api_key"] = ""
    snapshot["cartesia_voice_id"] = ""
    snapshot["enable_greeting_audio"] = False
    snapshot["enable_silence_fill"] = False
    return Settings(**snapshot)


class ConsoleSink(SinkProtocol):
    """SinkProtocol implementation that prints textual chunks to stdout."""

    def __init__(self, *, show_marks: bool = False) -> None:
        self._lock = asyncio.Lock()
        self._assistant_active = False
        self._show_marks = show_marks

    async def send(self, payload: bytes) -> None:
        text = payload.decode("utf-8", errors="ignore").replace("\x00", "")
        if not text:
            return
        async with self._lock:
            if not self._assistant_active:
                print("\nassistant>", end=" ", flush=True)
                self._assistant_active = True
            print(text, end="", flush=True)

    async def send_mark(self, name: str) -> None:
        async with self._lock:
            if self._show_marks:
                print(f"\n[mark {name}]", flush=True)
            if name.startswith("assistant-end") and self._assistant_active:
                print("", flush=True)
                self._assistant_active = False

    async def clear(self) -> None:
        async with self._lock:
            if self._show_marks:
                print("\n[sink cleared]", flush=True)
            self._assistant_active = False

    async def play_clip(
        self,
        frames,
        *,
        start_mark: str | None = None,
        end_mark: str | None = None,
    ) -> None:
        async with self._lock:
            if self._show_marks:
                clip = start_mark or "greeting"
                print(f"\n[play clip {clip}]", flush=True)


class ManualTTSSession(TTSSessionProtocol):
    """TTS session that surfaces raw text chunks instead of μ-law audio."""

    def __init__(
        self,
        context_id: str,
        cancel: tts.CancellationContext,
    ) -> None:
        self._context_id = context_id
        self._cancel = cancel
        self._text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._closed = False

    def start(self) -> None:
        # No background tasks needed; queue drives iteration.
        return None

    async def send_text(self, text: str) -> None:
        if self._closed:
            return
        await self._text_queue.put(text)

    async def finish_input(self) -> None:
        if self._closed:
            return
        await self._text_queue.put(None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._text_queue.put(None)

    async def get_audio(self) -> AsyncIterator[bytes]:
        while True:
            if self._cancel.cancelled():
                break
            text = await self._text_queue.get()
            if text is None:
                break
            if not text:
                continue
            yield text.encode("utf-8")


async def _async_input(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    def _read() -> str:
        try:
            return input(prompt)
        except EOFError:
            return "/quit"

    return await loop.run_in_executor(None, _read)


async def _wait_for_turn_advance(actor: SessionActor, turn_snapshot: int) -> None:
    while actor.turn_id == turn_snapshot:
        await asyncio.sleep(0.05)


async def _submit_utterance(
    actor: SessionActor,
    text: str,
    seq_counter: int,
    *,
    echo: bool = False,
    log_file: TextIO | None = None,
) -> int:
    utterance = text.strip()
    if not utterance:
        return seq_counter

    if echo:
        print(f"you> {utterance}", flush=True)
    if log_file is not None:
        log_file.write(f"[user] {utterance}\n")
        log_file.flush()

    history_baseline = len(actor.conversation_history)

    current_turn = actor.turn_id
    seq_counter += 1
    event = SttPartial(
        call_sid=actor.call_sid,
        turn_id=current_turn,
        seq=seq_counter,
        ts_mono=time.monotonic(),
        text=utterance,
        is_stable=True,
        is_final=True,
        segment_id=f"manual-{seq_counter}",
        partial_seq=0,
    )
    await actor.on_event(event)
    await _wait_for_turn_advance(actor, current_turn)

    if log_file is not None:
        for msg in actor.conversation_history[history_baseline:]:
            if msg.get("role") == "assistant":
                content = (msg.get("content") or "").strip()
                if content:
                    log_file.write(f"[assistant] {content}\n")
                    log_file.flush()

    return seq_counter


def _manual_tts_factory(
    context_id: str,
    cancel: tts.CancellationContext,
    *,
    trace_id: str | None = None,
) -> TTSSessionProtocol:
    return ManualTTSSession(context_id, cancel)


async def run_manual_session(args: argparse.Namespace, settings: Settings) -> None:
    log_handle: TextIO | None = None
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")

    sink = ConsoleSink(show_marks=args.show_marks)
    call_sid = args.call_sid or f"manual-{uuid.uuid4()}"
    actor = SessionActor(
        call_sid=call_sid,
        sink=sink,
        caller_phone=args.caller_phone or "",
        settings=settings,
        tts_factory=_manual_tts_factory,
    )

    actor_task = asyncio.create_task(actor.start(), name="manual-session-actor")
    seq_counter = 0

    print(
        f"[manual harness] call_sid={call_sid} trace_id={actor.trace_id} "
        "(type /quit to exit)",
        flush=True,
    )

    try:
        if log_handle is not None:
            log_handle.write(
                f"[session] call_sid={call_sid} trace_id={actor.trace_id}\n"
            )
            log_handle.flush()

        if actor.turn_id == 0:
            await _wait_for_turn_advance(actor, actor.turn_id)

        if args.initial_prompt:
            seq_counter = await _submit_utterance(
                actor,
                args.initial_prompt,
                seq_counter,
                echo=not args.stay_open,
                log_file=log_handle,
            )
            if not args.stay_open:
                return
            print("(stay-open) initial prompt sent; interactive mode ready", flush=True)

        while True:
            utterance = (await _async_input("you> ")).strip()
            if not utterance:
                continue
            if utterance in {"/quit", ":q", ":quit", ":exit", "/exit"}:
                print("Exiting manual harness.", flush=True)
                break
            if utterance in {"/history", ":history"}:
                print("\n--- Conversation History ---", flush=True)
                for msg in actor.conversation_history:
                    role = msg.get("role")
                    content = msg.get("content", "").strip()
                    print(f"{role}: {content}", flush=True)
                print("--- end ---\n", flush=True)
                continue
            if len(utterance) < 3:
                print("Input too short, provide at least 3 characters.", flush=True)
                continue
            seq_counter = await _submit_utterance(
                actor,
                utterance,
                seq_counter,
                log_file=log_handle,
            )
    finally:
        actor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await actor_task
        if log_handle is not None:
            log_handle.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual token harness for voice agent")
    parser.add_argument("--caller-phone", help="Optional caller phone (E.164)")
    parser.add_argument("--call-sid", help="Override call SID")
    parser.add_argument(
        "--initial-prompt",
        "--initial_prompt",
        help="Submit this text automatically as the first user utterance",
    )
    parser.add_argument(
        "--stay-open",
        action="store_true",
        help="Remain interactive after sending --initial-prompt",
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to append a plain-text transcript of the session",
    )
    parser.add_argument(
        "--show-marks",
        action="store_true",
        help="Print mark events (assistant-start, filler-start, etc.)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python log level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    repo_root = _ensure_repo_on_path()
    _load_env(repo_root)

    args = _parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.getLogger("voice_agent_v4.cartesia").setLevel(logging.ERROR)
    logging.getLogger("voice_agent_v4.openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    # Settings is frozen; create a manual-mode copy with external providers disabled.
    settings = _manual_mode_settings(SETTINGS)

    try:
        asyncio.run(run_manual_session(args, settings))
    except KeyboardInterrupt:  # pragma: no cover - interactive exit path
        print("Interrupted by user.", flush=True)


if __name__ == "__main__":
    main()
