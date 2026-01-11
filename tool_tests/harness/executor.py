"""
Pipecat Test Harness - Executor

Coordinates tool execution against the Pipecat tools adapter with safety patches.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable, List
import contextlib
from unittest.mock import patch

from .context import (
    HarnessContext,
    OutcomeStatus,
    SampleData,
    ToolExecutionResult,
    ToolTestCase,
)

logger = logging.getLogger("pipecat.test_harness")


class PipecatToolTestHarness:
    """
    Coordinates tool execution against the Pipecat tools adapter.
    """

    def __init__(self) -> None:
        self.sample_data: SampleData | None = None
        self.context: HarnessContext | None = None
        self.handlers: dict = {}
        self._temp_dir: Path | None = None
        self._restore_callbacks: List[Callable[[], None]] = []
        self._initialized = False

    async def __aenter__(self) -> "PipecatToolTestHarness":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def start(self) -> None:
        """Initialize the harness."""
        if self._initialized:
            return

        # Import Pipecat tools
        from voice_agent.tools import get_tool_handlers

        self.handlers = get_tool_handlers()
        logger.info("Loaded %d tool handlers", len(self.handlers))

        # Create temp directory
        self._temp_dir = Path(tempfile.mkdtemp(prefix="pipecat_harness_"))

        # Load sample data
        self.sample_data = SampleData.load_defaults()

        # Create context
        self.context = HarnessContext(
            handlers=self.handlers,
            temp_root=self._temp_dir,
            resources={"sample_data": self.sample_data},
        )

        # Apply safety patches
        self._apply_safety_patches()

        self._initialized = True
        logger.info("Pipecat test harness ready")

    async def shutdown(self) -> None:
        """Clean up harness resources."""
        if not self._initialized:
            return

        # Cleanup temp directory
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass

        # Restore patches
        for restore in reversed(self._restore_callbacks):
            try:
                restore()
            except Exception:
                logger.exception("Failed to restore patch")

        self._restore_callbacks.clear()
        self.context = None
        self.handlers = {}
        self._initialized = False

    def _monkey_patch(self, module: object, attribute: str, value: object) -> None:
        """Apply a monkey patch with automatic restore on shutdown."""
        original = getattr(module, attribute)
        setattr(module, attribute, value)

        def _restore() -> None:
            setattr(module, attribute, original)

        self._restore_callbacks.append(_restore)

    def _apply_safety_patches(self) -> None:
        """Patch external integrations to avoid real network traffic."""
        # Force Twilio backend for testing
        import os
        original_backend = os.environ.get("WHATSAPP_BACKEND")
        os.environ["WHATSAPP_BACKEND"] = "twilio"
        
        def _restore_env():
            if original_backend is None:
                os.environ.pop("WHATSAPP_BACKEND", None)
            else:
                os.environ["WHATSAPP_BACKEND"] = original_backend
                
        self._restore_callbacks.append(_restore_env)

        # Email SMTP stubs
        try:
            import voice_agent.tools.email as email_module

            def _stub_send_email_blocking(
                to: str,
                subject: str,
                body: str,
                cc: str | None = None,
                bcc: str | None = None,
                html: bool = False,
                *,
                correlation_id: str | None = None,
            ) -> dict:
                cid = correlation_id or str(uuid.uuid4())
                return {
                    "success": True,
                    "message": "[STUB] Simulated email dispatch",
                    "to": to,
                    "subject": subject,
                    "correlation_id": cid,
                    "channel": "email",
                }

            self._monkey_patch(email_module, "_send_email_blocking", _stub_send_email_blocking)
            logger.debug("Applied email stub")
        except ImportError:
            pass

        # WhatsApp / Twilio stubs
        try:
            import voice_agent.tools.whatsapp as whatsapp_module

            class _StubTwilioMessage:
                def __init__(self) -> None:
                    self.sid = f"SM{uuid.uuid4().hex[:24]}"
                    self.status = "queued"

            class _StubTwilioMessages:
                def create(self, **kwargs) -> _StubTwilioMessage:
                    return _StubTwilioMessage()

            class _StubTwilioClient:
                def __init__(self) -> None:
                    self.messages = _StubTwilioMessages()

            def _stub_get_twilio_client() -> _StubTwilioClient:
                return _StubTwilioClient()

            self._monkey_patch(whatsapp_module, "get_twilio_client", _stub_get_twilio_client)
            self._monkey_patch(whatsapp_module, "media_base_url_configured", lambda: True)
            logger.debug("Applied WhatsApp stub")
        except ImportError:
            pass

        # Google Calendar stubs
        try:
            import voice_agent.tools.google_calendar as calendar_module

            class _StubOperation:
                def __init__(self, payload: dict) -> None:
                    self._payload = payload

                def execute(self) -> dict:
                    return self._payload

            class _StubEvents:
                def __init__(self) -> None:
                    self._storage: list = []

                def insert(self, calendarId: str, body: dict) -> _StubOperation:
                    event_id = body.get("id") or f"evt-{uuid.uuid4().hex[:10]}"
                    event = {
                        "id": event_id,
                        "summary": body.get("summary", "Stub Event"),
                        "htmlLink": f"https://calendar.example.com/event/{event_id}",
                    }
                    self._storage.append(event)
                    return _StubOperation(event)

                def list(self, **kwargs) -> _StubOperation:
                    return _StubOperation({"items": list(self._storage)})

                def delete(self, calendarId: str, eventId: str) -> _StubOperation:
                    return _StubOperation({})

            class _StubFreeBusy:
                def query(self, body: dict) -> _StubOperation:
                    return _StubOperation({"calendars": {"primary": {"busy": []}}})

            class _StubCalendarService:
                def __init__(self) -> None:
                    self._events = _StubEvents()

                def events(self) -> _StubEvents:
                    return self._events

                def freebusy(self) -> _StubFreeBusy:
                    return _StubFreeBusy()

            def _stub_get_calendar_service() -> _StubCalendarService:
                return _StubCalendarService()

            self._monkey_patch(calendar_module, "get_calendar_service", _stub_get_calendar_service)
            logger.debug("Applied Calendar stub")
        except ImportError:
            pass

    async def run_cases(self, cases: Iterable[ToolTestCase]) -> List[ToolExecutionResult]:
        """Run a list of test cases."""
        if not self._initialized:
            raise RuntimeError("Harness not started. Call await harness.start() first.")

        results: List[ToolExecutionResult] = []
        for case in cases:
            result = await self._run_single_case(case)
            results.append(result)
        return results

    async def run_single_tool(
        self,
        tool_name: str,
        arguments: dict | None = None,
        description: str = "",
    ) -> ToolExecutionResult:
        """Run a single tool test."""
        case = ToolTestCase(
            tool_name=tool_name,
            description=description or f"Test {tool_name}",
            arguments=arguments or {},
        )
        return await self._run_single_case(case)

    def _determine_success(
        self,
        raw: dict,
        expect_success: bool,
        allowed_errors: tuple[str, ...],
    ) -> tuple[bool, OutcomeStatus, str, str | None]:
        """Determine the success status of a tool execution."""
        outer_success = bool(raw.get("success"))
        payload = raw.get("result")
        payload_dict = payload if isinstance(payload, dict) else {}

        derived_error: str | None = None
        success = outer_success

        if success and isinstance(payload_dict, dict):
            if payload_dict.get("success") is False:
                success = False
                derived_error = str(payload_dict.get("error") or payload_dict.get("message") or "")
            elif "error" in payload_dict:
                success = False
                derived_error = str(payload_dict.get("error"))
        elif not success:
            derived_error = str(raw.get("error") or "")

        if success:
            if expect_success:
                return True, OutcomeStatus.SUCCESS, "Tool execution succeeded", None
            return False, OutcomeStatus.UNEXPECTED_FAILURE, "Expected failure but tool succeeded", None

        error_text = derived_error or ""
        if not expect_success:
            if allowed_errors:
                for token in allowed_errors:
                    if token.lower() in error_text.lower():
                        return False, OutcomeStatus.EXPECTED_FAILURE, error_text, error_text
                return False, OutcomeStatus.UNEXPECTED_FAILURE, error_text, error_text
            return False, OutcomeStatus.EXPECTED_FAILURE, error_text, error_text

        if allowed_errors:
            for token in allowed_errors:
                if token.lower() in error_text.lower():
                    return False, OutcomeStatus.EXPECTED_FAILURE, error_text, error_text

        return False, OutcomeStatus.UNEXPECTED_FAILURE, error_text or "Unknown failure", error_text or None

    async def _run_single_case(self, case: ToolTestCase) -> ToolExecutionResult:
        """Run a single test case with context-aware backend patching."""
        assert self.context is not None

        # Prepare backend switching context
        backend_patches = contextlib.ExitStack()
        
        # Determine if we need to switch WhatsApp backend
        backend = case.metadata.get("backend")
        
        if backend == "twilio":
            try:
                import voice_agent.tools.whatsapp as whatsapp
                
                # Force _USE_EVOLUTION to False
                backend_patches.enter_context(patch.object(whatsapp, "_USE_EVOLUTION", False))
                # Swap implementation references to Twilio versions
                tools_to_patch = [
                    ("send_text_message", "send_text_message_twilio"),
                    ("send_media_message", "send_media_message_twilio"),
                    ("send_image", "send_image_twilio"),
                    ("send_video", "send_video_twilio"),
                    ("send_document", "send_document_twilio"),
                    ("send_audio", "send_audio_twilio"),
                    ("send_location", "send_location_twilio"),
                    ("send_generated_audio", "send_generated_audio_twilio"),
                    ("send_template_message", "send_template_message_twilio"),
                ]
                
                # We need to map tool_name to the async Twilio wrappers
                tool_map = {
                    "send_whatsapp_message": "send_text_message_async_twilio",
                    "send_whatsapp_image": "send_image_async_twilio",
                    "send_whatsapp_video": "send_video_async_twilio",
                    "send_whatsapp_document": "send_document_async_twilio",
                    "send_whatsapp_audio": "send_audio_async_twilio",
                    "send_whatsapp_location": "send_location_async_twilio",
                    "send_whatsapp_generated_audio": "send_generated_audio_twilio",
                    "send_whatsapp_template": "send_template_message_async_twilio",
                }
                
                # 1. Patch module level functions
                for original, fallback in tools_to_patch:
                    if hasattr(whatsapp, fallback):
                        backend_patches.enter_context(patch.object(whatsapp, original, getattr(whatsapp, fallback)))
                
                # 2. Manually patch the handlers dictionary to bypass wrapper/mock issues
                original_handlers = {}
                for tool_name, func_name in tool_map.items():
                    if hasattr(whatsapp, func_name) and tool_name in self.handlers:
                        original_handlers[tool_name] = self.handlers[tool_name]
                        self.handlers[tool_name] = getattr(whatsapp, func_name)
                        # Ensure context sees it 
                        self.context.handlers[tool_name] = getattr(whatsapp, func_name)
                
                def restore_handlers():
                    for name, handler in original_handlers.items():
                        self.handlers[name] = handler
                        self.context.handlers[name] = handler
                        
                backend_patches.callback(restore_handlers)

            except ImportError:
                logger.warning("Could not patch WhatsApp backend to Twilio")

        # Activate patches
        backend_patches.__enter__()

        try:
            logger.info("▶️  %s — %s", case.tool_name, case.description)

            start_time = time.perf_counter()
            duration = 0.0

            # Run setup
            try:
                if case.setup:
                    await self.context.maybe_await(case.setup(self.context))
            except Exception as exc:
                duration = time.perf_counter() - start_time
                message = f"Setup failed: {exc}"
                logger.exception("Setup failed for %s", case.tool_name)
                return ToolExecutionResult(
                    tool_name=case.tool_name,
                    arguments=case.arguments,
                    raw_result={"success": False, "error": message},
                    success=False,
                    status=OutcomeStatus.UNEXPECTED_FAILURE,
                    message=message,
                    duration_s=duration,
                    error=message,
                )

            # Get arguments
            arguments = case.arguments
            try:
                if case.arguments_factory is not None:
                    arguments = await self.context.maybe_await(case.arguments_factory(self.context))
                raw_result = await self.context.call_tool(case.tool_name, arguments)
                duration = time.perf_counter() - start_time
            except Exception as exc:
                duration = time.perf_counter() - start_time
                message = f"Tool invocation raised exception: {exc}"
                logger.exception("Tool %s raised an exception", case.tool_name)
                result = ToolExecutionResult(
                    tool_name=case.tool_name,
                    arguments=arguments,
                    raw_result={"success": False, "error": message},
                    success=False,
                    status=OutcomeStatus.UNEXPECTED_FAILURE,
                    message=message,
                    duration_s=duration,
                    error=message,
                )
                if case.teardown:
                    try:
                        await self.context.maybe_await(case.teardown(self.context, result))
                    except Exception:
                        logger.exception("Teardown failed for %s", case.tool_name)
                return result

            # Determine success
            success, status, message, error_text = self._determine_success(
                raw_result,
                expect_success=case.expect_success,
                allowed_errors=case.allow_error_substrings,
            )

            result = ToolExecutionResult(
                tool_name=case.tool_name,
                arguments=arguments,
                raw_result=raw_result,
                success=success,
                status=status,
                message=message,
                duration_s=duration,
                error=error_text,
            )

            # Run validator
            if case.validator and status != OutcomeStatus.UNEXPECTED_FAILURE:
                try:
                    await self.context.maybe_await(case.validator(self.context, result))
                except AssertionError as assertion_error:
                    result.success = False
                    result.status = OutcomeStatus.UNEXPECTED_FAILURE
                    result.message = f"Validator assertion failed: {assertion_error}"
                    result.error = str(assertion_error)
                except Exception as exc:
                    result.success = False
                    result.status = OutcomeStatus.UNEXPECTED_FAILURE
                    result.message = f"Validator raised exception: {exc}"
                    result.error = str(exc)

            # Run teardown
            if case.teardown:
                try:
                    await self.context.maybe_await(case.teardown(self.context, result))
                except Exception as exc:
                    logger.exception("Teardown raised for %s: %s", case.tool_name, exc)
                    if result.status == OutcomeStatus.SUCCESS:
                        result.status = OutcomeStatus.UNEXPECTED_FAILURE
                        result.success = False
                        result.message = f"Teardown raised exception: {exc}"
                        result.error = str(exc)

            # Log result
            icon = {
                OutcomeStatus.SUCCESS: "✅",
                OutcomeStatus.EXPECTED_FAILURE: "⚠️",
                OutcomeStatus.UNEXPECTED_FAILURE: "❌",
                OutcomeStatus.SKIPPED: "⏭️",
            }.get(result.status, "ℹ️")

            log_fn = (
                logger.info
                if result.status in {OutcomeStatus.SUCCESS, OutcomeStatus.EXPECTED_FAILURE, OutcomeStatus.SKIPPED}
                else logger.error
            )
            log_fn("%s %s [%s] (%.3fs)", icon, case.tool_name, result.status.value.upper(), result.duration_s)

            return result
            
        finally:
            backend_patches.close()


__all__ = ["PipecatToolTestHarness"]
