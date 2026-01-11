from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Set

from tool_tests.logger import configure_logger

from .context import OutcomeStatus, ToolExecutionResult
from .executor import PipecatToolTestHarness as ToolTestHarness
from .scenarios import build_tool_test_cases


async def _run_schema_validation() -> None:
    from tool_tests.tool_validator import run_validation

    await run_validation()


def _summarize(results: Sequence[ToolExecutionResult]) -> dict[str, int]:
    counter = Counter(result.status.value for result in results)
    return dict(counter)


def _filter_tools_arg(tool_args: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if not tool_args:
        return None
    return {name.strip() for entry in tool_args for name in entry.split(",") if name.strip()}


async def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run MCP tool validation harness.")
    parser.add_argument(
        "--tool",
        action="append",
        dest="tools",
        help="Only run specified tool(s). Can be used multiple times or comma separated.",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip Responses API schema validation.",
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Emit machine-readable JSON summary to stdout.",
    )
    args = parser.parse_args(argv)

    requested_tools = _filter_tools_arg(args.tools)
    logger = configure_logger("tool_tests.runner")
    repo_root = Path(__file__).resolve().parents[2]

    if not args.skip_schema and (requested_tools is None or requested_tools):
        logger.info("Running OpenAI Responses schema validation…")
        await _run_schema_validation()

    async with ToolTestHarness() as harness:
        if harness.sample_data is None:
            raise RuntimeError("Harness failed to load sample data.")

        cases = build_tool_test_cases(
            harness.sample_data,
            include=requested_tools,
        )
        if not cases:
            logger.warning("No tool test cases matched the provided filters.")
            return 0

        logger.info("Executing %d tool scenario(s)…", len(cases))
        results = await harness.run_cases(cases)

    summary = _summarize(results)
    total = len(results)
    success_count = sum(1 for r in results if r.status == OutcomeStatus.SUCCESS)
    unexpected_failures = [r for r in results if r.status == OutcomeStatus.UNEXPECTED_FAILURE]

    logger.info("Tool execution summary: %s", summary)
    logger.info("Successful scenarios: %d / %d", success_count, total)

    if unexpected_failures:
        logger.error("Unexpected failures detected:")
        for failure in unexpected_failures:
            logger.error(
                " - %s (%s): %s",
                failure.tool_name,
                failure.arguments,
                failure.message,
            )

    if args.summary_json:
        payload = {
            "total": total,
            "summary": summary,
            "results": [
                {
                    "tool": result.tool_name,
                    "status": result.status.value,
                    "success": result.success,
                    "message": result.message,
                    "arguments": result.arguments,
                    "error": result.error,
                    "duration_s": result.duration_s,
                }
                for result in results
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if not unexpected_failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
