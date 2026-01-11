#!/usr/bin/env python3
"""Run the Pipecat tool test harness.

This script runs tool tests against the Pipecat tools adapter using the
same test harness patterns as the original voice_agent_v4.

Usage:
    source pipecat_venv/bin/activate
    python scripts/run_pipecat_harness.py --suite utility
    python scripts/run_pipecat_harness.py --suite crm
    python scripts/run_pipecat_harness.py --suite all
    python scripts/run_pipecat_harness.py --tool get_current_time
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

# Setup path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


from rich.console import Console
from rich.table import Table

from tool_tests.harness import (
    PipecatToolTestHarness,
    SampleData,
    ToolTestCase,
    ToolExecutionResult,
    OutcomeStatus,
)
from tool_tests.harness.scenarios import (
    build_tool_test_cases,
    get_utility_cases,
    get_crm_cases,
    get_communication_cases,
    get_communication_twilio_cases,
    get_calendar_cases,
)

console = Console()
logger = logging.getLogger("pipecat_harness")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pipecat tool test harness")
    parser.add_argument(
        "--suite",
        choices=["utility", "crm", "communication", "communication_twilio", "calendar", "all"],
        default="utility",
        help="Test suite to run (default: utility)",
    )
    parser.add_argument(
        "--tool",
        type=str,
        help="Run a specific tool by name",
    )
    parser.add_argument(
        "--args",
        type=str,
        help="JSON arguments for --tool",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write results to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Log level (default: WARNING)",
    )
    return parser.parse_args(argv)


def get_test_cases(suite: str) -> List[ToolTestCase]:
    """Get test cases for a suite."""
    if suite == "utility":
        return get_utility_cases()
    elif suite == "crm":
        return get_crm_cases()
    elif suite == "communication":
        return get_communication_cases()
    elif suite == "communication_twilio":
        return get_communication_twilio_cases()
    elif suite == "calendar":
        return get_calendar_cases()
    elif suite == "all":
        sample = SampleData.load_defaults()
        return build_tool_test_cases(sample)
    return []


def print_results(results: List[ToolExecutionResult]) -> None:
    """Print results as a table."""
    table = Table(title="Test Results")
    table.add_column("Tool", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Duration", justify="right")
    table.add_column("Message")

    for result in results:
        status_style = {
            OutcomeStatus.SUCCESS: "green",
            OutcomeStatus.EXPECTED_FAILURE: "yellow",
            OutcomeStatus.UNEXPECTED_FAILURE: "red",
            OutcomeStatus.SKIPPED: "dim",
        }.get(result.status, "")
        
        icon = {
            OutcomeStatus.SUCCESS: "✅",
            OutcomeStatus.EXPECTED_FAILURE: "⚠️",
            OutcomeStatus.UNEXPECTED_FAILURE: "❌",
            OutcomeStatus.SKIPPED: "⏭️",
        }.get(result.status, "")
        
        status_text = f"{icon} {result.status.value.upper()}"
        duration = f"{result.duration_s*1000:.1f}ms"
        message = result.message[:50] + "..." if len(result.message) > 50 else result.message
        
        table.add_row(
            result.tool_name,
            f"[{status_style}]{status_text}[/{status_style}]",
            duration,
            message,
        )

    console.print(table)


def print_summary(results: List[ToolExecutionResult]) -> None:
    """Print summary statistics."""
    total = len(results)
    passed = sum(1 for r in results if r.status == OutcomeStatus.SUCCESS)
    expected_fail = sum(1 for r in results if r.status == OutcomeStatus.EXPECTED_FAILURE)
    failed = sum(1 for r in results if r.status == OutcomeStatus.UNEXPECTED_FAILURE)
    total_time = sum(r.duration_s for r in results) * 1000

    console.print()
    console.print(f"[bold]Summary:[/bold] {passed}/{total} passed, {expected_fail} expected failures, {failed} unexpected failures")
    console.print(f"[bold]Total time:[/bold] {total_time:.1f}ms")


async def run_harness(args: argparse.Namespace) -> int:
    """Run the test harness."""
    async with PipecatToolTestHarness() as harness:
        
        if args.tool:
            # Single tool test
            tool_args = {}
            if args.args:
                try:
                    tool_args = json.loads(args.args)
                except json.JSONDecodeError as e:
                    console.print(f"[red]Error parsing --args: {e}[/red]")
                    return 1
            
            console.print(f"[bold]Testing tool:[/bold] {args.tool}")
            result = await harness.run_single_tool(args.tool, tool_args)
            results = [result]
            
        else:
            # Run suite
            cases = get_test_cases(args.suite)
            if not cases:
                console.print(f"[yellow]No test cases found for suite: {args.suite}[/yellow]")
                return 0
            
            console.print(f"[bold]Running suite:[/bold] {args.suite} ({len(cases)} tests)")
            console.print()
            results = await harness.run_cases(cases)
        
        print_results(results)
        print_summary(results)
        
        # Save to file if requested
        if args.output:
            output_data = [
                {
                    "tool_name": r.tool_name,
                    "status": r.status.value,
                    "success": r.success,
                    "duration_s": r.duration_s,
                    "message": r.message,
                    "error": r.error,
                }
                for r in results
            ]
            args.output.write_text(json.dumps(output_data, indent=2))
            console.print(f"\n[dim]Results saved to {args.output}[/dim]")
        
        # Return exit code based on results
        unexpected = sum(1 for r in results if r.status == OutcomeStatus.UNEXPECTED_FAILURE)
        return 1 if unexpected > 0 else 0


def main(argv: Sequence[str] = ()) -> int:
    args = parse_args(argv or sys.argv[1:])
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s: %(message)s",
    )
    
    return asyncio.run(run_harness(args))


if __name__ == "__main__":
    sys.exit(main())
