"""Run individual tool tests for the Pipecat voice agent.

This script allows testing individual MCP tools or groups of tools
using the Pipecat tools adapter.

Usage:
    source venv/bin/activate
    python scripts/pipecat_tool_tests.py --tool get_current_time
    python scripts/pipecat_tool_tests.py --tool get_customer_by_phone --args '{"phone": "+5511999999999"}'
    python scripts/pipecat_tool_tests.py --category crm
    python scripts/pipecat_tool_tests.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _load_env(repo_root: Path) -> None:
    if load_dotenv is None:
        return
    env_path = repo_root / ".env"
    load_dotenv(dotenv_path=str(env_path), override=False)


_ensure_repo_on_path()

logger = logging.getLogger("pipecat_tool_tests")


# ──────────────────────────────────────────────────────────────────────────────
# Tool Categories
# ──────────────────────────────────────────────────────────────────────────────

TOOL_CATEGORIES = {
    "crm": [
        "get_customer_by_phone",
        "create_customer",
        "update_customer_by_phone",
        "update_customer_status",
        "get_customer_facts",
        "add_customer_fact",
        "get_facts_by_type",
        "delete_customer_fact",
    ],
    "calendar": [
        "list_calendar_attendees",
        "get_calendar_availability_for_attendee",
        "check_calendar_availability",
        "book_appointment",
        "create_calendar_event",
        "list_upcoming_calendar_events",
        "search_calendar_events",
        "get_calendar_event",
        "update_calendar_event",
        "cancel_calendar_event",
    ],
    "whatsapp": [
        "send_whatsapp_message",
        "send_whatsapp_image",
        "send_whatsapp_video",
        "send_whatsapp_audio",
        "send_whatsapp_document",
        "send_whatsapp_location",
        "send_whatsapp_template",
    ],
    "email": [
        "send_email",
        "send_bulk_email",
    ],
    "utility": [
        "get_current_time",
        "calculate",
        "get_system_info",
        "format_date",
        "get_weather",
    ],
    "calls": [
        "get_call_history",
        "get_last_call",
        "save_call_summary",
        "update_call_transcript",
        "get_calls_by_outcome",
    ],
    "tasks": [
        "create_todo",
        "list_todos",
        "complete_todo",
        "delete_todo",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Test Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolTestResult:
    tool_name: str
    success: bool
    duration_ms: float
    result: Any
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Tool Tester
# ──────────────────────────────────────────────────────────────────────────────

class PipecatToolTester:
    """Test harness for Pipecat tools."""
    
    def __init__(self) -> None:
        self.tools: Dict[str, Any] = {}
        self.handlers: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Load tools from Pipecat adapter."""
        if self._initialized:
            return
        
        from voice_agent.tools import get_all_tools, get_tool_handlers
        
        tools_list = get_all_tools()
        self.handlers = get_tool_handlers()
        
        for tool in tools_list:
            name = tool.get("function", {}).get("name")
            if name:
                self.tools[name] = tool
        
        self._initialized = True
        logger.info("Loaded %d tools", len(self.tools))
    
    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        self.initialize()
        return sorted(self.tools.keys())
    
    def list_tools_by_category(self) -> Dict[str, List[str]]:
        """Get tools grouped by category."""
        self.initialize()
        result = {}
        for category, tool_names in TOOL_CATEGORIES.items():
            available = [t for t in tool_names if t in self.tools]
            if available:
                result[category] = available
        
        # Add uncategorized
        all_categorized = set()
        for tools in TOOL_CATEGORIES.values():
            all_categorized.update(tools)
        
        uncategorized = [t for t in self.tools if t not in all_categorized]
        if uncategorized:
            result["other"] = sorted(uncategorized)
        
        return result
    
    async def test_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolTestResult:
        """Test a single tool."""
        self.initialize()
        
        if tool_name not in self.handlers:
            return ToolTestResult(
                tool_name=tool_name,
                success=False,
                duration_ms=0,
                result=None,
                error=f"Tool not found: {tool_name}",
            )
        
        handler = self.handlers[tool_name]
        args = arguments or {}
        
        start = time.perf_counter()
        try:
            result = await handler(**args)
            duration = (time.perf_counter() - start) * 1000
            
            # Check if result indicates error
            if isinstance(result, dict) and result.get("error"):
                return ToolTestResult(
                    tool_name=tool_name,
                    success=False,
                    duration_ms=duration,
                    result=result,
                    error=result.get("error"),
                )
            
            return ToolTestResult(
                tool_name=tool_name,
                success=True,
                duration_ms=duration,
                result=result,
            )
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return ToolTestResult(
                tool_name=tool_name,
                success=False,
                duration_ms=duration,
                result=None,
                error=str(e),
            )
    
    async def test_category(self, category: str) -> List[ToolTestResult]:
        """Test all tools in a category."""
        self.initialize()
        
        tool_names = TOOL_CATEGORIES.get(category, [])
        if not tool_names:
            logger.warning("Unknown category: %s", category)
            return []
        
        results = []
        for name in tool_names:
            if name in self.handlers:
                result = await self.test_tool(name)
                results.append(result)
        
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ──────────────────────────────────────────────────────────────────────────────

def print_result(result: ToolTestResult, verbose: bool = False) -> None:
    """Print a test result."""
    icon = "✅" if result.success else "❌"
    status = "PASS" if result.success else "FAIL"
    
    print(f"{icon} {result.tool_name} [{status}] ({result.duration_ms:.1f}ms)")
    
    if result.error:
        print(f"   Error: {result.error}")
    
    if verbose and result.result:
        result_str = json.dumps(result.result, indent=2, default=str)[:500]
        print(f"   Result: {result_str}")


def print_summary(results: List[ToolTestResult]) -> None:
    """Print test summary."""
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    total_time = sum(r.duration_ms for r in results)
    
    print()
    print("=" * 50)
    print(f"Summary: {passed}/{len(results)} passed, {failed} failed")
    print(f"Total time: {total_time:.1f}ms")
    print("=" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipecat tool tester")
    parser.add_argument(
        "--tool",
        help="Specific tool to test",
    )
    parser.add_argument(
        "--args",
        help="JSON string of arguments for the tool",
    )
    parser.add_argument(
        "--category",
        choices=list(TOOL_CATEGORIES.keys()),
        help="Test all tools in a category",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tools",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed results",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Python log level",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    tester = PipecatToolTester()
    
    if args.list:
        categories = tester.list_tools_by_category()
        print("\nAvailable Tools by Category:\n")
        for category, tools in categories.items():
            print(f"  {category.upper()}:")
            for tool in tools:
                print(f"    - {tool}")
            print()
        return 0
    
    if args.tool:
        # Test single tool
        tool_args = {}
        if args.args:
            try:
                tool_args = json.loads(args.args)
            except json.JSONDecodeError as e:
                print(f"Error parsing --args: {e}")
                return 1
        
        print(f"\nTesting tool: {args.tool}")
        print("-" * 40)
        
        result = await tester.test_tool(args.tool, tool_args)
        print_result(result, verbose=args.verbose)
        
        return 0 if result.success else 1
    
    if args.category:
        # Test category
        print(f"\nTesting category: {args.category}")
        print("-" * 40)
        
        results = await tester.test_category(args.category)
        for result in results:
            print_result(result, verbose=args.verbose)
        
        print_summary(results)
        
        failed = sum(1 for r in results if not r.success)
        return 1 if failed > 0 else 0
    
    # Default: test utility tools
    print("\nNo tool/category specified. Testing utility tools...")
    print("-" * 40)
    
    results = await tester.test_category("utility")
    for result in results:
        print_result(result, verbose=args.verbose)
    
    print_summary(results)
    return 0


def main() -> int:
    repo_root = _ensure_repo_on_path()
    _load_env(repo_root)
    
    args = _parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
