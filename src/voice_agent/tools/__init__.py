# MCP Tools package
"""
MCP Tools for Pipecat Voice Agent

This package provides all tool implementations:
- CRM tools (customer, facts, knowledge, tasks, calls)
- Calendar tools (Google Calendar)
- Communication tools (WhatsApp, Email)
"""

from typing import TYPE_CHECKING

from voice_agent.tools.server import MCPToolsServer, get_mcp_server

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

__all__ = ["MCPToolsServer", "get_mcp_server"]


# ──────────────────────────────────────────────────────────────────────────────
# Public API for Pipecat integration
# ──────────────────────────────────────────────────────────────────────────────

def get_all_tools():
    """Get all tool definitions in OpenAI function calling format."""
    server = get_mcp_server()
    tools = []
    for tool in server.list_tools():
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
        })
    return tools


def get_tool_handlers():
    """Get handler functions for all tools."""
    server = get_mcp_server()
    handlers = {}
    
    for tool in server.list_tools():
        tool_name = tool["name"]
        
        async def make_handler(name):
            async def handler(**kwargs):
                result = await server.call_tool_async(name, kwargs)
                if result.get("success"):
                    return result.get("result", {})
                return {"error": result.get("error", "Unknown error")}
            return handler
        
        import asyncio
        handlers[tool_name] = asyncio.get_event_loop().run_until_complete(
            make_handler(tool_name)
        ) if False else _create_handler(server, tool_name)
    
    return handlers


def _create_handler(server, tool_name):
    """Create an async handler for a specific tool.
    
    Args:
        server: The MCP server instance
        tool_name: Name of the tool to create a handler for
        
    Returns:
        An async function that accepts FunctionCallParams and calls the tool
    """
    async def handler(params: "FunctionCallParams"):
        """Handler that accepts Pipecat FunctionCallParams.
        
        Uses params.result_callback() to properly notify Pipecat of the result,
        which updates the LLM context and triggers the next LLM inference.
        """
        result = await server.call_tool_async(tool_name, params.arguments)
        if result.get("success"):
            await params.result_callback(result.get("result", {}))
        else:
            await params.result_callback({"error": result.get("error", "Unknown error")})
    
    handler.__name__ = f"handler_{tool_name}"
    return handler


def get_tool_count():
    """Get the number of available tools."""
    server = get_mcp_server()
    return len(server.list_tools())


# Essential tools subset for constrained contexts
ESSENTIAL_TOOL_NAMES = [
    "get_customer_by_phone",
    "create_customer",
    "update_customer_status",
    "get_customer_facts",
    "add_customer_fact",
    "list_calendar_attendees",
    "get_calendar_availability_for_attendee",
    "book_appointment",
    "cancel_calendar_event",
    "send_whatsapp_message",
    "end_call",
    "get_current_time",
]


def get_essential_tools():
    """Get only essential tools for reduced token usage."""
    all_tools = get_all_tools()
    return [t for t in all_tools if t.get("function", {}).get("name") in ESSENTIAL_TOOL_NAMES]


def get_essential_handlers():
    """Get handlers for essential tools only."""
    all_handlers = get_tool_handlers()
    return {n: h for n, h in all_handlers.items() if n in ESSENTIAL_TOOL_NAMES}
