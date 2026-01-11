"""
Simple MCP Tools Server for Voice Agent v4

Exposes useful tools that the LLM can invoke during conversations.
Uses the Model Context Protocol (MCP) specification.
"""

import asyncio
import datetime
import importlib.util
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("voice_agent_v4.mcp_tools")


class MCPToolsServer:
    """
    Simple MCP Tools Server that exposes callable tools to LLMs.
    
    Tools are discovered and invoked by the LLM during conversations.
    Each tool has a name, description, and parameter schema.
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.mcp_manager = None
        self._register_default_tools()
        self._load_custom_tools()
        logger.info("MCP Tools Server initialized with %d tools", len(self.tools))
    
    def _register_default_tools(self):
        """Register built-in tools."""
        self.register_tool(
            name="get_current_time",
            description="Get the current date and time",
            parameters={},
            handler=self._get_current_time
        )
        
        self.register_tool(
            name="calculate",
            description="Perform basic arithmetic calculations (add, subtract, multiply, divide)",
            parameters={
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            required=["operation", "a", "b"],
            handler=self._calculate
        )
        
        self.register_tool(
            name="get_system_info",
            description="Get information about the system (OS, Python version, etc.)",
            parameters={},
            handler=self._get_system_info
        )
        
        self.register_tool(
            name="format_date",
            description="Format a date string in different formats",
            parameters={
                "date_string": {
                    "type": "string",
                    "description": "Date string to format (YYYY-MM-DD)"
                },
                "format": {
                    "type": "string",
                    "enum": ["full", "short", "iso", "brazilian"],
                    "description": "Output format"
                }
            },
            required=["date_string", "format"],
            handler=self._format_date
        )
    
    def _load_custom_tools(self) -> None:
        """Load custom tools from same directory as this module."""
        # Tools are in the same directory as this server.py
        tools_dir = Path(__file__).parent
        
        if not tools_dir.exists():
            logger.debug("Tools directory not found: %s", tools_dir)
            return
        
        python_files = list(tools_dir.glob("*.py"))
        if not python_files:
            logger.debug("No custom tool files found in %s", tools_dir)
            return
        
        logger.info("Loading custom tools from %s", tools_dir)
        
        for py_file in python_files:
            if py_file.name.startswith("_"):
                continue
            # Skip this file and __init__.py
            if py_file.name in ("server.py", "__init__.py"):
                continue
            
            try:
                self._load_tool_from_file(py_file)
            except Exception as e:
                logger.error("Failed to load tools from %s: %s", py_file.name, e, exc_info=True)
    
    def _load_tool_from_file(self, file_path: Path) -> None:
        """Load tool definitions from a Python file."""
        module_name = f"voice_agent.tools.{file_path.stem}"
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            logger.warning("Could not load spec for %s", file_path)
            return
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        definitions = []
        if hasattr(module, "TOOL_DEFINITION"):
            definitions = [module.TOOL_DEFINITION]
        elif hasattr(module, "TOOL_DEFINITIONS"):
            definitions = module.TOOL_DEFINITIONS
        else:
            logger.warning("No TOOL_DEFINITION or TOOL_DEFINITIONS found in %s", file_path.name)
            return
        
        for tool_def in definitions:
            try:
                self.register_tool(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    parameters=tool_def.get("parameters", {}),
                    handler=tool_def["handler"],
                    required=tool_def.get("required")
                )
                logger.info("Loaded custom tool: %s from %s", tool_def["name"], file_path.name)
            except Exception as e:
                logger.error("Failed to register tool from %s: %s", file_path.name, e)
    
    async def _load_external_mcp_servers(self) -> None:
        """Load tools from external MCP servers via mcp_client"""
        try:
            from mcp_client import get_mcp_manager  # type: ignore
            
            self.mcp_manager = get_mcp_manager()
            await self.mcp_manager.load_config()
            
            external_tools = self.mcp_manager.get_all_tools()
            
            for tool in external_tools:
                server_name = tool.pop("_mcp_server", "unknown")
                tool_name = tool.get("name")
                
                if not tool_name:
                    logger.warning("External tool missing name from %s", server_name)
                    continue
                
                def make_handler(tn: str):
                    async def handler(**kwargs: Any) -> Any:
                        if self.mcp_manager is None:
                            return {"error": "MCP manager not initialized"}
                        result = await self.mcp_manager.call_tool(tn, kwargs)
                        if result.get("success"):
                            return result.get("result")
                        else:
                            return {"error": result.get("error")}
                    return handler
                
                self.register_tool(
                    name=tool_name,
                    description=tool.get("description", "External MCP tool"),
                    parameters=tool.get("inputSchema", {}).get("properties", {}),
                    handler=make_handler(tool_name),
                    required=tool.get("inputSchema", {}).get("required", [])
                )
                
                logger.info("Loaded external MCP tool: %s from %s", tool_name, server_name)
            
            logger.info("Loaded %d external MCP tools", len(external_tools))
            
        except ImportError:
            logger.debug("mcp_client not available, skipping external MCP servers")
        except Exception as e:
            logger.error("Failed to load external MCP servers: %s", e, exc_info=True)
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Any,  # Callable type
        required: List[str] | None = None
    ):
        """
        Register a new tool.
        
        Args:
            name: Tool name (unique identifier)
            description: Human-readable description
            parameters: JSON Schema for parameters
            handler: Function to execute when tool is called
            required: List of required parameter names
        """
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": parameters,
            "additionalProperties": False,
        }

        if required:
            schema["required"] = required

        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": schema,
            "handler": handler,
        }
        logger.info("Registered tool: %s", name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of all available tools.
        
        Returns:
            List of tool definitions (name, description, parameters)
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in self.tools.values()
        ]
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool by name with arguments.
        
        Args:
            name: Tool name
            arguments: Tool arguments as dictionary
            
        Returns:
            Tool execution result
        """
        if name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {name}"
            }
        
        tool = self.tools[name]
        
        try:
            # Validate required parameters
            required = tool["parameters"].get("required", [])
            for param in required:
                if param not in arguments:
                    return {
                        "success": False,
                        "error": f"Missing required parameter: {param}"
                    }
            
            # Call handler (handle both sync and async)
            result = tool["handler"](**arguments)
            
            # If result is a coroutine, run it
            if asyncio.iscoroutine(result):
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context
                    raise RuntimeError(
                        "Cannot call async tool from sync context - use call_tool_async"
                    )
                else:
                    result = loop.run_until_complete(result)
            
            return {
                "success": True,
                "result": result
            }
        
        except Exception as e:
            logger.error("Tool %s execution failed: %s", name, e, exc_info=True)
            return {
                "success": False,
                "error": f"Tool execution error: {str(e)}"
            }
    
    async def call_tool_async(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool by name with arguments (async version).
        
        Args:
            name: Tool name
            arguments: Tool arguments as dictionary
            
        Returns:
            Tool execution result
        """
        if name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {name}"
            }
        
        tool = self.tools[name]
        
        try:
            # Validate required parameters
            required = tool["parameters"].get("required", [])
            for param in required:
                if param not in arguments:
                    return {
                        "success": False,
                        "error": f"Missing required parameter: {param}"
                    }
            
            # Call handler
            result = tool["handler"](**arguments)
            
            # If result is a coroutine, await it
            if asyncio.iscoroutine(result):
                result = await result
            
            return {
                "success": True,
                "result": result
            }
        
        except Exception as e:
            logger.error("Tool %s execution failed: %s", name, e, exc_info=True)
            return {
                "success": False,
                "error": f"Tool execution error: {str(e)}"
            }
    
    # Tool Handlers
    
    def _get_current_time(self) -> str:
        """Get current date and time."""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    def _calculate(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """Perform arithmetic calculation."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None
        }
        
        if operation not in operations:
            return {"error": f"Unknown operation: {operation}"}
        
        result = operations[operation](a, b)
        
        if result is None:
            return {"error": "Division by zero"}
        
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        }
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
    
    def _format_date(self, date_string: str, format: str) -> str:
        """Format a date string."""
        try:
            date = datetime.datetime.strptime(date_string, "%Y-%m-%d")
            
            formats = {
                "full": "%A, %B %d, %Y",
                "short": "%m/%d/%Y",
                "iso": "%Y-%m-%d",
                "brazilian": "%d/%m/%Y"
            }
            
            if format not in formats:
                return f"Unknown format: {format}"
            
            return date.strftime(formats[format])
        
        except ValueError as e:
            return f"Invalid date string: {str(e)}"


# Global instance
_mcp_server = None
_mcp_server_initialized = False


def get_mcp_server() -> MCPToolsServer:
    """Get or create global MCP tools server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPToolsServer()
    return _mcp_server


async def get_mcp_server_async() -> MCPToolsServer:
    """Get or create global MCP tools server instance with async initialization."""
    global _mcp_server, _mcp_server_initialized
    if _mcp_server is None:
        _mcp_server = MCPToolsServer()
    
    if not _mcp_server_initialized:
        await _mcp_server._load_external_mcp_servers()
        _mcp_server_initialized = True
    
    return _mcp_server


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    server = MCPToolsServer()
    
    # List tools
    print("Available tools:")
    for tool in server.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Test tools
    print("\nTesting tools:")
    
    result = server.call_tool("get_current_time", {})
    print(f"Current time: {result}")
    
    result = server.call_tool("calculate", {"operation": "add", "a": 5, "b": 3})
    print(f"Calculate 5 + 3: {result}")
    
    result = server.call_tool("get_system_info", {})
    print(f"System info: {json.dumps(result, indent=2)}")
    
    result = server.call_tool("format_date", {"date_string": "2025-10-06", "format": "brazilian"})
    print(f"Format date: {result}")
