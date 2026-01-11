#!/usr/bin/env python3
"""Interactive text-based chat with the Pipecat agent.

This script replaces Twilio audio with terminal text input, allowing you
to chat with the full agent (all 63 tools) via keyboard instead of voice.

Usage:
    source pipecat_venv/bin/activate
    python scripts/pipecat_chat.py
    python scripts/pipecat_chat.py --caller-phone +5511999999999
    python scripts/pipecat_chat.py --log-file chat_session.log

Commands:
    /quit, /exit    - Exit the chat
    /history        - Show conversation history
    /tools          - List available tools
    /context        - Show current context
    /clear          - Clear conversation history
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

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
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()
logger = logging.getLogger("pipecat_chat")


class PipecatChatAgent:
    """Interactive text-based agent using Pipecat LLM with all tools."""
    
    def __init__(
        self,
        caller_phone: str = "",
        log_file: Optional[Path] = None,
    ) -> None:
        self.caller_phone = caller_phone
        self.call_sid = f"chat-{uuid.uuid4().hex[:8]}"
        self.log_file = log_file
        self.log_handle: Optional[TextIO] = None
        
        self.llm = None
        self.context = None
        self.tools: List[Dict] = []
        self.handlers: Dict[str, Any] = {}
        self.customer_context = None
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the agent with LLM, tools, and customer context."""
        if self._initialized:
            return
        
        console.print("[dim]Initializing Pipecat agent...[/dim]")
        
        # Open log file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = self.log_file.open("a", encoding="utf-8")
            self.log_handle.write(f"\n=== Session {self.call_sid} ===\n")
        
        # Import Pipecat components
        from voice_agent.config import SETTINGS
        from voice_agent.tools import get_all_tools, get_tool_handlers
        from voice_agent.context import (
            prefetch_customer_context,
            build_system_prompt_with_context,
        )
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        
        # Store settings for later use
        self._settings = SETTINGS
        
        # Load tools
        self.tools = get_all_tools()
        self.handlers = get_tool_handlers()
        console.print(f"[green]✓ Loaded {len(self.tools)} tools[/green]")
        
        # Prefetch customer context
        if self.caller_phone:
            try:
                self.customer_context = await asyncio.wait_for(
                    prefetch_customer_context(self.caller_phone),
                    timeout=5.0,
                )
                if self.customer_context.is_known:
                    console.print(f"[green]✓ Customer: {self.customer_context.name}[/green]")
                else:
                    console.print("[yellow]⚠ Unknown customer[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠ Prefetch failed: {e}[/yellow]")
        
        # Build system prompt
        base_prompt = SETTINGS.get_system_prompt()
        system_prompt = build_system_prompt_with_context(
            base_prompt,
            customer_context=self.customer_context,
            caller_phone=self.caller_phone,
        )
        
        # Initialize LLM
        self.llm = OpenAILLMService(
            api_key=SETTINGS.openai_api_key,
            model=SETTINGS.openai_model,
            base_url=SETTINGS.openai_base_url,
        )
        
        # Register tool handlers
        for tool in self.tools:
            tool_name = tool.get("function", {}).get("name")
            if tool_name and tool_name in self.handlers:
                self.llm.register_function(tool_name, self.handlers[tool_name])
        
        # Initialize context
        messages = [{"role": "system", "content": system_prompt}]
        self.context = OpenAILLMContext(messages=messages, tools=self.tools)
        
        console.print(f"[green]✓ LLM: {SETTINGS.openai_model}[/green]")
        console.print(f"[dim]Call SID: {self.call_sid}[/dim]")
        
        self._initialized = True
    
    async def chat(self, user_message: str) -> str:
        """Send a message and get the agent's response."""
        if not self._initialized:
            await self.initialize()
        
        # Log user message
        if self.log_handle:
            self.log_handle.write(f"[user] {user_message}\n")
            self.log_handle.flush()
        
        # Add user message to context
        self.context.add_message({"role": "user", "content": user_message})
        
        # Get LLM response
        response_text = ""
        tool_calls_made = []
        
        try:
            # Process through LLM - this handles tool calls automatically
            async for chunk in self._process_with_tools():
                response_text += chunk
        except Exception as e:
            response_text = f"[Error: {e}]"
            logger.exception("LLM error")
        
        # Add assistant response to context
        if response_text:
            self.context.add_message({"role": "assistant", "content": response_text})
            if self.log_handle:
                self.log_handle.write(f"[assistant] {response_text}\n")
                self.log_handle.flush()
        
        return response_text
    
    async def _process_with_tools(self) -> Any:
        """Process LLM response, handling any tool calls."""
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url or None,
        )
        
        messages = self.context.messages.copy()
        
        while True:
            # Get completion
            response = await client.chat.completions.create(
                model=self._settings.openai_model,
                messages=messages,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
            )
            
            choice = response.choices[0]
            message = choice.message
            
            # Check for tool calls
            if message.tool_calls:
                # Add assistant message with tool calls
                messages.append(message.model_dump())
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    
                    console.print(f"[dim]→ Calling {tool_name}({json.dumps(args, ensure_ascii=False)[:60]}...)[/dim]")
                    
                    # Execute tool
                    if tool_name in self.handlers:
                        try:
                            result = await self.handlers[tool_name](**args)
                            result_str = json.dumps(result, ensure_ascii=False, default=str)
                        except Exception as e:
                            result_str = json.dumps({"error": str(e)})
                    else:
                        result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})
                    
                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
                    
                    if self.log_handle:
                        self.log_handle.write(f"[tool:{tool_name}] {result_str[:200]}\n")
                
                # Continue loop to get final response
                continue
            
            # No tool calls - return content
            if message.content:
                yield message.content
            break
    
    def show_history(self) -> None:
        """Display conversation history."""
        console.print("\n[bold]Conversation History[/bold]")
        console.print("─" * 40)
        for msg in self.context.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                console.print(f"[dim]system: {content[:100]}...[/dim]")
            elif role == "user":
                console.print(f"[cyan]you:[/cyan] {content}")
            elif role == "assistant":
                console.print(f"[green]agent:[/green] {content}")
        console.print("─" * 40 + "\n")
    
    def show_tools(self) -> None:
        """Display available tools."""
        console.print("\n[bold]Available Tools[/bold]")
        console.print("─" * 40)
        for i, tool in enumerate(self.tools[:30], 1):
            name = tool.get("function", {}).get("name", "?")
            desc = tool.get("function", {}).get("description", "")[:50]
            console.print(f"  {i:2}. [cyan]{name}[/cyan] - {desc}")
        if len(self.tools) > 30:
            console.print(f"  ... and {len(self.tools) - 30} more")
        console.print("─" * 40 + "\n")
    
    def show_context(self) -> None:
        """Display current context info."""
        console.print("\n[bold]Current Context[/bold]")
        console.print("─" * 40)
        console.print(f"Call SID: {self.call_sid}")
        console.print(f"Caller: {self.caller_phone or 'Not set'}")
        if self.customer_context:
            console.print(f"Customer: {self.customer_context.name or 'Unknown'}")
            console.print(f"Customer ID: {self.customer_context.customer_id or 'N/A'}")
        console.print(f"Messages: {len(self.context.messages)}")
        console.print(f"Tools: {len(self.tools)}")
        console.print("─" * 40 + "\n")
    
    def clear_history(self) -> None:
        """Clear conversation history (keep system message)."""
        if self.context and self.context.messages:
            system_msg = self.context.messages[0]
            self.context.messages.clear()
            self.context.messages.append(system_msg)
        console.print("[yellow]Conversation cleared.[/yellow]")
    
    def close(self) -> None:
        """Close the agent."""
        if self.log_handle:
            self.log_handle.close()


async def run_chat(args: argparse.Namespace) -> None:
    """Run the interactive chat loop."""
    
    log_path = Path(args.log_file) if args.log_file else None
    agent = PipecatChatAgent(
        caller_phone=args.caller_phone or "",
        log_file=log_path,
    )
    
    try:
        await agent.initialize()
        
        console.print()
        console.print(Panel(
            "[bold]Pipecat Interactive Chat[/bold]\n"
            "Type your message and press Enter.\n"
            "Commands: /quit, /history, /tools, /context, /clear",
            title="Ready",
            border_style="green",
        ))
        console.print()
        
        while True:
            try:
                user_input = Prompt.ask("[cyan]you[/cyan]")
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /quit to exit[/yellow]")
                continue
            
            user_input = user_input.strip()
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in {"/quit", "/exit", ":q", ":quit"}:
                console.print("[dim]Goodbye![/dim]")
                break
            
            if user_input.lower() == "/history":
                agent.show_history()
                continue
            
            if user_input.lower() == "/tools":
                agent.show_tools()
                continue
            
            if user_input.lower() == "/context":
                agent.show_context()
                continue
            
            if user_input.lower() == "/clear":
                agent.clear_history()
                continue
            
            if user_input.startswith("/"):
                console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                continue
            
            # Get agent response
            console.print("[dim]Thinking...[/dim]", end="\r")
            start = time.time()
            response = await agent.chat(user_input)
            duration = time.time() - start
            
            # Display response
            console.print(" " * 20, end="\r")  # Clear "Thinking..."
            console.print(f"[green]agent[/green] ({duration:.1f}s):")
            console.print(Markdown(response))
            console.print()
    
    finally:
        agent.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Pipecat chat")
    parser.add_argument(
        "--caller-phone",
        help="Caller phone number for CRM context (E.164 format)",
    )
    parser.add_argument(
        "--log-file",
        help="Path to save conversation transcript",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Python log level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    try:
        asyncio.run(run_chat(args))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")


if __name__ == "__main__":
    main()
