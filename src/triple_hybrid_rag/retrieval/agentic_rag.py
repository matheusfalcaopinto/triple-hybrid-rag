"""
Agentic RAG Implementation

Tool-using RAG that can decide which tools to use, when to retrieve,
and how to combine multiple information sources.

Reference:
- ReAct: "ReAct: Synergizing Reasoning and Acting in Language Models"
- Toolformer patterns
"""

from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ToolType(Enum):
    """Types of tools available to the agent."""
    SEARCH = "search"
    RETRIEVE = "retrieve"
    CALCULATE = "calculate"
    LOOKUP = "lookup"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    EXTRACT = "extract"

@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0

@dataclass
class AgentAction:
    """An action the agent decides to take."""
    thought: str
    action: str
    action_input: str

@dataclass
class AgentStep:
    """A step in the agent's reasoning."""
    action: AgentAction
    observation: str
    tool_result: Optional[ToolResult] = None

@dataclass 
class AgenticRAGConfig:
    """Configuration for Agentic RAG."""
    
    # Agent settings
    max_iterations: int = 5
    max_tool_calls: int = 10
    
    # Reasoning
    enable_chain_of_thought: bool = True
    enable_self_reflection: bool = True
    
    # Tool settings
    parallel_tools: bool = False
    tool_timeout_seconds: float = 30.0
    
    # Generation
    max_tokens: int = 1000
    temperature: float = 0.7

@dataclass
class AgenticRAGResult:
    """Result from agentic RAG."""
    query: str
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    total_iterations: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class Tool(ABC):
    """Base class for agent tools."""
    
    name: str
    description: str
    
    @abstractmethod
    async def execute(self, input_text: str) -> ToolResult:
        """Execute the tool with given input."""
        pass

class SearchTool(Tool):
    """Tool for semantic/vector search."""
    
    name = "search"
    description = "Search the knowledge base for relevant documents"
    
    def __init__(self, search_fn: Callable):
        self.search_fn = search_fn
    
    async def execute(self, input_text: str) -> ToolResult:
        import time
        start = time.time()
        try:
            results = await self.search_fn(input_text)
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=results,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000,
            )

class CalculateTool(Tool):
    """Tool for mathematical calculations."""
    
    name = "calculate"
    description = "Perform mathematical calculations"
    
    async def execute(self, input_text: str) -> ToolResult:
        import time
        start = time.time()
        try:
            # Safe evaluation (limited operations)
            result = self._safe_eval(input_text)
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000,
            )
    
    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate a mathematical expression."""
        import re
        # Only allow numbers and basic operators
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expr):
            raise ValueError("Invalid expression")
        return eval(expr)  # noqa: S307

class SummarizeTool(Tool):
    """Tool for text summarization."""
    
    name = "summarize"
    description = "Summarize text content"
    
    def __init__(self, llm_fn: Callable):
        self.llm_fn = llm_fn
    
    async def execute(self, input_text: str) -> ToolResult:
        import time
        start = time.time()
        try:
            prompt = f"Summarize the following text concisely:\n\n{input_text[:2000]}"
            result = await self.llm_fn(prompt)
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000,
            )

class AgenticRAG:
    """
    Agentic RAG with tool-using capabilities.
    
    Uses ReAct-style reasoning to:
    1. Think about what information is needed
    2. Decide which tool to use
    3. Execute the tool
    4. Observe the result
    5. Continue or provide final answer
    
    Usage:
        agent = AgenticRAG(
            llm_fn=generate_fn,
            tools=[SearchTool(search_fn), CalculateTool()],
        )
        result = await agent.run("What is the revenue growth rate?")
    """
    
    def __init__(
        self,
        llm_fn: Callable,
        tools: Optional[List[Tool]] = None,
        config: Optional[AgenticRAGConfig] = None,
    ):
        """Initialize Agentic RAG."""
        self.llm_fn = llm_fn
        self.tools = {t.name: t for t in (tools or [])}
        self.config = config or AgenticRAGConfig()
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent."""
        self.tools[tool.name] = tool
    
    async def run(self, query: str) -> AgenticRAGResult:
        """
        Run the agent to answer a query.
        
        Args:
            query: User query
            
        Returns:
            AgenticRAGResult with answer and reasoning steps
        """
        result = AgenticRAGResult(query=query, answer="")
        
        steps: List[AgentStep] = []
        iteration = 0
        tool_calls = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            
            # Generate next action
            action = await self._generate_action(query, steps)
            
            if action is None:
                break
            
            # Check if agent wants to finish
            if action.action.lower() in ["finish", "final answer", "answer"]:
                result.answer = action.action_input
                break
            
            # Execute tool
            if action.action.lower() in self.tools:
                if tool_calls >= self.config.max_tool_calls:
                    observation = "Maximum tool calls reached"
                else:
                    tool = self.tools[action.action.lower()]
                    tool_result = await asyncio.wait_for(
                        tool.execute(action.action_input),
                        timeout=self.config.tool_timeout_seconds,
                    )
                    tool_calls += 1
                    result.tools_used.append(action.action)
                    
                    if tool_result.success:
                        observation = self._format_result(tool_result.result)
                    else:
                        observation = f"Tool error: {tool_result.error}"
            else:
                observation = f"Unknown action: {action.action}"
            
            step = AgentStep(
                action=action,
                observation=observation,
            )
            steps.append(step)
        
        result.steps = steps
        result.total_iterations = iteration
        
        # If no explicit answer, generate one from observations
        if not result.answer:
            result.answer = await self._generate_final_answer(query, steps)
        
        return result
    
    async def _generate_action(
        self,
        query: str,
        steps: List[AgentStep],
    ) -> Optional[AgentAction]:
        """Generate next action using LLM."""
        # Build prompt
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        history = ""
        for step in steps:
            history += f"\nThought: {step.action.thought}"
            history += f"\nAction: {step.action.action}"
            history += f"\nAction Input: {step.action.action_input}"
            history += f"\nObservation: {step.observation}\n"
        
        prompt = f"""Answer the following question using the available tools.

Available Tools:
{tool_descriptions}
- finish: Provide the final answer

Question: {query}
{history}
Think step by step. Use this format:
Thought: [your reasoning about what to do next]
Action: [the tool to use or 'finish']
Action Input: [input to the tool or final answer]

"""
        
        try:
            response = await self.llm_fn(prompt)
            return self._parse_action(response)
        except Exception as e:
            logger.error(f"Action generation failed: {e}")
            return None
    
    def _parse_action(self, response: str) -> Optional[AgentAction]:
        """Parse LLM response into action."""
        import re
        
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        action_match = re.search(r'Action:\s*(.+?)(?=Action Input:|$)', response, re.DOTALL)
        input_match = re.search(r'Action Input:\s*(.+?)$', response, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else "finish"
        action_input = input_match.group(1).strip() if input_match else response
        
        return AgentAction(
            thought=thought,
            action=action,
            action_input=action_input,
        )
    
    def _format_result(self, result: Any) -> str:
        """Format tool result for observation."""
        if isinstance(result, list):
            if len(result) == 0:
                return "No results found"
            # Format list of results
            formatted = []
            for i, item in enumerate(result[:5]):  # Limit to 5
                if hasattr(item, 'text'):
                    formatted.append(f"[{i+1}] {item.text[:200]}")
                else:
                    formatted.append(f"[{i+1}] {str(item)[:200]}")
            return "\n".join(formatted)
        return str(result)[:500]
    
    async def _generate_final_answer(
        self,
        query: str,
        steps: List[AgentStep],
    ) -> str:
        """Generate final answer from observations."""
        observations = "\n".join([
            f"- {step.observation}"
            for step in steps
            if step.observation
        ])
        
        prompt = f"""Based on the following observations, provide a comprehensive answer to the question.

Question: {query}

Observations:
{observations}

Answer:"""
        
        try:
            return await self.llm_fn(prompt)
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            return "Unable to generate an answer."

class StreamingRAG:
    """
    Streaming RAG with token-by-token output.
    
    Supports:
    - Streaming retrieval results
    - Progressive answer generation
    - Real-time context updates
    
    Usage:
        streaming_rag = StreamingRAG(
            retrieve_fn=retriever.retrieve,
            generate_fn=llm.generate_stream,
        )
        
        async for chunk in streaming_rag.stream("What is Python?"):
            print(chunk, end="")
    """
    
    def __init__(
        self,
        retrieve_fn: Callable,
        generate_fn: Callable,
        stream_fn: Optional[Callable] = None,
    ):
        """Initialize Streaming RAG."""
        self.retrieve = retrieve_fn
        self.generate = generate_fn
        self.stream = stream_fn
    
    async def stream_response(
        self,
        query: str,
        top_k: int = 5,
    ):
        """
        Stream response token by token.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Yields:
            Response chunks
        """
        # First, retrieve context
        try:
            documents = await self.retrieve(query, top_k=top_k)
        except Exception as e:
            yield f"[Error retrieving context: {e}]"
            return
        
        # Build context
        context = self._build_context(documents)
        
        # Build prompt
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        
        # Stream generation
        if self.stream:
            try:
                async for chunk in self.stream(prompt):
                    yield chunk
            except Exception as e:
                yield f"[Error streaming response: {e}]"
        else:
            # Fallback to non-streaming
            try:
                response = await self.generate(prompt)
                yield response
            except Exception as e:
                yield f"[Error generating response: {e}]"
    
    def _build_context(self, documents: List[Any]) -> str:
        """Build context from documents."""
        if not documents:
            return "No relevant documents found."
        
        parts = []
        for i, doc in enumerate(documents[:5]):
            text = getattr(doc, 'text', str(doc))
            parts.append(f"[{i+1}] {text[:400]}")
        
        return "\n\n".join(parts)
    
    async def retrieve_and_generate(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Non-streaming retrieve and generate.
        
        Args:
            query: User query
            top_k: Number of documents
            
        Returns:
            Dict with response and metadata
        """
        documents = await self.retrieve(query, top_k=top_k)
        context = self._build_context(documents)
        
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        
        response = await self.generate(prompt)
        
        return {
            'query': query,
            'response': response,
            'documents': documents,
            'context': context,
        }

class RAGOrchestrator:
    """
    Orchestrator for complex RAG workflows.
    
    Coordinates multiple RAG strategies:
    - Simple retrieval
    - Multi-hop reasoning
    - Agentic exploration
    - Streaming responses
    
    Selects strategy based on query complexity.
    """
    
    def __init__(
        self,
        simple_rag: Optional[Callable] = None,
        agentic_rag: Optional[AgenticRAG] = None,
        streaming_rag: Optional[StreamingRAG] = None,
        classify_fn: Optional[Callable] = None,
    ):
        """Initialize orchestrator."""
        self.simple_rag = simple_rag
        self.agentic_rag = agentic_rag
        self.streaming_rag = streaming_rag
        self.classify = classify_fn
    
    async def run(
        self,
        query: str,
        strategy: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Run appropriate RAG strategy.
        
        Args:
            query: User query
            strategy: Force specific strategy (simple, agentic, streaming)
            stream: Enable streaming output
            
        Returns:
            Response from selected strategy
        """
        # Auto-select strategy if not specified
        if strategy is None:
            strategy = await self._select_strategy(query)
        
        if strategy == "agentic" and self.agentic_rag:
            return await self.agentic_rag.run(query)
        
        elif strategy == "streaming" and self.streaming_rag and stream:
            return self.streaming_rag.stream_response(query)
        
        elif self.simple_rag:
            return await self.simple_rag(query)
        
        else:
            raise ValueError("No RAG strategy available")
    
    async def _select_strategy(self, query: str) -> str:
        """Select strategy based on query complexity."""
        if self.classify:
            return await self.classify(query)
        
        # Simple heuristics
        query_lower = query.lower()
        
        # Complex queries benefit from agentic
        if any(w in query_lower for w in ['compare', 'analyze', 'calculate', 'find all']):
            return "agentic"
        
        # Simple queries use basic retrieval
        if any(w in query_lower for w in ['what is', 'who is', 'define']):
            return "simple"
        
        return "simple"
