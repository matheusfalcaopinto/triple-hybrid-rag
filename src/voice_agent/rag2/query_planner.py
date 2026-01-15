"""
RAG 2.0 Query Planner Agent

Uses GPT-5 to analyze user queries and generate:
- Keywords for lexical (BM25) search
- Semantic query text for vector search
- Cypher query for graph traversal (when enabled)
- Per-channel top_k and weights
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """Plan for multi-channel retrieval."""
    # Original query
    original_query: str
    
    # Lexical channel
    keywords: List[str] = field(default_factory=list)
    lexical_top_k: int = 50
    
    # Semantic channel
    semantic_query_text: str = ""
    semantic_top_k: int = 100
    
    # Graph channel (optional)
    cypher_query: Optional[str] = None
    graph_top_k: int = 50
    
    # Channel weights for RRF
    weights: Dict[str, float] = field(default_factory=lambda: {
        "lexical": 0.7,
        "semantic": 0.8,
        "graph": 1.0,
    })
    
    # Detected intent
    intent: str = "general"
    requires_graph: bool = False


# noqa: E501 - Prompt strings are intentionally long for LLM readability
QUERY_PLANNER_SYSTEM_PROMPT = """\
You are a query planning agent for a RAG system.

Given a user query, analyze it and produce a structured plan for multi-channel retrieval.

The system has three retrieval channels:
1. **Lexical (BM25)**: Full-text search matching exact keywords
2. **Semantic (Vector)**: Embedding-based similarity search
3. **Graph (Cypher)**: Knowledge graph traversal for entity relationships

Your task is to output a JSON object with:
- `keywords`: List of important keywords/phrases for lexical search (Portuguese or English)
- `semantic_query_text`: Reformulated query optimized for semantic similarity
- `cypher_query`: A Cypher query for graph traversal (or null if not needed)
- `requires_graph`: Whether this query would benefit from graph traversal
- `intent`: The type of query (factual, procedural, comparative, entity_lookup, relational)
- `weights`: Suggested weights for each channel based on query type

Examples:

Query: "What are the payment terms in contract X?"
{
  "keywords": ["payment", "terms", "contract X", "prazo", "pagamento"],
  "semantic_query_text": "Payment terms and conditions in contract X",
  "cypher_query": "MATCH (c:Contract)-[:HAS_CLAUSE]->(cl) WHERE cl.type='payment' RETURN cl",
  "requires_graph": true,
  "intent": "factual",
  "weights": {"lexical": 0.7, "semantic": 0.8, "graph": 1.0}
}

Query: "How do I reset my password?"
{
  "keywords": ["reset", "password", "senha", "redefinir"],
  "semantic_query_text": "Step-by-step instructions for resetting user password",
  "cypher_query": null,
  "requires_graph": false,
  "intent": "procedural",
  "weights": {"lexical": 0.8, "semantic": 0.9, "graph": 0.0}
}

Always output valid JSON only, no explanations."""


class QueryPlanner:
    """
    Query planning agent using GPT-5.
    
    Analyzes queries and produces structured retrieval plans.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the query planner.
        
        Args:
            model: Model to use for planning
            temperature: Temperature for generation
        """
        self.model = model or SETTINGS.rag2_query_planner_model
        self.temperature = temperature
        self._client: Optional[OpenAI] = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=SETTINGS.openai_api_key,
                base_url=SETTINGS.openai_base_url,
            )
        return self._client
    
    def plan(self, query: str, collection: Optional[str] = None) -> QueryPlan:
        """
        Create a query plan for multi-channel retrieval.
        
        Args:
            query: User query
            collection: Optional collection context
            
        Returns:
            QueryPlan object
        """
        try:
            # Build messages
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": QUERY_PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ]
            
            if collection:
                messages[1]["content"] += f"\nCollection context: {collection}"
            
            # Call GPT-5
            response = self.client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            
            # Parse response
            content = response.choices[0].message.content or ""
            data = json.loads(content)
            
            return QueryPlan(
                original_query=query,
                keywords=data.get("keywords", []),
                semantic_query_text=data.get("semantic_query_text", query),
                cypher_query=data.get("cypher_query"),
                requires_graph=data.get("requires_graph", False),
                intent=data.get("intent", "general"),
                weights=data.get("weights", {
                    "lexical": SETTINGS.rag2_lexical_weight,
                    "semantic": SETTINGS.rag2_semantic_weight,
                    "graph": SETTINGS.rag2_graph_weight,
                }),
                lexical_top_k=SETTINGS.rag2_lexical_top_k,
                semantic_top_k=SETTINGS.rag2_semantic_top_k,
                graph_top_k=SETTINGS.rag2_graph_top_k,
            )
            
        except Exception as e:
            logger.warning(f"Query planning failed, using defaults: {e}")
            
            # Fallback to simple plan
            return QueryPlan(
                original_query=query,
                keywords=query.split(),
                semantic_query_text=query,
                lexical_top_k=SETTINGS.rag2_lexical_top_k,
                semantic_top_k=SETTINGS.rag2_semantic_top_k,
            )
    
    async def plan_async(self, query: str, collection: Optional[str] = None) -> QueryPlan:
        """Async version of plan."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.plan, query, collection)


def get_query_planner(**kwargs: Any) -> QueryPlanner:
    """Get a configured query planner instance."""
    return QueryPlanner(**kwargs)
