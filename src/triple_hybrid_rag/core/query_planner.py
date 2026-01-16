"""
Query Planner for Triple-Hybrid-RAG

Uses GPT-5 to decompose queries into optimized components for each search channel:
- Lexical: Extract keywords for BM25/FTS
- Semantic: Rephrase for embedding similarity
- Graph: Generate Cypher query for entity relationships
"""

import json
import logging
from typing import Dict, List, Optional

import httpx

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import QueryPlan

logger = logging.getLogger(__name__)

QUERY_PLANNER_PROMPT = """You are a query planner for a triple-hybrid RAG system. Analyze the user's query and decompose it for three search channels.

**Channels:**
1. **Lexical (BM25)**: Full-text search with keywords
2. **Semantic (Vector)**: Dense embedding similarity
3. **Graph (Cypher)**: Knowledge graph traversal for entity relationships

**Output JSON:**
```json
{
  "keywords": ["keyword1", "keyword2"],
  "semantic_query_text": "Rephrased query optimized for embedding similarity",
  "requires_graph": true/false,
  "cypher_query": "MATCH (e:Entity)-[r]->(related) WHERE ... RETURN ...",
  "intent": "factual|procedural|comparative|entity_lookup|relational",
  "weights": {
    "lexical": 0.7,
    "semantic": 0.8,
    "graph": 1.0
  }
}
```

**Intent Types:**
- `factual`: Looking for specific facts ("What is X?")
- `procedural`: Looking for how-to information ("How do I...?")
- `comparative`: Comparing entities ("What's the difference between...?")
- `entity_lookup`: Finding information about a specific entity
- `relational`: Understanding relationships between entities (use graph!)

**Cypher Guidelines:**
- Use `Entity` nodes with properties: `id`, `name`, `canonical_name`, `entity_type`
- Use `MENTIONED_IN` edges to connect entities to `Chunk` nodes
- Use `RELATED_TO` edges between entities
- Return chunk IDs for retrieval

**User Query:** {query}

Return ONLY valid JSON, no explanations."""

class QueryPlanner:
    """
    Query planner using GPT-5 for intelligent query decomposition.
    
    Analyzes user queries and creates optimized search strategies
    for each retrieval channel (lexical, semantic, graph).
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the query planner."""
        self.config = config or get_settings()
        self.api_key = self.config.openai_api_key
        self.api_base = self.config.openai_base_url.rstrip("/")
        self.model = self.config.rag_query_planner_model
        self.temperature = self.config.rag_query_planner_temperature
        self.enabled = self.config.rag_query_planner_enabled
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def plan(self, query: str) -> QueryPlan:
        """
        Create a query plan for the given user query.
        
        Args:
            query: User's search query
            
        Returns:
            QueryPlan with optimized components for each channel
        """
        if not self.enabled:
            return self._simple_plan(query)
        
        try:
            return await self._gpt_plan(query)
        except Exception as e:
            logger.warning(f"Query planner failed, using simple plan: {e}")
            return self._simple_plan(query)
    
    async def _gpt_plan(self, query: str) -> QueryPlan:
        """Use GPT to create an intelligent query plan."""
        client = await self._get_client()
        
        prompt = QUERY_PLANNER_PROMPT.format(query=query)
        
        response = await client.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a query planning assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": 500,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Parse JSON from response
        plan_data = self._parse_json(content)
        
        return QueryPlan(
            original_query=query,
            keywords=plan_data.get("keywords", self._extract_keywords(query)),
            semantic_query_text=plan_data.get("semantic_query_text", query),
            cypher_query=plan_data.get("cypher_query"),
            requires_graph=plan_data.get("requires_graph", False),
            intent=plan_data.get("intent", "general"),
            weights=plan_data.get("weights", {
                "lexical": self.config.rag_lexical_weight,
                "semantic": self.config.rag_semantic_weight,
                "graph": self.config.rag_graph_weight,
            }),
            lexical_top_k=self.config.rag_lexical_top_k,
            semantic_top_k=self.config.rag_semantic_top_k,
            graph_top_k=self.config.rag_graph_top_k,
        )
    
    def _simple_plan(self, query: str) -> QueryPlan:
        """Create a simple query plan without GPT."""
        keywords = self._extract_keywords(query)
        
        # Detect if graph search might be useful
        requires_graph = any(
            indicator in query.lower()
            for indicator in [
                "relationship", "related", "connected", "between",
                "who", "what company", "which organization",
                "works for", "belongs to", "part of"
            ]
        )
        
        # Detect intent
        intent = "general"
        if query.lower().startswith(("what is", "what are", "define")):
            intent = "factual"
        elif query.lower().startswith(("how do", "how to", "how can")):
            intent = "procedural"
        elif "difference" in query.lower() or "compare" in query.lower():
            intent = "comparative"
        elif requires_graph:
            intent = "relational"
        
        return QueryPlan(
            original_query=query,
            keywords=keywords,
            semantic_query_text=query,
            cypher_query=None,
            requires_graph=requires_graph,
            intent=intent,
            weights={
                "lexical": self.config.rag_lexical_weight,
                "semantic": self.config.rag_semantic_weight,
                "graph": self.config.rag_graph_weight if requires_graph else 0.5,
            },
            lexical_top_k=self.config.rag_lexical_top_k,
            semantic_top_k=self.config.rag_semantic_top_k,
            graph_top_k=self.config.rag_graph_top_k,
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for lexical search."""
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "what", "which", "who", "whom", "this",
            "that", "these", "those", "am", "i", "me", "my", "myself",
            "we", "our", "ours", "ourselves", "you", "your", "yours",
        }
        
        # Tokenize and filter
        words = query.lower().split()
        keywords = [
            word.strip(".,!?;:'\"()[]{}") 
            for word in words 
            if word.lower() not in stop_words and len(word) > 2
        ]
        
        return list(dict.fromkeys(keywords))  # Remove duplicates, preserve order
    
    def _parse_json(self, content: str) -> Dict:
        """Parse JSON from GPT response, handling markdown code blocks."""
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from content
            import re
            match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}
