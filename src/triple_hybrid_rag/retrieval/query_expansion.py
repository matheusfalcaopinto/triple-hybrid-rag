"""
Query Expansion for Triple-Hybrid-RAG

Implements multiple query expansion strategies:
1. Multi-Query Generation: Generate query variants using LLM
2. Pseudo-Relevance Feedback (PRF): Use top-k results to expand query
3. Synonym Expansion: WordNet-style synonym injection
4. Query Decomposition: Break complex queries into sub-queries

Reference: 
- Wang et al. "Query2Doc: Query Expansion with Large Language Models"
- RAG-Fusion: Raudaschl "Forget RAG, the Future is RAG-Fusion"
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import SearchResult

logger = logging.getLogger(__name__)

# Multi-query generation prompt
MULTI_QUERY_PROMPT = """You are an AI assistant helping to improve search results.
Given the user's query, generate {num_queries} alternative versions of the query 
that would help retrieve relevant documents. Each query should capture a different 
aspect or phrasing of the original question.

Original query: {query}

Generate {num_queries} alternative queries, one per line:"""

# Query decomposition prompt
DECOMPOSITION_PROMPT = """Break down the following complex query into simpler sub-queries.
Each sub-query should be answerable independently and together they should cover the original question.

Original query: {query}

Generate 2-4 simpler sub-queries, one per line:"""

# PRF expansion prompt
PRF_EXPANSION_PROMPT = """Given the original query and relevant document snippets, 
generate an expanded query that incorporates key terms and concepts from the documents.

Original query: {query}

Relevant snippets:
{snippets}

Generate a single expanded query that combines the original query with relevant terms from the snippets:"""

# Models that don't support temperature parameter (only default temperature=1.0)
# gpt-5-nano and similar reasoning models have this limitation
MODELS_WITHOUT_TEMPERATURE_SUPPORT = frozenset({
    "gpt-5-nano",
    "o1",
    "o1-preview", 
    "o1-mini",
    "o3",
    "o3-mini",
})

# Models that require max_completion_tokens instead of max_tokens
MODELS_REQUIRING_MAX_COMPLETION_TOKENS = frozenset({
    "gpt-5-nano",
    "o1",
    "o1-preview", 
    "o1-mini",
    "o3",
    "o3-mini",
})

def _model_supports_temperature(model: str) -> bool:
    """Check if the model supports the temperature parameter."""
    # Check exact match first
    if model.lower() in MODELS_WITHOUT_TEMPERATURE_SUPPORT:
        return False
    # Check if model name contains any of the unsupported model prefixes
    model_lower = model.lower()
    for unsupported in MODELS_WITHOUT_TEMPERATURE_SUPPORT:
        if unsupported in model_lower:
            return False
    return True

def _model_requires_max_completion_tokens(model: str) -> bool:
    """Check if the model requires max_completion_tokens instead of max_tokens."""
    model_lower = model.lower()
    for model_name in MODELS_REQUIRING_MAX_COMPLETION_TOKENS:
        if model_name in model_lower:
            return True
    return False

# Models that support reasoning_effort parameter (to control thinking depth)
MODELS_WITH_REASONING_EFFORT = frozenset({
    "gpt-5-nano",
    "o1",
    "o1-preview", 
    "o1-mini",
    "o3",
    "o3-mini",
})

def _model_supports_reasoning_effort(model: str) -> bool:
    """Check if the model supports reasoning_effort parameter."""
    model_lower = model.lower()
    for model_name in MODELS_WITH_REASONING_EFFORT:
        if model_name in model_lower:
            return True
    return False

@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion."""
    
    enabled: bool = True
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 200
    timeout: float = 30.0
    
    # Multi-query settings
    multi_query_enabled: bool = True
    num_query_variants: int = 3
    
    # PRF settings
    prf_enabled: bool = True
    prf_top_k: int = 3  # Top-k results to use for expansion
    prf_max_terms: int = 10
    
    # Decomposition settings
    decomposition_enabled: bool = True
    decomposition_threshold: int = 10  # Min words to trigger decomposition
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    # Reasoning model settings (gpt-5-nano, o1, o3 use internal reasoning tokens)
    reasoning_model_max_tokens: int = 2000  # Higher limit for reasoning models
    # reasoning_effort: controls how much thinking the model does
    # "low" = faster + cheaper, "medium" = balanced, "high" = most thorough
    # For query expansion, "low" is usually sufficient
    reasoning_effort: str = "low"

@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    prf_terms: List[str] = field(default_factory=list)
    expansion_time_ms: float = 0.0
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_queries(self) -> List[str]:
        """Get all query variants including original."""
        queries = [self.original_query]
        queries.extend(self.expanded_queries)
        queries.extend(self.sub_queries)
        return list(dict.fromkeys(queries))  # Dedupe while preserving order
    
    @property
    def num_expansions(self) -> int:
        """Total number of query expansions."""
        return len(self.expanded_queries) + len(self.sub_queries)

class QueryExpander:
    """
    Advanced query expansion for improved retrieval.
    
    Supports multiple expansion strategies:
    - Multi-query: Generate query variants using LLM
    - PRF: Use initial results to expand query
    - Decomposition: Break complex queries into simpler ones
    
    Usage:
        expander = QueryExpander()
        expanded = await expander.expand(query="What is the refund policy?")
        # Use expanded.all_queries for retrieval
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        expansion_config: Optional[QueryExpansionConfig] = None,
    ):
        """Initialize the query expander."""
        self.config = config or get_settings()
        self.expansion_config = expansion_config or QueryExpansionConfig()
        
        # Use AsyncOpenAI client (consistent with entity_extractor and hyde)
        self._client = AsyncOpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            timeout=self.expansion_config.timeout,
        )
        self._cache: Dict[str, ExpandedQuery] = {}
    
    async def close(self):
        """Close the client (no-op for AsyncOpenAI, included for API consistency)."""
        pass
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    async def expand(
        self,
        query: str,
        include_multi_query: bool = True,
        include_decomposition: bool = True,
    ) -> ExpandedQuery:
        """
        Expand a query using multiple strategies.
        
        Args:
            query: Original user query
            include_multi_query: Generate query variants
            include_decomposition: Decompose complex queries
            
        Returns:
            ExpandedQuery with all expansions
        """
        import time
        start_time = time.perf_counter()
        
        if not self.expansion_config.enabled:
            return ExpandedQuery(original_query=query)
        
        # Check cache
        cache_key: Optional[str] = None
        if self.expansion_config.cache_enabled:
            cache_key = self._get_cache_key(query)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.cache_hit = True
                return cached
        
        result = ExpandedQuery(original_query=query)
        
        # Extract keywords
        result.keywords = self._extract_keywords(query)
        
        # Run expansion strategies concurrently
        tasks = []
        
        if include_multi_query and self.expansion_config.multi_query_enabled:
            tasks.append(("multi_query", self._generate_multi_query(query)))
        
        if include_decomposition and self.expansion_config.decomposition_enabled:
            if self._should_decompose(query):
                tasks.append(("decomposition", self._decompose_query(query)))
        
        if tasks:
            task_results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True,
            )
            
            for (task_name, _), task_result in zip(tasks, task_results):
                if isinstance(task_result, Exception):
                    logger.warning(f"Query expansion task {task_name} failed: {task_result}")
                    continue
                
                if task_name == "multi_query":
                    result.expanded_queries.extend(task_result)
                elif task_name == "decomposition":
                    result.sub_queries.extend(task_result)
        
        result.expansion_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Cache result
        if self.expansion_config.cache_enabled and cache_key is not None:
            self._cache[cache_key] = result
        
        logger.debug(
            f"Query expansion: {len(result.expanded_queries)} variants, "
            f"{len(result.sub_queries)} sub-queries, "
            f"time={result.expansion_time_ms:.1f}ms"
        )
        
        return result
    
    async def expand_with_prf(
        self,
        query: str,
        initial_results: List[SearchResult],
    ) -> ExpandedQuery:
        """
        Expand query using Pseudo-Relevance Feedback.
        
        Uses the text from top-k initial results to expand the query.
        
        Args:
            query: Original user query
            initial_results: Initial retrieval results for feedback
            
        Returns:
            ExpandedQuery with PRF-based expansion
        """
        import time
        start_time = time.perf_counter()
        
        if not self.expansion_config.prf_enabled:
            return ExpandedQuery(original_query=query)
        
        result = ExpandedQuery(original_query=query)
        result.keywords = self._extract_keywords(query)
        
        # Use top-k results for PRF
        top_k = self.expansion_config.prf_top_k
        feedback_texts = [r.text for r in initial_results[:top_k] if r.text]
        
        if feedback_texts:
            try:
                # Extract key terms from feedback
                prf_terms = self._extract_prf_terms(query, feedback_texts)
                result.prf_terms = prf_terms
                
                # Generate expanded query using LLM
                expanded = await self._generate_prf_expansion(query, feedback_texts)
                if expanded:
                    result.expanded_queries.append(expanded)
                
            except Exception as e:
                logger.warning(f"PRF expansion failed: {e}")
        
        result.expansion_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Common stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again", "then",
            "once", "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "and", "but", "if", "or", "because", "until",
            "while", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "i", "me", "my", "myself", "we",
            "our", "ours", "ourselves", "you", "your", "yours",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(dict.fromkeys(keywords))  # Dedupe
    
    def _should_decompose(self, query: str) -> bool:
        """Check if query should be decomposed."""
        # Decompose long/complex queries
        word_count = len(query.split())
        if word_count < self.expansion_config.decomposition_threshold:
            return False
        
        # Check for multi-part indicators
        multi_indicators = ["and", "or", "also", "as well as", "in addition", "both", "either"]
        query_lower = query.lower()
        
        return any(ind in query_lower for ind in multi_indicators)
    
    def _extract_prf_terms(
        self,
        query: str,
        feedback_texts: List[str],
    ) -> List[str]:
        """Extract expansion terms from feedback documents."""
        query_keywords = set(self._extract_keywords(query))
        
        # Count term frequencies in feedback
        term_counts: Dict[str, int] = {}
        
        for text in feedback_texts:
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if len(word) > 2 and word not in query_keywords:
                    term_counts[word] = term_counts.get(word, 0) + 1
        
        # Sort by frequency and take top terms
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        max_terms = self.expansion_config.prf_max_terms
        
        return [term for term, _ in sorted_terms[:max_terms]]
    
    async def _generate_multi_query(self, query: str) -> List[str]:
        """Generate query variants using LLM."""
        num_queries = self.expansion_config.num_query_variants
        
        prompt = MULTI_QUERY_PROMPT.format(
            query=query,
            num_queries=num_queries,
        )
        
        # Build request kwargs - some models have specific parameter requirements
        request_kwargs: Dict[str, Any] = {
            "model": self.expansion_config.model,
            "messages": [
                {"role": "system", "content": "You generate alternative search queries."},
                {"role": "user", "content": prompt},
            ],
        }
        
        # Use max_completion_tokens for models that require it
        # Reasoning models need higher token limits for internal reasoning
        if _model_requires_max_completion_tokens(self.expansion_config.model):
            tokens = self.expansion_config.reasoning_model_max_tokens
            request_kwargs["max_completion_tokens"] = tokens
            logger.debug(f"Model {self.expansion_config.model} is a reasoning model, using max_completion_tokens={tokens}")
        else:
            request_kwargs["max_tokens"] = self.expansion_config.max_tokens
        
        # Only add temperature if the model supports it
        if _model_supports_temperature(self.expansion_config.model):
            request_kwargs["temperature"] = self.expansion_config.temperature
        else:
            logger.debug(f"Model {self.expansion_config.model} doesn't support temperature parameter, skipping")
        
        # Add reasoning_effort for reasoning models (gpt-5-nano, o1, o3)
        # "low" = faster + cheaper (good for query expansion), "medium" = balanced, "high" = most thorough
        if _model_supports_reasoning_effort(self.expansion_config.model):
            request_kwargs["reasoning_effort"] = self.expansion_config.reasoning_effort
            logger.debug(f"Model {self.expansion_config.model} supports reasoning_effort, using: {self.expansion_config.reasoning_effort}")
        
        response = await self._client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        
        # Parse queries (one per line)
        queries = []
        for line in content.strip().split("\n"):
            line = line.strip()
            # Remove numbering if present (1. query, - query, etc.)
            line = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)
            if line and line != query:
                queries.append(line)
        
        return queries[:num_queries]
    
    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        
        # Build request kwargs - some models have specific parameter requirements
        request_kwargs: Dict[str, Any] = {
            "model": self.expansion_config.model,
            "messages": [
                {"role": "system", "content": "You decompose complex queries into simpler ones."},
                {"role": "user", "content": prompt},
            ],
        }
        
        # Use max_completion_tokens for models that require it
        # Reasoning models need higher token limits for internal reasoning
        if _model_requires_max_completion_tokens(self.expansion_config.model):
            tokens = self.expansion_config.reasoning_model_max_tokens
            request_kwargs["max_completion_tokens"] = tokens
        else:
            request_kwargs["max_tokens"] = self.expansion_config.max_tokens
        
        # Only add temperature if the model supports it
        # Use lower temperature (0.3) for decomposition, but only if supported
        if _model_supports_temperature(self.expansion_config.model):
            request_kwargs["temperature"] = 0.3
        else:
            logger.debug(f"Model {self.expansion_config.model} doesn't support temperature parameter, skipping")
        
        # Add reasoning_effort for reasoning models
        if _model_supports_reasoning_effort(self.expansion_config.model):
            request_kwargs["reasoning_effort"] = self.expansion_config.reasoning_effort
        
        response = await self._client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        
        # Parse sub-queries
        sub_queries = []
        for line in content.strip().split("\n"):
            line = line.strip()
            line = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)
            if line and line != query:
                sub_queries.append(line)
        
        return sub_queries[:4]  # Max 4 sub-queries
    
    async def _generate_prf_expansion(
        self,
        query: str,
        feedback_texts: List[str],
    ) -> Optional[str]:
        """Generate expanded query using PRF."""
        # Truncate feedback texts
        snippets = []
        for i, text in enumerate(feedback_texts):
            snippet = text[:500] + "..." if len(text) > 500 else text
            snippets.append(f"{i+1}. {snippet}")
        
        prompt = PRF_EXPANSION_PROMPT.format(
            query=query,
            snippets="\n".join(snippets),
        )
        
        # Build request kwargs - some models have specific parameter requirements
        request_kwargs: Dict[str, Any] = {
            "model": self.expansion_config.model,
            "messages": [
                {"role": "system", "content": "You expand search queries using relevant context."},
                {"role": "user", "content": prompt},
            ],
        }
        
        # Use max_completion_tokens for models that require it
        # Reasoning models need higher token limits for internal reasoning
        if _model_requires_max_completion_tokens(self.expansion_config.model):
            tokens = self.expansion_config.reasoning_model_max_tokens
            request_kwargs["max_completion_tokens"] = tokens
        else:
            request_kwargs["max_tokens"] = 150
        
        # Only add temperature if the model supports it
        if _model_supports_temperature(self.expansion_config.model):
            request_kwargs["temperature"] = 0.3
        else:
            logger.debug(f"Model {self.expansion_config.model} doesn't support temperature parameter, skipping")
        
        # Add reasoning_effort for reasoning models
        if _model_supports_reasoning_effort(self.expansion_config.model):
            request_kwargs["reasoning_effort"] = self.expansion_config.reasoning_effort
        
        response = await self._client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        
        return content.strip()
    
    def clear_cache(self):
        """Clear the expansion cache."""
        self._cache.clear()

class RAGFusion:
    """
    RAG-Fusion: Multi-query retrieval with reciprocal rank aggregation.
    
    Generates multiple query variants, retrieves for each, then fuses
    results using reciprocal rank fusion.
    
    Reference: Raudaschl "Forget RAG, the Future is RAG-Fusion"
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        num_queries: int = 4,
        rrf_k: int = 60,
    ):
        """
        Initialize RAG-Fusion.
        
        Args:
            config: RAG configuration
            num_queries: Number of query variants to generate
            rrf_k: RRF constant (default 60)
        """
        self.config = config or get_settings()
        self.num_queries = num_queries
        self.rrf_k = rrf_k
        
        expansion_config = QueryExpansionConfig(
            num_query_variants=num_queries,
            decomposition_enabled=False,  # Use only multi-query
        )
        self.expander = QueryExpander(config, expansion_config)
    
    async def generate_queries(self, query: str) -> List[str]:
        """Generate query variants for fusion."""
        expanded = await self.expander.expand(query)
        return expanded.all_queries
    
    def fuse_results(
        self,
        results_per_query: List[List[SearchResult]],
    ) -> List[SearchResult]:
        """
        Fuse results from multiple queries using RRF.
        
        Args:
            results_per_query: List of result lists, one per query variant
            
        Returns:
            Fused and ranked results
        """
        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}
        
        for query_results in results_per_query:
            for rank, result in enumerate(query_results, start=1):
                key = str(result.chunk_id)
                score = 1.0 / (self.rrf_k + rank)
                
                rrf_scores[key] = rrf_scores.get(key, 0.0) + score
                
                if key not in result_map:
                    result_map[key] = result
        
        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        fused = []
        for key in sorted_keys:
            result = result_map[key]
            result.rrf_score = rrf_scores[key]
            result.final_score = rrf_scores[key]
            fused.append(result)
        
        return fused
    
    async def close(self):
        """Close resources."""
        await self.expander.close()
