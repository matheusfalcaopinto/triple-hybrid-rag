"""
HyDE - Hypothetical Document Embeddings for Triple-Hybrid-RAG

HyDE (Hypothetical Document Embeddings) bridges the semantic gap between 
questions and documents by generating a hypothetical answer first, then 
embedding that answer for retrieval.

This technique significantly improves retrieval quality for:
- Knowledge-intensive queries
- Questions with implicit context
- Technical/domain-specific queries

Reference: Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from triple_hybrid_rag.config import RAGConfig, get_settings

logger = logging.getLogger(__name__)

# HyDE prompt templates for different query intents
HYDE_PROMPTS = {
    "default": """Answer the following question in a detailed, factual paragraph. 
Write as if you are writing a passage from a document that would contain the answer.
Do not say "I don't know" - provide a plausible, detailed answer.

Question: {query}

Answer:""",

    "factual": """Write a detailed factual paragraph that would answer the following question.
Include specific details, definitions, and context as if from an authoritative source.

Question: {query}

Passage:""",

    "procedural": """Write a step-by-step procedural passage that explains how to accomplish the following.
Include specific steps, considerations, and best practices as if from a technical manual.

Question: {query}

Procedure:""",

    "comparative": """Write an analytical passage comparing and contrasting the subjects in the following question.
Include specific differences, similarities, advantages, and disadvantages as if from a comprehensive guide.

Question: {query}

Analysis:""",

    "entity_lookup": """Write an informative passage about the entity or concept mentioned in the following question.
Include definitions, properties, relationships, and context as if from an encyclopedia.

Question: {query}

Description:""",

    "relational": """Write a passage explaining the relationships and connections relevant to the following question.
Include entities, their relationships, and context as if from a knowledge base or reference document.

Question: {query}

Explanation:""",

    "technical": """Write a technical passage that would answer the following question.
Include technical details, specifications, code examples if relevant, and implementation considerations.

Question: {query}

Technical Documentation:""",
}


@dataclass
class HyDEConfig:
    """Configuration for HyDE generation."""
    
    enabled: bool = True
    model: str = "gpt-4o-mini"  # Fast, cheap model for hypothetical generation
    temperature: float = 0.7  # Some creativity for diverse hypotheticals
    max_tokens: int = 300  # Reasonable passage length (increased for reasoning models)
    num_hypotheticals: int = 1  # Can generate multiple for ensemble
    timeout: float = 30.0
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour cache
    
    # Intent-specific settings
    use_intent_prompts: bool = True
    fallback_to_original: bool = True  # Also search with original query
    
    # Reasoning model settings (gpt-5-nano, o1, o3 use internal reasoning tokens)
    reasoning_model_max_tokens: int = 3000  # Higher limit for reasoning models
    # reasoning_effort: controls how much thinking the model does
    # "low" = faster + cheaper, "medium" = balanced, "high" = most thorough
    # For HyDE, "low" is usually sufficient since we just need a plausible document
    reasoning_effort: str = "low"


@dataclass
class HyDEResult:
    """Result of HyDE generation."""
    
    original_query: str
    hypothetical_documents: List[str] = field(default_factory=list)
    intent: str = "default"
    embeddings: List[List[float]] = field(default_factory=list)
    generation_time_ms: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def primary_hypothetical(self) -> Optional[str]:
        """Get the first hypothetical document."""
        return self.hypothetical_documents[0] if self.hypothetical_documents else None
    
    @property
    def has_hypotheticals(self) -> bool:
        """Check if hypothetical documents were generated."""
        return len(self.hypothetical_documents) > 0


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

# Models that support reasoning_effort parameter (to control thinking depth)
MODELS_WITH_REASONING_EFFORT = frozenset({
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

def _model_supports_reasoning_effort(model: str) -> bool:
    """Check if the model supports reasoning_effort parameter."""
    model_lower = model.lower()
    for model_name in MODELS_WITH_REASONING_EFFORT:
        if model_name in model_lower:
            return True
    return False


class HyDEGenerator:
    """
    Hypothetical Document Embeddings generator.
    
    HyDE improves retrieval by:
    1. Taking a user query
    2. Generating a hypothetical document that would answer the query
    3. Embedding the hypothetical for retrieval (instead of the query)
    
    This bridges the semantic gap between question-style queries and 
    document-style corpus entries.
    
    Usage:
        hyde = HyDEGenerator()
        result = await hyde.generate(query="What is the refund policy?")
        # Use result.hypothetical_documents[0] for embedding
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        hyde_config: Optional[HyDEConfig] = None,
    ):
        """Initialize the HyDE generator."""
        self.config = config or get_settings()
        self.hyde_config = hyde_config or HyDEConfig()
        
        # Use AsyncOpenAI client (consistent with entity_extractor)
        self._client = AsyncOpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            timeout=self.hyde_config.timeout,
        )
        self._cache: Dict[str, HyDEResult] = {}
    
    async def close(self):
        """Close the client (no-op for AsyncOpenAI, included for API consistency)."""
        pass
    
    def _get_cache_key(self, query: str, intent: str) -> str:
        """Generate cache key for query+intent."""
        content = f"{query}:{intent}:{self.hyde_config.model}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_prompt(self, query: str, intent: str) -> str:
        """Get the appropriate prompt for the query intent."""
        if self.hyde_config.use_intent_prompts and intent in HYDE_PROMPTS:
            template = HYDE_PROMPTS[intent]
        else:
            template = HYDE_PROMPTS["default"]
        
        return template.format(query=query)
    
    def detect_intent(self, query: str) -> str:
        """
        Detect query intent for prompt selection.
        
        This is a simple heuristic-based detection. For production,
        consider using a trained classifier or the query planner's intent.
        """
        query_lower = query.lower()
        
        # Procedural queries
        if any(kw in query_lower for kw in ["how to", "how do", "how can", "steps to", "guide for"]):
            return "procedural"
        
        # Comparative queries
        if any(kw in query_lower for kw in ["difference between", "compare", "vs", "versus", "better than"]):
            return "comparative"
        
        # Technical queries
        if any(kw in query_lower for kw in ["implement", "code", "api", "function", "error", "bug", "configure"]):
            return "technical"
        
        # Entity lookup
        if any(kw in query_lower for kw in ["what is", "who is", "define", "meaning of"]):
            return "entity_lookup"
        
        # Relational queries
        if any(kw in query_lower for kw in ["relationship", "connected to", "related to", "works with"]):
            return "relational"
        
        # Factual queries (default for most "what", "when", "where" questions)
        if any(query_lower.startswith(kw) for kw in ["what", "when", "where", "which"]):
            return "factual"
        
        return "default"
    
    async def generate(
        self,
        query: str,
        intent: Optional[str] = None,
        num_hypotheticals: Optional[int] = None,
    ) -> HyDEResult:
        """
        Generate hypothetical documents for a query.
        
        Args:
            query: User's search query
            intent: Query intent (auto-detected if not provided)
            num_hypotheticals: Number of hypotheticals to generate (default: 1)
            
        Returns:
            HyDEResult with generated hypothetical documents
        """
        import time
        start_time = time.perf_counter()
        
        if not self.hyde_config.enabled:
            return HyDEResult(
                original_query=query,
                hypothetical_documents=[query],  # Fallback to original
                intent=intent or "disabled",
                metadata={"hyde_disabled": True},
            )
        
        # Auto-detect intent if not provided
        if intent is None:
            intent = self.detect_intent(query)
        
        # Check cache
        cache_key: Optional[str] = None
        if self.hyde_config.cache_enabled:
            cache_key = self._get_cache_key(query, intent)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.cache_hit = True
                return cached
        
        num_hypotheticals = num_hypotheticals or self.hyde_config.num_hypotheticals
        
        try:
            hypotheticals = await self._generate_hypotheticals(
                query=query,
                intent=intent,
                count=num_hypotheticals,
            )
            
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            
            result = HyDEResult(
                original_query=query,
                hypothetical_documents=hypotheticals,
                intent=intent,
                generation_time_ms=generation_time_ms,
                metadata={
                    "model": self.hyde_config.model,
                    "num_generated": len(hypotheticals),
                },
            )
            
            # Cache result
            if self.hyde_config.cache_enabled and cache_key is not None:
                self._cache[cache_key] = result
            
            logger.debug(
                f"HyDE generated {len(hypotheticals)} hypotheticals for query "
                f"(intent={intent}, time={generation_time_ms:.1f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            
            # Fallback to original query
            if self.hyde_config.fallback_to_original:
                return HyDEResult(
                    original_query=query,
                    hypothetical_documents=[query],
                    intent=intent,
                    error=str(e),
                    generation_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            
            return HyDEResult(
                original_query=query,
                intent=intent,
                error=str(e),
                generation_time_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    async def _generate_hypotheticals(
        self,
        query: str,
        intent: str,
        count: int,
    ) -> List[str]:
        """Generate hypothetical documents using LLM."""
        prompt = self._get_prompt(query, intent)
        
        # For multiple hypotheticals, we can either:
        # 1. Make multiple API calls (slower, more diverse)
        # 2. Ask for multiple in one call (faster, may be less diverse)
        
        if count == 1:
            return [await self._single_generation(prompt)]
        
        # Generate multiple hypotheticals concurrently
        tasks = [self._single_generation(prompt) for _ in range(count)]
        hypotheticals = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        return [h for h in hypotheticals if isinstance(h, str) and h.strip()]
    
    async def _single_generation(self, prompt: str) -> str:
        """Generate a single hypothetical document using AsyncOpenAI client."""
        # Build request kwargs - some models have specific parameter requirements
        request_kwargs: Dict[str, Any] = {
            "model": self.hyde_config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates informative document passages. Write detailed, factual content as if from a real document.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        
        # Use max_completion_tokens for models that require it (gpt-5-nano, o1, o3, etc.)
        # Reasoning models need higher token limits because they use tokens for internal reasoning
        if _model_requires_max_completion_tokens(self.hyde_config.model):
            # Use the higher reasoning_model_max_tokens for reasoning models
            tokens = self.hyde_config.reasoning_model_max_tokens
            request_kwargs["max_completion_tokens"] = tokens
            logger.debug(f"Model {self.hyde_config.model} is a reasoning model, using max_completion_tokens={tokens}")
        else:
            request_kwargs["max_tokens"] = self.hyde_config.max_tokens
        
        # Only add temperature if the model supports it
        # gpt-5-nano and o1/o3 reasoning models don't support temperature
        if _model_supports_temperature(self.hyde_config.model):
            request_kwargs["temperature"] = self.hyde_config.temperature
        else:
            logger.debug(f"Model {self.hyde_config.model} doesn't support temperature parameter, skipping")
        
        # Add reasoning_effort for reasoning models (gpt-5-nano, o1, o3)
        # "low" = faster + cheaper (good for HyDE), "medium" = balanced, "high" = most thorough
        if _model_supports_reasoning_effort(self.hyde_config.model):
            request_kwargs["reasoning_effort"] = self.hyde_config.reasoning_effort
            logger.debug(f"Model {self.hyde_config.model} supports reasoning_effort, using: {self.hyde_config.reasoning_effort}")
        
        response = await self._client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        
        return content.strip()
    
    async def generate_with_embeddings(
        self,
        query: str,
        embedder,  # MultimodalEmbedder
        intent: Optional[str] = None,
    ) -> HyDEResult:
        """
        Generate hypotheticals and embed them in one step.
        
        This is a convenience method that combines HyDE generation
        with embedding, ready for retrieval.
        
        Args:
            query: User's search query
            embedder: Embedder instance for generating vectors
            intent: Query intent (auto-detected if not provided)
            
        Returns:
            HyDEResult with both hypotheticals and their embeddings
        """
        result = await self.generate(query, intent)
        
        if result.has_hypotheticals:
            # Embed all hypothetical documents
            embeddings = await embedder.embed_texts(result.hypothetical_documents)
            result.embeddings = embeddings
            
            # Optionally also include original query embedding
            if self.hyde_config.fallback_to_original:
                original_embedding = await embedder.embed_text(query)
                result.metadata["original_query_embedding"] = original_embedding
        
        return result
    
    def clear_cache(self):
        """Clear the HyDE cache."""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class HyDEEnsemble:
    """
    Ensemble HyDE for improved retrieval.
    
    Generates multiple hypotheticals with different prompts/temperatures
    and combines their embeddings for more robust retrieval.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        num_variants: int = 3,
        temperatures: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble HyDE.
        
        Args:
            config: RAG configuration
            num_variants: Number of hypothetical variants to generate
            temperatures: Temperature for each variant (default: [0.3, 0.7, 1.0])
        """
        self.config = config or get_settings()
        self.num_variants = num_variants
        self.temperatures = temperatures or [0.3, 0.7, 1.0][:num_variants]
        
        # Create generators with different temperatures
        self.generators = []
        for temp in self.temperatures:
            hyde_config = HyDEConfig(
                temperature=temp,
                num_hypotheticals=1,
            )
            self.generators.append(HyDEGenerator(config, hyde_config))
    
    async def generate(
        self,
        query: str,
        intent: Optional[str] = None,
    ) -> List[HyDEResult]:
        """Generate ensemble of hypotheticals."""
        tasks = [g.generate(query, intent) for g in self.generators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, HyDEResult) and r.has_hypotheticals]
    
    async def generate_with_embeddings(
        self,
        query: str,
        embedder,
        intent: Optional[str] = None,
    ) -> Tuple[List[List[float]], List[HyDEResult]]:
        """
        Generate ensemble hypotheticals with embeddings.
        
        Returns:
            Tuple of (all_embeddings, hyde_results)
        """
        results = await self.generate(query, intent)
        
        all_texts = []
        for result in results:
            all_texts.extend(result.hypothetical_documents)
        
        if all_texts:
            embeddings = await embedder.embed_texts(all_texts)
        else:
            embeddings = []
        
        return embeddings, results
    
    async def close(self):
        """Close all generators."""
        for generator in self.generators:
            await generator.close()
