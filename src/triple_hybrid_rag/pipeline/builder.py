"""
Pipeline Builder

Fluent API for configuring and building enhanced RAG pipelines.
"""

import logging
from typing import Optional, Dict, Any, Callable, List

from triple_hybrid_rag.pipeline.enhanced_rag import (
    EnhancedRAGPipeline,
    PipelineConfig,
)

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """Registry for pipeline components with lazy initialization."""
    
    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
    
    def register(self, name: str, factory: Callable) -> None:
        """Register a component factory."""
        self._factories[name] = factory
    
    def get(self, name: str, **kwargs) -> Any:
        """Get or create a component."""
        if name not in self._instances:
            if name not in self._factories:
                return None
            self._instances[name] = self._factories[name](**kwargs)
        return self._instances[name]
    
    def clear(self) -> None:
        """Clear all instances."""
        self._instances.clear()

class PipelineBuilder:
    """
    Fluent builder for creating enhanced RAG pipelines.
    
    Example:
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(my_search_fn)
            .with_lexical_search(bm25_search)
            .enable_hyde(llm_fn=generate_fn)
            .enable_reranking()
            .enable_caching()
            .with_config(final_top_k=10)
            .build()
        )
    """
    
    def __init__(self):
        self._config = PipelineConfig()
        self._semantic_search: Optional[Callable] = None
        self._lexical_search: Optional[Callable] = None
        self._graph_search: Optional[Callable] = None
        self._embed_fn: Optional[Callable] = None
        self._llm_fn: Optional[Callable] = None
        
        # Components
        self._hyde = None
        self._expander = None
        self._router = None
        self._splade = None
        self._fusion = None
        self._reranker = None
        self._diversity = None
        self._compressor = None
        self._cache = None
        self._observer = None
    
    def with_config(self, **kwargs) -> "PipelineBuilder":
        """Set configuration options."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown config option: {key}")
        return self
    
    def with_semantic_search(self, fn: Callable) -> "PipelineBuilder":
        """Set semantic search function."""
        self._semantic_search = fn
        return self
    
    def with_lexical_search(self, fn: Callable) -> "PipelineBuilder":
        """Set lexical/BM25 search function."""
        self._lexical_search = fn
        return self
    
    def with_graph_search(self, fn: Callable) -> "PipelineBuilder":
        """Set graph search function."""
        self._graph_search = fn
        return self
    
    def with_embed_fn(self, fn: Callable) -> "PipelineBuilder":
        """Set embedding function."""
        self._embed_fn = fn
        return self
    
    def with_llm_fn(self, fn: Callable) -> "PipelineBuilder":
        """Set LLM function for generation tasks."""
        self._llm_fn = fn
        return self
    
    def enable_hyde(
        self,
        generator: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable HyDE generation."""
        self._config.enable_hyde = True
        
        if generator:
            self._hyde = generator
        # HyDE requires llm_fn to be passed externally
        return self
    
    def enable_query_expansion(
        self,
        expander: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable query expansion."""
        self._config.enable_query_expansion = True
        
        if expander:
            self._expander = expander
        # QueryExpander requires llm_fn to be passed externally
        return self
    
    def enable_routing(
        self,
        router: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable query routing."""
        self._config.enable_routing = True
        
        if router:
            self._router = router
        else:
            from triple_hybrid_rag.retrieval import QueryRouter, RouterConfig
            config = RouterConfig(**kwargs) if kwargs else None
            self._router = QueryRouter(config=config)
        return self
    
    def enable_splade(
        self,
        retriever: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable SPLADE retrieval."""
        self._config.enable_splade = True
        
        if retriever:
            self._splade = retriever
        # SPLADE requires model, so only set if provided
        return self
    
    def enable_adaptive_fusion(
        self,
        fusion: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable adaptive RRF fusion."""
        if fusion:
            self._fusion = fusion
        else:
            from triple_hybrid_rag.retrieval import AdaptiveRRFFusion, AdaptiveFusionConfig
            config = AdaptiveFusionConfig(**kwargs) if kwargs else None
            self._fusion = AdaptiveRRFFusion(config=config)
        return self
    
    def enable_reranking(
        self,
        reranker: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable multi-stage reranking."""
        self._config.enable_reranking = True
        
        if reranker:
            self._reranker = reranker
        # If no reranker provided, component remains None
        # Pipeline will skip reranking but track the stage
        return self
    
    def enable_diversity(
        self,
        optimizer: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable diversity optimization."""
        self._config.enable_diversity = True
        
        if optimizer:
            self._diversity = optimizer
        else:
            from triple_hybrid_rag.retrieval import DiversityOptimizer, DiversityConfig
            config = DiversityConfig(**kwargs) if kwargs else None
            self._diversity = DiversityOptimizer(config=config)
        return self
    
    def enable_compression(
        self,
        compressor: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable context compression."""
        self._config.enable_compression = True
        
        if compressor:
            self._compressor = compressor
        else:
            from triple_hybrid_rag.retrieval import ContextCompressor, CompressionConfig
            config = CompressionConfig(**kwargs) if kwargs else None
            self._compressor = ContextCompressor(config=config)
        return self
    
    def enable_caching(
        self,
        cache: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Enable query caching."""
        self._config.enable_caching = True
        
        if cache:
            self._cache = cache
        else:
            from triple_hybrid_rag.retrieval import QueryCache, CacheConfig
            config = CacheConfig(**kwargs) if kwargs else None
            self._cache = QueryCache(config=config, embed_fn=self._embed_fn)
        return self
    
    def enable_observability(
        self,
        observer: Optional[Any] = None,
    ) -> "PipelineBuilder":
        """Enable observability."""
        self._config.enable_observability = True
        
        if observer:
            self._observer = observer
        else:
            from triple_hybrid_rag.retrieval import RAGObserver
            self._observer = RAGObserver()
        return self
    
    def disable_hyde(self) -> "PipelineBuilder":
        """Disable HyDE."""
        self._config.enable_hyde = False
        return self
    
    def disable_query_expansion(self) -> "PipelineBuilder":
        """Disable query expansion."""
        self._config.enable_query_expansion = False
        return self
    
    def disable_routing(self) -> "PipelineBuilder":
        """Disable routing."""
        self._config.enable_routing = False
        return self
    
    def disable_splade(self) -> "PipelineBuilder":
        """Disable SPLADE."""
        self._config.enable_splade = False
        return self
    
    def disable_reranking(self) -> "PipelineBuilder":
        """Disable reranking."""
        self._config.enable_reranking = False
        return self
    
    def disable_diversity(self) -> "PipelineBuilder":
        """Disable diversity."""
        self._config.enable_diversity = False
        return self
    
    def disable_compression(self) -> "PipelineBuilder":
        """Disable compression."""
        self._config.enable_compression = False
        return self
    
    def disable_caching(self) -> "PipelineBuilder":
        """Disable caching."""
        self._config.enable_caching = False
        return self
    
    def minimal(self) -> "PipelineBuilder":
        """Create minimal pipeline with only basic retrieval."""
        self._config.enable_hyde = False
        self._config.enable_query_expansion = False
        self._config.enable_splade = False
        self._config.enable_reranking = False
        self._config.enable_diversity = False
        self._config.enable_compression = False
        self._config.enable_routing = False
        return self
    
    def full(self) -> "PipelineBuilder":
        """Enable all enhancement features."""
        return (
            self
            .enable_hyde()
            .enable_query_expansion()
            .enable_routing()
            .enable_adaptive_fusion()
            .enable_reranking()
            .enable_diversity()
            .enable_compression()
            .enable_caching()
            .enable_observability()
        )
    
    def build(self) -> EnhancedRAGPipeline:
        """Build the configured pipeline."""
        return EnhancedRAGPipeline(
            config=self._config,
            semantic_search_fn=self._semantic_search,
            lexical_search_fn=self._lexical_search,
            graph_search_fn=self._graph_search,
            hyde_generator=self._hyde if self._config.enable_hyde else None,
            query_expander=self._expander if self._config.enable_query_expansion else None,
            query_router=self._router if self._config.enable_routing else None,
            splade_retriever=self._splade if self._config.enable_splade else None,
            adaptive_fusion=self._fusion,
            reranker=self._reranker if self._config.enable_reranking else None,
            diversity_optimizer=self._diversity if self._config.enable_diversity else None,
            context_compressor=self._compressor if self._config.enable_compression else None,
            cache=self._cache if self._config.enable_caching else None,
            observer=self._observer if self._config.enable_observability else None,
            embed_fn=self._embed_fn,
        )

def create_pipeline(
    semantic_search: Callable,
    lexical_search: Optional[Callable] = None,
    graph_search: Optional[Callable] = None,
    preset: str = "balanced",
    **kwargs,
) -> EnhancedRAGPipeline:
    """
    Factory function to create a pipeline with a preset configuration.
    
    Presets:
    - "minimal": Only basic retrieval, no enhancements
    - "fast": Light enhancements for speed
    - "balanced": Good balance of quality and speed (default)
    - "quality": Maximum quality, slower
    
    Args:
        semantic_search: Semantic search function
        lexical_search: Optional BM25/lexical search
        graph_search: Optional graph search
        preset: Configuration preset
        **kwargs: Additional config overrides
    """
    builder = PipelineBuilder()
    builder.with_semantic_search(semantic_search)
    
    if lexical_search:
        builder.with_lexical_search(lexical_search)
    if graph_search:
        builder.with_graph_search(graph_search)
    
    if preset == "minimal":
        builder.minimal()
    elif preset == "fast":
        builder.enable_routing()
        builder.enable_caching()
    elif preset == "balanced":
        builder.enable_routing()
        builder.enable_reranking()
        builder.enable_diversity()
        builder.enable_caching()
        builder.enable_observability()
    elif preset == "quality":
        builder.full()
    else:
        logger.warning(f"Unknown preset: {preset}, using balanced")
        builder.enable_routing()
        builder.enable_reranking()
        builder.enable_diversity()
        builder.enable_caching()
    
    # Apply any config overrides
    if kwargs:
        builder.with_config(**kwargs)
    
    return builder.build()
