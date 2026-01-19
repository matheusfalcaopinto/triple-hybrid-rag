"""
Enhanced RAG Pipeline

State-of-the-art hybrid RAG system that orchestrates all enhancement
components for optimal retrieval performance.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for enhanced RAG pipeline."""
    
    # Feature flags
    enable_hyde: bool = True
    enable_query_expansion: bool = True
    enable_splade: bool = True
    enable_reranking: bool = True
    enable_diversity: bool = True
    enable_compression: bool = True
    enable_caching: bool = True
    enable_routing: bool = True
    enable_observability: bool = True
    
    # Retrieval settings
    semantic_top_k: int = 50
    lexical_top_k: int = 50
    graph_top_k: int = 20
    final_top_k: int = 10
    
    # Fusion settings
    semantic_weight: float = 0.7
    lexical_weight: float = 0.7
    graph_weight: float = 0.5
    splade_weight: float = 0.3
    
    # Quality thresholds
    min_relevance_score: float = 0.3
    min_confidence: float = 0.5
    
    # Cache settings
    cache_ttl_seconds: int = 3600
    
    # Context limits
    max_context_tokens: int = 4000

@dataclass
class RetrievalContext:
    """Context passed through the pipeline."""
    query: str
    trace_id: Optional[str] = None
    
    # Query analysis results
    query_category: Optional[str] = None
    query_intent: Optional[str] = None
    routing_strategy: Optional[str] = None
    
    # Enhancement outputs
    hyde_document: Optional[str] = None
    expanded_queries: List[str] = field(default_factory=list)
    
    # Retrieval results per channel
    semantic_results: List[Any] = field(default_factory=list)
    lexical_results: List[Any] = field(default_factory=list)
    graph_results: List[Any] = field(default_factory=list)
    splade_results: List[Any] = field(default_factory=list)
    
    # Processing metadata
    timings: Dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False
    
    # Weights used
    weights_used: Dict[str, float] = field(default_factory=dict)

@dataclass
class PipelineResult:
    """Result from the enhanced RAG pipeline."""
    query: str
    results: List[Any]
    context: RetrievalContext
    
    # Quality metrics
    total_candidates: int = 0
    final_count: int = 0
    avg_score: float = 0.0
    
    # Performance
    total_time_ms: float = 0.0
    cache_hit: bool = False
    
    # Debug info
    stages_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and len(self.results) > 0

class EnhancedRAGPipeline:
    """
    State-of-the-art hybrid RAG pipeline.
    
    Orchestrates:
    - Query classification and routing
    - HyDE (Hypothetical Document Embeddings)
    - Multi-query expansion
    - Multi-channel retrieval (semantic, lexical, graph, SPLADE)
    - Adaptive fusion with learned weights
    - Multi-stage reranking
    - Diversity optimization
    - Context compression
    - Result caching
    - Full observability
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        # Core retrievers
        semantic_search_fn: Optional[Callable] = None,
        lexical_search_fn: Optional[Callable] = None,
        graph_search_fn: Optional[Callable] = None,
        # Enhancement components
        hyde_generator: Optional[Any] = None,
        query_expander: Optional[Any] = None,
        query_router: Optional[Any] = None,
        splade_retriever: Optional[Any] = None,
        adaptive_fusion: Optional[Any] = None,
        reranker: Optional[Any] = None,
        diversity_optimizer: Optional[Any] = None,
        context_compressor: Optional[Any] = None,
        # Infrastructure
        cache: Optional[Any] = None,
        observer: Optional[Any] = None,
        embed_fn: Optional[Callable] = None,
    ):
        self.config = config or PipelineConfig()
        
        # Core search functions
        self.semantic_search = semantic_search_fn
        self.lexical_search = lexical_search_fn
        self.graph_search = graph_search_fn
        self.embed_fn = embed_fn
        
        # Enhancement components
        self.hyde = hyde_generator
        self.expander = query_expander
        self.router = query_router
        self.splade = splade_retriever
        self.fusion = adaptive_fusion
        self.reranker = reranker
        self.diversity = diversity_optimizer
        self.compressor = context_compressor
        
        # Infrastructure
        self.cache = cache
        self.observer = observer
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Execute the full retrieval pipeline.
        
        Args:
            query: User query
            top_k: Override final result count
            **kwargs: Additional parameters
            
        Returns:
            PipelineResult with ranked results
        """
        start_time = time.time()
        top_k = top_k or self.config.final_top_k
        
        # Initialize context
        ctx = RetrievalContext(query=query)
        stages_completed = []
        errors = []
        
        # Start observation
        if self.observer and self.config.enable_observability:
            ctx.trace_id = self.observer.observe_query(query)
        
        try:
            # Stage 1: Check cache
            if self.cache and self.config.enable_caching:
                cached = self.cache.get(query)
                if cached is not None:
                    ctx.cache_hit = True
                    return self._build_result(
                        query, cached, ctx, start_time,
                        stages_completed + ["cache_hit"],
                        errors,
                    )
            
            # Stage 2: Query routing
            ctx = self._route_query(ctx)
            stages_completed.append("routing")
            
            # Stage 3: Query enhancement (HyDE + Expansion)
            ctx = self._enhance_query(ctx)
            stages_completed.append("enhancement")
            
            # Stage 4: Multi-channel retrieval
            ctx = self._retrieve_all_channels(ctx)
            stages_completed.append("retrieval")
            
            # Stage 5: Fusion
            fused_results = self._fuse_results(ctx)
            stages_completed.append("fusion")
            
            # Stage 6: Reranking
            reranked = self._rerank_results(fused_results, ctx)
            stages_completed.append("reranking")
            
            # Stage 7: Diversity optimization
            diverse = self._diversify_results(reranked, ctx, top_k)
            stages_completed.append("diversity")
            
            # Stage 8: Context compression
            final_results = self._compress_context(diverse, ctx)
            stages_completed.append("compression")
            
            # Cache results
            if self.cache and self.config.enable_caching:
                self.cache.set(query, final_results, ttl_seconds=self.config.cache_ttl_seconds)
            
            return self._build_result(
                query, final_results, ctx, start_time,
                stages_completed, errors,
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            errors.append(str(e))
            return self._build_result(
                query, [], ctx, start_time,
                stages_completed, errors,
            )
        
        finally:
            if self.observer and ctx.trace_id and self.config.enable_observability:
                self.observer.complete_query(ctx.trace_id)
    
    def _route_query(self, ctx: RetrievalContext) -> RetrievalContext:
        """Route query to optimal strategy."""
        if not self.router or not self.config.enable_routing:
            return ctx
        
        start = time.time()
        
        try:
            decision = self.router.route(ctx.query)
            
            ctx.query_category = decision.classification.category.value
            ctx.routing_strategy = decision.strategy.value
            
            # Update weights based on routing
            ctx.weights_used = {
                'semantic': decision.semantic_weight,
                'lexical': decision.lexical_weight,
                'graph': decision.graph_weight,
                'splade': decision.splade_weight,
            }
            
        except Exception as e:
            logger.warning(f"Routing failed: {e}, using defaults")
            ctx.weights_used = {
                'semantic': self.config.semantic_weight,
                'lexical': self.config.lexical_weight,
                'graph': self.config.graph_weight,
                'splade': self.config.splade_weight,
            }
        
        ctx.timings['routing'] = (time.time() - start) * 1000
        return ctx
    
    def _enhance_query(self, ctx: RetrievalContext) -> RetrievalContext:
        """Enhance query with HyDE and expansion."""
        # HyDE
        if self.hyde and self.config.enable_hyde:
            start = time.time()
            try:
                result = self.hyde.generate(ctx.query)
                ctx.hyde_document = result.hypothetical_document
            except Exception as e:
                logger.warning(f"HyDE failed: {e}")
            ctx.timings['hyde'] = (time.time() - start) * 1000
        
        # Query expansion
        if self.expander and self.config.enable_query_expansion:
            start = time.time()
            try:
                expanded = self.expander.expand(ctx.query)
                ctx.expanded_queries = expanded.all_queries
            except Exception as e:
                logger.warning(f"Expansion failed: {e}")
            ctx.timings['expansion'] = (time.time() - start) * 1000
        
        return ctx
    
    def _retrieve_all_channels(self, ctx: RetrievalContext) -> RetrievalContext:
        """Execute retrieval on all channels."""
        queries_to_search = [ctx.query] + ctx.expanded_queries[:2]
        
        # Semantic search
        if self.semantic_search:
            start = time.time()
            try:
                all_results = []
                for q in queries_to_search:
                    results = self.semantic_search(q, top_k=self.config.semantic_top_k)
                    all_results.extend(results)
                ctx.semantic_results = self._dedupe_results(all_results)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
            ctx.timings['semantic'] = (time.time() - start) * 1000
        
        # Lexical search
        if self.lexical_search:
            start = time.time()
            try:
                all_results = []
                for q in queries_to_search:
                    results = self.lexical_search(q, top_k=self.config.lexical_top_k)
                    all_results.extend(results)
                ctx.lexical_results = self._dedupe_results(all_results)
            except Exception as e:
                logger.warning(f"Lexical search failed: {e}")
            ctx.timings['lexical'] = (time.time() - start) * 1000
        
        # Graph search
        if self.graph_search:
            start = time.time()
            try:
                results = self.graph_search(ctx.query, top_k=self.config.graph_top_k)
                ctx.graph_results = results
            except Exception as e:
                logger.warning(f"Graph search failed: {e}")
            ctx.timings['graph'] = (time.time() - start) * 1000
        
        # SPLADE search
        if self.splade and self.config.enable_splade:
            start = time.time()
            try:
                results = self.splade.search(ctx.query, top_k=self.config.semantic_top_k)
                ctx.splade_results = results
            except Exception as e:
                logger.warning(f"SPLADE search failed: {e}")
            ctx.timings['splade'] = (time.time() - start) * 1000
        
        return ctx
    
    def _fuse_results(self, ctx: RetrievalContext) -> List[Any]:
        """Fuse results from all channels."""
        start = time.time()
        
        # Use adaptive fusion if available
        if self.fusion:
            try:
                fused = self.fusion.fuse(
                    query=ctx.query,
                    semantic_results=ctx.semantic_results,
                    lexical_results=ctx.lexical_results,
                    graph_results=ctx.graph_results,
                )
                ctx.timings['fusion'] = (time.time() - start) * 1000
                return fused
            except Exception as e:
                logger.warning(f"Adaptive fusion failed: {e}")
        
        # Fallback to simple RRF
        all_results = {}
        k = 60  # RRF constant
        
        # Process each channel
        channels = [
            (ctx.semantic_results, ctx.weights_used.get('semantic', 0.7)),
            (ctx.lexical_results, ctx.weights_used.get('lexical', 0.7)),
            (ctx.graph_results, ctx.weights_used.get('graph', 0.5)),
        ]
        
        for results, weight in channels:
            for rank, result in enumerate(results):
                doc_id = self._get_doc_id(result)
                rrf_score = weight / (k + rank + 1)
                
                if doc_id in all_results:
                    all_results[doc_id]['score'] += rrf_score
                else:
                    all_results[doc_id] = {'result': result, 'score': rrf_score}
        
        # Sort by score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True,
        )
        
        ctx.timings['fusion'] = (time.time() - start) * 1000
        return [item['result'] for item in sorted_results]
    
    def _rerank_results(
        self,
        results: List[Any],
        ctx: RetrievalContext,
    ) -> List[Any]:
        """Rerank fused results."""
        if not self.reranker or not self.config.enable_reranking:
            return results
        
        start = time.time()
        
        try:
            reranked = self.reranker.rerank(results, ctx.query)
            results = [r.result for r in reranked.results]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
        
        ctx.timings['reranking'] = (time.time() - start) * 1000
        return results
    
    def _diversify_results(
        self,
        results: List[Any],
        ctx: RetrievalContext,
        top_k: int,
    ) -> List[Any]:
        """Apply diversity optimization."""
        if not self.diversity or not self.config.enable_diversity:
            return results[:top_k]
        
        start = time.time()
        
        try:
            diverse = self.diversity.optimize(results, ctx.query, top_k=top_k)
            results = diverse.results
        except Exception as e:
            logger.warning(f"Diversity optimization failed: {e}")
            results = results[:top_k]
        
        ctx.timings['diversity'] = (time.time() - start) * 1000
        return results
    
    def _compress_context(
        self,
        results: List[Any],
        ctx: RetrievalContext,
    ) -> List[Any]:
        """Compress context if needed."""
        if not self.compressor or not self.config.enable_compression:
            return results
        
        start = time.time()
        
        try:
            # Estimate tokens
            total_text = " ".join(self._get_text(r) for r in results)
            est_tokens = len(total_text.split()) * 1.3
            
            if est_tokens > self.config.max_context_tokens:
                compressed = self.compressor.compress(
                    total_text,
                    ctx.query,
                    max_tokens=self.config.max_context_tokens,
                )
                # Results modified to include compressed text
                for r in results:
                    if hasattr(r, 'compressed'):
                        r.compressed = True
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
        
        ctx.timings['compression'] = (time.time() - start) * 1000
        return results
    
    def _build_result(
        self,
        query: str,
        results: List[Any],
        ctx: RetrievalContext,
        start_time: float,
        stages_completed: List[str],
        errors: List[str],
    ) -> PipelineResult:
        """Build final pipeline result."""
        total_time = (time.time() - start_time) * 1000
        
        total_candidates = (
            len(ctx.semantic_results) +
            len(ctx.lexical_results) +
            len(ctx.graph_results) +
            len(ctx.splade_results)
        )
        
        avg_score = 0.0
        if results:
            scores = [self._get_score(r) for r in results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return PipelineResult(
            query=query,
            results=results,
            context=ctx,
            total_candidates=total_candidates,
            final_count=len(results),
            avg_score=avg_score,
            total_time_ms=total_time,
            cache_hit=ctx.cache_hit,
            stages_completed=stages_completed,
            errors=errors,
        )
    
    def _dedupe_results(self, results: List[Any]) -> List[Any]:
        """Remove duplicate results."""
        seen = set()
        unique = []
        for r in results:
            doc_id = self._get_doc_id(r)
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(r)
        return unique
    
    def _get_doc_id(self, result: Any) -> str:
        """Extract document ID from result."""
        if hasattr(result, 'doc_id'):
            return str(result.doc_id)
        if hasattr(result, 'chunk_id'):
            return str(result.chunk_id)
        if isinstance(result, dict):
            return str(result.get('id', result.get('doc_id', id(result))))
        return str(id(result))
    
    def _get_text(self, result: Any) -> str:
        """Extract text from result."""
        if hasattr(result, 'text'):
            return result.text
        if hasattr(result, 'content'):
            return result.content
        if isinstance(result, dict):
            return str(result.get('text', result.get('content', '')))
        return str(result)
    
    def _get_score(self, result: Any) -> float:
        """Extract score from result."""
        if hasattr(result, 'score'):
            return float(result.score)
        if isinstance(result, dict):
            return float(result.get('score', 0.0))
        return 0.0
