"""
Advanced Retrieval Components for Triple-Hybrid-RAG

This module contains state-of-the-art retrieval enhancements:
- HyDE: Hypothetical Document Embeddings
- Query Expansion: Multi-query generation with PRF
- Multi-Stage Reranking: Progressive refinement pipeline
- Diversity Optimization: MMR and DPP algorithms
"""

from triple_hybrid_rag.retrieval.hyde import (
    HyDEGenerator,
    HyDEConfig,
    HyDEResult,
    HyDEEnsemble,
)
from triple_hybrid_rag.retrieval.query_expansion import (
    QueryExpander,
    QueryExpansionConfig,
    ExpandedQuery,
    RAGFusion,
)
from triple_hybrid_rag.retrieval.multi_stage_rerank import (
    MultiStageReranker,
    RerankerConfig,
    RerankedResult,
    ListwiseReranker,
)
from triple_hybrid_rag.retrieval.diversity import (
    DiversityOptimizer,
    DiversityConfig,
    DiversityResult,
    IntentDiversifier,
)
from triple_hybrid_rag.retrieval.adaptive_fusion import (
    AdaptiveRRFFusion,
    AdaptiveFusionConfig,
    AdaptiveFusionPredictor,
    QueryFeatureExtractor,
    FusionWeights,
    QueryFeatures,
    QueryIntent,
)
from triple_hybrid_rag.retrieval.context_compression import (
    ContextCompressor,
    CompressionConfig,
    CompressionResult,
    CompressionStrategy,
    LongContextHandler,
)
from triple_hybrid_rag.retrieval.splade import (
    SpladeEncoder,
    SpladeRetriever,
    SpladeConfig,
    SpladeResult,
    SparseVector,
)
from triple_hybrid_rag.retrieval.query_router import (
    QueryRouter,
    QueryClassifier,
    QueryClassification,
    QueryCategory,
    RetrievalStrategy,
    RoutingDecision,
    RouterConfig,
    AdaptiveRouter,
)
from triple_hybrid_rag.retrieval.caching import (
    QueryCache,
    CacheConfig,
    CacheStrategy,
    CacheEntry,
    CacheStats,
    MultiLevelCache,
)
from triple_hybrid_rag.retrieval.observability import (
    RAGObserver,
    MetricsCollector,
    Tracer,
    SpanContext,
    PipelineMetrics,
    ComponentType,
    traced,
    get_global_observer,
)
from triple_hybrid_rag.retrieval.batch import (
    BatchProcessor,
    StreamingBatchProcessor,
    BatchConfig,
    BatchQuery,
    BatchResult,
    BatchStats,
    Priority,
)
from triple_hybrid_rag.retrieval.self_rag import (
    SelfRAG,
    SelfRAGConfig,
    AdaptiveRAG,
    RetrievalDecision,
    SupportLevel,
    RelevanceLevel,
    PassageEvaluation,
    GenerationResult,
)
from triple_hybrid_rag.retrieval.hierarchical import (
    ParentDocumentRetriever,
    ParentDocumentConfig,
    SentenceWindowRetriever,
    SentenceWindowConfig,
    AutoMergingRetriever,
    DocumentNode,
    HierarchicalChunk,
)
from triple_hybrid_rag.retrieval.corrective_rag import (
    CorrectiveRAG,
    CRAGConfig,
    CRAGResult,
    RetrievalQuality,
    CorrectionAction,
    RetrievalAssessment,
    KnowledgeRefiner,
)
from triple_hybrid_rag.retrieval.multimodal import (
    MultimodalRetriever,
    MultimodalConfig,
    MultimodalContent,
    MultimodalResult,
    ModalityType,
    ColBERTRetriever,
    ColBERTConfig,
    MultiVectorRetriever,
)
from triple_hybrid_rag.retrieval.agentic_rag import (
    AgenticRAG,
    AgenticRAGConfig,
    AgenticRAGResult,
    Tool,
    ToolResult,
    ToolType,
    SearchTool,
    CalculateTool,
    SummarizeTool,
    AgentAction,
    AgentStep,
    StreamingRAG,
    RAGOrchestrator,
)

__all__ = [
    # HyDE
    "HyDEGenerator",
    "HyDEConfig",
    "HyDEResult",
    "HyDEEnsemble",
    # Query Expansion
    "QueryExpander",
    "QueryExpansionConfig",
    "ExpandedQuery",
    "RAGFusion",
    # Multi-Stage Reranking
    "MultiStageReranker",
    "RerankerConfig",
    "RerankedResult",
    "ListwiseReranker",
    # Diversity
    "DiversityOptimizer",
    "DiversityConfig",
    "DiversityResult",
    "IntentDiversifier",
    # Adaptive Fusion
    "AdaptiveRRFFusion",
    "AdaptiveFusionConfig",
    "AdaptiveFusionPredictor",
    "QueryFeatureExtractor",
    "FusionWeights",
    "QueryFeatures",
    "QueryIntent",
    # Context Compression
    "ContextCompressor",
    "CompressionConfig",
    "CompressionResult",
    "CompressionStrategy",
    "LongContextHandler",
    # SPLADE
    "SpladeEncoder",
    "SpladeRetriever",
    "SpladeConfig",
    "SpladeResult",
    "SparseVector",
    # Query Router
    "QueryRouter",
    "QueryClassifier",
    "QueryClassification",
    "QueryCategory",
    "RetrievalStrategy",
    "RoutingDecision",
    "RouterConfig",
    "AdaptiveRouter",
    # Caching
    "QueryCache",
    "CacheConfig",
    "CacheStrategy",
    "CacheEntry",
    "CacheStats",
    "MultiLevelCache",
    # Observability
    "RAGObserver",
    "MetricsCollector",
    "Tracer",
    "SpanContext",
    "PipelineMetrics",
    "ComponentType",
    "traced",
    "get_global_observer",
    # Batch Processing
    "BatchProcessor",
    "StreamingBatchProcessor",
    "BatchConfig",
    "BatchQuery",
    "BatchResult",
    "BatchStats",
    "Priority",
    # Self-RAG
    "SelfRAG",
    "SelfRAGConfig",
    "AdaptiveRAG",
    "RetrievalDecision",
    "SupportLevel",
    "RelevanceLevel",
    "PassageEvaluation",
    "GenerationResult",
    # Hierarchical Retrieval
    "ParentDocumentRetriever",
    "ParentDocumentConfig",
    "SentenceWindowRetriever",
    "SentenceWindowConfig",
    "AutoMergingRetriever",
    "DocumentNode",
    "HierarchicalChunk",
    # Corrective RAG
    "CorrectiveRAG",
    "CRAGConfig",
    "CRAGResult",
    "RetrievalQuality",
    "CorrectionAction",
    "RetrievalAssessment",
    "KnowledgeRefiner",
    # Multimodal
    "MultimodalRetriever",
    "MultimodalConfig",
    "MultimodalContent",
    "MultimodalResult",
    "ModalityType",
    "ColBERTRetriever",
    "ColBERTConfig",
    "MultiVectorRetriever",
    # Agentic RAG
    "AgenticRAG",
    "AgenticRAGConfig",
    "AgenticRAGResult",
    "Tool",
    "ToolResult",
    "ToolType",
    "SearchTool",
    "CalculateTool",
    "SummarizeTool",
    "AgentAction",
    "AgentStep",
    "StreamingRAG",
    "RAGOrchestrator",
]
