"""
Triple-Hybrid-RAG Configuration

Pydantic-based settings with environment variable support.
All settings can be overridden via environment variables or .env file.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """
    Configuration for Triple-Hybrid-RAG.
    
    All settings can be overridden via environment variables.
    Prefix: RAG_ for most settings, or use the exact variable name.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATABASE
    # ═══════════════════════════════════════════════════════════════════════════
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:54332/rag_db",
        description="PostgreSQL connection URL",
    )
    database_pool_size: int = Field(default=10, description="Connection pool size")
    database_max_overflow: int = Field(default=20, description="Max overflow connections")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPENAI API
    # ═══════════════════════════════════════════════════════════════════════════
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    openai_model: str = Field(default="gpt-5", description="Default OpenAI model")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EMBEDDING API
    # ═══════════════════════════════════════════════════════════════════════════
    rag_embed_api_base: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="Embedding API base URL",
    )
    rag_embed_model: str = Field(
        default="qwen3-vl-embedding-2b",
        description="Embedding model name",
    )
    rag_embed_batch_size: int = Field(default=200, description="Batch size for embeddings")
    rag_embed_concurrent_batches: int = Field(
        default=8,
        description="Number of concurrent embedding batches",
    )
    rag_embed_timeout: float = Field(default=60.0, description="Embedding request timeout")
    rag_embed_dim_model: int = Field(
        default=2048,
        description="Full model embedding dimension",
    )
    rag_embed_dim_store: int = Field(
        default=1024,
        description="Truncated dimension for storage (Matryoshka)",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RERANKING API
    # ═══════════════════════════════════════════════════════════════════════════
    rag_rerank_api_base: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="Reranking API base URL",
    )
    rag_rerank_model: str = Field(
        default="qwen3-vl-reranker-2b",
        description="Reranking model name",
    )
    rag_rerank_timeout: float = Field(default=30.0, description="Reranking request timeout")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OCR API
    # ═══════════════════════════════════════════════════════════════════════════
    rag_ocr_api_base: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="OCR API base URL",
    )
    rag_ocr_model: str = Field(default="qwen3-vl-8b", description="OCR model name")
    rag_ocr_timeout: float = Field(default=60.0, description="OCR request timeout")
    
    # DeepSeek OCR (optional)
    rag_deepseek_ocr_api_base: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="DeepSeek OCR API base URL",
    )
    rag_deepseek_ocr_model: str = Field(
        default="deepseek-ocr",
        description="DeepSeek OCR model name",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OCR INGESTION MODE
    # ═══════════════════════════════════════════════════════════════════════════
    rag_ocr_mode: str = Field(
        default="auto",
        description="OCR ingestion mode: 'qwen' (Qwen3-VL), 'deepseek' (DeepSeek OCR), 'off' (text extraction only), 'auto' (system decides based on file analysis)",
    )
    rag_ocr_auto_preferred: str = Field(
        default="qwen",
        description="Preferred OCR provider when AUTO mode decides OCR is needed: 'qwen' or 'deepseek'",
    )
    rag_ocr_auto_threshold: float = Field(
        default=0.3,
        description="Score threshold (0-1) for AUTO mode to enable OCR. Lower = more aggressive OCR usage.",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUPPYGRAPH
    # ═══════════════════════════════════════════════════════════════════════════
    puppygraph_bolt_url: str = Field(
        default="bolt://localhost:7697",
        description="PuppyGraph Bolt protocol URL",
    )
    puppygraph_http_url: str = Field(
        default="http://localhost:8192",
        description="PuppyGraph HTTP API URL",
    )
    puppygraph_web_ui_url: str = Field(
        default="http://localhost:8091",
        description="PuppyGraph Web UI URL",
    )
    puppygraph_username: str = Field(default="admin", description="PuppyGraph username")
    puppygraph_password: str = Field(default="puppygraph123", description="PuppyGraph password")
    puppygraph_timeout: float = Field(default=30.0, description="PuppyGraph request timeout")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE FLAGS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Master switch
    rag_enabled: bool = Field(default=True, description="Master enable/disable")
    
    # Retrieval channels
    rag_lexical_enabled: bool = Field(default=True, description="Enable lexical (BM25) search")
    rag_semantic_enabled: bool = Field(default=True, description="Enable semantic (vector) search")
    rag_graph_enabled: bool = Field(default=True, description="Enable graph (PuppyGraph) search")
    
    # Processing features
    rag_rerank_enabled: bool = Field(default=True, description="Enable reranking")
    rag_denoise_enabled: bool = Field(default=True, description="Enable conformal denoising")
    rag_query_planner_enabled: bool = Field(default=True, description="Enable query planner")
    
    # Ingestion features
    rag_entity_extraction_enabled: bool = Field(
        default=True,
        description="Enable entity extraction during ingestion",
    )
    rag_ocr_enabled: bool = Field(default=True, description="Enable OCR for images")
    rag_deepseek_ocr_enabled: bool = Field(
        default=False,
        description="Use DeepSeek OCR instead of Qwen3-VL",
    )
    rag_multimodal_embedding_enabled: bool = Field(
        default=True,
        description="Enable direct image embeddings",
    )

    # Ingestion retry settings
    rag_ingest_embed_retry_attempts: int = Field(
        default=3,
        description="Retry attempts for embedding API calls during ingestion",
    )
    rag_ingest_db_retry_attempts: int = Field(
        default=3,
        description="Retry attempts for database writes during ingestion",
    )
    rag_ingest_retry_backoff_min: float = Field(
        default=1.0,
        description="Minimum backoff (seconds) for ingestion retries",
    )
    rag_ingest_retry_backoff_max: float = Field(
        default=10.0,
        description="Maximum backoff (seconds) for ingestion retries",
    )
    rag_db_batch_size: int = Field(
        default=1000,
        description="Batch size for database inserts (reduces network round-trips)",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EMBEDDING CACHE
    # ═══════════════════════════════════════════════════════════════════════════
    rag_embedding_cache_enabled: bool = Field(
        default=True,
        description="Enable embedding cache to skip re-embedding identical content",
    )
    rag_embedding_cache_backend: str = Field(
        default="memory",
        description="Cache backend: 'memory' (in-process) or 'redis' (distributed)",
    )
    rag_embedding_cache_max_size: int = Field(
        default=100_000,
        description="Max embeddings to cache (memory backend only)",
    )
    rag_embedding_cache_ttl: int = Field(
        default=604800,  # 7 days
        description="Cache TTL in seconds (redis backend only)",
    )
    rag_embedding_cache_redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for distributed embedding cache",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PIPELINED INGESTION
    # ═══════════════════════════════════════════════════════════════════════════
    rag_pipeline_enabled: bool = Field(
        default=True,
        description="Enable pipelined (overlapping) ingestion for large documents",
    )
    rag_pipeline_queue_size: int = Field(
        default=10,
        description="Max batches to buffer between pipeline stages",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH INFERENCE (OpenAI Batch API for ingestion)
    # ═══════════════════════════════════════════════════════════════════════════
    rag_batch_inference_enabled: bool = Field(
        default=False,
        description="Enable OpenAI Batch API for NER during ingestion. 50% cheaper but 24h completion window.",
    )
    rag_batch_inference_threshold: int = Field(
        default=10,
        description="Minimum number of chunks to trigger batch inference instead of real-time API.",
    )
    rag_batch_inference_check_interval: int = Field(
        default=60,
        description="Seconds between batch status checks.",
    )
    rag_batch_inference_max_wait: int = Field(
        default=86400,
        description="Maximum seconds to wait for batch completion (default: 24 hours).",
    )
    
    # Chunking features
    rag_parent_child_chunking: bool = Field(
        default=True,
        description="Enable hierarchical parent/child chunking",
    )
    rag_matryoshka_embeddings: bool = Field(
        default=True,
        description="Enable Matryoshka embedding truncation",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHUNKING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    rag_parent_chunk_tokens: int = Field(
        default=800,
        description="Target parent chunk size in tokens",
    )
    rag_parent_chunk_max_tokens: int = Field(
        default=1000,
        description="Maximum parent chunk size in tokens",
    )
    rag_child_chunk_tokens: int = Field(
        default=200,
        description="Target child chunk size in tokens",
    )
    rag_chunk_overlap_tokens: int = Field(
        default=50,
        description="Overlap between chunks in tokens",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RETRIEVAL CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Weights
    rag_lexical_weight: float = Field(default=0.7, description="Lexical channel RRF weight")
    rag_semantic_weight: float = Field(default=0.8, description="Semantic channel RRF weight")
    rag_graph_weight: float = Field(default=1.0, description="Graph channel RRF weight")
    
    # Safety & Denoising
    rag_safety_threshold: float = Field(
        default=0.6,
        description="Minimum similarity threshold",
    )
    rag_denoise_alpha: float = Field(default=0.6, description="Conformal denoising alpha")
    
    # Top-K settings
    rag_lexical_top_k: int = Field(default=50, description="Max results from FTS")
    rag_semantic_top_k: int = Field(default=100, description="Max results from vector search")
    rag_graph_top_k: int = Field(default=50, description="Max results from graph search")
    rag_rerank_top_k: int = Field(default=20, description="Results to rerank")
    rag_final_top_k: int = Field(default=5, description="Final results after rerank")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY PLANNER
    # ═══════════════════════════════════════════════════════════════════════════
    rag_query_planner_model: str = Field(default="gpt-5", description="Query planner model")
    rag_query_planner_temperature: float = Field(
        default=0.0,
        description="Query planner temperature",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENTITY EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════
    rag_ner_model: str = Field(default="gpt-5", description="NER model")
    rag_ner_temperature: float = Field(default=0.0, description="NER temperature")
    rag_ner_max_chunks_per_request: int = Field(
        default=8,
        description="Max chunks per NER request",
    )
    rag_ner_max_chars_per_chunk: int = Field(
        default=2000,
        description="Max characters per chunk sent to NER",
    )
    rag_ner_reasoning_effort: str = Field(
        default="low",
        description="Reasoning effort for NER with gpt-5-nano/o1/o3 models: 'low', 'medium', 'high'. 'low' is faster and cheaper.",
    )
    rag_entity_types: str = Field(
        default="PERSON,ORGANIZATION,PRODUCT,CLAUSE,DATE,MONEY,LOCATION,TECHNICAL_TERM",
        description="Comma-separated list of entity types to extract",
    )
    
    @field_validator("rag_entity_types")
    @classmethod
    def validate_entity_types(cls, v: str) -> str:
        """Validate entity types are comma-separated."""
        if not v:
            return "PERSON,ORGANIZATION"
        return v.upper()
    
    @property
    def entity_types_list(self) -> List[str]:
        """Get entity types as a list."""
        return [t.strip() for t in self.rag_entity_types.split(",") if t.strip()]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GUNDAM TILING OCR
    # ═══════════════════════════════════════════════════════════════════════════
    rag_gundam_tiling_enabled: bool = Field(
        default=True,
        description="Enable Gundam Tiling for large images",
    )
    rag_gundam_min_image_size: int = Field(
        default=1500,
        description="Min dimension (px) to trigger tiling",
    )
    rag_gundam_tile_size: int = Field(default=1024, description="Tile size in pixels")
    rag_gundam_overlap: int = Field(default=128, description="Overlap between tiles")
    rag_gundam_max_tiles: int = Field(default=16, description="Max tiles per image")
    rag_gundam_merge_strategy: str = Field(
        default="fuzzy",
        description="Merge strategy: fuzzy, concat, vote",
    )
    rag_gundam_fuzzy_threshold: float = Field(
        default=0.85,
        description="Similarity threshold for fuzzy merge",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS)
    # ═══════════════════════════════════════════════════════════════════════════
    rag_hyde_enabled: bool = Field(
        default=True,
        description="Enable HyDE for query transformation",
    )
    rag_hyde_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for HyDE generation",
    )
    rag_hyde_temperature: float = Field(
        default=0.7,
        description="Temperature for HyDE generation",
    )
    rag_hyde_num_hypotheticals: int = Field(
        default=1,
        description="Number of hypothetical documents to generate",
    )
    rag_hyde_cache_enabled: bool = Field(
        default=True,
        description="Enable caching of HyDE results",
    )
    rag_hyde_use_intent_prompts: bool = Field(
        default=True,
        description="Use intent-specific prompt templates",
    )
    rag_hyde_fallback_to_original: bool = Field(
        default=True,
        description="Fall back to original query on generation failure",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY EXPANSION
    # ═══════════════════════════════════════════════════════════════════════════
    rag_query_expansion_enabled: bool = Field(
        default=True,
        description="Enable query expansion for improved recall",
    )
    rag_query_expansion_num_variants: int = Field(
        default=3,
        description="Number of query variants to generate",
    )
    rag_query_expansion_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for query expansion",
    )
    rag_query_expansion_temperature: float = Field(
        default=0.7,
        description="Temperature for query expansion",
    )
    rag_query_prf_enabled: bool = Field(
        default=True,
        description="Enable Pseudo-Relevance Feedback",
    )
    rag_query_prf_top_k: int = Field(
        default=3,
        description="Top-k documents to use for PRF",
    )
    rag_query_prf_num_terms: int = Field(
        default=10,
        description="Number of expansion terms from PRF",
    )
    rag_query_decomposition_enabled: bool = Field(
        default=True,
        description="Enable query decomposition for complex queries",
    )
    rag_query_decomposition_threshold: int = Field(
        default=5,
        description="Word count threshold for decomposition",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-STAGE RERANKING
    # ═══════════════════════════════════════════════════════════════════════════
    rag_multistage_rerank_enabled: bool = Field(
        default=True,
        description="Enable multi-stage reranking pipeline",
    )
    rag_rerank_stage1_enabled: bool = Field(
        default=True,
        description="Enable Stage 1 (bi-encoder filtering)",
    )
    rag_rerank_stage1_top_k: int = Field(
        default=100,
        description="Top-k results after Stage 1",
    )
    rag_rerank_stage2_enabled: bool = Field(
        default=True,
        description="Enable Stage 2 (cross-encoder scoring)",
    )
    rag_rerank_stage2_top_k: int = Field(
        default=30,
        description="Top-k results after Stage 2",
    )
    rag_rerank_stage2_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for cross-encoder reranking",
    )
    rag_rerank_stage2_batch_size: int = Field(
        default=10,
        description="Batch size for Stage 2 reranking",
    )
    rag_rerank_stage3_enabled: bool = Field(
        default=True,
        description="Enable Stage 3 (MMR diversity)",
    )
    rag_rerank_mmr_lambda: float = Field(
        default=0.7,
        description="MMR lambda (0=diversity, 1=relevance)",
    )
    rag_rerank_stage4_enabled: bool = Field(
        default=True,
        description="Enable Stage 4 (score calibration)",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DIVERSITY OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    rag_diversity_enabled: bool = Field(
        default=True,
        description="Enable diversity optimization",
    )
    rag_diversity_mmr_lambda: float = Field(
        default=0.7,
        description="MMR lambda for diversity (0=max diversity, 1=max relevance)",
    )
    rag_diversity_max_per_document: int = Field(
        default=3,
        description="Maximum results from a single document",
    )
    rag_diversity_max_per_page: int = Field(
        default=2,
        description="Maximum results from a single page",
    )
    rag_diversity_min_similarity_threshold: float = Field(
        default=0.95,
        description="Min similarity to consider as duplicate",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LLM-AS-JUDGE EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    rag_judge_enabled: bool = Field(
        default=False,
        description="Enable LLM-as-Judge for evaluation",
    )
    rag_judge_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for judging",
    )
    rag_judge_temperature: float = Field(
        default=0.0,
        description="Temperature for judge (low for consistency)",
    )
    rag_judge_max_concurrent: int = Field(
        default=5,
        description="Max concurrent judge requests",
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OBSERVABILITY
    # ═══════════════════════════════════════════════════════════════════════════
    rag_metrics_enabled: bool = Field(default=False, description="Enable Prometheus metrics")
    rag_metrics_port: int = Field(default=9190, description="Metrics endpoint port")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════════════════
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format: json or text")


# Singleton instance
_settings: Optional[RAGConfig] = None


@lru_cache
def get_settings() -> RAGConfig:
    """
    Get the singleton settings instance.
    
    Uses lru_cache to ensure only one instance is created.
    Call get_settings.cache_clear() to reload settings.
    """
    global _settings
    if _settings is None:
        _settings = RAGConfig()
    return _settings


def reset_settings() -> None:
    """Reset the settings singleton (useful for testing)."""
    global _settings
    _settings = None
    get_settings.cache_clear()


# Convenience alias
SETTINGS = get_settings()
