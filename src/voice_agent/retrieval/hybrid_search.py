"""
Hybrid Search Module

Implements hybrid search combining:
- BM25 full-text search (via PostgreSQL FTS)
- Vector similarity search (via pgvector)
- RRF (Reciprocal Rank Fusion) for score combination
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voice_agent.config import SETTINGS
from voice_agent.ingestion.embedder import Embedder
from voice_agent.observability.rag_metrics import rag_metrics
from voice_agent.utils.db import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for hybrid search."""
    use_hybrid: bool = True
    use_vector: bool = True
    use_bm25: bool = True
    use_image_search: bool = False
    top_k_retrieve: int = 50
    top_k_image: int = 3
    top_k_final: int = 10
    rrf_k: int = 60  # RRF constant (typically 60)
    fts_language: str = "portuguese"
    category_filter: Optional[str] = None
    source_filter: Optional[str] = None
    min_similarity: float = 0.0
    
    @classmethod
    def from_settings(cls) -> "SearchConfig":
        """Create config from global settings."""
        return cls(
            use_hybrid=SETTINGS.rag_use_hybrid_bm25,
            top_k_retrieve=SETTINGS.rag_top_k_retrieve,
            top_k_image=SETTINGS.rag_top_k_image,
            fts_language=SETTINGS.rag_fts_language,
            use_image_search=SETTINGS.rag_enable_image_embeddings,
        )


@dataclass
class SearchResult:
    """A single search result with provenance."""
    chunk_id: str
    content: str
    modality: str
    source_document: str
    page: int
    chunk_index: int
    
    # Scores
    similarity_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    
    # Metadata
    ocr_confidence: Optional[float] = None
    is_table: bool = False
    table_context: Optional[str] = None
    alt_text: Optional[str] = None
    category: Optional[str] = None
    title: Optional[str] = None
    
    # For debugging
    retrieval_method: str = ""  # "vector", "bm25", "hybrid", "image"


class HybridSearcher:
    """
    Hybrid search combining BM25 and vector similarity.
    
    Uses Reciprocal Rank Fusion (RRF) to combine rankings from
    different retrieval methods.
    """
    
    def __init__(
        self,
        org_id: str,
        embedder: Optional[Embedder] = None,
        config: Optional[SearchConfig] = None,
    ):
        """
        Initialize the hybrid searcher.
        
        Args:
            org_id: Organization ID for data isolation
            embedder: Embedder for query vectorization
            config: Search configuration
        """
        self.org_id = org_id
        self.embedder = embedder or Embedder()
        self.config = config or SearchConfig.from_settings()
        self._supabase = None
    
    @property
    def supabase(self):
        """Lazy-load Supabase client."""
        if self._supabase is None:
            self._supabase = get_supabase_client()
        return self._supabase
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return (overrides config)
            category: Filter by category
            source_document: Filter by source document
            
        Returns:
            List of SearchResult ordered by relevance
        """
        start_time = time.perf_counter()
        top_k = top_k or self.config.top_k_final
        
        try:
            # Get query embeddings
            text_embedding, image_embedding = await self.embedder.embed_query(query)
            
            # Parallel retrieval
            tasks = []
            
            if self.config.use_vector:
                tasks.append(self._vector_search(
                    text_embedding,
                    category=category,
                    source_document=source_document,
                ))
            
            if self.config.use_bm25:
                tasks.append(self._bm25_search(
                    query,
                    category=category,
                    source_document=source_document,
                ))
            
            if self.config.use_image_search and image_embedding:
                tasks.append(self._image_search(image_embedding))
            
            results_lists = await asyncio.gather(*tasks)
            
            # Combine results
            if self.config.use_hybrid and len(results_lists) > 1:
                combined = self._rrf_fusion(results_lists)
            elif results_lists:
                combined = results_lists[0]
            else:
                combined = []
            
            # Apply filters and limit
            filtered = self._apply_filters(
                combined,
                category=category,
                source_document=source_document,
            )
            
            results = filtered[:top_k]
            
            # Record success metrics
            duration = time.perf_counter() - start_time
            rag_metrics.search_duration.observe(duration)
            rag_metrics.searches_total.inc()
            rag_metrics.search_results_count.observe(len(results))
            
            if self.config.use_hybrid:
                rag_metrics.hybrid_searches.inc()
            elif self.config.use_vector and not self.config.use_bm25:
                rag_metrics.vector_only_searches.inc()
            elif self.config.use_bm25 and not self.config.use_vector:
                rag_metrics.bm25_only_searches.inc()
            
            return results
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            rag_metrics.search_duration.observe(duration)
            rag_metrics.search_errors.inc()
            logger.error(f"Search failed: {e}")
            raise
    
    async def _vector_search(
        self,
        embedding: List[float],
        category: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform vector similarity search."""
        try:
            # Build query
            query = self.supabase.table("knowledge_base_chunks").select(
                "id, content, modality, source_document, page, chunk_index, "
                "ocr_confidence, is_table, table_context, alt_text, category, title"
            ).eq("org_id", self.org_id).not_.is_("vector_embedding", "null")
            
            if category:
                query = query.eq("category", category)
            if source_document:
                query = query.eq("source_document", source_document)
            
            # Execute with vector similarity
            # Note: Supabase doesn't support direct vector ops in Python client,
            # so we use RPC function
            response = await asyncio.to_thread(
                lambda: self.supabase.rpc(
                    "kb_chunks_vector_search",
                    {
                        "p_org_id": self.org_id,
                        "p_embedding": embedding,
                        "p_limit": self.config.top_k_retrieve,
                        "p_category": category,
                        "p_source_document": source_document,
                    }
                ).execute()
            )
            
            results = []
            for row in response.data:
                results.append(SearchResult(
                    chunk_id=row["id"],
                    content=row["content"],
                    modality=row["modality"],
                    source_document=row["source_document"],
                    page=row["page"] or 1,
                    chunk_index=row["chunk_index"] or 0,
                    similarity_score=row.get("similarity", 0.0),
                    ocr_confidence=row.get("ocr_confidence"),
                    is_table=row.get("is_table", False),
                    table_context=row.get("table_context"),
                    alt_text=row.get("alt_text"),
                    retrieval_method="vector",
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Fallback to simple query if RPC fails
            return await self._vector_search_fallback(embedding, category, source_document)
    
    async def _vector_search_fallback(
        self,
        embedding: List[float],
        category: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> List[SearchResult]:
        """Fallback vector search using direct query."""
        try:
            # Use raw SQL via RPC or fetch all and compute similarity in Python
            # This is less efficient but works as fallback
            query = self.supabase.table("knowledge_base_chunks").select(
                "id, content, modality, source_document, page, chunk_index, "
                "ocr_confidence, is_table, table_context, alt_text, category, title, "
                "vector_embedding"
            ).eq("org_id", self.org_id).not_.is_("vector_embedding", "null")
            
            if category:
                query = query.eq("category", category)
            if source_document:
                query = query.eq("source_document", source_document)
            
            response = await asyncio.to_thread(
                lambda: query.limit(self.config.top_k_retrieve * 2).execute()
            )
            
            import numpy as np
            query_vec = np.array(embedding)
            
            results = []
            for row in response.data:
                if not row.get("vector_embedding"):
                    continue
                
                doc_vec = np.array(row["vector_embedding"])
                # Cosine similarity (vectors should be normalized)
                similarity = float(np.dot(query_vec, doc_vec))
                
                results.append(SearchResult(
                    chunk_id=row["id"],
                    content=row["content"],
                    modality=row["modality"],
                    source_document=row["source_document"],
                    page=row.get("page") or 1,
                    chunk_index=row.get("chunk_index") or 0,
                    similarity_score=similarity,
                    ocr_confidence=row.get("ocr_confidence"),
                    is_table=row.get("is_table", False),
                    table_context=row.get("table_context"),
                    alt_text=row.get("alt_text"),
                    category=row.get("category"),
                    title=row.get("title"),
                    retrieval_method="vector_fallback",
                ))
            
            # Sort by similarity
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:self.config.top_k_retrieve]
            
        except Exception as e:
            logger.error(f"Vector search fallback failed: {e}")
            return []
    
    async def _bm25_search(
        self,
        query: str,
        category: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform BM25/FTS search."""
        try:
            # Use RPC function for FTS
            response = await asyncio.to_thread(
                lambda: self.supabase.rpc(
                    "kb_chunks_fts_pt",
                    {
                        "p_org_id": self.org_id,
                        "p_query": query,
                        "p_limit": self.config.top_k_retrieve,
                    }
                ).execute()
            )
            
            results = []
            for row in response.data:
                results.append(SearchResult(
                    chunk_id=row["id"],
                    content=row["content"],
                    modality=row["modality"],
                    source_document=row["source_document"],
                    page=row.get("page") or 1,
                    chunk_index=row.get("chunk_index") or 0,
                    bm25_score=row.get("rank", 0.0),
                    ocr_confidence=row.get("ocr_confidence"),
                    is_table=row.get("is_table", False),
                    table_context=row.get("table_context"),
                    alt_text=row.get("alt_text"),
                    retrieval_method="bm25",
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            # Fallback to ILIKE search
            return await self._bm25_search_fallback(query, category, source_document)
    
    async def _bm25_search_fallback(
        self,
        query: str,
        category: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> List[SearchResult]:
        """Fallback text search using ILIKE."""
        try:
            # Simple ILIKE search
            search_pattern = f"%{query}%"
            
            query_builder = self.supabase.table("knowledge_base_chunks").select(
                "id, content, modality, source_document, page, chunk_index, "
                "ocr_confidence, is_table, table_context, alt_text, category, title"
            ).eq("org_id", self.org_id).ilike("content", search_pattern)
            
            if category:
                query_builder = query_builder.eq("category", category)
            if source_document:
                query_builder = query_builder.eq("source_document", source_document)
            
            response = await asyncio.to_thread(
                lambda: query_builder.limit(self.config.top_k_retrieve).execute()
            )
            
            results = []
            for row in response.data:
                # Compute simple relevance score based on query term frequency
                content_lower = row["content"].lower()
                query_lower = query.lower()
                term_count = content_lower.count(query_lower)
                score = term_count / max(len(content_lower), 1) * 100
                
                results.append(SearchResult(
                    chunk_id=row["id"],
                    content=row["content"],
                    modality=row["modality"],
                    source_document=row["source_document"],
                    page=row.get("page") or 1,
                    chunk_index=row.get("chunk_index") or 0,
                    bm25_score=score,
                    ocr_confidence=row.get("ocr_confidence"),
                    is_table=row.get("is_table", False),
                    table_context=row.get("table_context"),
                    alt_text=row.get("alt_text"),
                    category=row.get("category"),
                    title=row.get("title"),
                    retrieval_method="ilike_fallback",
                ))
            
            results.sort(key=lambda x: x.bm25_score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"BM25 fallback failed: {e}")
            return []
    
    async def _image_search(
        self,
        image_embedding: List[float],
    ) -> List[SearchResult]:
        """Perform image vector search."""
        try:
            response = await asyncio.to_thread(
                lambda: self.supabase.rpc(
                    "kb_chunks_image_search",
                    {
                        "p_org_id": self.org_id,
                        "p_image_embedding": image_embedding,
                        "p_limit": self.config.top_k_image,
                    }
                ).execute()
            )
            
            results = []
            for row in response.data:
                results.append(SearchResult(
                    chunk_id=row["id"],
                    content=row["content"],
                    modality=row["modality"],
                    source_document=row["source_document"],
                    page=row.get("page") or 1,
                    chunk_index=0,
                    similarity_score=row.get("similarity", 0.0),
                    alt_text=row.get("alt_text"),
                    retrieval_method="image",
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []
    
    def _rrf_fusion(
        self,
        results_lists: List[List[SearchResult]],
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank_i)) for each result list
        """
        k = self.config.rrf_k
        
        # Map chunk_id to result and RRF scores
        chunk_results: Dict[str, SearchResult] = {}
        chunk_rrf_scores: Dict[str, float] = {}
        
        for results in results_lists:
            for rank, result in enumerate(results):
                chunk_id = result.chunk_id
                rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed
                
                if chunk_id not in chunk_results:
                    chunk_results[chunk_id] = result
                    chunk_rrf_scores[chunk_id] = 0.0
                
                chunk_rrf_scores[chunk_id] += rrf_score
                
                # Keep the best individual scores
                existing = chunk_results[chunk_id]
                if result.similarity_score > existing.similarity_score:
                    existing.similarity_score = result.similarity_score
                if result.bm25_score > existing.bm25_score:
                    existing.bm25_score = result.bm25_score
        
        # Update RRF scores and sort
        for chunk_id, result in chunk_results.items():
            result.rrf_score = chunk_rrf_scores[chunk_id]
            result.retrieval_method = "hybrid"
        
        combined = list(chunk_results.values())
        combined.sort(key=lambda x: x.rrf_score, reverse=True)
        
        return combined
    
    def _apply_filters(
        self,
        results: List[SearchResult],
        category: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> List[SearchResult]:
        """Apply additional filters to results."""
        filtered = results
        
        if self.config.min_similarity > 0:
            filtered = [
                r for r in filtered
                if r.similarity_score >= self.config.min_similarity
                or r.bm25_score > 0
            ]
        
        if category:
            filtered = [r for r in filtered if r.category == category]
        
        if source_document:
            filtered = [r for r in filtered if r.source_document == source_document]
        
        return filtered
