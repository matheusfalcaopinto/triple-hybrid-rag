"""RAG reranking routes."""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from control_plane.schemas.rag import (
    DocumentResult,
    RerankHealthResponse,
    RerankRequest,
    RerankResponse,
    RerankedDocument,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global reranker instance (lazy-loaded)
_reranker_instance = None
_reranker_model_name: str | None = None


def _get_reranker():
    """Get or create the global reranker instance."""
    global _reranker_instance, _reranker_model_name
    
    if _reranker_instance is not None:
        return _reranker_instance
    
    try:
        from sentence_transformers import CrossEncoder
        
        # Default model for reranking
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        _reranker_instance = CrossEncoder(model_name)
        _reranker_model_name = model_name
        logger.info(f"Loaded CrossEncoder model: {model_name}")
        return _reranker_instance
        
    except ImportError as e:
        logger.error(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="sentence-transformers not installed"
        ) from e
    except Exception as e:
        logger.error(f"Failed to load CrossEncoder model: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load CrossEncoder model: {str(e)}"
        ) from e


@router.get("/health", response_model=RerankHealthResponse)
async def rerank_health() -> RerankHealthResponse:
    """Check if the reranker service is healthy and model is loaded."""
    global _reranker_instance, _reranker_model_name
    
    model_loaded = _reranker_instance is not None
    model_name = _reranker_model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    return RerankHealthResponse(
        status="healthy" if model_loaded else "not_loaded",
        model_loaded=model_loaded,
        model_name=model_name,
    )


@router.post("/warmup", response_model=RerankHealthResponse)
async def warmup_reranker() -> RerankHealthResponse:
    """
    Pre-load the CrossEncoder model for faster inference.
    
    Call this endpoint during application startup to avoid
    cold-start latency on the first rerank request.
    """
    start_time = time.monotonic()
    
    model = _get_reranker()
    
    # Warm up with a dummy prediction
    model.predict([("test query", "test document")])
    
    warmup_time = (time.monotonic() - start_time) * 1000
    logger.info(f"CrossEncoder warmed up in {warmup_time:.1f}ms")
    
    return RerankHealthResponse(
        status="warmed_up",
        model_loaded=True,
        model_name=_reranker_model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """
    Rerank documents using CrossEncoder model.
    
    Takes a query and a list of documents, scores each query-document pair
    using a cross-encoder model, and returns the documents sorted by
    relevance score.
    
    The CrossEncoder model computes a relevance score for each (query, document)
    pair by processing them jointly, which provides more accurate relevance
    estimation than separate embedding comparison.
    
    Args:
        request: Rerank request with query and documents
        
    Returns:
        Reranked documents with scores
    """
    start_time = time.monotonic()
    
    if not request.documents:
        return RerankResponse(
            query=request.query,
            model=request.model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            documents=[],
            processing_time_ms=0.0,
        )
    
    # Get or load the model
    model = _get_reranker()
    
    # Prepare query-document pairs
    pairs = []
    for doc in request.documents:
        # Build document text with title if available
        doc_text = doc.content
        if doc.title:
            doc_text = f"Title: {doc.title}\n{doc_text}"
        pairs.append((request.query, doc_text))
    
    # Score pairs using CrossEncoder
    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.error(f"CrossEncoder prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reranking failed: {str(e)}"
        ) from e
    
    # Create scored documents with original ranks
    scored_docs = []
    for i, (doc, score) in enumerate(zip(request.documents, scores)):
        scored_docs.append(
            RerankedDocument(
                id=doc.id,
                content=doc.content,
                title=doc.title,
                metadata=doc.metadata,
                rerank_score=float(score),
                original_rank=i,
                similarity_score=doc.similarity_score,
                bm25_score=doc.bm25_score,
                rrf_score=doc.rrf_score,
            )
        )
    
    # Sort by rerank score (descending)
    scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
    
    # Return top-k results
    top_k = min(request.top_k, len(scored_docs))
    result_docs = scored_docs[:top_k]
    
    processing_time_ms = (time.monotonic() - start_time) * 1000
    
    logger.info(
        f"Reranked {len(request.documents)} docs -> top {top_k} in {processing_time_ms:.1f}ms"
    )
    
    return RerankResponse(
        query=request.query,
        model=_reranker_model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        documents=result_docs,
        processing_time_ms=processing_time_ms,
    )


@router.post("/score", response_model=dict)
async def score_pair(
    query: str,
    document: str,
) -> dict:
    """
    Score a single query-document pair.
    
    This is a convenience endpoint for testing and debugging.
    For batch processing, use the /rerank endpoint.
    
    Args:
        query: The search query
        document: The document text
        
    Returns:
        Relevance score for the pair
    """
    start_time = time.monotonic()
    
    model = _get_reranker()
    
    try:
        score = model.predict([(query, document)])[0]
    except Exception as e:
        logger.error(f"CrossEncoder scoring failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {str(e)}"
        ) from e
    
    processing_time_ms = (time.monotonic() - start_time) * 1000
    
    return {
        "query": query,
        "document": document[:200] + "..." if len(document) > 200 else document,
        "score": float(score),
        "model": _reranker_model_name,
        "processing_time_ms": processing_time_ms,
    }
