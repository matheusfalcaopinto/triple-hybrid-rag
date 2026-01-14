"""RAG (Retrieval-Augmented Generation) schemas."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentResult(BaseModel):
    """A document result for reranking."""
    
    id: str = Field(..., description="Unique identifier for the document/chunk")
    content: str = Field(..., description="Document text content")
    title: str | None = Field(None, description="Document title if available")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
    
    # Pre-existing scores (optional)
    similarity_score: float | None = Field(None, description="Vector similarity score")
    bm25_score: float | None = Field(None, description="BM25 relevance score")
    rrf_score: float | None = Field(None, description="RRF fusion score")


class RerankRequest(BaseModel):
    """Request to rerank search results using CrossEncoder."""
    
    query: str = Field(..., description="Original search query")
    documents: list[DocumentResult] = Field(..., description="List of documents to rerank")
    top_k: int = Field(5, ge=1, le=100, description="Number of top results to return")
    model: str | None = Field(None, description="CrossEncoder model name (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "documents": [
                    {
                        "id": "doc1",
                        "content": "Machine learning is a branch of AI...",
                        "title": "ML Guide",
                        "similarity_score": 0.85
                    },
                    {
                        "id": "doc2",
                        "content": "Pizza is a popular Italian dish...",
                        "title": "Recipes",
                        "similarity_score": 0.75
                    }
                ],
                "top_k": 5
            }
        }


class RerankedDocument(BaseModel):
    """A document with rerank score."""
    
    id: str = Field(..., description="Unique identifier for the document/chunk")
    content: str = Field(..., description="Document text content")
    title: str | None = Field(None, description="Document title if available")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
    
    # Scores
    rerank_score: float = Field(..., description="CrossEncoder rerank score")
    original_rank: int = Field(..., description="Original position before reranking")
    similarity_score: float | None = Field(None, description="Original similarity score")
    bm25_score: float | None = Field(None, description="Original BM25 score")
    rrf_score: float | None = Field(None, description="Original RRF score")


class RerankResponse(BaseModel):
    """Response from reranking endpoint."""
    
    query: str = Field(..., description="Original search query")
    model: str = Field(..., description="CrossEncoder model used")
    documents: list[RerankedDocument] = Field(..., description="Reranked documents")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "documents": [
                    {
                        "id": "doc1",
                        "content": "Machine learning is a branch of AI...",
                        "title": "ML Guide",
                        "rerank_score": 9.5,
                        "original_rank": 0,
                        "similarity_score": 0.85
                    }
                ],
                "processing_time_ms": 45.3
            }
        }


class RerankHealthResponse(BaseModel):
    """Health check response for reranker service."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the CrossEncoder model is loaded")
    model_name: str = Field(..., description="Name of the CrossEncoder model")
