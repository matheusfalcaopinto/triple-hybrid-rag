"""
RAG 2.0 Embedder Module

Matryoshka Representation Learning (MRL) embedder:
- Generates embeddings using Qwen3-VL-Embedding (4096d)
- Truncates to 1024d (prefix) for storage efficiency
- L2 normalization for cosine similarity search
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]  # Truncated & normalized
    full_embedding: Optional[List[float]] = None  # Original 4096d (if requested)
    error: Optional[str] = None


def normalize_l2(embedding: List[float]) -> List[float]:
    """L2-normalize an embedding vector."""
    arr = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return list(arr.tolist())


def truncate_matryoshka(
    embedding: List[float],
    target_dim: int = 1024,
    normalize: bool = True,
) -> List[float]:
    """
    Truncate embedding using Matryoshka Representation Learning (MRL).
    
    MRL models are trained so that the first N dimensions form a valid
    embedding at reduced dimensionality. We simply take the first 
    `target_dim` elements.
    
    Args:
        embedding: Original embedding (e.g., 4096d)
        target_dim: Target dimensions (e.g., 1024)
        normalize: Whether to L2-normalize after truncation
        
    Returns:
        Truncated (and optionally normalized) embedding
    """
    if len(embedding) <= target_dim:
        return normalize_l2(embedding) if normalize else embedding
    
    truncated = embedding[:target_dim]
    
    if normalize:
        return normalize_l2(truncated)
    
    return truncated


class RAG2Embedder:
    """
    RAG 2.0 Embedder with Matryoshka truncation.
    
    Uses Qwen3-VL-Embedding to generate 4096d embeddings,
    then truncates to 1024d for storage in pgvector.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        store_dim: Optional[int] = None,
        model_dim: Optional[int] = None,
        batch_size: int = 20,
        keep_full: bool = False,
    ):
        """
        Initialize the RAG2 embedder.
        
        Args:
            model: Embedding model name (default from config)
            api_base: API base URL for local embedding server
            store_dim: Target storage dimensions (default: 1024)
            model_dim: Model output dimensions (default: 4096)
            batch_size: Batch size for embedding requests
            keep_full: Whether to also return the full 4096d embedding
        """
        self.model = model or SETTINGS.rag_embed_model
        self.api_base = api_base or SETTINGS.rag_embed_api_base
        self.store_dim = store_dim or SETTINGS.rag2_embed_dim_store
        self.model_dim = model_dim or SETTINGS.rag2_embed_dim_model
        self.batch_size = batch_size
        self.keep_full = keep_full
        
        # Lazy-loaded client
        self._client: Optional[Any] = None
    
    @property
    def client(self) -> Any:
        """Lazy-load OpenAI client for local API."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.api_base,
                api_key="not-needed",
            )
        return self._client
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with truncated 1024d embedding
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
            )
            
            full_embedding = response.data[0].embedding
            
            # Verify model output dimension
            if len(full_embedding) != self.model_dim:
                logger.warning(
                    f"Unexpected embedding dim: got {len(full_embedding)}, "
                    f"expected {self.model_dim}"
                )
            
            # Truncate and normalize
            truncated = truncate_matryoshka(
                full_embedding,
                target_dim=self.store_dim,
                normalize=True,
            )
            
            return EmbeddingResult(
                text=text,
                embedding=truncated,
                full_embedding=full_embedding if self.keep_full else None,
            )
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return EmbeddingResult(
                text=text,
                embedding=[],
                error=str(e),
            )
    
    def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                )
                
                for j, item in enumerate(response.data):
                    full_embedding = item.embedding
                    
                    truncated = truncate_matryoshka(
                        full_embedding,
                        target_dim=self.store_dim,
                        normalize=True,
                    )
                    
                    results.append(EmbeddingResult(
                        text=batch[j],
                        embedding=truncated,
                        full_embedding=full_embedding if self.keep_full else None,
                    ))
                    
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Add error results for this batch
                for text in batch:
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=[],
                        error=str(e),
                    ))
        
        return results
    
    async def embed_text_async(self, text: str) -> EmbeddingResult:
        """Async version of embed_text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)
    
    async def embed_texts_async(self, texts: List[str]) -> List[EmbeddingResult]:
        """Async version of embed_texts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query for retrieval.
        
        Returns just the embedding vector (for use with pgvector).
        
        Args:
            query: Query text
            
        Returns:
            1024d embedding vector (normalized)
        """
        result = self.embed_text(query)
        if result.error:
            raise ValueError(f"Query embedding failed: {result.error}")
        return result.embedding


# Convenience function
def get_rag2_embedder(**kwargs: Any) -> RAG2Embedder:
    """Get a configured RAG2 embedder instance."""
    return RAG2Embedder(**kwargs)
