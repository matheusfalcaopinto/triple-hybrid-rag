"""
Multimodal Embedder for Triple-Hybrid-RAG

Supports:
- Text embeddings via OpenAI-compatible API (Qwen3-VL-Embedding-2B)
- Image embeddings for direct multimodal retrieval
- Mixed text+image embeddings
- Matryoshka truncation (2048d → 1024d)
"""

import base64
import logging
from io import BytesIO
from typing import List, Optional, Union

import httpx
import numpy as np

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import ChildChunk

logger = logging.getLogger(__name__)

class MultimodalEmbedder:
    """
    Multimodal embedder supporting text and image inputs.
    
    Uses Qwen3-VL-Embedding via OpenAI-compatible API for:
    - Text embeddings
    - Image embeddings (direct vision encoding)
    - Mixed text+image embeddings
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the embedder with configuration."""
        self.config = config or get_settings()
        self.api_base = self.config.rag_embed_api_base.rstrip("/")
        self.model = self.config.rag_embed_model
        self.batch_size = self.config.rag_embed_batch_size
        self.timeout = self.config.rag_embed_timeout
        self.dim_model = self.config.rag_embed_dim_model
        self.dim_store = self.config.rag_embed_dim_store
        self.use_matryoshka = self.config.rag_matryoshka_embeddings
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _truncate_embedding(self, embedding: List[float]) -> List[float]:
        """
        Truncate embedding using Matryoshka property.
        
        Matryoshka embeddings maintain semantic meaning when truncated
        to smaller dimensions (2048 → 1024).
        """
        if not self.use_matryoshka:
            return embedding
        
        if len(embedding) <= self.dim_store:
            return embedding
        
        # Truncate and normalize
        truncated = embedding[:self.dim_store]
        norm = np.linalg.norm(truncated)
        if norm > 0:
            truncated = (np.array(truncated) / norm).tolist()
        
        return truncated
    
    async def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings to embed
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        client = await self._get_client()
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = await client.post(
                    f"{self.api_base}/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                
                data = response.json()
                batch_embeddings = [item["embedding"] for item in data["data"]]
                
                # Apply Matryoshka truncation
                batch_embeddings = [self._truncate_embedding(e) for e in batch_embeddings]
                
                if normalize:
                    batch_embeddings = [
                        self._normalize(e) for e in batch_embeddings
                    ]
                
                all_embeddings.extend(batch_embeddings)
                
            except httpx.HTTPError as e:
                logger.error(f"Embedding API error: {e}")
                # Return zero embeddings for failed batch
                all_embeddings.extend([[0.0] * self.dim_store] * len(batch))
        
        return all_embeddings
    
    async def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """Embed a single text string."""
        embeddings = await self.embed_texts([text], normalize=normalize)
        return embeddings[0] if embeddings else [0.0] * self.dim_store
    
    async def embed_images(
        self,
        images: List[bytes],
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Embed a list of images directly (multimodal).
        
        Args:
            images: List of image bytes (PNG/JPEG)
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        if not images:
            return []
        
        if not self.config.rag_multimodal_embedding_enabled:
            logger.warning("Multimodal embeddings disabled, returning zero vectors")
            return [[0.0] * self.dim_store] * len(images)
        
        client = await self._get_client()
        all_embeddings = []
        
        for image_bytes in images:
            try:
                # Encode image as base64
                b64_image = base64.b64encode(image_bytes).decode("utf-8")
                
                # Use vision-compatible endpoint
                response = await client.post(
                    f"{self.api_base}/embeddings",
                    json={
                        "model": self.model,
                        "input": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}"
                                }
                            }
                        ],
                        "encoding_format": "float",
                    },
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data["data"][0]["embedding"]
                embedding = self._truncate_embedding(embedding)
                
                if normalize:
                    embedding = self._normalize(embedding)
                
                all_embeddings.append(embedding)
                
            except httpx.HTTPError as e:
                logger.error(f"Image embedding API error: {e}")
                all_embeddings.append([0.0] * self.dim_store)
        
        return all_embeddings
    
    async def embed_image(self, image: bytes, normalize: bool = True) -> List[float]:
        """Embed a single image."""
        embeddings = await self.embed_images([image], normalize=normalize)
        return embeddings[0] if embeddings else [0.0] * self.dim_store
    
    async def embed_mixed(
        self,
        text: str,
        image: bytes,
        normalize: bool = True,
    ) -> List[float]:
        """
        Embed text and image together (joint representation).
        
        Uses multimodal model to create a unified embedding
        that captures both textual and visual information.
        """
        if not self.config.rag_multimodal_embedding_enabled:
            return await self.embed_text(text, normalize=normalize)
        
        client = await self._get_client()
        
        try:
            b64_image = base64.b64encode(image).decode("utf-8")
            
            response = await client.post(
                f"{self.api_base}/embeddings",
                json={
                    "model": self.model,
                    "input": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            }
                        }
                    ],
                    "encoding_format": "float",
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = data["data"][0]["embedding"]
            embedding = self._truncate_embedding(embedding)
            
            if normalize:
                embedding = self._normalize(embedding)
            
            return embedding
            
        except httpx.HTTPError as e:
            logger.error(f"Mixed embedding API error: {e}")
            return [0.0] * self.dim_store
    
    async def embed_chunks(
        self,
        chunks: List[ChildChunk],
        include_images: bool = True,
    ) -> List[ChildChunk]:
        """
        Embed a list of child chunks (text and optionally images).
        
        Args:
            chunks: List of ChildChunk objects
            include_images: Whether to embed images separately
            
        Returns:
            Chunks with embeddings populated
        """
        # Separate text and image chunks
        text_chunks = []
        image_chunks = []
        
        for chunk in chunks:
            if chunk.image_data and include_images:
                image_chunks.append(chunk)
            else:
                text_chunks.append(chunk)
        
        # Embed text chunks
        if text_chunks:
            texts = [c.text for c in text_chunks]
            text_embeddings = await self.embed_texts(texts)
            
            for chunk, embedding in zip(text_chunks, text_embeddings):
                chunk.embedding = embedding
        
        # Embed image chunks
        if image_chunks and self.config.rag_multimodal_embedding_enabled:
            images = [c.image_data for c in image_chunks]
            image_embeddings = await self.embed_images(images)
            
            for chunk, embedding in zip(image_chunks, image_embeddings):
                chunk.image_embedding = embedding
                # Also embed alt text if available
                if chunk.text and chunk.text != "[Image]":
                    chunk.embedding = await self.embed_text(chunk.text)
        
        return chunks
    
    def _normalize(self, embedding: List[float]) -> List[float]:
        """L2 normalize an embedding vector."""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
    
    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(arr1, arr2) / (norm1 * norm2))
