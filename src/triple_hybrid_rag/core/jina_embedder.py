"""
Jina AI Embedder for Triple-Hybrid-RAG

Multimodal embedder using Jina AI API:
- Text embeddings via jina-embeddings-v4
- Image embeddings (direct multimodal support)
- Matryoshka dimension truncation
- Rate limiting and retry logic
"""

import asyncio
import base64
import logging
from typing import Callable, List, Optional, Union

import aiohttp
import httpx
import numpy as np

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import ChildChunk

logger = logging.getLogger(__name__)


class JinaEmbedder:
    """
    Jina AI multimodal embedder.
    
    Uses Jina AI API for:
    - Text embeddings (jina-embeddings-v4)
    - Image embeddings (multimodal support)
    - Late interaction embeddings
    
    Supports task-specific embeddings:
    - retrieval.query: For query embeddings
    - retrieval.passage: For document/passage embeddings
    """
    
    JINA_API_BASE = "https://api.jina.ai/v1"
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the Jina embedder with configuration."""
        self.config = config or get_settings()
        
        # Jina-specific settings
        self.api_key = self.config.jina_api_key
        self.api_base = self.config.jina_api_base.rstrip("/")
        self.model = self.config.jina_embed_model
        self.dimensions = self.config.jina_embed_dimensions
        self.task_query = self.config.jina_embed_task_query
        self.task_passage = self.config.jina_embed_task_passage
        self.batch_size = self.config.jina_embed_batch_size
        self.timeout = self.config.jina_embed_timeout
        
        # Rate limiting (500 RPM = ~8.3 req/sec, be conservative)
        self._rate_limit_delay = 0.15  # 150ms between requests
        self._last_request_time = 0.0
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        import time
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    async def embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        normalize: bool = True,
        raise_on_error: bool = False,
    ) -> List[List[float]]:
        """
        Embed a list of texts using Jina API.
        
        Args:
            texts: List of text strings to embed
            task: Task type ('retrieval.query' or 'retrieval.passage')
                  Defaults to retrieval.passage for documents
            normalize: Whether to L2 normalize embeddings
            raise_on_error: Raise exception on API error
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if not self.api_key:
            logger.error("JINA_API_KEY not configured")
            if raise_on_error:
                raise ValueError("JINA_API_KEY not configured")
            return [[0.0] * self.dimensions] * len(texts)
        
        task = task or self.task_passage
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                await self._rate_limit()
                client = await self._get_client()
                
                # Build Jina-format input
                input_data = [{"text": t} for t in batch]
                
                response = await client.post(
                    f"{self.api_base}/embeddings",
                    json={
                        "model": self.model,
                        "task": task,
                        "dimensions": self.dimensions,
                        "input": input_data,
                    },
                )
                response.raise_for_status()
                
                data = response.json()
                batch_embeddings = [item["embedding"] for item in data["data"]]
                
                if normalize:
                    batch_embeddings = [
                        self._normalize(e) for e in batch_embeddings
                    ]
                
                all_embeddings.extend(batch_embeddings)
                
            except httpx.HTTPError as e:
                logger.error(f"Jina Embedding API error: {e}")
                if raise_on_error:
                    raise
                # Return zero embeddings for failed batch
                all_embeddings.extend([[0.0] * self.dimensions] * len(batch))
        
        return all_embeddings
    
    async def embed_text(
        self,
        text: str,
        task: Optional[str] = None,
        normalize: bool = True,
    ) -> List[float]:
        """Embed a single text string."""
        task = task or self.task_passage
        embeddings = await self.embed_texts([text], task=task, normalize=normalize)
        return embeddings[0] if embeddings else [0.0] * self.dimensions
    
    async def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """
        Embed a query string using query-specific task.
        
        Uses 'retrieval.query' task for optimal query embeddings.
        """
        return await self.embed_text(query, task=self.task_query, normalize=normalize)
    
    async def embed_images(
        self,
        images: List[bytes],
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Embed images directly using Jina multimodal API.
        
        Args:
            images: List of image bytes (PNG/JPEG)
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        if not images:
            return []
        
        if not self.api_key:
            logger.error("JINA_API_KEY not configured")
            return [[0.0] * self.dimensions] * len(images)
        
        if not self.config.rag_multimodal_embedding_enabled:
            logger.warning("Multimodal embeddings disabled, returning zero vectors")
            return [[0.0] * self.dimensions] * len(images)
        
        all_embeddings = []
        
        for image_bytes in images:
            try:
                await self._rate_limit()
                client = await self._get_client()
                
                # Encode image as base64
                b64_image = base64.b64encode(image_bytes).decode("utf-8")
                
                response = await client.post(
                    f"{self.api_base}/embeddings",
                    json={
                        "model": self.model,
                        "dimensions": self.dimensions,
                        "input": [{"image": b64_image}],
                    },
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data["data"][0]["embedding"]
                
                if normalize:
                    embedding = self._normalize(embedding)
                
                all_embeddings.append(embedding)
                
            except httpx.HTTPError as e:
                logger.error(f"Jina Image Embedding API error: {e}")
                all_embeddings.append([0.0] * self.dimensions)
        
        return all_embeddings
    
    async def embed_image(self, image: bytes, normalize: bool = True) -> List[float]:
        """Embed a single image."""
        embeddings = await self.embed_images([image], normalize=normalize)
        return embeddings[0] if embeddings else [0.0] * self.dimensions
    
    async def embed_mixed(
        self,
        text: str,
        image: bytes,
        normalize: bool = True,
    ) -> List[float]:
        """
        Embed text and image together.
        
        Note: Jina API embeds each modality separately.
        For joint representation, we average the embeddings.
        """
        if not self.config.rag_multimodal_embedding_enabled:
            return await self.embed_text(text, normalize=normalize)
        
        # Get both embeddings
        text_emb = await self.embed_text(text, normalize=False)
        image_emb = await self.embed_image(image, normalize=False)
        
        # Average and normalize
        combined = [(t + i) / 2 for t, i in zip(text_emb, image_emb)]
        
        if normalize:
            combined = self._normalize(combined)
        
        return combined
    
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
            text_embeddings = await self.embed_texts(texts, task=self.task_passage)
            
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
                    chunk.embedding = await self.embed_text(chunk.text, task=self.task_passage)
        
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONCURRENT EMBEDDING (OPTIMIZATION)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _embed_batch_aiohttp(
        self,
        session: aiohttp.ClientSession,
        batch: List[str],
        batch_index: int,
        task: str,
    ) -> tuple[int, List[List[float]]]:
        """
        Embed a single batch using aiohttp (for concurrent processing).
        
        Returns:
            Tuple of (batch_index, embeddings) for ordered reconstruction
        """
        try:
            await self._rate_limit()
            
            input_data = [{"text": t} for t in batch]
            
            async with session.post(
                f"{self.api_base}/embeddings",
                json={
                    "model": self.model,
                    "task": task,
                    "dimensions": self.dimensions,
                    "input": input_data,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Jina embedding batch {batch_index} failed: {error_text}")
                    return (batch_index, [[0.0] * self.dimensions] * len(batch))
                
                data = await response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                embeddings = [self._normalize(e) for e in embeddings]
                
                return (batch_index, embeddings)
                
        except Exception as e:
            logger.error(f"Jina embedding batch {batch_index} error: {e}")
            return (batch_index, [[0.0] * self.dimensions] * len(batch))
    
    async def embed_texts_concurrent(
        self,
        texts: List[str],
        task: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        Embed texts with concurrent batch processing using aiohttp.
        
        Note: Respects Jina rate limits (500 RPM).
        
        Args:
            texts: List of text strings to embed
            task: Task type for embeddings
            progress_callback: Optional callback(completed, total) for progress
            
        Returns:
            List of embedding vectors (in same order as input)
        """
        if not texts:
            return []
        
        if not self.api_key:
            logger.error("JINA_API_KEY not configured")
            return [[0.0] * self.dimensions] * len(texts)
        
        task = task or self.task_passage
        
        # Limit concurrency to respect rate limits (max 4 concurrent)
        concurrent_batches = min(4, self.config.rag_embed_concurrent_batches)
        
        # Create batches
        batches: List[tuple[int, List[str]]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batches.append((len(batches), batch))
        
        total_batches = len(batches)
        logger.info(
            f"Jina: Embedding {len(texts)} texts in {total_batches} batches "
            f"({concurrent_batches} concurrent)"
        )
        
        # Connection pooling
        connector = aiohttp.TCPConnector(limit=concurrent_batches * 2)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        all_results: List[tuple[int, List[List[float]]]] = []
        completed = 0
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            # Process batches in groups
            for group_start in range(0, total_batches, concurrent_batches):
                group_end = min(group_start + concurrent_batches, total_batches)
                current_group = batches[group_start:group_end]
                
                # Fire concurrent requests
                tasks = [
                    self._embed_batch_aiohttp(session, batch, idx, task)
                    for idx, batch in current_group
                ]
                
                # Wait for group
                group_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in group_results:
                    if isinstance(result, BaseException):
                        logger.error(f"Batch failed with exception: {result}")
                        continue
                    all_results.append(result)  # type: ignore
                
                completed += len(current_group)
                if progress_callback:
                    progress_callback(completed, total_batches)
        
        # Sort by batch index and flatten
        all_results.sort(key=lambda x: x[0])
        embeddings = []
        for _, batch_embeddings in all_results:
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def embed_chunks_concurrent(
        self,
        chunks: List[ChildChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ChildChunk]:
        """
        Embed chunks with concurrent batch processing.
        
        Args:
            chunks: List of ChildChunk objects
            progress_callback: Optional callback for progress tracking
            
        Returns:
            Chunks with embeddings populated
        """
        # Separate text and image chunks
        text_chunks = []
        image_chunks = []
        
        for chunk in chunks:
            if chunk.image_data:
                image_chunks.append(chunk)
            else:
                text_chunks.append(chunk)
        
        # Embed text chunks concurrently
        if text_chunks:
            texts = [c.text for c in text_chunks]
            text_embeddings = await self.embed_texts_concurrent(
                texts, task=self.task_passage, progress_callback=progress_callback
            )
            
            for chunk, embedding in zip(text_chunks, text_embeddings):
                chunk.embedding = embedding
        
        # Embed image chunks (sequential due to rate limits)
        if image_chunks and self.config.rag_multimodal_embedding_enabled:
            images = [c.image_data for c in image_chunks]
            image_embeddings = await self.embed_images(images)
            
            for chunk, embedding in zip(image_chunks, image_embeddings):
                chunk.image_embedding = embedding
        
        return chunks
