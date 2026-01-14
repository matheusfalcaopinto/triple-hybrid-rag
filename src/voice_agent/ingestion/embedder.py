"""
Embedding Generation Module

Handles unified multi-modal embedding generation for RAG using Qwen3-VL-Embedding:
- Text embeddings via local Qwen3-VL-Embedding-8B (4096d)
- Image embeddings via Qwen3-VL-Embedding-8B (4096d)
- Single model handles both text and images with multilingual support
- Batching and normalization
"""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from voice_agent.config import SETTINGS
from voice_agent.ingestion.chunker import Chunk, ChunkType
from voice_agent.observability.rag_metrics import rag_metrics

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    chunk: Chunk
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    error: Optional[str] = None


def normalize_embedding(embedding: List[float]) -> List[float]:
    """L2-normalize an embedding vector."""
    arr = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def truncate_embedding(embedding: List[float], target_dim: int) -> List[float]:
    """
    Truncate embedding to target dimensions.
    
    This is used to fit 4096d Qwen3-VL embeddings into pgvector's 
    4000d halfvec HNSW index limit. Truncating the last 96 dimensions
    (2.3%) has negligible impact on retrieval quality.
    
    Args:
        embedding: Original embedding vector
        target_dim: Target number of dimensions
        
    Returns:
        Truncated embedding vector
    """
    if len(embedding) <= target_dim:
        return embedding
    return embedding[:target_dim]


class Embedder:
    """
    Generate embeddings for text and images using unified Qwen3-VL-Embedding model.
    
    Uses a single model (qwen3-vl-embedding-8b) for both text and image embeddings,
    which provides:
    - 4096-dimensional embeddings
    - Multilingual text support (30+ languages)
    - Native multimodal understanding
    - Consistent embedding space for text and images
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = 20,
        enable_image_embeddings: bool = False,
        normalize: bool = True,
        use_local: Optional[bool] = None,
        api_base: Optional[str] = None,
        # Legacy parameters for backward compatibility
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
    ):
        """
        Initialize the embedder.
        
        Args:
            model: Unified embedding model (default: qwen3-vl-embedding-8b)
            batch_size: Batch size for embedding requests
            enable_image_embeddings: Whether to generate image embeddings
            normalize: Whether to L2-normalize embeddings
            use_local: Whether to use local embedding API (default from config)
            api_base: Base URL for local embedding API (default from config)
            text_model: (Deprecated) Alias for model
            image_model: (Deprecated) Alias for model
        """
        # Use unified model - legacy params are ignored
        self.model = model or SETTINGS.rag_embed_model
        self.batch_size = batch_size or SETTINGS.rag_embed_batch_size
        self.enable_image_embeddings = enable_image_embeddings or SETTINGS.rag_enable_image_embeddings
        self.normalize = normalize
        self.use_local = use_local if use_local is not None else SETTINGS.rag_embed_use_local
        self.api_base = api_base or SETTINGS.rag_embed_api_base
        
        # Legacy aliases for backward compatibility
        self.text_model = self.model
        self.image_model = self.model
        
        # Lazy-loaded clients
        self._openai_client = None
        self._local_client = None
        self._siglip_model = None
        self._siglip_processor = None
        self._siglip_device = None
    
    @property
    def openai_client(self):
        """Lazy-load OpenAI client for OpenAI API."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=SETTINGS.openai_api_key)
        return self._openai_client
    
    @property
    def local_client(self):
        """Lazy-load OpenAI client for local API (LM Studio, etc.)."""
        if self._local_client is None:
            from openai import OpenAI
            self._local_client = OpenAI(
                base_url=self.api_base,
                api_key="not-needed",  # Local servers typically don't need API keys
            )
        return self._local_client
    
    def _load_siglip(self):
        """Lazy-load SigLIP model and processor."""
        if self._siglip_model is None:
            try:
                import torch
                from transformers import AutoProcessor
                
                # Try to import SigLIP-specific model first, fall back to AutoModel
                try:
                    from transformers import SiglipModel
                    model_class = SiglipModel
                except ImportError:
                    from transformers import AutoModel
                    model_class = AutoModel
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._siglip_processor = AutoProcessor.from_pretrained(self.image_model)
                self._siglip_model = model_class.from_pretrained(self.image_model).to(device)
                self._siglip_device = device
                logger.info(f"Loaded SigLIP model ({model_class.__name__}) on {device}")
            except Exception as e:
                logger.error(f"Failed to load SigLIP model: {e}")
                raise
    
    async def embed_chunks(
        self,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects
            progress_callback: Optional callback(processed, total) for progress
            
        Returns:
            List of EmbeddingResult objects
        """
        start_time = time.perf_counter()
        results = []
        total = len(chunks)
        
        # Separate text and image chunks
        text_chunks = [c for c in chunks if c.chunk_type in (ChunkType.TEXT, ChunkType.TABLE, ChunkType.IMAGE_CAPTION)]
        image_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMAGE and c.image_data]
        
        # Process text chunks in batches
        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i:i + self.batch_size]
            batch_results = await self._embed_text_batch(batch)
            results.extend(batch_results)
            
            if progress_callback:
                progress_callback(len(results), total)
        
        # Process image chunks if enabled
        if self.enable_image_embeddings and image_chunks:
            image_results = await self._embed_images(image_chunks)
            results.extend(image_results)
            
            if progress_callback:
                progress_callback(len(results), total)
        elif image_chunks:
            # Add image chunks without embeddings
            for chunk in image_chunks:
                results.append(EmbeddingResult(
                    chunk=chunk,
                    error="Image embeddings disabled",
                ))
        
        # Record metrics
        duration = time.perf_counter() - start_time
        successful = sum(1 for r in results if r.text_embedding is not None or r.image_embedding is not None)
        failed = len(results) - successful
        text_count = len(text_chunks)
        image_count = len(image_chunks) if self.enable_image_embeddings else 0
        
        rag_metrics.embedding_duration.observe(duration)
        rag_metrics.text_embeddings_generated.inc(text_count)
        if image_count > 0:
            rag_metrics.image_embeddings_generated.inc(image_count)
        if failed > 0:
            rag_metrics.embedding_errors.inc(failed)
        
        return results
    
    async def _embed_text_batch(
        self,
        chunks: List[Chunk],
    ) -> List[EmbeddingResult]:
        """Embed a batch of text chunks using OpenAI."""
        if not chunks:
            return []
        
        texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]
        
        try:
            # Run OpenAI API call in thread to avoid blocking
            response = await asyncio.to_thread(
                self._call_openai_embeddings,
                texts,
            )
            
            results = []
            for i, chunk in enumerate(chunks):
                embedding = response[i]
                if self.normalize:
                    embedding = normalize_embedding(embedding)
                # Truncate to fit pgvector halfvec index limit (4096 -> 4000)
                embedding = truncate_embedding(embedding, SETTINGS.rag_vector_dim)
                
                results.append(EmbeddingResult(
                    chunk=chunk,
                    text_embedding=embedding,
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return [
                EmbeddingResult(chunk=chunk, error=str(e))
                for chunk in chunks
            ]
    
    def _call_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Call embeddings API (sync) - uses local or OpenAI based on config."""
        client = self.local_client if self.use_local else self.openai_client
        response = client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]
    
    def _call_local_image_embeddings(self, image_data: bytes) -> List[float]:
        """
        Call local Qwen3-VL-Embedding model for image embeddings.
        
        Uses the same model as text embeddings (unified qwen3-vl-embedding-8b).
        The model natively supports image inputs via the embeddings API.
        """
        # Encode image to base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # Use the embeddings endpoint with image
        try:
            # Try embeddings API with multimodal input
            response = self.local_client.embeddings.create(
                model=self.model,  # Same model for text and images
                input=[{  # type: ignore[arg-type]
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                }],
                encoding_format="float",
            )
            return response.data[0].embedding
        except Exception as e:
            logger.debug(f"Multimodal embeddings API failed, trying alternative: {e}")
            # Fallback: Some servers use a different format
            try:
                response = self.local_client.embeddings.create(
                    model=self.model,
                    input=f"<image>{image_b64}</image>",
                    encoding_format="float",
                )
                return response.data[0].embedding
            except Exception as e2:
                logger.error(f"Image embedding failed: {e2}")
                raise
    
    def _prepare_text_for_embedding(self, chunk: Chunk) -> str:
        """Prepare chunk text for embedding, including context."""
        parts = []
        
        # Add heading context if available
        if chunk.heading_context:
            parts.append(f"Section: {chunk.heading_context}")
        
        # Add table context for table chunks
        if chunk.is_table and chunk.table_context:
            parts.append(f"Table: {chunk.table_context}")
        
        # Add main content
        parts.append(chunk.content)
        
        return "\n".join(parts)
    
    async def _embed_images(
        self,
        chunks: List[Chunk],
    ) -> List[EmbeddingResult]:
        """Embed image chunks using local Qwen3-VL-Embedding or SigLIP."""
        if not chunks:
            return []
        
        results = []
        
        # Use local Qwen3-VL-Embedding API
        if self.use_local:
            for chunk in chunks:
                try:
                    if chunk.image_data is None:
                        results.append(EmbeddingResult(
                            chunk=chunk,
                            error="No image data",
                        ))
                        continue
                    
                    embedding = await asyncio.to_thread(
                        self._call_local_image_embeddings,
                        chunk.image_data,
                    )
                    
                    if self.normalize:
                        embedding = normalize_embedding(embedding)
                    # Truncate to fit pgvector halfvec index limit (4096 -> 4000)
                    embedding = truncate_embedding(embedding, SETTINGS.rag_vector_dim)
                    
                    results.append(EmbeddingResult(
                        chunk=chunk,
                        image_embedding=embedding,
                    ))
                    
                except Exception as e:
                    logger.error(f"Image embedding failed for chunk {chunk.chunk_index}: {e}")
                    results.append(EmbeddingResult(
                        chunk=chunk,
                        error=str(e),
                    ))
            
            return results
        
        # Fallback to SigLIP for non-local mode
        try:
            self._load_siglip()
            
            import torch
            from PIL import Image
            import io
            
            for chunk in chunks:
                try:
                    if chunk.image_data is None:
                        results.append(EmbeddingResult(
                            chunk=chunk,
                            error="No image data",
                        ))
                        continue
                    
                    # Load image from bytes
                    image = Image.open(io.BytesIO(chunk.image_data))
                    
                    # Process and embed
                    with torch.no_grad():
                        inputs = self._siglip_processor(  # type: ignore[misc]
                            images=[image],
                            return_tensors="pt",
                        ).to(self._siglip_device)
                        
                        embeddings = self._siglip_model.get_image_features(**inputs)  # type: ignore[union-attr]
                        
                        # Normalize
                        if self.normalize:
                            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                        
                        embedding = embeddings[0].cpu().numpy().tolist()
                        # Truncate to fit pgvector halfvec index limit
                        embedding = truncate_embedding(embedding, SETTINGS.rag_vector_dim)
                    
                    results.append(EmbeddingResult(
                        chunk=chunk,
                        image_embedding=embedding,
                    ))
                    
                except Exception as e:
                    logger.error(f"Image embedding failed for chunk {chunk.chunk_index}: {e}")
                    results.append(EmbeddingResult(
                        chunk=chunk,
                        error=str(e),
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"SigLIP loading failed: {e}")
            return [
                EmbeddingResult(chunk=chunk, error=str(e))
                for chunk in chunks
            ]
    
    async def embed_query(self, query: str) -> Tuple[List[float], Optional[List[float]]]:
        """
        Embed a search query.
        
        Uses the unified qwen3-vl-embedding-8b model for text embedding.
        Image embedding for text-to-image search is only available with SigLIP fallback.
        
        Args:
            query: Query text
            
        Returns:
            Tuple of (text_embedding, image_embedding or None)
        """
        # Text embedding (uses unified Qwen3-VL-Embedding model)
        response = await asyncio.to_thread(
            self._call_openai_embeddings,
            [query],
        )
        text_embedding = response[0]
        if self.normalize:
            text_embedding = normalize_embedding(text_embedding)
        # Truncate to fit pgvector halfvec index limit (4096 -> 4000)
        text_embedding = truncate_embedding(text_embedding, SETTINGS.rag_vector_dim)
        
        # Image embedding (for text-to-image search) - only with SigLIP in non-local mode
        # Note: Qwen3-VL-Embedding text queries work for image search in the same embedding space
        image_embedding = None
        if self.enable_image_embeddings and not self.use_local:
            try:
                self._load_siglip()
                import torch
                
                with torch.no_grad():
                    inputs = self._siglip_processor(  # type: ignore[misc]
                        text=[query],
                        return_tensors="pt",
                    ).to(self._siglip_device)
                    
                    embeddings = self._siglip_model.get_text_features(**inputs)  # type: ignore[union-attr]
                    
                    if self.normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                    
                    image_embedding = embeddings[0].cpu().numpy().tolist()
                    # Truncate to fit pgvector halfvec index limit
                    image_embedding = truncate_embedding(image_embedding, SETTINGS.rag_vector_dim)
            except Exception as e:
                logger.warning(f"Image query embedding failed: {e}")
        
        return text_embedding, image_embedding
    
    def embed_text_sync(self, text: str) -> List[float]:
        """
        Synchronously embed a single text (for simple use cases).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (truncated to rag_vector_dim)
        """
        embeddings = self._call_openai_embeddings([text])
        embedding = embeddings[0]
        if self.normalize:
            embedding = normalize_embedding(embedding)
        # Truncate to fit pgvector halfvec index limit (4096 -> 4000)
        embedding = truncate_embedding(embedding, SETTINGS.rag_vector_dim)
        return embedding
