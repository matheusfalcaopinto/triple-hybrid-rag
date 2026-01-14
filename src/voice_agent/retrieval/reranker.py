"""
Qwen3-VL Multimodal Reranker Module

Provides precision reranking of search results using Qwen3-VL-Reranker-8B.
Supports both text-only and multimodal (text + image) reranking.

The reranker uses a yes/no classification approach:
- Sends query-document pairs to the VL model
- Model judges relevance and outputs "yes" or "no"
- Uses logprobs to extract confidence scores
"""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

from voice_agent.config import SETTINGS
from voice_agent.observability.rag_metrics import rag_metrics
from voice_agent.retrieval.hybrid_search import SearchResult

logger = logging.getLogger(__name__)

# System prompt for Qwen3-VL-Reranker
# Note: /no_think disables Qwen3's internal thinking mode for consistent yes/no responses
RERANKER_SYSTEM_PROMPT = """Decide if the document is relevant to the query. Answer yes or no only. /no_think"""

# Default instruction for retrieval tasks
DEFAULT_INSTRUCTION = "Retrieve relevant documents that answer the user's query"


@dataclass
class RerankResult:
    """Result of reranking with score."""
    search_result: SearchResult
    rerank_score: float
    original_rank: int


class Qwen3VLReranker:
    """
    Multimodal reranker using Qwen3-VL-Reranker.
    
    Uses the Qwen3-VL-Reranker model to score query-document pairs
    for more precise relevance ranking. Supports text, images, and
    mixed modality documents.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        top_k: Optional[int] = None,
        enabled: bool = True,
        use_local: Optional[bool] = None,
        instruction: Optional[str] = None,
    ):
        """
        Initialize the Qwen3-VL reranker.
        
        Args:
            model_name: Model name (default: qwen3-vl-reranker-8b)
            api_base: API endpoint (default: http://127.0.0.1:1234/v1)
            top_k: Number of results to return after reranking
            enabled: Whether reranking is enabled
            use_local: Whether to use local API (True) or skip (False)
            instruction: Custom instruction for the task
        """
        self.model_name = model_name or SETTINGS.rag_rerank_model
        self.api_base = api_base or SETTINGS.rag_rerank_api_base
        self.top_k = top_k or SETTINGS.rag_top_k_rerank
        self.enabled = enabled if enabled is not None else SETTINGS.rag_reranking_enabled
        self.use_local = use_local if use_local is not None else SETTINGS.rag_rerank_use_local
        self.instruction = instruction or DEFAULT_INSTRUCTION
        
        self._client: Optional[AsyncOpenAI] = None
        self._sync_client: Optional[OpenAI] = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-load async OpenAI client for local API."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.api_base,
                api_key="lm-studio",  # LM Studio accepts any key
            )
        return self._client
    
    @property
    def sync_client(self) -> OpenAI:
        """Lazy-load sync OpenAI client for local API."""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                base_url=self.api_base,
                api_key="lm-studio",
            )
        return self._sync_client
    
    def _build_rerank_prompt(
        self,
        query: str,
        document: str,
        image_base64: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the prompt for reranking a single query-document pair.
        
        Uses a simple query/document format that works reliably with
        GGUF models in LM Studio.
        
        Args:
            query: The search query
            document: The document text
            image_base64: Optional base64-encoded image
            
        Returns:
            List of message dicts for chat completion
        """
        # Build the user message content
        content_parts: List[Dict[str, Any]] = []
        
        # Simple, direct prompt format works better with GGUF models
        prompt_text = (
            f"Query: {query}\n\n"
            f"Document: {document}\n\n"
            f"Is this document relevant to the query?"
        )
        content_parts.append({"type": "text", "text": prompt_text})
        
        # Add image if provided (multimodal reranking)
        if image_base64:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
        
        messages = [
            {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]
        
        return messages
    
    def _prepare_document(self, result: SearchResult) -> Tuple[str, Optional[str]]:
        """
        Prepare document text and optional image for reranking.
        
        Args:
            result: Search result to prepare
            
        Returns:
            Tuple of (document_text, image_base64 or None)
        """
        parts = []
        
        # Add title if available
        if result.title:
            parts.append(f"Title: {result.title}")
        
        # Add table context if it's a table
        if result.is_table and result.table_context:
            parts.append(f"Table: {result.table_context}")
        
        # Add alt text for images
        if result.alt_text:
            parts.append(f"Description: {result.alt_text}")
        
        # Add main content
        parts.append(result.content)
        
        document_text = "\n".join(parts)
        
        # Get image if available (for multimodal reranking)
        image_base64 = None
        if hasattr(result, 'image_data') and result.image_data:
            # image_data should already be base64 encoded
            if isinstance(result.image_data, bytes):
                image_base64 = base64.b64encode(result.image_data).decode('utf-8')
            elif isinstance(result.image_data, str):
                image_base64 = result.image_data
        
        return document_text, image_base64
    
    async def _score_pair(
        self,
        query: str,
        document: str,
        image_base64: Optional[str] = None,
    ) -> float:
        """
        Score a single query-document pair.
        
        Uses the model's response to determine relevance.
        If model says "yes", score is high. If "no", score is low.
        
        Args:
            query: Search query
            document: Document text
            image_base64: Optional image
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        messages = self._build_rerank_prompt(query, document, image_base64)
        
        try:
            # Call the model - use small temperature for more varied output
            # Note: GGUF models in LM Studio may not return logprobs properly
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                max_tokens=10,  # Allow more tokens for better parsing
                temperature=0.1,  # Small temperature helps with GGUF models
                logprobs=True,
                top_logprobs=5,  # Get top 5 token logprobs
            )
            
            # Extract the response
            choice = response.choices[0]
            answer = choice.message.content.strip().lower() if choice.message.content else ""
            
            # Try to get logprobs for more precise scoring
            if choice.logprobs and choice.logprobs.content:
                token_logprobs = choice.logprobs.content[0]
                # Find yes/no in top logprobs
                yes_logprob = float('-inf')
                no_logprob = float('-inf')
                
                if token_logprobs.top_logprobs:
                    for lp in token_logprobs.top_logprobs:
                        token = lp.token.lower().strip()
                        if token in ('yes', 'sim', 'y'):
                            yes_logprob = lp.logprob
                        elif token in ('no', 'não', 'n'):
                            no_logprob = lp.logprob
                
                # Convert logprobs to probability using softmax
                import math
                if yes_logprob > float('-inf') or no_logprob > float('-inf'):
                    # Normalize
                    max_logprob = max(yes_logprob, no_logprob)
                    yes_prob = math.exp(yes_logprob - max_logprob)
                    no_prob = math.exp(no_logprob - max_logprob)
                    total = yes_prob + no_prob
                    
                    if total > 0:
                        return yes_prob / total
            
            # Fallback: parse the text response
            # Handle cases like "yes\nno, but..." by looking at first word
            first_word = answer.split()[0] if answer.split() else ""
            first_word = first_word.rstrip('.,!?;:\n').lower()
            
            # Check for yes/no in first word
            if first_word.startswith('yes') or first_word in ('sim', 'y'):
                return 0.9
            elif first_word.startswith('no') or first_word in ('não', 'n'):
                # Also check if "yes" appears later in the response
                if 'yes' in answer.lower():
                    logger.debug(f"Mixed response (has yes): '{answer[:50]}...'")
                    return 0.6  # Mixed signal, slightly positive
                return 0.1
            else:
                # Look for yes/no anywhere in response
                if 'yes' in answer.lower():
                    return 0.7
                elif 'no' in answer.lower():
                    return 0.3
                # Unknown response, use neutral score
                logger.warning(f"Unexpected reranker response: '{answer[:100]}'")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error scoring pair: {e}")
            return 0.5  # Neutral score on error
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank search results using Qwen3-VL-Reranker.
        
        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked list of SearchResult
        """
        if not self.enabled or not self.use_local or not results:
            return results[:top_k or self.top_k]
        
        start_time = time.perf_counter()
        top_k = top_k or self.top_k
        
        # Only rerank top candidates to save computation
        candidates = results[:min(len(results), 50)]
        
        try:
            # Score all pairs concurrently (with concurrency limit)
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
            
            async def score_with_limit(idx: int, result: SearchResult) -> Tuple[int, float]:
                async with semaphore:
                    doc_text, image_b64 = self._prepare_document(result)
                    score = await self._score_pair(query, doc_text, image_b64)
                    return idx, score
            
            tasks = [
                score_with_limit(i, result)
                for i, result in enumerate(candidates)
            ]
            
            scored_pairs = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Create rerank results
            reranked = []
            for result in scored_pairs:
                if isinstance(result, Exception):
                    logger.error(f"Reranking task failed: {result}")
                    continue
                idx, score = result
                candidates[idx].rerank_score = score
                reranked.append(RerankResult(
                    search_result=candidates[idx],
                    rerank_score=score,
                    original_rank=idx,
                ))
            
            # Sort by rerank score
            reranked.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Return top-k with updated scores
            final_results = []
            for rr in reranked[:top_k]:
                result = rr.search_result
                result.rerank_score = rr.rerank_score
                final_results.append(result)
            
            # Record success metrics
            duration = time.perf_counter() - start_time
            rag_metrics.rerank_duration.observe(duration)
            rag_metrics.reranks_total.inc()
            rag_metrics.rerank_results_count.observe(len(final_results))
            
            logger.debug(
                f"Reranked {len(candidates)} docs in {duration:.2f}s, "
                f"returning top {len(final_results)}"
            )
            
            return final_results
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            rag_metrics.rerank_duration.observe(duration)
            rag_metrics.rerank_errors.inc()
            logger.error(f"Reranking failed: {e}")
            # Fallback to original order
            rag_metrics.rerank_fallbacks.inc()
            return results[:top_k]
    
    def rerank_sync(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Synchronous version of rerank.
        
        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked list of SearchResult
        """
        if not self.enabled or not self.use_local or not results:
            return results[:top_k or self.top_k]
        
        top_k = top_k or self.top_k
        candidates = results[:min(len(results), 50)]
        
        try:
            # Score pairs sequentially (sync version)
            for i, result in enumerate(candidates):
                doc_text, image_b64 = self._prepare_document(result)
                messages = self._build_rerank_prompt(query, doc_text, image_b64)
                
                try:
                    response = self.sync_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,  # type: ignore
                        max_tokens=1,
                        temperature=0.0,
                    )
                    
                    answer = response.choices[0].message.content.strip().lower() if response.choices[0].message.content else ""
                    
                    if answer in ('yes', 'sim', 'y'):
                        result.rerank_score = 0.9
                    elif answer in ('no', 'não', 'n'):
                        result.rerank_score = 0.1
                    else:
                        result.rerank_score = 0.5
                        
                except Exception as e:
                    logger.warning(f"Error scoring document {i}: {e}")
                    result.rerank_score = 0.5
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Sync reranking failed: {e}")
            return results[:top_k]


# Backward compatibility alias
Reranker = Qwen3VLReranker


class LightweightReranker:
    """
    Lightweight reranker using simple scoring heuristics.
    
    Used as fallback when Qwen3-VL-Reranker is not available.
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank using lightweight heuristics.
        
        Scoring based on:
        - Term overlap between query and content
        - RRF score boost
        - Table/image bonus for relevant queries
        """
        top_k = top_k or self.top_k
        
        if not results:
            return []
        
        query_terms = set(query.lower().split())
        
        for result in results:
            content_terms = set(result.content.lower().split())
            
            # Term overlap score
            overlap = len(query_terms & content_terms)
            overlap_score = overlap / max(len(query_terms), 1)
            
            # Combine with existing scores
            combined_score = (
                result.rrf_score * 0.5 +
                result.similarity_score * 0.3 +
                overlap_score * 0.2
            )
            
            # Boost for tables if query seems to ask for data
            data_keywords = {"table", "data", "numbers", "statistics", "chart", "tabela", "dados"}
            if result.is_table and query_terms & data_keywords:
                combined_score *= 1.2
            
            result.rerank_score = combined_score
        
        # Sort by combined score
        results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        
        return results[:top_k]


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.
    
    Legacy reranker for backward compatibility when sentence-transformers
    is installed and RAG_RERANK_USE_LOCAL=false.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        enabled: bool = True,
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name
            top_k: Number of results to return after reranking
            enabled: Whether reranking is enabled
        """
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.top_k = top_k or SETTINGS.rag_top_k_rerank
        self.enabled = enabled if enabled is not None else SETTINGS.rag_reranking_enabled
        
        self._model = None
    
    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder reranker: {self.model_name}")
            rag_metrics.crossencoder_model_loaded.set(1)
            
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.enabled = False
    
    def _prepare_document(self, result: SearchResult) -> str:
        """Prepare document text for cross-encoder scoring."""
        parts = []
        
        if result.title:
            parts.append(f"Title: {result.title}")
        
        if result.is_table and result.table_context:
            parts.append(f"Table: {result.table_context}")
        
        if result.alt_text:
            parts.append(f"Description: {result.alt_text}")
        
        parts.append(result.content)
        
        return "\n".join(parts)
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked list of SearchResult
        """
        if not self.enabled or not results:
            return results[:top_k or self.top_k]
        
        start_time = time.perf_counter()
        top_k = top_k or self.top_k
        
        candidates = results[:min(len(results), 100)]
        
        try:
            await asyncio.to_thread(self._load_model)
            
            if not self.enabled or self._model is None:
                rag_metrics.rerank_fallbacks.inc()
                return results[:top_k]
            
            pairs = [
                (query, self._prepare_document(r))
                for r in candidates
            ]
            
            scores = await asyncio.to_thread(
                self._model.predict,
                pairs,
            )
            
            reranked = []
            for i, (result, score) in enumerate(zip(candidates, scores)):
                result.rerank_score = float(score)
                reranked.append(RerankResult(
                    search_result=result,
                    rerank_score=float(score),
                    original_rank=i,
                ))
            
            reranked.sort(key=lambda x: x.rerank_score, reverse=True)
            
            final_results = []
            for rr in reranked[:top_k]:
                result = rr.search_result
                result.rerank_score = rr.rerank_score
                final_results.append(result)
            
            duration = time.perf_counter() - start_time
            rag_metrics.rerank_duration.observe(duration)
            rag_metrics.reranks_total.inc()
            rag_metrics.rerank_results_count.observe(len(final_results))
            
            return final_results
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            rag_metrics.rerank_duration.observe(duration)
            rag_metrics.rerank_errors.inc()
            logger.error(f"Cross-encoder reranking failed: {e}")
            return results[:top_k]
    
    def rerank_sync(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Synchronous version of rerank."""
        if not self.enabled or not results:
            return results[:top_k or self.top_k]
        
        top_k = top_k or self.top_k
        candidates = results[:min(len(results), 100)]
        
        try:
            self._load_model()
            
            if not self.enabled or self._model is None:
                return results[:top_k]
            
            pairs = [
                (query, self._prepare_document(r))
                for r in candidates
            ]
            
            scores = self._model.predict(pairs)
            
            for result, score in zip(candidates, scores):
                result.rerank_score = float(score)
            
            candidates.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Sync cross-encoder reranking failed: {e}")
            return results[:top_k]


def get_reranker(
    use_local: Optional[bool] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> Qwen3VLReranker | CrossEncoderReranker | LightweightReranker:
    """
    Factory function to get the appropriate reranker.
    
    Args:
        use_local: Use Qwen3-VL-Reranker (True) or CrossEncoder (False)
        model_name: Model name override
        **kwargs: Additional arguments for reranker
        
    Returns:
        Reranker instance
    """
    use_local = use_local if use_local is not None else SETTINGS.rag_rerank_use_local
    
    if use_local:
        return Qwen3VLReranker(model_name=model_name, use_local=True, **kwargs)
    else:
        # Try CrossEncoder, fall back to Lightweight
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
            return CrossEncoderReranker(model_name=model_name, **kwargs)
        except ImportError:
            logger.warning(
                "sentence-transformers not available, using lightweight reranker"
            )
            return LightweightReranker(**kwargs)
