"""
Context Compression Module

Compresses retrieved context to fit within LLM context windows
while preserving relevant information.

Techniques:
- Extractive compression: Select most relevant sentences
- Abstractive compression: Summarize passages with LLM
- Token-aware truncation: Smart truncation respecting boundaries
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Tuple
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Available compression strategies."""
    EXTRACTIVE = "extractive"  # Select relevant sentences
    ABSTRACTIVE = "abstractive"  # LLM summarization
    HYBRID = "hybrid"  # Extract then summarize
    TRUNCATE = "truncate"  # Smart truncation


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    
    enabled: bool = True
    strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE
    
    # Token limits
    max_tokens: int = 4096
    target_compression_ratio: float = 0.5  # Target 50% of original
    
    # Extractive settings
    sentences_per_chunk: int = 5  # Top sentences to keep per chunk
    min_sentence_length: int = 10  # Min chars for valid sentence
    
    # Abstractive settings
    summary_max_tokens: int = 200  # Max tokens per summary
    preserve_quotes: bool = True  # Keep quoted text intact
    preserve_code: bool = True  # Keep code blocks intact
    
    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class CompressedChunk:
    """A compressed chunk with metadata."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy_used: CompressionStrategy
    preserved_elements: List[str] = field(default_factory=list)  # code, quotes, etc.
    relevance_scores: List[float] = field(default_factory=list)  # sentence scores
    

@dataclass
class CompressionResult:
    """Result of context compression."""
    original_context: str
    compressed_context: str
    chunks: List[CompressedChunk]
    
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    
    strategy_used: CompressionStrategy = CompressionStrategy.EXTRACTIVE
    cache_hit: bool = False


class SentenceScorer:
    """Score sentences for relevance to query."""
    
    def __init__(
        self,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.embed_fn = embed_fn
    
    def score_sentences(
        self,
        sentences: List[str],
        query: str,
    ) -> List[Tuple[str, float]]:
        """
        Score sentences by relevance to query.
        
        Returns list of (sentence, score) tuples sorted by score descending.
        """
        if not sentences:
            return []
        
        if self.embed_fn:
            return self._score_with_embeddings(sentences, query)
        else:
            return self._score_with_heuristics(sentences, query)
    
    def _score_with_embeddings(
        self,
        sentences: List[str],
        query: str,
    ) -> List[Tuple[str, float]]:
        """Score using embedding similarity."""
        try:
            # Get query embedding
            query_emb = self.embed_fn([query])[0]
            
            # Get sentence embeddings
            sentence_embs = self.embed_fn(sentences)
            
            # Calculate cosine similarity
            scored = []
            for sent, emb in zip(sentences, sentence_embs):
                sim = self._cosine_similarity(query_emb, emb)
                scored.append((sent, sim))
            
            return sorted(scored, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.warning(f"Embedding scoring failed: {e}, using heuristics")
            return self._score_with_heuristics(sentences, query)
    
    def _score_with_heuristics(
        self,
        sentences: List[str],
        query: str,
    ) -> List[Tuple[str, float]]:
        """Score using keyword overlap and position."""
        query_words = set(query.lower().split())
        
        scored = []
        for i, sent in enumerate(sentences):
            sent_words = set(sent.lower().split())
            
            # Keyword overlap
            overlap = len(query_words & sent_words) / max(len(query_words), 1)
            
            # Position bonus (earlier sentences often more important)
            position_score = 1.0 / (1 + i * 0.1)
            
            # Length penalty (very short or very long sentences)
            word_count = len(sent.split())
            length_score = min(word_count / 20, 1.0) if word_count < 50 else 0.8
            
            # Combined score
            score = 0.5 * overlap + 0.3 * position_score + 0.2 * length_score
            scored.append((sent, score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        if not vec1 or not vec2:
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)


class ContextCompressor:
    """
    Compress retrieved context to fit LLM context windows.
    
    Uses extractive, abstractive, or hybrid strategies to reduce
    token count while preserving relevant information.
    """
    
    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.config = config or CompressionConfig()
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.sentence_scorer = SentenceScorer(embed_fn)
        self._cache: dict = {}
    
    def compress(
        self,
        context: str,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """
        Compress context to fit within token limit.
        
        Args:
            context: The full context to compress
            query: The query for relevance scoring
            max_tokens: Optional override for max tokens
            
        Returns:
            CompressionResult with compressed context
        """
        if not self.config.enabled:
            return CompressionResult(
                original_context=context,
                compressed_context=context,
                chunks=[],
                original_tokens=self._estimate_tokens(context),
                compressed_tokens=self._estimate_tokens(context),
                compression_ratio=1.0,
            )
        
        max_tokens = max_tokens or self.config.max_tokens
        
        # Check cache
        cache_key = self._cache_key(context, query, max_tokens)
        if self.config.enable_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.cache_hit = True
            return cached
        
        # Estimate current tokens
        original_tokens = self._estimate_tokens(context)
        
        # If already within limit, return as-is
        if original_tokens <= max_tokens:
            result = CompressionResult(
                original_context=context,
                compressed_context=context,
                chunks=[],
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
            )
            return result
        
        # Select compression strategy
        strategy = self.config.strategy
        
        if strategy == CompressionStrategy.EXTRACTIVE:
            result = self._extractive_compress(context, query, max_tokens)
        elif strategy == CompressionStrategy.ABSTRACTIVE:
            result = self._abstractive_compress(context, query, max_tokens)
        elif strategy == CompressionStrategy.HYBRID:
            result = self._hybrid_compress(context, query, max_tokens)
        else:  # TRUNCATE
            result = self._smart_truncate(context, max_tokens)
        
        # Cache result
        if self.config.enable_cache:
            self._cache[cache_key] = result
        
        logger.debug(
            f"Context compression: {original_tokens} -> {result.compressed_tokens} tokens "
            f"({result.compression_ratio:.1%})"
        )
        
        return result
    
    def _extractive_compress(
        self,
        context: str,
        query: str,
        max_tokens: int,
    ) -> CompressionResult:
        """Extract most relevant sentences."""
        # Split into chunks (paragraphs or sections)
        chunks = self._split_into_chunks(context)
        
        compressed_chunks = []
        total_compressed = ""
        current_tokens = 0
        
        for chunk_text in chunks:
            # Extract preserved elements (code, quotes)
            preserved, clean_text = self._extract_preserved(chunk_text)
            
            # Split into sentences
            sentences = self._split_sentences(clean_text)
            
            # Score and rank sentences
            scored = self.sentence_scorer.score_sentences(sentences, query)
            
            # Select top sentences within budget
            selected = []
            chunk_tokens = 0
            
            for sent, score in scored[:self.config.sentences_per_chunk]:
                sent_tokens = self._estimate_tokens(sent)
                if current_tokens + chunk_tokens + sent_tokens <= max_tokens:
                    selected.append((sent, score))
                    chunk_tokens += sent_tokens
            
            # Reconstruct chunk maintaining order
            original_order = []
            for sent, score in selected:
                orig_idx = sentences.index(sent) if sent in sentences else 0
                original_order.append((orig_idx, sent, score))
            original_order.sort(key=lambda x: x[0])
            
            compressed_text = " ".join(s for _, s, _ in original_order)
            
            # Add back preserved elements
            for preserved_text in preserved:
                preserved_tokens = self._estimate_tokens(preserved_text)
                if current_tokens + chunk_tokens + preserved_tokens <= max_tokens:
                    compressed_text += f"\n\n{preserved_text}"
                    chunk_tokens += preserved_tokens
            
            if compressed_text.strip():
                compressed_chunks.append(CompressedChunk(
                    original_text=chunk_text,
                    compressed_text=compressed_text,
                    original_tokens=self._estimate_tokens(chunk_text),
                    compressed_tokens=chunk_tokens,
                    compression_ratio=chunk_tokens / max(self._estimate_tokens(chunk_text), 1),
                    strategy_used=CompressionStrategy.EXTRACTIVE,
                    preserved_elements=preserved,
                    relevance_scores=[s for _, _, s in original_order],
                ))
                
                total_compressed += compressed_text + "\n\n"
                current_tokens += chunk_tokens
        
        return CompressionResult(
            original_context=context,
            compressed_context=total_compressed.strip(),
            chunks=compressed_chunks,
            original_tokens=self._estimate_tokens(context),
            compressed_tokens=current_tokens,
            compression_ratio=current_tokens / max(self._estimate_tokens(context), 1),
            strategy_used=CompressionStrategy.EXTRACTIVE,
        )
    
    def _abstractive_compress(
        self,
        context: str,
        query: str,
        max_tokens: int,
    ) -> CompressionResult:
        """Summarize context using LLM."""
        if not self.llm_fn:
            logger.warning("No LLM function provided, falling back to extractive")
            return self._extractive_compress(context, query, max_tokens)
        
        # Split into chunks for summarization
        chunks = self._split_into_chunks(context)
        
        compressed_chunks = []
        summaries = []
        current_tokens = 0
        
        for chunk_text in chunks:
            # Extract preserved elements
            preserved, clean_text = self._extract_preserved(chunk_text)
            
            # Calculate target summary length
            target_tokens = min(
                self.config.summary_max_tokens,
                (max_tokens - current_tokens) // max(len(chunks), 1),
            )
            
            if target_tokens < 50:
                continue
            
            # Generate summary
            prompt = f"""Summarize the following text in approximately {target_tokens} words, 
focusing on information relevant to the query: "{query}"

Text:
{clean_text}

Summary:"""
            
            try:
                summary = self.llm_fn(prompt)
                summary_tokens = self._estimate_tokens(summary)
                
                # Add preserved elements
                final_text = summary
                for preserved_text in preserved:
                    preserved_tokens = self._estimate_tokens(preserved_text)
                    if current_tokens + summary_tokens + preserved_tokens <= max_tokens:
                        final_text += f"\n\n{preserved_text}"
                        summary_tokens += preserved_tokens
                
                compressed_chunks.append(CompressedChunk(
                    original_text=chunk_text,
                    compressed_text=final_text,
                    original_tokens=self._estimate_tokens(chunk_text),
                    compressed_tokens=summary_tokens,
                    compression_ratio=summary_tokens / max(self._estimate_tokens(chunk_text), 1),
                    strategy_used=CompressionStrategy.ABSTRACTIVE,
                    preserved_elements=preserved,
                ))
                
                summaries.append(final_text)
                current_tokens += summary_tokens
                
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")
                # Fall back to extractive for this chunk
                extracted = self._extractive_compress(chunk_text, query, target_tokens)
                if extracted.chunks:
                    compressed_chunks.extend(extracted.chunks)
                    summaries.append(extracted.compressed_context)
                    current_tokens += extracted.compressed_tokens
        
        compressed_context = "\n\n".join(summaries)
        
        return CompressionResult(
            original_context=context,
            compressed_context=compressed_context,
            chunks=compressed_chunks,
            original_tokens=self._estimate_tokens(context),
            compressed_tokens=current_tokens,
            compression_ratio=current_tokens / max(self._estimate_tokens(context), 1),
            strategy_used=CompressionStrategy.ABSTRACTIVE,
        )
    
    def _hybrid_compress(
        self,
        context: str,
        query: str,
        max_tokens: int,
    ) -> CompressionResult:
        """First extract, then summarize if needed."""
        # First pass: extractive
        extracted = self._extractive_compress(
            context, query, int(max_tokens * 1.5)  # Allow some slack
        )
        
        # If still too long, summarize
        if extracted.compressed_tokens > max_tokens and self.llm_fn:
            return self._abstractive_compress(
                extracted.compressed_context, query, max_tokens
            )
        
        return extracted
    
    def _smart_truncate(
        self,
        context: str,
        max_tokens: int,
    ) -> CompressionResult:
        """Truncate at sentence boundaries."""
        sentences = self._split_sentences(context)
        
        truncated = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens <= max_tokens:
                truncated.append(sent)
                current_tokens += sent_tokens
            else:
                break
        
        compressed = " ".join(truncated)
        
        return CompressionResult(
            original_context=context,
            compressed_context=compressed,
            chunks=[],
            original_tokens=self._estimate_tokens(context),
            compressed_tokens=current_tokens,
            compression_ratio=current_tokens / max(self._estimate_tokens(context), 1),
            strategy_used=CompressionStrategy.TRUNCATE,
        )
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into logical chunks (paragraphs/sections)."""
        # Split on double newlines or section markers
        chunks = re.split(r'\n\s*\n|\n(?=#{1,6}\s)', text)
        return [c.strip() for c in chunks if c.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [
            s.strip() for s in sentences 
            if s.strip() and len(s.strip()) >= self.config.min_sentence_length
        ]
    
    def _extract_preserved(self, text: str) -> Tuple[List[str], str]:
        """Extract elements that should be preserved (code, quotes)."""
        preserved = []
        clean_text = text
        
        # Extract code blocks
        if self.config.preserve_code:
            code_pattern = r'```[\w]*\n.*?```'
            code_blocks = re.findall(code_pattern, text, re.DOTALL)
            preserved.extend(code_blocks)
            clean_text = re.sub(code_pattern, '', clean_text, flags=re.DOTALL)
        
        # Extract quotes
        if self.config.preserve_quotes:
            quote_pattern = r'"[^"]{50,}"'  # Quoted text > 50 chars
            quotes = re.findall(quote_pattern, clean_text)
            preserved.extend(quotes)
        
        return preserved, clean_text.strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters for English
        return len(text) // 4
    
    def _cache_key(self, context: str, query: str, max_tokens: int) -> str:
        """Generate cache key."""
        content = f"{context}:{query}:{max_tokens}"
        return hashlib.md5(content.encode()).hexdigest()


class LongContextHandler:
    """
    Handle documents that exceed context limits.
    
    Implements strategies for processing long documents:
    - Chunk and summarize
    - Hierarchical compression
    - Map-reduce summarization
    """
    
    def __init__(
        self,
        compressor: Optional[ContextCompressor] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_tokens: int = 4096,
    ):
        self.compressor = compressor or ContextCompressor()
        self.llm_fn = llm_fn
        self.max_tokens = max_tokens
    
    def process_long_context(
        self,
        documents: List[str],
        query: str,
    ) -> str:
        """
        Process multiple documents that together exceed context limit.
        
        Args:
            documents: List of document texts
            query: Query for relevance scoring
            
        Returns:
            Compressed context string
        """
        # First, compress each document individually
        compressed_docs = []
        per_doc_budget = self.max_tokens // max(len(documents), 1)
        
        for doc in documents:
            result = self.compressor.compress(
                doc, query, max_tokens=per_doc_budget
            )
            compressed_docs.append(result.compressed_context)
        
        combined = "\n\n---\n\n".join(compressed_docs)
        
        # If still too long, do final compression pass
        if self.compressor._estimate_tokens(combined) > self.max_tokens:
            final_result = self.compressor.compress(
                combined, query, max_tokens=self.max_tokens
            )
            return final_result.compressed_context
        
        return combined
    
    def map_reduce_summarize(
        self,
        documents: List[str],
        query: str,
    ) -> str:
        """
        Map-reduce style summarization for very long content.
        
        Map: Summarize each document
        Reduce: Combine summaries into final answer
        """
        if not self.llm_fn:
            return self.process_long_context(documents, query)
        
        # Map phase: summarize each document
        summaries = []
        for doc in documents:
            if self.compressor._estimate_tokens(doc) > 500:
                result = self.compressor.compress(doc, query, max_tokens=500)
                summaries.append(result.compressed_context)
            else:
                summaries.append(doc)
        
        # Reduce phase: combine summaries
        combined = "\n\n".join(summaries)
        
        if self.compressor._estimate_tokens(combined) <= self.max_tokens:
            return combined
        
        # Final reduction with LLM
        prompt = f"""Given these document summaries, create a comprehensive summary 
that answers the query: "{query}"

Summaries:
{combined}

Final Summary:"""
        
        try:
            return self.llm_fn(prompt)
        except Exception as e:
            logger.warning(f"Map-reduce failed: {e}")
            return self.process_long_context(documents, query)


def create_context_compressor(
    config: Optional[CompressionConfig] = None,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> ContextCompressor:
    """Factory function to create a context compressor."""
    return ContextCompressor(config=config, embed_fn=embed_fn, llm_fn=llm_fn)
