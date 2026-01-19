"""
Multimodal RAG Implementation

Support for text, images, tables, and structured data retrieval.
Includes late interaction, cross-modal fusion, and ColBERT-style retrieval.

Reference:
- ColBERT: "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"
- Multi-Vector Retrieval patterns
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types of content modalities."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    AUDIO = "audio"
    VIDEO = "video"

@dataclass
class MultimodalContent:
    """Content with modality information."""
    content_id: str
    modality: ModalityType
    content: Any  # Text string, image bytes, table dict, etc.
    text_representation: Optional[str] = None  # Text fallback/description
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultimodalResult:
    """Result from multimodal retrieval."""
    content_id: str
    modality: ModalityType
    text: str
    score: float
    original_content: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultimodalConfig:
    """Configuration for multimodal retrieval."""
    
    # Modality weights
    text_weight: float = 1.0
    image_weight: float = 0.8
    table_weight: float = 0.9
    code_weight: float = 0.85
    
    # Processing options
    generate_text_for_images: bool = True
    extract_table_text: bool = True
    parse_code_structure: bool = True
    
    # Fusion
    cross_modal_fusion: bool = True
    late_interaction: bool = False
    
    # Retrieval
    top_k_per_modality: int = 20
    final_top_k: int = 10

class MultimodalRetriever:
    """
    Multimodal Retriever supporting text, images, tables, and code.
    
    Features:
    - Modality-specific embedding and retrieval
    - Cross-modal fusion
    - Late interaction scoring
    - Unified ranking
    
    Usage:
        retriever = MultimodalRetriever(
            text_embed_fn=text_embedder,
            text_search_fn=vector_search,
        )
        
        # Add content
        retriever.add_content(text_content)
        retriever.add_content(image_content)
        
        # Retrieve
        results = retriever.retrieve("query about images and text")
    """
    
    def __init__(
        self,
        text_embed_fn: Optional[Callable] = None,
        text_search_fn: Optional[Callable] = None,
        image_embed_fn: Optional[Callable] = None,
        image_search_fn: Optional[Callable] = None,
        config: Optional[MultimodalConfig] = None,
    ):
        """Initialize multimodal retriever."""
        self.text_embed_fn = text_embed_fn
        self.text_search_fn = text_search_fn
        self.image_embed_fn = image_embed_fn
        self.image_search_fn = image_search_fn
        self.config = config or MultimodalConfig()
        
        # Content storage
        self.content_store: Dict[str, MultimodalContent] = {}
        self.modality_index: Dict[ModalityType, List[str]] = {
            m: [] for m in ModalityType
        }
    
    def add_content(
        self,
        content: Union[str, MultimodalContent],
        modality: ModalityType = ModalityType.TEXT,
        content_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Add content to the retriever.
        
        Args:
            content: Content to add (string or MultimodalContent)
            modality: Content modality type
            content_id: Optional ID
            metadata: Optional metadata
        """
        if isinstance(content, MultimodalContent):
            mm_content = content
        else:
            # Create MultimodalContent from raw content
            content_id = content_id or self._generate_id(str(content))
            mm_content = MultimodalContent(
                content_id=content_id,
                modality=modality,
                content=content,
                text_representation=self._get_text_representation(content, modality),
                metadata=metadata or {},
            )
        
        # Store content
        self.content_store[mm_content.content_id] = mm_content
        self.modality_index[mm_content.modality].append(mm_content.content_id)
        
        # Generate embedding if applicable
        if mm_content.modality == ModalityType.TEXT and self.text_embed_fn:
            text = mm_content.text_representation or str(mm_content.content)
            mm_content.embedding = self.text_embed_fn(text)
        elif mm_content.modality == ModalityType.IMAGE and self.image_embed_fn:
            mm_content.embedding = self.image_embed_fn(mm_content.content)
    
    def retrieve(
        self,
        query: str,
        modalities: Optional[List[ModalityType]] = None,
        top_k: Optional[int] = None,
    ) -> List[MultimodalResult]:
        """
        Retrieve content across modalities.
        
        Args:
            query: Search query
            modalities: Filter to specific modalities (None = all)
            top_k: Number of results
            
        Returns:
            List of MultimodalResult
        """
        top_k = top_k or self.config.final_top_k
        modalities = modalities or list(ModalityType)
        
        all_results: List[MultimodalResult] = []
        
        # Retrieve from each modality
        for modality in modalities:
            modality_results = self._retrieve_modality(
                query,
                modality,
                self.config.top_k_per_modality,
            )
            all_results.extend(modality_results)
        
        # Apply cross-modal fusion if enabled
        if self.config.cross_modal_fusion:
            all_results = self._cross_modal_fusion(query, all_results)
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def _retrieve_modality(
        self,
        query: str,
        modality: ModalityType,
        top_k: int,
    ) -> List[MultimodalResult]:
        """Retrieve from a specific modality."""
        results = []
        
        # Get modality weight
        weight = self._get_modality_weight(modality)
        
        if modality == ModalityType.TEXT and self.text_search_fn:
            search_results = self.text_search_fn(query, top_k=top_k)
            for r in search_results:
                content_id = getattr(r, 'id', getattr(r, 'content_id', str(r)))
                content = self.content_store.get(content_id)
                
                results.append(MultimodalResult(
                    content_id=content_id,
                    modality=modality,
                    text=content.text_representation if content else str(r),
                    score=getattr(r, 'score', 0.5) * weight,
                    original_content=content.content if content else None,
                    metadata=content.metadata if content else {},
                ))
        
        elif modality == ModalityType.IMAGE and self.image_search_fn:
            # For images, search using text query against image embeddings
            search_results = self.image_search_fn(query, top_k=top_k)
            for r in search_results:
                content_id = getattr(r, 'id', str(r))
                content = self.content_store.get(content_id)
                
                results.append(MultimodalResult(
                    content_id=content_id,
                    modality=modality,
                    text=content.text_representation if content else "[Image]",
                    score=getattr(r, 'score', 0.5) * weight,
                    original_content=content.content if content else None,
                    metadata=content.metadata if content else {},
                ))
        
        else:
            # Fallback: search using text representations
            content_ids = self.modality_index.get(modality, [])
            for content_id in content_ids[:top_k]:
                content = self.content_store.get(content_id)
                if content and content.text_representation:
                    # Simple keyword matching as fallback
                    score = self._simple_score(query, content.text_representation)
                    results.append(MultimodalResult(
                        content_id=content_id,
                        modality=modality,
                        text=content.text_representation,
                        score=score * weight,
                        original_content=content.content,
                        metadata=content.metadata,
                    ))
        
        return results
    
    def _cross_modal_fusion(
        self,
        query: str,
        results: List[MultimodalResult],
    ) -> List[MultimodalResult]:
        """Apply cross-modal fusion to results."""
        # Group by content ID to combine scores
        content_scores: Dict[str, MultimodalResult] = {}
        
        for result in results:
            if result.content_id in content_scores:
                # Combine scores (max strategy)
                existing = content_scores[result.content_id]
                if result.score > existing.score:
                    content_scores[result.content_id] = result
            else:
                content_scores[result.content_id] = result
        
        return list(content_scores.values())
    
    def _get_modality_weight(self, modality: ModalityType) -> float:
        """Get weight for modality."""
        weights = {
            ModalityType.TEXT: self.config.text_weight,
            ModalityType.IMAGE: self.config.image_weight,
            ModalityType.TABLE: self.config.table_weight,
            ModalityType.CODE: self.config.code_weight,
        }
        return weights.get(modality, 1.0)
    
    def _get_text_representation(
        self,
        content: Any,
        modality: ModalityType,
    ) -> Optional[str]:
        """Get text representation of content."""
        if modality == ModalityType.TEXT:
            return str(content)
        elif modality == ModalityType.TABLE:
            if isinstance(content, dict):
                return self._table_to_text(content)
            return str(content)
        elif modality == ModalityType.CODE:
            return str(content)
        elif modality == ModalityType.IMAGE:
            # Would use vision model in production
            return None
        return None
    
    def _table_to_text(self, table: Dict) -> str:
        """Convert table to text representation."""
        lines = []
        if 'headers' in table:
            lines.append(" | ".join(str(h) for h in table['headers']))
        if 'rows' in table:
            for row in table['rows'][:10]:  # Limit rows
                lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)
    
    def _simple_score(self, query: str, text: str) -> float:
        """Simple keyword matching score."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = len(query_words & text_words)
        return overlap / max(len(query_words), 1)
    
    def _generate_id(self, content: str) -> str:
        """Generate content ID from hash."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class ColBERTConfig:
    """Configuration for ColBERT-style retrieval."""
    
    # Token-level settings
    max_query_tokens: int = 32
    max_doc_tokens: int = 180
    
    # Interaction
    interaction_type: str = "maxsim"  # maxsim, sumsim
    
    # Retrieval
    top_k: int = 10
    candidate_k: int = 100

class ColBERTRetriever:
    """
    ColBERT-style late interaction retriever.
    
    Instead of single-vector similarity, computes token-level
    late interaction between query and document embeddings.
    
    Features:
    - Token-level embeddings
    - MaxSim late interaction
    - Efficient candidate retrieval + reranking
    """
    
    def __init__(
        self,
        tokenize_fn: Optional[Callable] = None,
        embed_tokens_fn: Optional[Callable] = None,
        config: Optional[ColBERTConfig] = None,
    ):
        """Initialize ColBERT retriever."""
        self.tokenize = tokenize_fn
        self.embed_tokens = embed_tokens_fn
        self.config = config or ColBERTConfig()
        
        # Document storage
        self.documents: Dict[str, Dict] = {}
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None,
    ):
        """Add document with token embeddings."""
        if self.tokenize and self.embed_tokens:
            tokens = self.tokenize(text)[:self.config.max_doc_tokens]
            embeddings = self.embed_tokens(tokens)
        else:
            tokens = text.split()[:self.config.max_doc_tokens]
            embeddings = None
        
        self.documents[doc_id] = {
            'text': text,
            'tokens': tokens,
            'embeddings': embeddings,
            'metadata': metadata or {},
        }
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve using late interaction.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of results with scores
        """
        top_k = top_k or self.config.top_k
        
        # Tokenize and embed query
        if self.tokenize and self.embed_tokens:
            query_tokens = self.tokenize(query)[:self.config.max_query_tokens]
            query_embeddings = self.embed_tokens(query_tokens)
        else:
            query_tokens = query.lower().split()[:self.config.max_query_tokens]
            query_embeddings = None
        
        # Score all documents
        scores = []
        for doc_id, doc in self.documents.items():
            if query_embeddings is not None and doc['embeddings'] is not None:
                score = self._late_interaction_score(
                    query_embeddings, doc['embeddings']
                )
            else:
                # Fallback to token overlap
                score = self._token_overlap_score(query_tokens, doc['tokens'])
            
            scores.append({
                'doc_id': doc_id,
                'text': doc['text'],
                'score': score,
                'metadata': doc['metadata'],
            })
        
        # Sort and limit
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def _late_interaction_score(
        self,
        query_embeddings: List[List[float]],
        doc_embeddings: List[List[float]],
    ) -> float:
        """Compute late interaction score (MaxSim)."""
        total_score = 0.0
        
        for q_emb in query_embeddings:
            max_sim = 0.0
            for d_emb in doc_embeddings:
                sim = self._cosine_similarity(q_emb, d_emb)
                max_sim = max(max_sim, sim)
            total_score += max_sim
        
        return total_score
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Compute cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def _token_overlap_score(
        self,
        query_tokens: List[str],
        doc_tokens: List[str],
    ) -> float:
        """Fallback token overlap scoring."""
        query_set = set(t.lower() for t in query_tokens)
        doc_set = set(t.lower() for t in doc_tokens)
        overlap = len(query_set & doc_set)
        return overlap / max(len(query_set), 1)

class MultiVectorRetriever:
    """
    Multi-vector retrieval using multiple embeddings per document.
    
    Each document is represented by multiple vectors:
    - Dense embedding of full text
    - Embeddings of key passages
    - Embeddings of extracted entities
    - Summary embedding
    """
    
    def __init__(
        self,
        embed_fn: Optional[Callable] = None,
        search_fn: Optional[Callable] = None,
        aggregation: str = "max",  # max, mean, weighted
    ):
        """Initialize multi-vector retriever."""
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.aggregation = aggregation
        
        # Storage
        self.documents: Dict[str, Dict] = {}
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        key_passages: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        summary: Optional[str] = None,
    ):
        """Add document with multiple representations."""
        vectors = []
        
        # Full text embedding
        if self.embed_fn:
            full_embedding = self.embed_fn(text)
            vectors.append(('full', full_embedding))
        
        # Key passage embeddings
        if key_passages and self.embed_fn:
            for i, passage in enumerate(key_passages[:5]):
                vectors.append((f'passage_{i}', self.embed_fn(passage)))
        
        # Entity embeddings
        if entities and self.embed_fn:
            entity_text = " ".join(entities)
            vectors.append(('entities', self.embed_fn(entity_text)))
        
        # Summary embedding
        if summary and self.embed_fn:
            vectors.append(('summary', self.embed_fn(summary)))
        
        self.documents[doc_id] = {
            'text': text,
            'vectors': vectors,
            'key_passages': key_passages or [],
            'entities': entities or [],
            'summary': summary,
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """Retrieve using multi-vector scoring."""
        if not self.embed_fn:
            return []
        
        query_embedding = self.embed_fn(query)
        
        scores = []
        for doc_id, doc in self.documents.items():
            vector_scores = []
            for vec_type, vec in doc['vectors']:
                sim = self._cosine_similarity(query_embedding, vec)
                vector_scores.append(sim)
            
            # Aggregate scores
            if self.aggregation == "max":
                final_score = max(vector_scores) if vector_scores else 0.0
            elif self.aggregation == "mean":
                final_score = sum(vector_scores) / len(vector_scores) if vector_scores else 0.0
            else:
                final_score = max(vector_scores) if vector_scores else 0.0
            
            scores.append({
                'doc_id': doc_id,
                'text': doc['text'],
                'score': final_score,
            })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Compute cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
