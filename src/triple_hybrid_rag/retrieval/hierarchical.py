"""
Hierarchical Retrieval Strategies

Implements advanced retrieval patterns:
- Parent Document Retriever: Store small chunks, retrieve parent documents
- Sentence Window Retriever: Retrieve sentences with surrounding context
- Auto-Merging Retriever: Automatically merge related chunks

Reference:
- LlamaIndex hierarchical retrieval patterns
- LangChain parent document retriever
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class DocumentNode:
    """Node in the document hierarchy."""
    id: str
    text: str
    level: int  # 0 = document, 1 = section, 2 = paragraph, 3 = sentence
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

@dataclass
class HierarchicalChunk:
    """Chunk with hierarchical context."""
    chunk_id: str
    text: str
    parent_text: Optional[str] = None
    parent_id: Optional[str] = None
    children_texts: List[str] = field(default_factory=list)
    window_text: Optional[str] = None
    level: int = 2
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ParentDocumentConfig:
    """Configuration for parent document retriever."""
    
    # Child chunk settings
    child_chunk_size: int = 200
    child_chunk_overlap: int = 50
    
    # Parent chunk settings
    parent_chunk_size: int = 1000
    parent_chunk_overlap: int = 200
    
    # Retrieval settings
    child_top_k: int = 20
    final_top_k: int = 5
    
    # Deduplication
    dedupe_parents: bool = True

class ParentDocumentRetriever:
    """
    Parent Document Retriever.
    
    Strategy:
    1. Index small chunks for precise matching
    2. On retrieval, return the parent (larger) chunks
    3. Provides better context while maintaining precision
    
    Usage:
        retriever = ParentDocumentRetriever(
            embed_fn=embedding_function,
            search_fn=vector_search,
        )
        
        # Build index
        retriever.add_documents(documents)
        
        # Retrieve
        results = retriever.retrieve("query", top_k=5)
    """
    
    def __init__(
        self,
        embed_fn: Optional[Callable] = None,
        search_fn: Optional[Callable] = None,
        config: Optional[ParentDocumentConfig] = None,
    ):
        """Initialize parent document retriever."""
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.config = config or ParentDocumentConfig()
        
        # Storage
        self.parent_chunks: Dict[str, DocumentNode] = {}
        self.child_chunks: Dict[str, DocumentNode] = {}
        self.child_to_parent: Dict[str, str] = {}
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Add documents to the retriever.
        
        Args:
            documents: List of document texts
            doc_ids: Optional document IDs
            metadata: Optional metadata for each document
        """
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            doc_meta = metadata[i] if metadata else {}
            
            # Create parent chunks
            parent_chunks = self._create_chunks(
                doc,
                self.config.parent_chunk_size,
                self.config.parent_chunk_overlap,
                level=1,
                doc_id=doc_id,
            )
            
            for parent in parent_chunks:
                self.parent_chunks[parent.id] = parent
                
                # Create child chunks for each parent
                child_chunks = self._create_chunks(
                    parent.text,
                    self.config.child_chunk_size,
                    self.config.child_chunk_overlap,
                    level=2,
                    doc_id=parent.id,
                )
                
                for child in child_chunks:
                    child.parent_id = parent.id
                    child.metadata = {**doc_meta, 'parent_id': parent.id}
                    self.child_chunks[child.id] = child
                    self.child_to_parent[child.id] = parent.id
                    parent.children_ids.append(child.id)
        
        logger.info(
            f"Added {len(documents)} documents, "
            f"{len(self.parent_chunks)} parent chunks, "
            f"{len(self.child_chunks)} child chunks"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[HierarchicalChunk]:
        """
        Retrieve parent documents based on child chunk matches.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of HierarchicalChunk with parent context
        """
        top_k = top_k or self.config.final_top_k
        
        if not self.search_fn:
            logger.warning("No search function provided")
            return []
        
        # Search child chunks
        child_results = self.search_fn(
            query,
            top_k=self.config.child_top_k,
        )
        
        if not child_results:
            return []
        
        # Map to parents
        parent_scores: Dict[str, Tuple[float, List[str]]] = defaultdict(lambda: (0.0, []))
        
        for result in child_results:
            child_id = getattr(result, 'chunk_id', getattr(result, 'id', None))
            score = getattr(result, 'score', 0.5)
            
            if child_id and child_id in self.child_to_parent:
                parent_id = self.child_to_parent[child_id]
                current_score, child_ids = parent_scores[parent_id]
                # Aggregate scores (max strategy)
                parent_scores[parent_id] = (
                    max(current_score, score),
                    child_ids + [child_id],
                )
        
        # Sort parents by score
        sorted_parents = sorted(
            parent_scores.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )[:top_k]
        
        # Build results
        results = []
        for parent_id, (score, matching_child_ids) in sorted_parents:
            parent = self.parent_chunks.get(parent_id)
            if not parent:
                continue
            
            # Get matching children texts
            children_texts = [
                self.child_chunks[cid].text
                for cid in matching_child_ids
                if cid in self.child_chunks
            ]
            
            results.append(HierarchicalChunk(
                chunk_id=parent_id,
                text=parent.text,
                parent_text=None,  # Could link to document level
                parent_id=parent.parent_id,
                children_texts=children_texts,
                level=parent.level,
                score=score,
                metadata=parent.metadata,
            ))
        
        return results
    
    def _create_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        level: int,
        doc_id: str,
    ) -> List[DocumentNode]:
        """Create fixed-size chunks from text."""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size // 2:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            chunk_id = f"{doc_id}_l{level}_c{chunk_idx}"
            
            chunks.append(DocumentNode(
                id=chunk_id,
                text=chunk_text.strip(),
                level=level,
                metadata={'doc_id': doc_id},
            ))
            
            start = end - overlap
            chunk_idx += 1
        
        return chunks

@dataclass
class SentenceWindowConfig:
    """Configuration for sentence window retriever."""
    
    # Window settings
    window_size: int = 3  # Sentences before and after
    
    # Sentence splitting
    sentence_delimiters: str = '.!?'
    min_sentence_length: int = 10
    
    # Retrieval
    top_k: int = 10
    
    # Merging
    merge_adjacent: bool = True
    merge_threshold: int = 2  # Max gap between sentences to merge

class SentenceWindowRetriever:
    """
    Sentence Window Retriever.
    
    Strategy:
    1. Index individual sentences for precise matching
    2. On retrieval, expand to include surrounding sentences
    3. Provides focused context around exact matches
    
    Usage:
        retriever = SentenceWindowRetriever(
            embed_fn=embedding_function,
            search_fn=vector_search,
        )
        
        # Build index
        retriever.add_documents(documents)
        
        # Retrieve
        results = retriever.retrieve("query")
    """
    
    def __init__(
        self,
        embed_fn: Optional[Callable] = None,
        search_fn: Optional[Callable] = None,
        config: Optional[SentenceWindowConfig] = None,
    ):
        """Initialize sentence window retriever."""
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.config = config or SentenceWindowConfig()
        
        # Storage
        self.documents: Dict[str, str] = {}
        self.sentences: Dict[str, DocumentNode] = {}
        self.doc_sentences: Dict[str, List[str]] = defaultdict(list)
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ):
        """Add documents to the retriever."""
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            self.documents[doc_id] = doc
            
            # Split into sentences
            sentences = self._split_sentences(doc)
            
            for j, sent in enumerate(sentences):
                sent_id = f"{doc_id}_s{j}"
                self.sentences[sent_id] = DocumentNode(
                    id=sent_id,
                    text=sent,
                    level=3,
                    metadata={
                        'doc_id': doc_id,
                        'sentence_idx': j,
                        'total_sentences': len(sentences),
                    },
                )
                self.doc_sentences[doc_id].append(sent_id)
        
        logger.info(
            f"Added {len(documents)} documents, "
            f"{len(self.sentences)} sentences"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[HierarchicalChunk]:
        """
        Retrieve sentences with surrounding window.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of HierarchicalChunk with window context
        """
        top_k = top_k or self.config.top_k
        
        if not self.search_fn:
            return []
        
        # Search sentences
        sentence_results = self.search_fn(query, top_k=top_k * 2)
        
        if not sentence_results:
            return []
        
        # Expand to windows and potentially merge
        results = []
        processed_sentences: Set[str] = set()
        
        for result in sentence_results:
            sent_id = getattr(result, 'id', getattr(result, 'chunk_id', None))
            score = getattr(result, 'score', 0.5)
            
            if not sent_id or sent_id in processed_sentences:
                continue
            
            sentence = self.sentences.get(sent_id)
            if not sentence:
                continue
            
            # Get window
            doc_id = sentence.metadata.get('doc_id')
            sent_idx = sentence.metadata.get('sentence_idx', 0)
            total_sentences = sentence.metadata.get('total_sentences', 1)
            
            # Calculate window bounds
            start_idx = max(0, sent_idx - self.config.window_size)
            end_idx = min(total_sentences, sent_idx + self.config.window_size + 1)
            
            # Get window sentences
            doc_sent_ids = self.doc_sentences.get(doc_id, [])
            window_sent_ids = doc_sent_ids[start_idx:end_idx]
            
            # Mark as processed
            for sid in window_sent_ids:
                processed_sentences.add(sid)
            
            # Build window text
            window_sentences = [
                self.sentences[sid].text
                for sid in window_sent_ids
                if sid in self.sentences
            ]
            window_text = ' '.join(window_sentences)
            
            results.append(HierarchicalChunk(
                chunk_id=sent_id,
                text=sentence.text,
                window_text=window_text,
                level=3,
                score=score,
                metadata={
                    'doc_id': doc_id,
                    'window_start': start_idx,
                    'window_end': end_idx,
                    'window_size': len(window_sentences),
                },
            ))
            
            if len(results) >= top_k:
                break
        
        # Optionally merge adjacent results
        if self.config.merge_adjacent:
            results = self._merge_adjacent_windows(results)
        
        return results[:top_k]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting
        pattern = f'[{re.escape(self.config.sentence_delimiters)}]+'
        raw_sentences = re.split(pattern, text)
        
        # Filter and clean
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            if len(sent) >= self.config.min_sentence_length:
                sentences.append(sent)
        
        return sentences
    
    def _merge_adjacent_windows(
        self,
        results: List[HierarchicalChunk],
    ) -> List[HierarchicalChunk]:
        """Merge adjacent window results."""
        if len(results) <= 1:
            return results
        
        # Group by document
        doc_results: Dict[str, List[HierarchicalChunk]] = defaultdict(list)
        for r in results:
            doc_id = r.metadata.get('doc_id')
            if doc_id:
                doc_results[doc_id].append(r)
        
        merged = []
        for doc_id, doc_chunks in doc_results.items():
            # Sort by window start
            doc_chunks.sort(key=lambda x: x.metadata.get('window_start', 0))
            
            current = doc_chunks[0]
            for next_chunk in doc_chunks[1:]:
                current_end = current.metadata.get('window_end', 0)
                next_start = next_chunk.metadata.get('window_start', 0)
                
                # Check if should merge
                gap = next_start - current_end
                if gap <= self.config.merge_threshold:
                    # Merge
                    current = HierarchicalChunk(
                        chunk_id=f"{current.chunk_id}+{next_chunk.chunk_id}",
                        text=f"{current.text} {next_chunk.text}",
                        window_text=f"{current.window_text or current.text} {next_chunk.window_text or next_chunk.text}",
                        level=current.level,
                        score=max(current.score, next_chunk.score),
                        metadata={
                            **current.metadata,
                            'window_end': next_chunk.metadata.get('window_end'),
                            'merged': True,
                        },
                    )
                else:
                    merged.append(current)
                    current = next_chunk
            
            merged.append(current)
        
        # Re-sort by score
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged

class AutoMergingRetriever:
    """
    Auto-Merging Retriever.
    
    Strategy:
    1. Build hierarchical document tree
    2. Index leaf nodes
    3. On retrieval, automatically merge to parent if majority of children match
    
    This provides adaptive granularity based on query needs.
    """
    
    def __init__(
        self,
        embed_fn: Optional[Callable] = None,
        search_fn: Optional[Callable] = None,
        merge_threshold: float = 0.5,
    ):
        """Initialize auto-merging retriever."""
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.merge_threshold = merge_threshold
        
        # Hierarchy storage
        self.nodes: Dict[str, DocumentNode] = {}
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ):
        """Build hierarchical index from documents."""
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            
            # Level 0: Document
            doc_node = DocumentNode(
                id=doc_id,
                text=doc,
                level=0,
            )
            self.nodes[doc_id] = doc_node
            
            # Level 1: Sections (split by double newlines or headers)
            sections = self._split_sections(doc)
            for j, section in enumerate(sections):
                section_id = f"{doc_id}_sec_{j}"
                section_node = DocumentNode(
                    id=section_id,
                    text=section,
                    level=1,
                    parent_id=doc_id,
                )
                self.nodes[section_id] = section_node
                self.parent_to_children[doc_id].append(section_id)
                doc_node.children_ids.append(section_id)
                
                # Level 2: Paragraphs
                paragraphs = self._split_paragraphs(section)
                for k, para in enumerate(paragraphs):
                    para_id = f"{section_id}_p_{k}"
                    para_node = DocumentNode(
                        id=para_id,
                        text=para,
                        level=2,
                        parent_id=section_id,
                    )
                    self.nodes[para_id] = para_node
                    self.parent_to_children[section_id].append(para_id)
                    section_node.children_ids.append(para_id)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[HierarchicalChunk]:
        """Retrieve with automatic merging."""
        if not self.search_fn:
            return []
        
        # Search leaf nodes (paragraphs)
        leaf_results = self.search_fn(query, top_k=top_k * 3)
        
        if not leaf_results:
            return []
        
        # Track matches per parent
        parent_matches: Dict[str, Dict] = defaultdict(
            lambda: {'score': 0.0, 'count': 0, 'children': []}
        )
        
        for result in leaf_results:
            node_id = getattr(result, 'id', None)
            score = getattr(result, 'score', 0.5)
            
            node = self.nodes.get(node_id)
            if not node or not node.parent_id:
                continue
            
            parent_id = node.parent_id
            parent_matches[parent_id]['score'] = max(
                parent_matches[parent_id]['score'], score
            )
            parent_matches[parent_id]['count'] += 1
            parent_matches[parent_id]['children'].append(node_id)
        
        # Decide whether to merge
        results = []
        processed_nodes: Set[str] = set()
        
        for parent_id, match_info in parent_matches.items():
            parent = self.nodes.get(parent_id)
            if not parent:
                continue
            
            total_children = len(parent.children_ids)
            match_ratio = match_info['count'] / total_children if total_children > 0 else 0
            
            if match_ratio >= self.merge_threshold:
                # Return parent (merged)
                if parent_id not in processed_nodes:
                    results.append(HierarchicalChunk(
                        chunk_id=parent_id,
                        text=parent.text,
                        children_texts=[
                            self.nodes[cid].text
                            for cid in match_info['children']
                            if cid in self.nodes
                        ],
                        level=parent.level,
                        score=match_info['score'],
                        metadata={
                            'merged': True,
                            'merge_ratio': match_ratio,
                            'matched_children': match_info['count'],
                        },
                    ))
                    processed_nodes.add(parent_id)
                    processed_nodes.update(match_info['children'])
            else:
                # Return individual children
                for child_id in match_info['children']:
                    if child_id not in processed_nodes:
                        child = self.nodes.get(child_id)
                        if child:
                            results.append(HierarchicalChunk(
                                chunk_id=child_id,
                                text=child.text,
                                parent_id=parent_id,
                                parent_text=parent.text[:200] + "...",
                                level=child.level,
                                score=match_info['score'],
                                metadata={'merged': False},
                            ))
                            processed_nodes.add(child_id)
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _split_sections(self, text: str) -> List[str]:
        """Split document into sections."""
        import re
        
        # Split by headers or double newlines
        sections = re.split(r'\n\n+|(?=^#+\s)', text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split section into paragraphs."""
        paragraphs = text.split('\n')
        return [p.strip() for p in paragraphs if len(p.strip()) > 50]
