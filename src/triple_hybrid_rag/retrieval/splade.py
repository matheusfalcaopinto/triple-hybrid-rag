"""
SPLADE Integration Module

SPLADE (Sparse Lexical and Expansion) provides learned sparse representations
for efficient lexical matching with semantic understanding.

Features:
- Sparse vector generation
- Term expansion for better recall
- Hybrid dense+sparse retrieval
"""

import re
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SpladeConfig:
    """Configuration for SPLADE retrieval."""
    
    enabled: bool = True
    
    # Model settings
    model_name: str = "naver/splade-cocondenser-ensembledistil"
    max_length: int = 256
    
    # Sparse vector settings
    top_k_tokens: int = 256  # Keep top K tokens per document
    min_weight: float = 0.01  # Minimum weight threshold
    
    # Retrieval settings
    default_top_k: int = 100
    use_expansion: bool = True  # Enable term expansion
    
    # Hybrid settings
    dense_weight: float = 0.7  # Weight for dense vectors in hybrid
    sparse_weight: float = 0.3  # Weight for sparse vectors in hybrid

@dataclass
class SparseVector:
    """A sparse vector representation."""
    indices: List[int]  # Token IDs or vocab indices
    values: List[float]  # Corresponding weights
    tokens: List[str] = field(default_factory=list)  # Optional token strings
    
    @property
    def dimension(self) -> int:
        """Get the number of non-zero elements."""
        return len(self.indices)
    
    def to_dict(self) -> Dict[int, float]:
        """Convert to dictionary format."""
        return dict(zip(self.indices, self.values))
    
    def top_tokens(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top K tokens by weight."""
        if not self.tokens:
            return []
        combined = list(zip(self.tokens, self.values))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]

@dataclass
class SpladeResult:
    """Result of SPLADE encoding."""
    text: str
    sparse_vector: SparseVector
    expanded_terms: List[str] = field(default_factory=list)
    encoding_time_ms: float = 0.0

@dataclass  
class SpladeSearchResult:
    """Result of SPLADE search."""
    doc_id: Any
    text: str
    score: float
    sparse_score: float = 0.0
    dense_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)

class SpladeEncoder:
    """
    SPLADE encoder for generating sparse representations.
    
    This is a simplified implementation that can work without
    the actual SPLADE model by using TF-IDF-like weighting with
    expansion heuristics.
    """
    
    def __init__(
        self,
        config: Optional[SpladeConfig] = None,
        model_fn: Optional[Callable[[str], Dict[str, float]]] = None,
    ):
        self.config = config or SpladeConfig()
        self.model_fn = model_fn
        
        # Vocabulary for token-to-id mapping
        self._vocab: Dict[str, int] = {}
        self._next_id = 0
        
        # IDF scores for fallback weighting
        self._idf: Dict[str, float] = {}
        self._doc_count = 0
    
    def encode(self, text: str) -> SpladeResult:
        """
        Encode text into SPLADE sparse vector.
        
        Args:
            text: Text to encode
            
        Returns:
            SpladeResult with sparse vector
        """
        import time
        start = time.time()
        
        if self.model_fn:
            # Use actual SPLADE model
            try:
                token_weights = self.model_fn(text)
                sparse_vector = self._weights_to_vector(token_weights)
                expanded = list(token_weights.keys())
            except Exception as e:
                logger.warning(f"SPLADE model failed: {e}, using fallback")
                sparse_vector, expanded = self._fallback_encode(text)
        else:
            # Use fallback encoding
            sparse_vector, expanded = self._fallback_encode(text)
        
        encoding_time = (time.time() - start) * 1000
        
        return SpladeResult(
            text=text,
            sparse_vector=sparse_vector,
            expanded_terms=expanded,
            encoding_time_ms=encoding_time,
        )
    
    def encode_batch(self, texts: List[str]) -> List[SpladeResult]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def _fallback_encode(self, text: str) -> Tuple[SparseVector, List[str]]:
        """
        Fallback encoding using TF-IDF-like weighting with expansion.
        
        This provides SPLADE-like behavior without the actual model.
        """
        # Tokenize
        tokens = self._tokenize(text)
        
        # Calculate term frequencies
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        # Apply log TF weighting
        token_weights = {}
        for token, count in tf.items():
            # Log TF
            tf_weight = 1 + math.log(count) if count > 0 else 0
            
            # IDF (use default if not available)
            idf_weight = self._idf.get(token, math.log(10))  # Default IDF
            
            # Combined weight
            weight = tf_weight * idf_weight
            
            if weight >= self.config.min_weight:
                token_weights[token] = weight
        
        # Term expansion (add related terms)
        expanded_terms = []
        if self.config.use_expansion:
            expanded_terms = self._expand_terms(list(tf.keys()))
            for term in expanded_terms:
                if term not in token_weights:
                    # Give expanded terms lower weight
                    token_weights[term] = self.config.min_weight * 2
        
        # Convert to sparse vector
        sparse_vector = self._weights_to_vector(token_weights)
        
        return sparse_vector, expanded_terms
    
    def _weights_to_vector(self, token_weights: Dict[str, float]) -> SparseVector:
        """Convert token weights to sparse vector."""
        # Sort by weight and keep top K
        sorted_tokens = sorted(
            token_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:self.config.top_k_tokens]
        
        indices = []
        values = []
        tokens = []
        
        for token, weight in sorted_tokens:
            if weight < self.config.min_weight:
                continue
                
            # Get or create token ID
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            
            indices.append(self._vocab[token])
            values.append(weight)
            tokens.append(token)
        
        return SparseVector(indices=indices, values=values, tokens=tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter stopwords (basic list)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'about', 'against', 'this', 'that', 'these',
            'those', 'it', 'its',
        }
        
        return [t for t in tokens if t not in stopwords and len(t) > 1]
    
    def _expand_terms(self, terms: List[str]) -> List[str]:
        """
        Expand terms with related terms.
        
        This is a simple heuristic expansion. Real SPLADE learns this.
        """
        expanded = set()
        
        # Simple expansion rules
        expansions = {
            # Technical terms
            'api': ['interface', 'endpoint', 'rest', 'http'],
            'database': ['db', 'sql', 'storage', 'data'],
            'authentication': ['auth', 'login', 'security', 'token'],
            'configuration': ['config', 'settings', 'setup'],
            'error': ['exception', 'bug', 'issue', 'problem'],
            'function': ['method', 'procedure', 'routine'],
            'parameter': ['argument', 'param', 'variable'],
            'return': ['response', 'result', 'output'],
            'create': ['add', 'insert', 'make', 'new'],
            'delete': ['remove', 'drop', 'erase'],
            'update': ['modify', 'change', 'edit'],
            'get': ['fetch', 'retrieve', 'read', 'query'],
            # Domain terms
            'policy': ['rule', 'guideline', 'regulation'],
            'refund': ['return', 'reimbursement', 'money'],
            'user': ['customer', 'client', 'member'],
            'product': ['item', 'goods', 'service'],
            'price': ['cost', 'rate', 'fee'],
        }
        
        for term in terms:
            term_lower = term.lower()
            if term_lower in expansions:
                expanded.update(expansions[term_lower])
        
        # Also add common suffixes
        for term in terms:
            if len(term) > 4:
                # Plural/singular
                if term.endswith('s'):
                    expanded.add(term[:-1])
                else:
                    expanded.add(term + 's')
                
                # -ing forms
                if term.endswith('ing'):
                    expanded.add(term[:-3])
                    expanded.add(term[:-3] + 'e')
        
        return list(expanded - set(terms))
    
    def update_idf(self, documents: List[str]) -> None:
        """Update IDF scores from document collection."""
        doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1
        
        self._doc_count = len(documents)
        
        for token, df in doc_freq.items():
            self._idf[token] = math.log(self._doc_count / (df + 1)) + 1

class SpladeRetriever:
    """
    SPLADE-based sparse retriever.
    
    Performs retrieval using sparse vector similarity.
    """
    
    def __init__(
        self,
        config: Optional[SpladeConfig] = None,
        encoder: Optional[SpladeEncoder] = None,
    ):
        self.config = config or SpladeConfig()
        self.encoder = encoder or SpladeEncoder(config)
        
        # Document index
        self._doc_vectors: Dict[Any, SparseVector] = {}
        self._doc_texts: Dict[Any, str] = {}
        
        # Inverted index for efficient retrieval
        self._inverted_index: Dict[int, List[Tuple[Any, float]]] = defaultdict(list)
    
    def index_documents(
        self,
        documents: List[Tuple[Any, str]],  # (doc_id, text) pairs
    ) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of (doc_id, text) tuples
        """
        # Update IDF from corpus
        self.encoder.update_idf([text for _, text in documents])
        
        # Encode and index each document
        for doc_id, text in documents:
            result = self.encoder.encode(text)
            self._doc_vectors[doc_id] = result.sparse_vector
            self._doc_texts[doc_id] = text
            
            # Add to inverted index
            for idx, value in zip(
                result.sparse_vector.indices,
                result.sparse_vector.values,
            ):
                self._inverted_index[idx].append((doc_id, value))
        
        logger.debug(f"Indexed {len(documents)} documents for SPLADE retrieval")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SpladeSearchResult]:
        """
        Search for documents matching query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results sorted by score
        """
        if not self.config.enabled:
            return []
        
        top_k = top_k or self.config.default_top_k
        
        # Encode query
        query_result = self.encoder.encode(query)
        query_vector = query_result.sparse_vector
        
        # Calculate scores using inverted index
        doc_scores: Dict[Any, float] = defaultdict(float)
        matched_terms: Dict[Any, List[str]] = defaultdict(list)
        
        query_weights = dict(zip(query_vector.indices, query_vector.values))
        query_tokens = dict(zip(query_vector.indices, query_vector.tokens))
        
        for q_idx, q_value in query_weights.items():
            if q_idx in self._inverted_index:
                for doc_id, d_value in self._inverted_index[q_idx]:
                    # Dot product contribution
                    doc_scores[doc_id] += q_value * d_value
                    
                    # Track matched terms
                    if q_idx in query_tokens:
                        matched_terms[doc_id].append(query_tokens[q_idx])
        
        # Sort by score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        
        # Build results
        results = []
        for doc_id, score in sorted_docs:
            results.append(SpladeSearchResult(
                doc_id=doc_id,
                text=self._doc_texts.get(doc_id, ""),
                score=score,
                sparse_score=score,
                matched_terms=matched_terms[doc_id],
            ))
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        dense_results: List[Tuple[Any, float]],  # (doc_id, score) from dense retrieval
        top_k: Optional[int] = None,
    ) -> List[SpladeSearchResult]:
        """
        Combine SPLADE sparse search with dense retrieval results.
        
        Args:
            query: Search query
            dense_results: Results from dense (embedding) retrieval
            top_k: Number of results to return
            
        Returns:
            Hybrid results combining sparse and dense scores
        """
        top_k = top_k or self.config.default_top_k
        
        # Get sparse results
        sparse_results = self.search(query, top_k=top_k * 2)
        
        # Create score maps
        sparse_scores = {r.doc_id: r.sparse_score for r in sparse_results}
        dense_scores = dict(dense_results)
        
        # Get all unique doc IDs
        all_docs = set(sparse_scores.keys()) | set(dense_scores.keys())
        
        # Calculate hybrid scores
        hybrid_scores = []
        for doc_id in all_docs:
            s_score = sparse_scores.get(doc_id, 0.0)
            d_score = dense_scores.get(doc_id, 0.0)
            
            # Weighted combination
            hybrid = (
                self.config.sparse_weight * s_score +
                self.config.dense_weight * d_score
            )
            
            hybrid_scores.append((doc_id, hybrid, s_score, d_score))
        
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        sparse_result_map = {r.doc_id: r for r in sparse_results}
        
        for doc_id, score, s_score, d_score in hybrid_scores[:top_k]:
            sparse_result = sparse_result_map.get(doc_id)
            
            results.append(SpladeSearchResult(
                doc_id=doc_id,
                text=self._doc_texts.get(doc_id, ""),
                score=score,
                sparse_score=s_score,
                dense_score=d_score,
                matched_terms=sparse_result.matched_terms if sparse_result else [],
            ))
        
        return results

class QueryTermAnalyzer:
    """
    Analyze query terms for SPLADE optimization.
    
    Identifies important terms and suggests expansions.
    """
    
    def __init__(self, encoder: Optional[SpladeEncoder] = None):
        self.encoder = encoder or SpladeEncoder()
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query terms.
        
        Returns:
            Analysis dict with term importance and suggestions
        """
        result = self.encoder.encode(query)
        
        # Get top terms
        top_terms = result.sparse_vector.top_tokens(10)
        
        # Categorize terms
        technical_terms = []
        domain_terms = []
        general_terms = []
        
        technical_patterns = [
            r'api', r'http', r'json', r'xml', r'sql', r'rest',
            r'config', r'auth', r'token', r'endpoint', r'database',
        ]
        
        for term, weight in top_terms:
            is_technical = any(
                re.search(p, term, re.IGNORECASE)
                for p in technical_patterns
            )
            
            if is_technical:
                technical_terms.append((term, weight))
            elif weight > 1.0:
                domain_terms.append((term, weight))
            else:
                general_terms.append((term, weight))
        
        return {
            'top_terms': top_terms,
            'technical_terms': technical_terms,
            'domain_terms': domain_terms,
            'general_terms': general_terms,
            'expanded_terms': result.expanded_terms,
            'total_terms': result.sparse_vector.dimension,
        }

def create_splade_retriever(
    config: Optional[SpladeConfig] = None,
    model_fn: Optional[Callable[[str], Dict[str, float]]] = None,
) -> SpladeRetriever:
    """Factory function to create SPLADE retriever."""
    encoder = SpladeEncoder(config=config, model_fn=model_fn)
    return SpladeRetriever(config=config, encoder=encoder)
