"""
Query Classification and Index Routing Module

Intelligently routes queries to the optimal retrieval strategy
based on query characteristics and intent.

Features:
- ML-based query classification
- Dynamic index selection
- Confidence-based routing
- Multi-strategy fallback
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Callable, Set
from enum import Enum
from collections import Counter

logger = logging.getLogger(__name__)

class QueryCategory(Enum):
    """High-level query categories."""
    FACTUAL = "factual"  # Who, what, when, where questions
    PROCEDURAL = "procedural"  # How-to questions
    CONCEPTUAL = "conceptual"  # Why, explain questions
    COMPARATIVE = "comparative"  # Compare/contrast
    EXPLORATORY = "exploratory"  # Open-ended exploration
    LOOKUP = "lookup"  # Direct entity/term lookup
    ANALYTICAL = "analytical"  # Analysis, trends, patterns

class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    SEMANTIC_HEAVY = "semantic_heavy"  # Prioritize semantic search
    LEXICAL_HEAVY = "lexical_heavy"  # Prioritize BM25/lexical
    GRAPH_HEAVY = "graph_heavy"  # Prioritize knowledge graph
    HYBRID_BALANCED = "hybrid_balanced"  # Equal weight all channels
    DENSE_SPARSE_HYBRID = "dense_sparse_hybrid"  # Dense + SPLADE
    SEMANTIC_ONLY = "semantic_only"  # Only semantic search
    LEXICAL_ONLY = "lexical_only"  # Only lexical search

class IndexType(Enum):
    """Types of indices available."""
    VECTOR = "vector"  # Dense vector index (semantic)
    BM25 = "bm25"  # BM25/lexical index
    SPLADE = "splade"  # Sparse learned embeddings
    GRAPH = "graph"  # Knowledge graph
    HYBRID = "hybrid"  # Combined index

@dataclass
class QueryClassification:
    """Classification result for a query."""
    query: str
    category: QueryCategory
    confidence: float
    
    # Secondary classifications
    secondary_categories: List[Tuple[QueryCategory, float]] = field(default_factory=list)
    
    # Query characteristics
    is_question: bool = False
    question_type: Optional[str] = None  # what, how, why, etc.
    has_entities: bool = False
    entities: List[str] = field(default_factory=list)
    has_technical_terms: bool = False
    technical_terms: List[str] = field(default_factory=list)
    complexity_score: float = 0.5  # 0-1, higher = more complex
    
    # Features
    word_count: int = 0
    specificity: float = 0.5  # 0-1, higher = more specific

@dataclass
class RoutingDecision:
    """Routing decision for a query."""
    query: str
    classification: QueryClassification
    
    # Primary strategy
    strategy: RetrievalStrategy
    strategy_confidence: float
    
    # Index selection
    primary_index: IndexType
    secondary_indices: List[IndexType] = field(default_factory=list)
    
    # Channel weights
    semantic_weight: float = 0.7
    lexical_weight: float = 0.7
    graph_weight: float = 0.5
    splade_weight: float = 0.3
    
    # Parameters
    top_k_multiplier: float = 1.0  # Multiply default top_k
    use_hyde: bool = False
    use_query_expansion: bool = False
    use_reranking: bool = True
    
    # Reasoning
    reasoning: str = ""

@dataclass
class RouterConfig:
    """Configuration for query router."""
    
    enabled: bool = True
    
    # Default settings when routing is disabled
    default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_BALANCED
    
    # Classification thresholds
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.3
    
    # Feature weights for classification
    use_ml_classifier: bool = False  # Use ML model if available
    
    # Strategy preferences
    prefer_semantic_for_conceptual: bool = True
    prefer_lexical_for_factual: bool = True
    prefer_graph_for_relational: bool = True
    
    # Index availability
    available_indices: List[IndexType] = field(
        default_factory=lambda: [IndexType.VECTOR, IndexType.BM25, IndexType.GRAPH]
    )

class QueryClassifier:
    """
    Classify queries into categories.
    
    Uses pattern matching and heuristics for classification.
    Can be extended with ML-based classification.
    """
    
    # Question patterns
    QUESTION_PATTERNS = {
        'what': r'\b(what|which)\b.*\??\s*$',
        'how': r'\bhow\b.*\??\s*$',
        'why': r'\bwhy\b.*\??\s*$',
        'when': r'\bwhen\b.*\??\s*$',
        'where': r'\bwhere\b.*\??\s*$',
        'who': r'\bwho\b.*\??\s*$',
        'can': r'\bcan\b.*\??\s*$',
        'is': r'\b(is|are|does|do)\b.*\??\s*$',
    }
    
    # Category patterns
    CATEGORY_PATTERNS = {
        QueryCategory.PROCEDURAL: [
            r'\bhow\s+(to|do|can|should)\b',
            r'\bsteps?\b.*\b(to|for)\b',
            r'\bprocess\b.*\b(for|of|to)\b',
            r'\bguide\b',
            r'\btutorial\b',
            r'\binstructions?\b',
            r'\bconfigure\b',
            r'\bsetup\b',
            r'\binstall\b',
        ],
        QueryCategory.FACTUAL: [
            r'\b(what|which)\s+(is|are|was|were)\b',
            r'\bwhen\s+(is|was|did)\b',
            r'\bwho\s+(is|was|are|were)\b',
            r'\bwhere\s+(is|was|are|were)\b',
            r'\bdefine\b',
            r'\bdefinition\b',
            r'\bmeaning\b',
        ],
        QueryCategory.CONCEPTUAL: [
            r'\bwhy\b',
            r'\bexplain\b',
            r'\bunderstand\b',
            r'\bconcept\b',
            r'\btheory\b',
            r'\bprinciple\b',
            r'\breason\b',
            r'\bcause\b',
        ],
        QueryCategory.COMPARATIVE: [
            r'\bdifference\b',
            r'\bcompare\b',
            r'\bversus\b|\bvs\.?\b',
            r'\bbetter\b',
            r'\bworse\b',
            r'\badvantage\b',
            r'\bdisadvantage\b',
            r'\bsimilar\b',
            r'\bunlike\b',
        ],
        QueryCategory.ANALYTICAL: [
            r'\banalyze\b',
            r'\banalysis\b',
            r'\btrend\b',
            r'\bpattern\b',
            r'\bstatistics?\b',
            r'\bmetrics?\b',
            r'\bmeasure\b',
            r'\bevaluate\b',
        ],
        QueryCategory.LOOKUP: [
            r'^["\'].*["\']$',  # Quoted exact match
            r'^\w+$',  # Single word
            r'\b(find|show|get|list)\b.*\bname\b',
            r'\bcode\b.*\b(for|of)\b',
        ],
    }
    
    # Technical terms
    TECHNICAL_PATTERNS = [
        r'\bapi\b', r'\bhttp\b', r'\bjson\b', r'\bxml\b', r'\bsql\b',
        r'\bdatabase\b', r'\bserver\b', r'\bclient\b', r'\bendpoint\b',
        r'\bfunction\b', r'\bmethod\b', r'\bclass\b', r'\bmodule\b',
        r'\bconfig(uration)?\b', r'\bparameter\b', r'\bvariable\b',
        r'\berror\b', r'\bexception\b', r'\bbug\b',
        r'\btoken\b', r'\bauth(entication)?\b', r'\bsecurity\b',
        r'\bpipeline\b', r'\bworkflow\b', r'\bprocess\b',
    ]
    
    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        ml_classifier: Optional[Callable[[str], Tuple[QueryCategory, float]]] = None,
    ):
        self.config = config or RouterConfig()
        self.ml_classifier = ml_classifier
    
    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query into categories.
        
        Args:
            query: The query text
            
        Returns:
            QueryClassification with category and metadata
        """
        query_lower = query.lower().strip()
        
        # Try ML classifier first
        if self.config.use_ml_classifier and self.ml_classifier:
            try:
                category, confidence = self.ml_classifier(query)
                return self._build_classification(query, category, confidence)
            except Exception as e:
                logger.warning(f"ML classifier failed: {e}, using rules")
        
        # Rule-based classification
        category_scores: Dict[QueryCategory, float] = {}
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1.0
            if score > 0:
                category_scores[category] = score / len(patterns)
        
        # Determine primary category
        if category_scores:
            sorted_categories = sorted(
                category_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            primary_category = sorted_categories[0][0]
            confidence = min(sorted_categories[0][1] + 0.3, 1.0)
        else:
            primary_category = QueryCategory.EXPLORATORY
            confidence = 0.3
        
        return self._build_classification(query, primary_category, confidence, category_scores)
    
    def _build_classification(
        self,
        query: str,
        category: QueryCategory,
        confidence: float,
        all_scores: Optional[Dict[QueryCategory, float]] = None,
    ) -> QueryClassification:
        """Build full classification result."""
        query_lower = query.lower()
        
        # Detect question type
        is_question = query.strip().endswith('?') or any(
            re.search(pattern, query_lower)
            for pattern in self.QUESTION_PATTERNS.values()
        )
        
        question_type = None
        for q_type, pattern in self.QUESTION_PATTERNS.items():
            if re.search(pattern, query_lower):
                question_type = q_type
                break
        
        # Detect technical terms
        technical_terms = []
        for pattern in self.TECHNICAL_PATTERNS:
            matches = re.findall(pattern, query_lower)
            technical_terms.extend(matches)
        
        # Detect entities (capitalized words, quoted strings)
        entities = re.findall(r'[A-Z][a-zA-Z]+', query)
        entities.extend(re.findall(r'["\']([^"\']+)["\']', query))
        
        # Calculate complexity
        word_count = len(query.split())
        has_multiple_clauses = bool(re.search(r'\b(and|or|but|if|when|while)\b', query_lower))
        complexity = min(
            (word_count / 20) * 0.5 + 
            (0.3 if has_multiple_clauses else 0) +
            (0.2 if len(technical_terms) > 2 else 0),
            1.0
        )
        
        # Calculate specificity
        specificity = min(
            (len(technical_terms) / 5) * 0.4 +
            (len(entities) / 3) * 0.3 +
            (word_count / 15) * 0.3,
            1.0
        )
        
        # Secondary categories
        secondary = []
        if all_scores:
            for cat, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[1:3]:
                if score > 0.1:
                    secondary.append((cat, score))
        
        return QueryClassification(
            query=query,
            category=category,
            confidence=confidence,
            secondary_categories=secondary,
            is_question=is_question,
            question_type=question_type,
            has_entities=len(entities) > 0,
            entities=entities,
            has_technical_terms=len(technical_terms) > 0,
            technical_terms=list(set(technical_terms)),
            complexity_score=complexity,
            word_count=word_count,
            specificity=specificity,
        )

class QueryRouter:
    """
    Route queries to optimal retrieval strategies.
    
    Based on classification, selects the best combination of:
    - Search indices
    - Retrieval algorithms
    - Enhancement techniques
    """
    
    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        classifier: Optional[QueryClassifier] = None,
    ):
        self.config = config or RouterConfig()
        self.classifier = classifier or QueryClassifier(config)
    
    def route(self, query: str) -> RoutingDecision:
        """
        Route a query to optimal retrieval strategy.
        
        Args:
            query: The query to route
            
        Returns:
            RoutingDecision with strategy and parameters
        """
        if not self.config.enabled:
            return self._default_routing(query)
        
        # Classify query
        classification = self.classifier.classify(query)
        
        # Select strategy based on classification
        strategy, strategy_confidence = self._select_strategy(classification)
        
        # Select indices
        primary_index, secondary_indices = self._select_indices(
            classification, strategy
        )
        
        # Determine channel weights
        weights = self._compute_weights(classification, strategy)
        
        # Determine enhancement settings
        enhancements = self._determine_enhancements(classification, strategy)
        
        # Build reasoning
        reasoning = self._build_reasoning(classification, strategy)
        
        return RoutingDecision(
            query=query,
            classification=classification,
            strategy=strategy,
            strategy_confidence=strategy_confidence,
            primary_index=primary_index,
            secondary_indices=secondary_indices,
            semantic_weight=weights['semantic'],
            lexical_weight=weights['lexical'],
            graph_weight=weights['graph'],
            splade_weight=weights.get('splade', 0.0),
            top_k_multiplier=enhancements['top_k_multiplier'],
            use_hyde=enhancements['use_hyde'],
            use_query_expansion=enhancements['use_query_expansion'],
            use_reranking=enhancements['use_reranking'],
            reasoning=reasoning,
        )
    
    def _select_strategy(
        self,
        classification: QueryClassification,
    ) -> Tuple[RetrievalStrategy, float]:
        """Select retrieval strategy based on classification."""
        category = classification.category
        confidence = classification.confidence
        
        # High-confidence category-specific strategies
        if confidence >= self.config.high_confidence_threshold:
            if category == QueryCategory.FACTUAL:
                if self.config.prefer_lexical_for_factual:
                    return RetrievalStrategy.LEXICAL_HEAVY, confidence
                return RetrievalStrategy.DENSE_SPARSE_HYBRID, confidence
            
            elif category == QueryCategory.CONCEPTUAL:
                if self.config.prefer_semantic_for_conceptual:
                    return RetrievalStrategy.SEMANTIC_HEAVY, confidence
                return RetrievalStrategy.HYBRID_BALANCED, confidence
            
            elif category == QueryCategory.PROCEDURAL:
                # Procedural often benefits from semantic understanding
                return RetrievalStrategy.SEMANTIC_HEAVY, confidence * 0.9
            
            elif category == QueryCategory.COMPARATIVE:
                # Need both semantic similarity and exact matching
                return RetrievalStrategy.HYBRID_BALANCED, confidence
            
            elif category == QueryCategory.LOOKUP:
                # Direct lookup benefits from lexical
                return RetrievalStrategy.LEXICAL_HEAVY, confidence
            
            elif category == QueryCategory.ANALYTICAL:
                # Analytical queries often need graph traversal
                if self.config.prefer_graph_for_relational:
                    return RetrievalStrategy.GRAPH_HEAVY, confidence * 0.8
                return RetrievalStrategy.HYBRID_BALANCED, confidence
        
        # Check for relational indicators
        if classification.has_entities and classification.complexity_score > 0.5:
            return RetrievalStrategy.GRAPH_HEAVY, 0.6
        
        # Low confidence or exploratory - use balanced approach
        return RetrievalStrategy.HYBRID_BALANCED, 0.5
    
    def _select_indices(
        self,
        classification: QueryClassification,
        strategy: RetrievalStrategy,
    ) -> Tuple[IndexType, List[IndexType]]:
        """Select primary and secondary indices."""
        available = set(self.config.available_indices)
        
        # Strategy-based primary index selection
        strategy_to_primary = {
            RetrievalStrategy.SEMANTIC_HEAVY: IndexType.VECTOR,
            RetrievalStrategy.LEXICAL_HEAVY: IndexType.BM25,
            RetrievalStrategy.GRAPH_HEAVY: IndexType.GRAPH,
            RetrievalStrategy.HYBRID_BALANCED: IndexType.HYBRID,
            RetrievalStrategy.DENSE_SPARSE_HYBRID: IndexType.VECTOR,
            RetrievalStrategy.SEMANTIC_ONLY: IndexType.VECTOR,
            RetrievalStrategy.LEXICAL_ONLY: IndexType.BM25,
        }
        
        primary = strategy_to_primary.get(strategy, IndexType.VECTOR)
        
        # Fall back if not available
        if primary not in available:
            primary = list(available)[0] if available else IndexType.VECTOR
        
        # Select secondary indices
        secondary = []
        
        if strategy in [
            RetrievalStrategy.HYBRID_BALANCED,
            RetrievalStrategy.DENSE_SPARSE_HYBRID,
        ]:
            for idx_type in [IndexType.VECTOR, IndexType.BM25, IndexType.GRAPH]:
                if idx_type in available and idx_type != primary:
                    secondary.append(idx_type)
        
        # Add SPLADE if available and beneficial
        if (
            IndexType.SPLADE in available and
            strategy in [
                RetrievalStrategy.LEXICAL_HEAVY,
                RetrievalStrategy.DENSE_SPARSE_HYBRID,
            ]
        ):
            if IndexType.SPLADE not in secondary:
                secondary.append(IndexType.SPLADE)
        
        return primary, secondary
    
    def _compute_weights(
        self,
        classification: QueryClassification,
        strategy: RetrievalStrategy,
    ) -> Dict[str, float]:
        """Compute channel weights."""
        # Base weights by strategy
        strategy_weights = {
            RetrievalStrategy.SEMANTIC_HEAVY: {
                'semantic': 1.0, 'lexical': 0.3, 'graph': 0.3, 'splade': 0.2,
            },
            RetrievalStrategy.LEXICAL_HEAVY: {
                'semantic': 0.4, 'lexical': 1.0, 'graph': 0.3, 'splade': 0.6,
            },
            RetrievalStrategy.GRAPH_HEAVY: {
                'semantic': 0.5, 'lexical': 0.4, 'graph': 1.0, 'splade': 0.2,
            },
            RetrievalStrategy.HYBRID_BALANCED: {
                'semantic': 0.7, 'lexical': 0.7, 'graph': 0.6, 'splade': 0.3,
            },
            RetrievalStrategy.DENSE_SPARSE_HYBRID: {
                'semantic': 0.8, 'lexical': 0.6, 'graph': 0.3, 'splade': 0.7,
            },
            RetrievalStrategy.SEMANTIC_ONLY: {
                'semantic': 1.0, 'lexical': 0.0, 'graph': 0.0, 'splade': 0.0,
            },
            RetrievalStrategy.LEXICAL_ONLY: {
                'semantic': 0.0, 'lexical': 1.0, 'graph': 0.0, 'splade': 0.0,
            },
        }
        
        weights = strategy_weights.get(
            strategy,
            {'semantic': 0.7, 'lexical': 0.7, 'graph': 0.5, 'splade': 0.3},
        ).copy()
        
        # Adjust based on classification
        if classification.has_technical_terms:
            weights['lexical'] *= 1.2
            weights['splade'] *= 1.3
        
        if classification.has_entities:
            weights['graph'] *= 1.2
        
        if classification.complexity_score > 0.7:
            weights['semantic'] *= 1.1
        
        # Normalize (optional)
        # total = sum(weights.values())
        # weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _determine_enhancements(
        self,
        classification: QueryClassification,
        strategy: RetrievalStrategy,
    ) -> Dict[str, Any]:
        """Determine which enhancements to enable."""
        enhancements = {
            'top_k_multiplier': 1.0,
            'use_hyde': False,
            'use_query_expansion': False,
            'use_reranking': True,
        }
        
        # HyDE for semantic-heavy strategies with conceptual queries
        if (
            strategy in [RetrievalStrategy.SEMANTIC_HEAVY, RetrievalStrategy.HYBRID_BALANCED] and
            classification.category in [QueryCategory.CONCEPTUAL, QueryCategory.PROCEDURAL]
        ):
            enhancements['use_hyde'] = True
        
        # Query expansion for complex or exploratory queries
        if (
            classification.complexity_score > 0.5 or
            classification.category == QueryCategory.EXPLORATORY
        ):
            enhancements['use_query_expansion'] = True
            enhancements['top_k_multiplier'] = 1.5
        
        # More results for comparative queries
        if classification.category == QueryCategory.COMPARATIVE:
            enhancements['top_k_multiplier'] = 2.0
        
        # Skip reranking for simple lookups
        if (
            classification.category == QueryCategory.LOOKUP and
            classification.confidence > 0.8
        ):
            enhancements['use_reranking'] = False
        
        return enhancements
    
    def _build_reasoning(
        self,
        classification: QueryClassification,
        strategy: RetrievalStrategy,
    ) -> str:
        """Build human-readable reasoning for the routing decision."""
        parts = []
        
        parts.append(f"Category: {classification.category.value} (conf: {classification.confidence:.2f})")
        parts.append(f"Strategy: {strategy.value}")
        
        if classification.has_technical_terms:
            parts.append(f"Technical: {', '.join(classification.technical_terms[:3])}")
        
        if classification.has_entities:
            parts.append(f"Entities: {', '.join(classification.entities[:3])}")
        
        if classification.complexity_score > 0.6:
            parts.append("Complex query")
        
        return " | ".join(parts)
    
    def _default_routing(self, query: str) -> RoutingDecision:
        """Return default routing when disabled."""
        classification = QueryClassification(
            query=query,
            category=QueryCategory.EXPLORATORY,
            confidence=0.5,
        )
        
        return RoutingDecision(
            query=query,
            classification=classification,
            strategy=self.config.default_strategy,
            strategy_confidence=1.0,
            primary_index=IndexType.HYBRID,
            semantic_weight=0.7,
            lexical_weight=0.7,
            graph_weight=0.5,
            reasoning="Default routing (router disabled)",
        )

class AdaptiveRouter:
    """
    Adaptive router that learns from feedback.
    
    Tracks query performance and adjusts routing decisions
    based on historical success rates.
    """
    
    def __init__(
        self,
        base_router: Optional[QueryRouter] = None,
        learning_rate: float = 0.1,
    ):
        self.base_router = base_router or QueryRouter()
        self.learning_rate = learning_rate
        
        # Track performance by category and strategy
        self._performance: Dict[Tuple[QueryCategory, RetrievalStrategy], List[float]] = {}
        self._query_count = 0
    
    def route(self, query: str) -> RoutingDecision:
        """Route with adaptive adjustments."""
        self._query_count += 1
        
        # Get base decision
        decision = self.base_router.route(query)
        
        # Check if we have enough data to adjust
        key = (decision.classification.category, decision.strategy)
        
        if key in self._performance and len(self._performance[key]) >= 10:
            avg_score = sum(self._performance[key]) / len(self._performance[key])
            
            # If performance is poor, try alternative
            if avg_score < 0.5:
                alternative = self._get_alternative_strategy(decision)
                if alternative:
                    decision.strategy = alternative
                    decision.reasoning += f" | Adapted: poor perf ({avg_score:.2f})"
        
        return decision
    
    def record_feedback(
        self,
        decision: RoutingDecision,
        score: float,  # 0-1, quality score
    ) -> None:
        """Record feedback for a routing decision."""
        key = (decision.classification.category, decision.strategy)
        
        if key not in self._performance:
            self._performance[key] = []
        
        # Keep rolling window of last 100 scores
        self._performance[key].append(score)
        if len(self._performance[key]) > 100:
            self._performance[key] = self._performance[key][-100:]
    
    def _get_alternative_strategy(
        self,
        decision: RoutingDecision,
    ) -> Optional[RetrievalStrategy]:
        """Get alternative strategy if current is performing poorly."""
        current = decision.strategy
        
        # Strategy alternatives
        alternatives = {
            RetrievalStrategy.SEMANTIC_HEAVY: RetrievalStrategy.HYBRID_BALANCED,
            RetrievalStrategy.LEXICAL_HEAVY: RetrievalStrategy.DENSE_SPARSE_HYBRID,
            RetrievalStrategy.GRAPH_HEAVY: RetrievalStrategy.HYBRID_BALANCED,
            RetrievalStrategy.HYBRID_BALANCED: RetrievalStrategy.SEMANTIC_HEAVY,
        }
        
        return alternatives.get(current)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = {
            'total_queries': self._query_count,
            'tracked_combinations': len(self._performance),
            'performance_by_category': {},
        }
        
        for (category, strategy), scores in self._performance.items():
            key = f"{category.value}_{strategy.value}"
            stats['performance_by_category'][key] = {
                'count': len(scores),
                'avg_score': sum(scores) / len(scores) if scores else 0,
            }
        
        return stats

def create_query_router(
    config: Optional[RouterConfig] = None,
    adaptive: bool = False,
) -> QueryRouter:
    """Factory function to create a query router."""
    router = QueryRouter(config=config)
    
    if adaptive:
        return AdaptiveRouter(base_router=router)
    
    return router
