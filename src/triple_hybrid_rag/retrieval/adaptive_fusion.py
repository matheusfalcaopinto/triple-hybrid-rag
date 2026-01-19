"""
Adaptive Query-Aware Fusion (AQAF)

Dynamically adjusts fusion weights based on query characteristics
instead of using static RRF weights.

Features:
- Query intent classification
- Query complexity analysis
- Entity density detection
- Temporal signal detection
- ML-predicted channel weights
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from triple_hybrid_rag.types import SearchResult

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent categories for weight prediction."""
    FACTUAL = "factual"           # Who, what, when, where facts
    PROCEDURAL = "procedural"     # How-to, step-by-step
    CONCEPTUAL = "conceptual"     # Explain, describe concepts
    COMPARATIVE = "comparative"   # Compare, contrast, vs
    NAVIGATIONAL = "navigational" # Find specific document/section
    RELATIONAL = "relational"     # Relationships between entities
    EXPLORATORY = "exploratory"   # Open-ended, research
    TECHNICAL = "technical"       # Code, API, technical details


@dataclass
class QueryFeatures:
    """Extracted features from a query for weight prediction."""
    
    # Basic features
    word_count: int = 0
    char_count: int = 0
    
    # Intent classification
    primary_intent: QueryIntent = QueryIntent.FACTUAL
    intent_confidence: float = 0.5
    
    # Complexity indicators
    has_multiple_clauses: bool = False
    question_type: str = "unknown"  # who, what, when, where, why, how
    
    # Entity features
    entity_count: int = 0
    has_named_entities: bool = False
    entity_types: List[str] = field(default_factory=list)
    
    # Temporal features
    has_temporal_reference: bool = False
    temporal_type: Optional[str] = None  # past, present, future, relative
    
    # Technical features
    has_code_reference: bool = False
    has_technical_terms: bool = False
    
    # Lexical features
    keyword_density: float = 0.0
    stopword_ratio: float = 0.0
    
    # Semantic features
    abstraction_level: str = "medium"  # low, medium, high


@dataclass
class FusionWeights:
    """Adaptive weights for multi-channel fusion."""
    lexical_weight: float = 0.7
    semantic_weight: float = 0.8
    graph_weight: float = 1.0
    splade_weight: float = 0.0  # 0 if SPLADE not enabled
    
    # Confidence in the prediction
    confidence: float = 0.5
    
    # Metadata
    query_features: Optional[QueryFeatures] = None
    reasoning: str = ""
    
    def normalize(self) -> "FusionWeights":
        """Normalize weights to sum to 1."""
        total = self.lexical_weight + self.semantic_weight + self.graph_weight + self.splade_weight
        if total > 0:
            return FusionWeights(
                lexical_weight=self.lexical_weight / total,
                semantic_weight=self.semantic_weight / total,
                graph_weight=self.graph_weight / total,
                splade_weight=self.splade_weight / total,
                confidence=self.confidence,
                query_features=self.query_features,
                reasoning=self.reasoning,
            )
        return self
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "lexical": self.lexical_weight,
            "semantic": self.semantic_weight,
            "graph": self.graph_weight,
            "splade": self.splade_weight,
        }


@dataclass
class AdaptiveFusionConfig:
    """Configuration for adaptive fusion."""
    enabled: bool = True
    
    # Default weights (fallback)
    default_lexical_weight: float = 0.7
    default_semantic_weight: float = 0.8
    default_graph_weight: float = 1.0
    default_splade_weight: float = 0.0
    
    # Intent-based weight adjustments
    use_intent_weights: bool = True
    
    # Feature extraction
    extract_entities: bool = True
    detect_temporal: bool = True
    
    # Confidence threshold for using adaptive weights
    min_confidence: float = 0.3


class QueryFeatureExtractor:
    """Extract features from queries for weight prediction."""
    
    # Question word patterns
    QUESTION_PATTERNS = {
        "who": r"\bwho\b",
        "what": r"\bwhat\b",
        "when": r"\bwhen\b",
        "where": r"\bwhere\b",
        "why": r"\bwhy\b",
        "how": r"\bhow\b",
        "which": r"\bwhich\b",
        "is": r"^is\b|^are\b|^was\b|^were\b",
        "can": r"^can\b|^could\b",
        "do": r"^do\b|^does\b|^did\b",
    }
    
    # Intent detection patterns
    INTENT_PATTERNS = {
        QueryIntent.PROCEDURAL: [
            r"\bhow\s+to\b", r"\bsteps?\s+to\b", r"\bguide\b", r"\btutorial\b",
            r"\bprocess\b", r"\bprocedure\b", r"\binstructions?\b",
        ],
        QueryIntent.COMPARATIVE: [
            r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bdifference\b",
            r"\bbetter\b", r"\bworse\b", r"\badvantages?\b", r"\bdisadvantages?\b",
        ],
        QueryIntent.CONCEPTUAL: [
            r"\bexplain\b", r"\bdescribe\b", r"\bwhat\s+is\b", r"\bdefine\b",
            r"\bmeaning\b", r"\bconcept\b", r"\bunderstand\b",
        ],
        QueryIntent.RELATIONAL: [
            r"\brelation\b", r"\bconnect\b", r"\blink\b", r"\bassociat\b",
            r"\brelated\s+to\b", r"\bbelong\b", r"\bpart\s+of\b",
        ],
        QueryIntent.NAVIGATIONAL: [
            r"\bfind\b", r"\blocate\b", r"\bwhere\s+is\b", r"\bsection\b",
            r"\bpage\b", r"\bdocument\b", r"\bfile\b",
        ],
        QueryIntent.TECHNICAL: [
            r"\bapi\b", r"\bcode\b", r"\bfunction\b", r"\berror\b",
            r"\bimplementat\b", r"\bconfigur\b", r"\bdebug\b", r"\bsyntax\b",
        ],
        QueryIntent.FACTUAL: [
            r"\bwho\b", r"\bwhen\b", r"\bwhere\b", r"\bhow\s+many\b",
            r"\bhow\s+much\b", r"\bdate\b", r"\btime\b", r"\bnumber\b",
        ],
    }
    
    # Temporal patterns
    TEMPORAL_PATTERNS = {
        "past": [r"\bwas\b", r"\bwere\b", r"\bprevious\b", r"\blast\b", r"\bago\b", r"\bhistor\b"],
        "present": [r"\bis\b", r"\bare\b", r"\bcurrent\b", r"\bnow\b", r"\btoday\b"],
        "future": [r"\bwill\b", r"\bfuture\b", r"\bnext\b", r"\bupcoming\b", r"\bplanned\b"],
        "relative": [r"\bbefore\b", r"\bafter\b", r"\bsince\b", r"\buntil\b", r"\bduring\b"],
    }
    
    # Technical terms (simplified)
    TECHNICAL_TERMS = {
        r"\bapi\b", r"\bjson\b", r"\bxml\b", r"\bsql\b", r"\bhttp\b",
        r"\brest\b", r"\bgraphql\b", r"\bdatabase\b", r"\bserver\b",
        r"\bclient\b", r"\bbackend\b", r"\bfrontend\b", r"\bfunction\b",
        r"\bmethod\b", r"\bclass\b", r"\bobject\b", r"\bvariable\b",
    }
    
    # Stop words for density calculation
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "but", "if", "or", "because", "until", "while", "this",
        "that", "these", "those", "what", "which", "who", "whom",
    }
    
    def extract(self, query: str) -> QueryFeatures:
        """Extract features from a query."""
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        features = QueryFeatures(
            word_count=len(words),
            char_count=len(query),
        )
        
        # Detect question type
        features.question_type = self._detect_question_type(query_lower)
        
        # Detect primary intent
        features.primary_intent, features.intent_confidence = self._detect_intent(query_lower)
        
        # Detect complexity
        features.has_multiple_clauses = self._has_multiple_clauses(query_lower)
        
        # Detect temporal references
        features.has_temporal_reference, features.temporal_type = self._detect_temporal(query_lower)
        
        # Detect technical content
        features.has_technical_terms = self._has_technical_terms(query_lower)
        features.has_code_reference = bool(re.search(r'`[^`]+`|```', query))
        
        # Calculate lexical features
        features.stopword_ratio = self._calculate_stopword_ratio(words)
        features.keyword_density = 1.0 - features.stopword_ratio
        
        # Estimate abstraction level
        features.abstraction_level = self._estimate_abstraction(features)
        
        return features
    
    def _detect_question_type(self, query: str) -> str:
        """Detect the question type."""
        for qtype, pattern in self.QUESTION_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                return qtype
        return "statement"
    
    def _detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Detect the primary intent of the query."""
        intent_scores: Dict[QueryIntent, int] = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query, re.IGNORECASE))
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return QueryIntent.EXPLORATORY, 0.3
        
        # Get the intent with highest score
        best_intent = max(intent_scores, key=lambda x: intent_scores[x])
        max_score = intent_scores[best_intent]
        
        # Calculate confidence based on score and specificity
        confidence = min(0.9, 0.4 + (max_score * 0.15))
        
        return best_intent, confidence
    
    def _has_multiple_clauses(self, query: str) -> bool:
        """Check if query has multiple clauses."""
        clause_indicators = [" and ", " or ", " but ", ",", ";"]
        return any(ind in query for ind in clause_indicators) and len(query.split()) > 5
    
    def _detect_temporal(self, query: str) -> Tuple[bool, Optional[str]]:
        """Detect temporal references in the query."""
        for temporal_type, patterns in self.TEMPORAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return True, temporal_type
        return False, None
    
    def _has_technical_terms(self, query: str) -> bool:
        """Check if query contains technical terms."""
        for pattern in self.TECHNICAL_TERMS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _calculate_stopword_ratio(self, words: List[str]) -> float:
        """Calculate the ratio of stop words in the query."""
        if not words:
            return 0.0
        stopword_count = sum(1 for w in words if w in self.STOP_WORDS)
        return stopword_count / len(words)
    
    def _estimate_abstraction(self, features: QueryFeatures) -> str:
        """Estimate the abstraction level of the query."""
        # High abstraction: conceptual, comparative queries
        if features.primary_intent in [QueryIntent.CONCEPTUAL, QueryIntent.COMPARATIVE]:
            return "high"
        
        # Low abstraction: factual, navigational queries
        if features.primary_intent in [QueryIntent.FACTUAL, QueryIntent.NAVIGATIONAL]:
            return "low"
        
        # Medium for everything else
        return "medium"


class AdaptiveFusionPredictor:
    """
    Predict optimal fusion weights based on query features.
    
    Uses a rule-based system with learned adjustments.
    Can be extended to use a trained ML model.
    """
    
    # Base weights by intent
    INTENT_WEIGHTS: Dict[QueryIntent, Dict[str, float]] = {
        QueryIntent.FACTUAL: {
            "lexical": 0.9,   # High lexical for exact facts
            "semantic": 0.6,
            "graph": 0.8,    # Entities often involved
            "splade": 0.7,
        },
        QueryIntent.PROCEDURAL: {
            "lexical": 0.6,
            "semantic": 0.9,  # High semantic for understanding steps
            "graph": 0.5,
            "splade": 0.6,
        },
        QueryIntent.CONCEPTUAL: {
            "lexical": 0.5,
            "semantic": 1.0,  # Very high semantic for concepts
            "graph": 0.6,
            "splade": 0.7,
        },
        QueryIntent.COMPARATIVE: {
            "lexical": 0.6,
            "semantic": 0.9,
            "graph": 0.7,    # Relationships matter
            "splade": 0.8,
        },
        QueryIntent.NAVIGATIONAL: {
            "lexical": 1.0,   # Very high lexical for navigation
            "semantic": 0.5,
            "graph": 0.4,
            "splade": 0.6,
        },
        QueryIntent.RELATIONAL: {
            "lexical": 0.5,
            "semantic": 0.7,
            "graph": 1.0,    # Very high graph for relationships
            "splade": 0.6,
        },
        QueryIntent.EXPLORATORY: {
            "lexical": 0.7,
            "semantic": 0.8,
            "graph": 0.7,
            "splade": 0.7,
        },
        QueryIntent.TECHNICAL: {
            "lexical": 0.8,   # High for exact terms
            "semantic": 0.7,
            "graph": 0.6,
            "splade": 0.9,   # SPLADE good for technical
        },
    }
    
    def __init__(self, config: Optional[AdaptiveFusionConfig] = None):
        self.config = config or AdaptiveFusionConfig()
        self.feature_extractor = QueryFeatureExtractor()
    
    def predict(self, query: str, features: Optional[QueryFeatures] = None) -> FusionWeights:
        """Predict optimal fusion weights for a query."""
        if not self.config.enabled:
            return self._default_weights()
        
        # Extract features if not provided
        if features is None:
            features = self.feature_extractor.extract(query)
        
        # Get base weights from intent
        base_weights = self._get_intent_weights(features.primary_intent)
        
        # Apply adjustments based on other features
        adjusted_weights = self._apply_adjustments(base_weights, features)
        
        # Build reasoning
        reasoning = self._build_reasoning(features, adjusted_weights)
        
        return FusionWeights(
            lexical_weight=adjusted_weights["lexical"],
            semantic_weight=adjusted_weights["semantic"],
            graph_weight=adjusted_weights["graph"],
            splade_weight=adjusted_weights["splade"] if self.config.default_splade_weight > 0 else 0,
            confidence=features.intent_confidence,
            query_features=features,
            reasoning=reasoning,
        )
    
    def _default_weights(self) -> FusionWeights:
        """Return default static weights."""
        return FusionWeights(
            lexical_weight=self.config.default_lexical_weight,
            semantic_weight=self.config.default_semantic_weight,
            graph_weight=self.config.default_graph_weight,
            splade_weight=self.config.default_splade_weight,
            confidence=1.0,
            reasoning="Using default static weights (adaptive fusion disabled)",
        )
    
    def _get_intent_weights(self, intent: QueryIntent) -> Dict[str, float]:
        """Get base weights for an intent."""
        return self.INTENT_WEIGHTS.get(intent, self.INTENT_WEIGHTS[QueryIntent.EXPLORATORY]).copy()
    
    def _apply_adjustments(
        self,
        weights: Dict[str, float],
        features: QueryFeatures,
    ) -> Dict[str, float]:
        """Apply feature-based adjustments to weights."""
        adjusted = weights.copy()
        
        # Temporal queries: boost lexical (dates, times are exact)
        if features.has_temporal_reference:
            adjusted["lexical"] *= 1.1
            adjusted["graph"] *= 1.05  # Temporal relations
        
        # Technical queries: boost lexical and SPLADE
        if features.has_technical_terms:
            adjusted["lexical"] *= 1.1
            adjusted["splade"] *= 1.15
        
        # Code references: heavy lexical bias
        if features.has_code_reference:
            adjusted["lexical"] *= 1.2
            adjusted["semantic"] *= 0.8
        
        # High keyword density: boost lexical
        if features.keyword_density > 0.7:
            adjusted["lexical"] *= 1.1
        
        # Multiple clauses: boost semantic
        if features.has_multiple_clauses:
            adjusted["semantic"] *= 1.1
        
        # High abstraction: boost semantic
        if features.abstraction_level == "high":
            adjusted["semantic"] *= 1.15
            adjusted["lexical"] *= 0.9
        
        # Low abstraction: boost lexical
        if features.abstraction_level == "low":
            adjusted["lexical"] *= 1.1
            adjusted["semantic"] *= 0.95
        
        # Short queries: boost semantic (need more interpretation)
        if features.word_count <= 3:
            adjusted["semantic"] *= 1.1
        
        # Long queries: boost lexical (more specific terms)
        if features.word_count >= 10:
            adjusted["lexical"] *= 1.05
        
        # Clamp weights to reasonable range
        for key in adjusted:
            adjusted[key] = max(0.1, min(1.5, adjusted[key]))
        
        return adjusted
    
    def _build_reasoning(self, features: QueryFeatures, weights: Dict[str, float]) -> str:
        """Build a human-readable reasoning string."""
        reasons = [f"Intent: {features.primary_intent.value} (conf: {features.intent_confidence:.2f})"]
        
        if features.has_temporal_reference:
            reasons.append(f"temporal:{features.temporal_type}")
        if features.has_technical_terms:
            reasons.append("technical")
        if features.has_code_reference:
            reasons.append("code")
        if features.has_multiple_clauses:
            reasons.append("complex")
        
        reasons.append(f"abstraction:{features.abstraction_level}")
        
        return " | ".join(reasons)


class AdaptiveRRFFusion:
    """
    Adaptive Reciprocal Rank Fusion with query-aware weights.
    
    Extends standard RRF to use dynamically predicted weights.
    """
    
    def __init__(
        self,
        config: Optional[AdaptiveFusionConfig] = None,
        rrf_k: int = 60,
    ):
        self.config = config or AdaptiveFusionConfig()
        self.predictor = AdaptiveFusionPredictor(config)
        self.rrf_k = rrf_k
    
    def fuse(
        self,
        query: str,
        lexical_results: List[SearchResult],
        semantic_results: List[SearchResult],
        graph_results: List[SearchResult],
        splade_results: Optional[List[SearchResult]] = None,
    ) -> Tuple[List[SearchResult], FusionWeights]:
        """
        Fuse results from multiple channels using adaptive weights.
        
        Returns:
            Tuple of (fused results, weights used)
        """
        # Predict optimal weights
        weights = self.predictor.predict(query)
        
        # Collect all results by chunk_id
        result_scores: Dict[str, Dict[str, Any]] = {}
        
        # Process each channel
        self._add_channel_scores(
            result_scores, lexical_results, "lexical", weights.lexical_weight
        )
        self._add_channel_scores(
            result_scores, semantic_results, "semantic", weights.semantic_weight
        )
        self._add_channel_scores(
            result_scores, graph_results, "graph", weights.graph_weight
        )
        
        if splade_results and weights.splade_weight > 0:
            self._add_channel_scores(
                result_scores, splade_results, "splade", weights.splade_weight
            )
        
        # Compute final RRF scores
        fused_results = []
        for chunk_id, data in result_scores.items():
            result = data["result"]
            rrf_score = sum(data["scores"].values())
            result.rrf_score = rrf_score
            fused_results.append(result)
        
        # Sort by RRF score
        fused_results.sort(key=lambda r: r.rrf_score or 0, reverse=True)
        
        logger.debug(
            f"Adaptive fusion: {weights.reasoning} | "
            f"weights={weights.to_dict()} | "
            f"results={len(fused_results)}"
        )
        
        return fused_results, weights
    
    def _add_channel_scores(
        self,
        result_scores: Dict[str, Dict[str, Any]],
        results: List[SearchResult],
        channel: str,
        weight: float,
    ) -> None:
        """Add RRF scores from a channel."""
        for rank, result in enumerate(results, start=1):
            chunk_id = str(result.chunk_id)
            rrf_score = weight * (1.0 / (self.rrf_k + rank))
            
            if chunk_id not in result_scores:
                result_scores[chunk_id] = {
                    "result": result,
                    "scores": {},
                }
            
            result_scores[chunk_id]["scores"][channel] = rrf_score
