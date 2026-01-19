"""
Self-RAG Implementation

Self-RAG (Self-Reflective Retrieval-Augmented Generation) that decides
when to retrieve, evaluates retrieved passages, and critiques its own outputs.

Reference:
- Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class RetrievalDecision(Enum):
    """Whether to retrieve documents."""
    YES = "yes"
    NO = "no"
    CONTINUE = "continue"

class SupportLevel(Enum):
    """How well a passage supports the response."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NO_SUPPORT = "no_support"

class RelevanceLevel(Enum):
    """Relevance of retrieved passage to query."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"

@dataclass
class SelfRAGConfig:
    """Configuration for Self-RAG."""
    
    # Retrieval decision
    retrieval_threshold: float = 0.6
    max_retrieval_rounds: int = 3
    
    # Relevance filtering
    min_relevance_score: float = 0.5
    
    # Support evaluation
    min_support_score: float = 0.5
    
    # Generation settings
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Critique settings
    enable_critique: bool = True
    critique_threshold: float = 0.7

@dataclass
class PassageEvaluation:
    """Evaluation of a retrieved passage."""
    passage_id: str
    text: str
    relevance: RelevanceLevel
    relevance_score: float
    support: Optional[SupportLevel] = None
    support_score: Optional[float] = None

@dataclass
class GenerationResult:
    """Result of self-reflective generation."""
    query: str
    response: str
    retrieved_passages: List[PassageEvaluation] = field(default_factory=list)
    retrieval_rounds: int = 0
    critique_score: float = 0.0
    needs_revision: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class SelfRAG:
    """
    Self-Reflective RAG that decides when to retrieve and evaluates outputs.
    
    Pipeline:
    1. Decide if retrieval is needed
    2. If yes, retrieve and evaluate relevance
    3. Generate response with supported passages
    4. Evaluate if response is supported
    5. Critique and optionally revise
    
    Usage:
        self_rag = SelfRAG(
            retrieve_fn=retriever.retrieve,
            generate_fn=llm.generate,
        )
        result = await self_rag.generate("What is Python?")
    """
    
    def __init__(
        self,
        retrieve_fn: Callable,
        generate_fn: Callable,
        config: Optional[SelfRAGConfig] = None,
    ):
        """Initialize Self-RAG."""
        self.retrieve = retrieve_fn
        self.generate = generate_fn
        self.config = config or SelfRAGConfig()
    
    async def __call__(self, query: str) -> GenerationResult:
        """Generate response with self-reflection."""
        return await self.generate_with_reflection(query)
    
    async def generate_with_reflection(self, query: str) -> GenerationResult:
        """
        Generate response with self-reflective retrieval.
        
        Args:
            query: User query
            
        Returns:
            GenerationResult with response and evaluations
        """
        result = GenerationResult(query=query, response="")
        
        # Step 1: Decide if retrieval is needed
        needs_retrieval = await self._should_retrieve(query)
        
        if needs_retrieval == RetrievalDecision.NO:
            # Generate without retrieval
            result.response = await self._generate_response(query, [])
            result.metadata['retrieval_decision'] = 'no'
            return result
        
        # Step 2: Iterative retrieval and generation
        passages: List[PassageEvaluation] = []
        round_num = 0
        
        while round_num < self.config.max_retrieval_rounds:
            round_num += 1
            result.retrieval_rounds = round_num
            
            # Retrieve passages
            new_passages = await self._retrieve_and_evaluate(query, passages)
            
            if not new_passages:
                break
            
            # Filter by relevance
            relevant_passages = [
                p for p in new_passages
                if p.relevance_score >= self.config.min_relevance_score
            ]
            
            if not relevant_passages:
                break
            
            passages.extend(relevant_passages)
            
            # Generate response
            response = await self._generate_response(query, passages)
            
            # Evaluate support
            passages = await self._evaluate_support(response, passages)
            
            # Check if we have sufficient support
            supported_passages = [
                p for p in passages
                if p.support_score and p.support_score >= self.config.min_support_score
            ]
            
            if len(supported_passages) >= 1:
                result.response = response
                result.retrieved_passages = passages
                break
            
            # Decide if we need more retrieval
            continue_decision = await self._should_continue_retrieval(
                query, response, passages
            )
            
            if continue_decision != RetrievalDecision.CONTINUE:
                result.response = response
                result.retrieved_passages = passages
                break
        
        # Step 3: Critique and potentially revise
        if self.config.enable_critique and result.response:
            critique = await self._critique_response(query, result.response, passages)
            result.critique_score = critique['score']
            result.needs_revision = critique['score'] < self.config.critique_threshold
            
            if result.needs_revision:
                result.response = await self._revise_response(
                    query, result.response, passages, critique['feedback']
                )
                result.metadata['revised'] = True
        
        return result
    
    async def _should_retrieve(self, query: str) -> RetrievalDecision:
        """Decide if retrieval is needed for the query."""
        prompt = f"""Analyze if this query requires retrieving external information to answer accurately.

Query: {query}

Consider:
1. Is this a factual question that needs specific data?
2. Is this about recent events or specific entities?
3. Can this be answered from general knowledge alone?

Respond with only: YES, NO, or CONTINUE
"""
        
        try:
            response = await self.generate(prompt)
            response_text = response.strip().upper()
            
            if "YES" in response_text:
                return RetrievalDecision.YES
            elif "NO" in response_text:
                return RetrievalDecision.NO
            else:
                return RetrievalDecision.CONTINUE
        except Exception as e:
            logger.warning(f"Retrieval decision failed: {e}")
            return RetrievalDecision.YES  # Default to retrieve
    
    async def _retrieve_and_evaluate(
        self,
        query: str,
        existing_passages: List[PassageEvaluation],
    ) -> List[PassageEvaluation]:
        """Retrieve passages and evaluate their relevance."""
        try:
            # Get existing IDs to avoid duplicates
            existing_ids = {p.passage_id for p in existing_passages}
            
            # Retrieve
            results = await self.retrieve(query)
            if not results:
                return []
            
            # Evaluate each passage
            evaluations = []
            for i, result in enumerate(results[:10]):  # Limit to top 10
                passage_id = getattr(result, 'doc_id', str(i))
                
                if passage_id in existing_ids:
                    continue
                
                text = getattr(result, 'text', str(result))
                
                relevance = await self._evaluate_relevance(query, text)
                
                evaluations.append(PassageEvaluation(
                    passage_id=passage_id,
                    text=text,
                    relevance=relevance['level'],
                    relevance_score=relevance['score'],
                ))
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def _evaluate_relevance(
        self,
        query: str,
        passage: str,
    ) -> Dict[str, Any]:
        """Evaluate relevance of a passage to the query."""
        prompt = f"""Rate the relevance of this passage to the query.

Query: {query}

Passage: {passage[:500]}

Rate on a scale of 0-10 where:
- 0-3: Irrelevant
- 4-6: Partially relevant
- 7-10: Highly relevant

Respond with only a number (0-10):"""

        try:
            response = await self.generate(prompt)
            
            # Parse score
            match = re.search(r'\d+', response)
            score = int(match.group()) if match else 5
            score = min(10, max(0, score))
            
            # Convert to normalized score and level
            normalized_score = score / 10.0
            
            if normalized_score >= 0.7:
                level = RelevanceLevel.RELEVANT
            elif normalized_score >= 0.4:
                level = RelevanceLevel.PARTIALLY_RELEVANT
            else:
                level = RelevanceLevel.IRRELEVANT
            
            return {'level': level, 'score': normalized_score}
            
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return {'level': RelevanceLevel.PARTIALLY_RELEVANT, 'score': 0.5}
    
    async def _generate_response(
        self,
        query: str,
        passages: List[PassageEvaluation],
    ) -> str:
        """Generate response using relevant passages."""
        if passages:
            context = "\n\n".join([
                f"[{i+1}] {p.text[:400]}"
                for i, p in enumerate(passages[:5])
            ])
            
            prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Provide a comprehensive answer based on the context. If the context doesn't contain relevant information, say so.

Answer:"""
        else:
            prompt = f"""Answer the following question based on your knowledge.

Question: {query}

Answer:"""
        
        try:
            return await self.generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I apologize, but I was unable to generate a response."
    
    async def _evaluate_support(
        self,
        response: str,
        passages: List[PassageEvaluation],
    ) -> List[PassageEvaluation]:
        """Evaluate how well passages support the response."""
        for passage in passages:
            prompt = f"""Evaluate how well this passage supports the response.

Response: {response[:400]}

Passage: {passage.text[:400]}

Rate support level:
- FULLY_SUPPORTED: The passage directly supports all claims in the response
- PARTIALLY_SUPPORTED: The passage supports some claims in the response
- NO_SUPPORT: The passage does not support the response

Respond with only: FULLY_SUPPORTED, PARTIALLY_SUPPORTED, or NO_SUPPORT"""

            try:
                result = await self.generate(prompt)
                result_upper = result.strip().upper()
                
                if "FULLY" in result_upper:
                    passage.support = SupportLevel.FULLY_SUPPORTED
                    passage.support_score = 1.0
                elif "PARTIAL" in result_upper:
                    passage.support = SupportLevel.PARTIALLY_SUPPORTED
                    passage.support_score = 0.6
                else:
                    passage.support = SupportLevel.NO_SUPPORT
                    passage.support_score = 0.0
                    
            except Exception as e:
                logger.warning(f"Support evaluation failed: {e}")
                passage.support = SupportLevel.PARTIALLY_SUPPORTED
                passage.support_score = 0.5
        
        return passages
    
    async def _should_continue_retrieval(
        self,
        query: str,
        current_response: str,
        passages: List[PassageEvaluation],
    ) -> RetrievalDecision:
        """Decide if more retrieval is needed."""
        prompt = f"""Evaluate if the current response adequately answers the query or if more information retrieval is needed.

Query: {query}

Current Response: {current_response[:400]}

Number of supporting passages: {len([p for p in passages if p.support_score and p.support_score > 0.5])}

Should we retrieve more information? Respond with: YES, NO, or CONTINUE"""

        try:
            response = await self.generate(prompt)
            response_upper = response.strip().upper()
            
            if "YES" in response_upper or "CONTINUE" in response_upper:
                return RetrievalDecision.CONTINUE
            else:
                return RetrievalDecision.NO
                
        except Exception as e:
            logger.warning(f"Continue decision failed: {e}")
            return RetrievalDecision.NO
    
    async def _critique_response(
        self,
        query: str,
        response: str,
        passages: List[PassageEvaluation],
    ) -> Dict[str, Any]:
        """Critique the generated response."""
        prompt = f"""Critique this response for quality and accuracy.

Query: {query}

Response: {response[:500]}

Evaluate on these criteria (rate each 0-10):
1. Accuracy: Are the facts correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it clear and well-written?
4. Relevance: Does it stay on topic?

Provide:
- Overall score (0-10):
- Main issues (if any):
- Suggestions for improvement:"""

        try:
            result = await self.generate(prompt)
            
            # Parse overall score
            score_match = re.search(r'Overall\s*score[:\s]*(\d+)', result, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1)) / 10.0
            else:
                # Try to find any number
                nums = re.findall(r'\d+', result[:100])
                score = int(nums[0]) / 10.0 if nums else 0.7
            
            score = min(1.0, max(0.0, score))
            
            return {
                'score': score,
                'feedback': result,
            }
            
        except Exception as e:
            logger.warning(f"Critique failed: {e}")
            return {'score': 0.7, 'feedback': ''}
    
    async def _revise_response(
        self,
        query: str,
        original_response: str,
        passages: List[PassageEvaluation],
        critique: str,
    ) -> str:
        """Revise the response based on critique."""
        context = "\n".join([
            f"[{i+1}] {p.text[:300]}"
            for i, p in enumerate(passages[:3])
            if p.support_score and p.support_score > 0.5
        ])
        
        prompt = f"""Revise this response based on the critique.

Query: {query}

Original Response: {original_response[:400]}

Critique: {critique[:300]}

Context for reference:
{context}

Provide an improved response that addresses the issues identified in the critique:"""

        try:
            return await self.generate(prompt)
        except Exception as e:
            logger.warning(f"Revision failed: {e}")
            return original_response

class AdaptiveRAG:
    """
    Adaptive RAG that learns optimal retrieval strategies.
    
    Features:
    - Learns from feedback which queries benefit from retrieval
    - Adapts retrieval depth based on query complexity
    - Tracks performance metrics to optimize decisions
    """
    
    def __init__(
        self,
        retrieve_fn: Callable,
        generate_fn: Callable,
        config: Optional[SelfRAGConfig] = None,
    ):
        """Initialize Adaptive RAG."""
        self.retrieve = retrieve_fn
        self.generate = generate_fn
        self.config = config or SelfRAGConfig()
        
        # Learning components
        self.query_stats: Dict[str, Dict] = {}
        self.category_performance: Dict[str, Dict] = {
            'factual': {'retrieval_benefit': 0.8},
            'analytical': {'retrieval_benefit': 0.6},
            'creative': {'retrieval_benefit': 0.3},
            'conversational': {'retrieval_benefit': 0.2},
        }
    
    def classify_query(self, query: str) -> str:
        """Classify query into category."""
        query_lower = query.lower()
        
        # Simple rule-based classification
        if any(w in query_lower for w in ['what is', 'who is', 'when did', 'where is']):
            return 'factual'
        elif any(w in query_lower for w in ['why', 'how does', 'explain', 'analyze']):
            return 'analytical'
        elif any(w in query_lower for w in ['create', 'write', 'generate', 'imagine']):
            return 'creative'
        else:
            return 'conversational'
    
    def should_retrieve(self, query: str) -> Tuple[bool, float]:
        """Decide if retrieval is beneficial for this query."""
        category = self.classify_query(query)
        benefit = self.category_performance.get(category, {}).get('retrieval_benefit', 0.5)
        
        return benefit >= 0.5, benefit
    
    def update_performance(
        self,
        query: str,
        used_retrieval: bool,
        quality_score: float,
    ):
        """Update performance tracking based on feedback."""
        category = self.classify_query(query)
        
        # Update category statistics
        if category not in self.category_performance:
            self.category_performance[category] = {'retrieval_benefit': 0.5}
        
        current = self.category_performance[category]['retrieval_benefit']
        
        # Adjust benefit estimate based on outcome
        if used_retrieval:
            # If retrieval was used and quality is high, increase benefit
            adjustment = (quality_score - 0.7) * 0.1
        else:
            # If no retrieval and quality is high, decrease benefit
            adjustment = -(quality_score - 0.7) * 0.1
        
        new_benefit = max(0.0, min(1.0, current + adjustment))
        self.category_performance[category]['retrieval_benefit'] = new_benefit
