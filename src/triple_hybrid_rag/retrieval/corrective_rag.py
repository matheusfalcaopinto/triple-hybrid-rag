"""
Corrective RAG Implementation

CRAG (Corrective Retrieval-Augmented Generation) evaluates retrieved documents
and takes corrective actions when retrieval quality is poor.

Reference:
- Yan et al. "Corrective Retrieval Augmented Generation"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class RetrievalQuality(Enum):
    """Assessment of retrieval quality."""
    CORRECT = "correct"  # Retrieved docs are relevant
    INCORRECT = "incorrect"  # Retrieved docs are not relevant
    AMBIGUOUS = "ambiguous"  # Uncertain relevance

class CorrectionAction(Enum):
    """Corrective action to take."""
    USE_RETRIEVED = "use_retrieved"
    WEB_SEARCH = "web_search"
    REFINE_QUERY = "refine_query"
    KNOWLEDGE_ONLY = "knowledge_only"
    HYBRID = "hybrid"

@dataclass
class CRAGConfig:
    """Configuration for Corrective RAG."""
    
    # Quality thresholds
    correct_threshold: float = 0.7
    ambiguous_threshold: float = 0.4
    
    # Correction settings
    enable_web_search: bool = False
    enable_query_refinement: bool = True
    max_refinement_attempts: int = 2
    
    # Knowledge refinement
    strip_irrelevant: bool = True
    decompose_complex: bool = True
    
    # Generation
    max_tokens: int = 500

@dataclass
class RetrievalAssessment:
    """Assessment of retrieved documents."""
    quality: RetrievalQuality
    confidence: float
    relevant_docs: List[Any] = field(default_factory=list)
    irrelevant_docs: List[Any] = field(default_factory=list)
    action: CorrectionAction = CorrectionAction.USE_RETRIEVED

@dataclass
class CRAGResult:
    """Result from Corrective RAG."""
    query: str
    response: str
    assessment: RetrievalAssessment
    used_documents: List[Any] = field(default_factory=list)
    correction_applied: bool = False
    refined_query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CorrectiveRAG:
    """
    Corrective Retrieval-Augmented Generation.
    
    Pipeline:
    1. Retrieve documents
    2. Evaluate retrieval quality
    3. Take corrective action if needed:
       - CORRECT: Use retrieved documents
       - AMBIGUOUS: Combine retrieved with refined/web search
       - INCORRECT: Refine query or use web search
    4. Generate response with corrected context
    
    Usage:
        crag = CorrectiveRAG(
            retrieve_fn=retriever.retrieve,
            generate_fn=llm.generate,
        )
        result = await crag.generate("What is Python?")
    """
    
    def __init__(
        self,
        retrieve_fn: Callable,
        generate_fn: Callable,
        web_search_fn: Optional[Callable] = None,
        config: Optional[CRAGConfig] = None,
    ):
        """Initialize Corrective RAG."""
        self.retrieve = retrieve_fn
        self.generate = generate_fn
        self.web_search = web_search_fn
        self.config = config or CRAGConfig()
    
    async def __call__(self, query: str) -> CRAGResult:
        """Generate response with corrective retrieval."""
        return await self.generate_with_correction(query)
    
    async def generate_with_correction(self, query: str) -> CRAGResult:
        """
        Generate response with corrective retrieval.
        
        Args:
            query: User query
            
        Returns:
            CRAGResult with response and correction info
        """
        result = CRAGResult(
            query=query,
            response="",
            assessment=RetrievalAssessment(
                quality=RetrievalQuality.AMBIGUOUS,
                confidence=0.0,
            ),
        )
        
        # Step 1: Initial retrieval
        documents = await self._retrieve(query)
        
        if not documents:
            # No documents retrieved - use corrective action
            result.assessment.quality = RetrievalQuality.INCORRECT
            result.assessment.action = CorrectionAction.KNOWLEDGE_ONLY
            result.response = await self._generate_response(query, [])
            return result
        
        # Step 2: Evaluate retrieval quality
        assessment = await self._evaluate_retrieval(query, documents)
        result.assessment = assessment
        
        # Step 3: Take corrective action based on quality
        if assessment.quality == RetrievalQuality.CORRECT:
            # Use retrieved documents directly
            result.used_documents = assessment.relevant_docs
            result.response = await self._generate_response(
                query, assessment.relevant_docs
            )
            
        elif assessment.quality == RetrievalQuality.INCORRECT:
            # Take corrective action
            result.correction_applied = True
            
            if self.config.enable_query_refinement:
                # Try query refinement
                refined_query, refined_docs = await self._refine_and_retrieve(query)
                result.refined_query = refined_query
                
                if refined_docs:
                    result.used_documents = refined_docs
                    result.response = await self._generate_response(
                        query, refined_docs
                    )
                else:
                    # Fall back to knowledge only
                    result.response = await self._generate_response(query, [])
            
            elif self.config.enable_web_search and self.web_search:
                # Try web search
                web_docs = await self._web_search(query)
                result.used_documents = web_docs
                result.response = await self._generate_response(query, web_docs)
            
            else:
                # Generate from knowledge only
                result.response = await self._generate_response(query, [])
        
        else:  # AMBIGUOUS
            # Hybrid approach: combine relevant retrieved with refinement
            result.correction_applied = True
            
            # Use relevant docs
            docs_to_use = assessment.relevant_docs.copy()
            
            # Try to get additional context
            if self.config.enable_query_refinement:
                refined_query, refined_docs = await self._refine_and_retrieve(query)
                result.refined_query = refined_query
                # Add any new docs
                existing_ids = {self._get_doc_id(d) for d in docs_to_use}
                for doc in refined_docs:
                    if self._get_doc_id(doc) not in existing_ids:
                        docs_to_use.append(doc)
            
            result.used_documents = docs_to_use
            result.response = await self._generate_response(query, docs_to_use)
        
        return result
    
    async def _retrieve(self, query: str) -> List[Any]:
        """Retrieve documents."""
        try:
            results = await self.retrieve(query)
            return results if results else []
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def _evaluate_retrieval(
        self,
        query: str,
        documents: List[Any],
    ) -> RetrievalAssessment:
        """Evaluate quality of retrieved documents."""
        relevant = []
        irrelevant = []
        total_score = 0.0
        
        for doc in documents:
            # Evaluate each document
            score = await self._evaluate_document(query, doc)
            total_score += score
            
            if score >= self.config.correct_threshold:
                relevant.append(doc)
            elif score < self.config.ambiguous_threshold:
                irrelevant.append(doc)
            else:
                # Ambiguous - still include as relevant
                relevant.append(doc)
        
        # Calculate overall quality
        avg_score = total_score / len(documents) if documents else 0.0
        
        if avg_score >= self.config.correct_threshold and len(relevant) >= len(documents) // 2:
            quality = RetrievalQuality.CORRECT
            action = CorrectionAction.USE_RETRIEVED
        elif avg_score < self.config.ambiguous_threshold or len(relevant) == 0:
            quality = RetrievalQuality.INCORRECT
            action = CorrectionAction.REFINE_QUERY if self.config.enable_query_refinement else CorrectionAction.KNOWLEDGE_ONLY
        else:
            quality = RetrievalQuality.AMBIGUOUS
            action = CorrectionAction.HYBRID
        
        return RetrievalAssessment(
            quality=quality,
            confidence=avg_score,
            relevant_docs=relevant,
            irrelevant_docs=irrelevant,
            action=action,
        )
    
    async def _evaluate_document(self, query: str, doc: Any) -> float:
        """Evaluate a single document's relevance."""
        text = self._get_doc_text(doc)
        
        prompt = f"""Rate how relevant this document is to the query.

Query: {query}

Document: {text[:500]}

Rate relevance from 0 to 10:
- 0-3: Not relevant
- 4-6: Partially relevant  
- 7-10: Highly relevant

Respond with only a number (0-10):"""

        try:
            response = await self.generate(prompt)
            import re
            match = re.search(r'\d+', response)
            score = int(match.group()) if match else 5
            return min(10, max(0, score)) / 10.0
        except Exception as e:
            logger.warning(f"Document evaluation failed: {e}")
            return 0.5
    
    async def _refine_and_retrieve(
        self,
        query: str,
        attempt: int = 0,
    ) -> tuple:
        """Refine query and retrieve again."""
        if attempt >= self.config.max_refinement_attempts:
            return query, []
        
        # Generate refined query
        prompt = f"""Rewrite this query to be more specific and searchable.
Keep the same intent but make it clearer and more precise.

Original query: {query}

Refined query:"""

        try:
            refined = await self.generate(prompt)
            refined = refined.strip().strip('"').strip("'")
            
            if refined and refined != query:
                # Retrieve with refined query
                docs = await self._retrieve(refined)
                
                if docs:
                    # Evaluate quality
                    assessment = await self._evaluate_retrieval(refined, docs)
                    
                    if assessment.quality in [RetrievalQuality.CORRECT, RetrievalQuality.AMBIGUOUS]:
                        return refined, assessment.relevant_docs
                    
                    # Try again
                    return await self._refine_and_retrieve(refined, attempt + 1)
            
            return query, []
            
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return query, []
    
    async def _web_search(self, query: str) -> List[Any]:
        """Perform web search for additional context."""
        if not self.web_search:
            return []
        
        try:
            results = await self.web_search(query)
            return results if results else []
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []
    
    async def _generate_response(
        self,
        query: str,
        documents: List[Any],
    ) -> str:
        """Generate response from documents."""
        if documents:
            context = "\n\n".join([
                f"[{i+1}] {self._get_doc_text(doc)[:400]}"
                for i, doc in enumerate(documents[:5])
            ])
            
            prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Provide a comprehensive answer. If the context is not sufficient, indicate what information is missing.

Answer:"""
        else:
            prompt = f"""Answer this question based on your knowledge.
Note: No relevant documents were found, so this answer is based on general knowledge.

Question: {query}

Answer:"""
        
        try:
            return await self.generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I apologize, but I was unable to generate a response."
    
    def _get_doc_text(self, doc: Any) -> str:
        """Extract text from document."""
        if hasattr(doc, 'text'):
            return doc.text
        if hasattr(doc, 'content'):
            return doc.content
        if isinstance(doc, dict):
            return str(doc.get('text', doc.get('content', '')))
        return str(doc)
    
    def _get_doc_id(self, doc: Any) -> str:
        """Extract ID from document."""
        if hasattr(doc, 'id'):
            return str(doc.id)
        if hasattr(doc, 'doc_id'):
            return str(doc.doc_id)
        if isinstance(doc, dict):
            return str(doc.get('id', id(doc)))
        return str(id(doc))

class KnowledgeRefiner:
    """
    Refines retrieved knowledge before use.
    
    Operations:
    - Strip irrelevant information
    - Decompose into atomic facts
    - Verify against query
    """
    
    def __init__(self, generate_fn: Callable):
        """Initialize knowledge refiner."""
        self.generate = generate_fn
    
    async def refine(
        self,
        query: str,
        documents: List[Any],
    ) -> List[str]:
        """
        Refine documents to extract relevant knowledge.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            List of refined knowledge statements
        """
        refined_knowledge = []
        
        for doc in documents:
            text = self._get_text(doc)
            
            # Extract relevant parts
            prompt = f"""Extract only the information from this text that is relevant to the query.
Output as a list of key facts, one per line.

Query: {query}

Text: {text[:600]}

Relevant facts:"""

            try:
                response = await self.generate(prompt)
                
                # Parse facts
                facts = [
                    line.strip().lstrip('-').lstrip('•').strip()
                    for line in response.split('\n')
                    if line.strip() and len(line.strip()) > 10
                ]
                
                refined_knowledge.extend(facts)
                
            except Exception as e:
                logger.warning(f"Knowledge refinement failed: {e}")
        
        return refined_knowledge
    
    async def decompose(self, text: str) -> List[str]:
        """Decompose text into atomic facts."""
        prompt = f"""Decompose this text into atomic facts (single, independent statements).
Output one fact per line.

Text: {text[:500]}

Atomic facts:"""

        try:
            response = await self.generate(prompt)
            return [
                line.strip().lstrip('-').strip()
                for line in response.split('\n')
                if line.strip() and len(line.strip()) > 10
            ]
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
            return [text]
    
    def _get_text(self, doc: Any) -> str:
        if hasattr(doc, 'text'):
            return doc.text
        if isinstance(doc, dict):
            return str(doc.get('text', ''))
        return str(doc)
