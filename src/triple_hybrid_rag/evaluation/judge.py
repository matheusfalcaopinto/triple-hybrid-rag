"""
LLM-as-Judge for Triple-Hybrid-RAG Evaluation

Uses LLMs to automatically assess retrieval quality:
- Relevance judgment: Score document-query relevance
- Answer faithfulness: Check if answers are grounded
- Pairwise comparison: Compare two retrieval results

Reference:
- Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
- Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import SearchResult

logger = logging.getLogger(__name__)

class RelevanceLevel(Enum):
    """Graded relevance levels."""
    NOT_RELEVANT = 0
    MARGINALLY_RELEVANT = 1
    RELEVANT = 2
    HIGHLY_RELEVANT = 3
    PERFECT = 4
    
    @classmethod
    def from_score(cls, score: float) -> "RelevanceLevel":
        """Convert numeric score to relevance level."""
        if score < 0.2:
            return cls.NOT_RELEVANT
        elif score < 0.4:
            return cls.MARGINALLY_RELEVANT
        elif score < 0.6:
            return cls.RELEVANT
        elif score < 0.8:
            return cls.HIGHLY_RELEVANT
        else:
            return cls.PERFECT

@dataclass
class JudgmentResult:
    """Result of LLM judgment."""
    
    query: str
    document_text: str
    relevance_score: float  # 0.0 to 1.0
    relevance_level: RelevanceLevel = RelevanceLevel.NOT_RELEVANT
    reasoning: str = ""
    confidence: float = 1.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FaithfulnessResult:
    """Result of faithfulness assessment."""
    
    answer: str
    context: str
    faithfulness_score: float  # 0.0 to 1.0
    supported_claims: int = 0
    unsupported_claims: int = 0
    reasoning: str = ""
    error: Optional[str] = None

@dataclass 
class PairwiseResult:
    """Result of pairwise comparison."""
    
    query: str
    winner: str  # "A", "B", or "tie"
    confidence: float = 1.0
    reasoning: str = ""
    error: Optional[str] = None

# Relevance judgment prompt
RELEVANCE_PROMPT = """You are an expert relevance assessor. Given a query and a document passage, 
rate the relevance of the passage to the query on a scale of 0-4:

0 - NOT RELEVANT: The passage has no connection to the query
1 - MARGINALLY RELEVANT: The passage has slight connection but doesn't address the query
2 - RELEVANT: The passage addresses the query but may be incomplete
3 - HIGHLY RELEVANT: The passage clearly addresses the query with good information
4 - PERFECT: The passage perfectly answers or addresses the query

Query: {query}

Passage: {passage}

Provide your assessment in this format:
SCORE: [0-4]
REASONING: [Your brief reasoning]"""

# Faithfulness judgment prompt
FAITHFULNESS_PROMPT = """You are an expert fact-checker. Given an answer and supporting context,
assess whether the answer is faithful to (grounded in) the provided context.

Context:
{context}

Answer:
{answer}

Analyze the answer and determine:
1. Is each claim in the answer supported by the context?
2. Are there any hallucinated claims not in the context?

Provide your assessment:
FAITHFULNESS_SCORE: [0.0-1.0] (1.0 = fully faithful)
SUPPORTED_CLAIMS: [number]
UNSUPPORTED_CLAIMS: [number]
REASONING: [brief explanation]"""

# Pairwise comparison prompt
PAIRWISE_PROMPT = """Compare two retrieval results for the given query.
Which result is more relevant and useful for answering the query?

Query: {query}

Result A:
{result_a}

Result B:
{result_b}

Choose the better result:
WINNER: [A/B/tie]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""

class LLMJudge:
    """
    LLM-based relevance judge for retrieval evaluation.
    
    Uses LLMs to automatically assess document relevance,
    answer faithfulness, and compare retrieval results.
    
    Usage:
        judge = LLMJudge()
        result = await judge.judge_relevance(query, document_text)
        print(f"Relevance: {result.relevance_level.name}")
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
    ):
        """Initialize LLM judge."""
        self.config = config or get_settings()
        self.model = model
        self.timeout = timeout
        
        self.api_key = self.config.openai_api_key
        self.api_base = self.config.openai_base_url.rstrip("/")
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def judge_relevance(
        self,
        query: str,
        document_text: str,
    ) -> JudgmentResult:
        """
        Judge the relevance of a document to a query.
        
        Args:
            query: User query
            document_text: Document passage text
            
        Returns:
            JudgmentResult with relevance assessment
        """
        prompt = RELEVANCE_PROMPT.format(
            query=query,
            passage=document_text[:2000],  # Truncate for context limits
        )
        
        try:
            response_text = await self._call_llm(prompt)
            
            # Parse response
            score = 2  # Default to relevant
            reasoning = ""
            
            for line in response_text.strip().split("\n"):
                if line.startswith("SCORE:"):
                    try:
                        score = int(line.split(":")[1].strip())
                        score = max(0, min(4, score))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            
            # Normalize to 0-1
            relevance_score = score / 4.0
            
            return JudgmentResult(
                query=query,
                document_text=document_text,
                relevance_score=relevance_score,
                relevance_level=RelevanceLevel(score),
                reasoning=reasoning,
                metadata={"model": self.model},
            )
            
        except Exception as e:
            logger.error(f"Relevance judgment failed: {e}")
            return JudgmentResult(
                query=query,
                document_text=document_text,
                relevance_score=0.0,
                error=str(e),
            )
    
    async def judge_relevance_batch(
        self,
        query: str,
        documents: List[str],
        max_concurrent: int = 5,
    ) -> List[JudgmentResult]:
        """
        Judge relevance for multiple documents.
        
        Args:
            query: User query
            documents: List of document texts
            max_concurrent: Maximum concurrent judgments
            
        Returns:
            List of JudgmentResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def judge_one(doc: str) -> JudgmentResult:
            async with semaphore:
                return await self.judge_relevance(query, doc)
        
        tasks = [judge_one(doc) for doc in documents]
        return await asyncio.gather(*tasks)
    
    async def judge_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> FaithfulnessResult:
        """
        Assess whether an answer is faithful to the provided context.
        
        Args:
            answer: Generated answer
            context: Source context used to generate answer
            
        Returns:
            FaithfulnessResult with assessment
        """
        prompt = FAITHFULNESS_PROMPT.format(
            context=context[:3000],
            answer=answer[:1000],
        )
        
        try:
            response_text = await self._call_llm(prompt)
            
            # Parse response
            faithfulness_score = 0.5
            supported = 0
            unsupported = 0
            reasoning = ""
            
            for line in response_text.strip().split("\n"):
                if line.startswith("FAITHFULNESS_SCORE:"):
                    try:
                        faithfulness_score = float(line.split(":")[1].strip())
                        faithfulness_score = max(0.0, min(1.0, faithfulness_score))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("SUPPORTED_CLAIMS:"):
                    try:
                        supported = int(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("UNSUPPORTED_CLAIMS:"):
                    try:
                        unsupported = int(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            
            return FaithfulnessResult(
                answer=answer,
                context=context,
                faithfulness_score=faithfulness_score,
                supported_claims=supported,
                unsupported_claims=unsupported,
                reasoning=reasoning,
            )
            
        except Exception as e:
            logger.error(f"Faithfulness judgment failed: {e}")
            return FaithfulnessResult(
                answer=answer,
                context=context,
                faithfulness_score=0.0,
                error=str(e),
            )
    
    async def pairwise_compare(
        self,
        query: str,
        result_a: str,
        result_b: str,
    ) -> PairwiseResult:
        """
        Compare two retrieval results head-to-head.
        
        Args:
            query: User query
            result_a: First retrieval result
            result_b: Second retrieval result
            
        Returns:
            PairwiseResult indicating winner
        """
        prompt = PAIRWISE_PROMPT.format(
            query=query,
            result_a=result_a[:1500],
            result_b=result_b[:1500],
        )
        
        try:
            response_text = await self._call_llm(prompt)
            
            # Parse response
            winner = "tie"
            confidence = 0.5
            reasoning = ""
            
            for line in response_text.strip().split("\n"):
                if line.startswith("WINNER:"):
                    w = line.split(":")[1].strip().upper()
                    if w in ["A", "B", "TIE"]:
                        winner = w.lower() if w != "TIE" else "tie"
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":")[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            
            return PairwiseResult(
                query=query,
                winner=winner,
                confidence=confidence,
                reasoning=reasoning,
            )
            
        except Exception as e:
            logger.error(f"Pairwise comparison failed: {e}")
            return PairwiseResult(
                query=query,
                winner="tie",
                error=str(e),
            )
    
    async def auto_label_results(
        self,
        query: str,
        results: List[SearchResult],
    ) -> Dict[str, float]:
        """
        Automatically generate relevance labels for search results.
        
        Useful for creating evaluation datasets without manual labeling.
        
        Args:
            query: User query
            results: List of SearchResult objects
            
        Returns:
            Dict mapping chunk_id to relevance score
        """
        judgments = await self.judge_relevance_batch(
            query,
            [r.text or "" for r in results],
        )
        
        labels = {}
        for result, judgment in zip(results, judgments):
            labels[str(result.chunk_id)] = judgment.relevance_score
        
        return labels
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response text."""
        client = await self._get_client()
        
        response = await client.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert relevance assessor. Provide concise, structured assessments.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]

class SyntheticQueryGenerator:
    """
    Generate synthetic queries for evaluation.
    
    Creates test queries from document content to enable
    automated evaluation without manual query creation.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize query generator."""
        self.config = config or get_settings()
        self.model = model
        self.api_key = self.config.openai_api_key
        self.api_base = self.config.openai_base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def generate_queries(
        self,
        document_text: str,
        num_queries: int = 3,
        query_types: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate synthetic queries for a document.
        
        Args:
            document_text: Source document text
            num_queries: Number of queries to generate
            query_types: Types of queries to generate
            
        Returns:
            List of generated queries
        """
        query_types = query_types or ["factual", "procedural", "comparative"]
        
        prompt = f"""Generate {num_queries} diverse search queries that this document passage would answer.
Include different query types: {', '.join(query_types)}.

Document passage:
{document_text[:2000]}

Generate {num_queries} queries, one per line:"""

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You generate realistic search queries."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 300,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse queries
            queries = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith(("#", "-", "*", "Query")):
                    # Remove numbering
                    import re
                    line = re.sub(r'^[\d\.\)\-]+\s*', '', line)
                    if line:
                        queries.append(line)
            
            return queries[:num_queries]
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return []
    
    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
