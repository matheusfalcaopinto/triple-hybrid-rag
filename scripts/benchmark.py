#!/usr/bin/env python3
"""
Performance Benchmark Suite for Triple-Hybrid-RAG

Measures performance of:
- Chunking (parent/child creation rate)
- Token estimation (with/without caching)
- Embedding throughput
- Database storage rate (simulated)
- Full pipeline throughput
- NEW: HyDE generation
- NEW: Query expansion
- NEW: Multi-stage reranking simulation
- NEW: Diversity optimization

Usage:
    uv run python scripts/benchmark.py [--size small|medium|large] [--output results.json]
"""

import argparse
import asyncio
import gc
import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triple_hybrid_rag.core.chunker import HierarchicalChunker
from triple_hybrid_rag.config import RAGConfig, get_settings

@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    duration_seconds: float
    items_processed: int
    rate_per_second: float
    memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    timestamp: str
    system_info: Dict[str, Any]
    test_data_info: Dict[str, Any]
    results: List[BenchmarkResult]
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "test_data_info": self.test_data_info,
            "results": [asdict(r) for r in self.results],
            "total_duration_seconds": self.total_duration_seconds,
        }

    def print_summary(self):
        """Print a formatted summary of results."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Duration: {self.total_duration_seconds:.2f}s")
        print("-" * 70)
        print(f"{'Benchmark':<40} {'Rate':>12} {'Duration':>10} {'Memory':>8}")
        print("-" * 70)
        
        for result in self.results:
            rate_str = f"{result.rate_per_second:,.1f}/s"
            dur_str = f"{result.duration_seconds:.3f}s"
            mem_str = f"{result.memory_mb:.1f}MB" if result.memory_mb > 0 else "N/A"
            print(f"{result.name:<40} {rate_str:>12} {dur_str:>10} {mem_str:>8}")
        
        print("=" * 70)

def generate_test_text(num_lines: int, avg_words_per_line: int = 15) -> str:
    """Generate synthetic test text for benchmarking."""
    import random
    
    # Common words for realistic text
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "document", "system", "data", "information", "process", "analysis", "report",
        "performance", "optimization", "algorithm", "function", "method", "class",
        "variable", "parameter", "configuration", "database", "query", "result",
        "retrieval", "embedding", "vector", "semantic", "lexical", "graph",
    ]
    
    lines = []
    for i in range(num_lines):
        # Vary line length
        line_words = random.randint(avg_words_per_line - 5, avg_words_per_line + 10)
        line = " ".join(random.choices(words, k=line_words))
        
        # Add some structure (paragraphs, sections)
        if i % 50 == 0 and i > 0:
            lines.append("")  # Paragraph break
        if i % 200 == 0 and i > 0:
            lines.append(f"\n## Section {i // 200}\n")
        
        lines.append(line.capitalize() + ".")
    
    return "\n".join(lines)

def get_system_info() -> Dict[str, Any]:
    """Collect system information for benchmark context."""
    import platform
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

def measure_memory() -> float:
    """Get current memory usage in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024

class Benchmarker:
    """Runs performance benchmarks."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_settings()
        self.results: List[BenchmarkResult] = []
    
    def benchmark_token_estimation(self, text: str, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark token counting performance."""
        chunker = HierarchicalChunker(config=self.config)
        
        # Warm up
        for _ in range(10):
            chunker.count_tokens(text[:1000])
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        for _ in range(iterations):
            chunker.count_tokens(text[:1000])
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="token_estimation",
            duration_seconds=duration,
            items_processed=iterations,
            rate_per_second=iterations / duration,
            memory_mb=memory,
            metadata={"text_length": len(text[:1000])},
        )
    
    def benchmark_parent_chunking(self, text: str) -> BenchmarkResult:
        """Benchmark parent chunk creation."""
        chunker = HierarchicalChunker(config=self.config)
        document_id = uuid4()
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        parent_chunks = chunker.split_into_parents(text, document_id, "benchmark")
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="parent_chunking",
            duration_seconds=duration,
            items_processed=len(parent_chunks),
            rate_per_second=len(parent_chunks) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "text_length": len(text),
                "text_lines": text.count("\n"),
            },
        )
    
    def benchmark_child_chunking(self, text: str) -> BenchmarkResult:
        """Benchmark full hierarchical chunking (parent + child)."""
        chunker = HierarchicalChunker(config=self.config)
        document_id = uuid4()
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        parent_chunks, child_chunks = chunker.split_document(text, document_id, "benchmark")
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="full_chunking",
            duration_seconds=duration,
            items_processed=len(child_chunks),
            rate_per_second=len(child_chunks) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "parent_chunks": len(parent_chunks),
                "child_chunks": len(child_chunks),
                "text_length": len(text),
            },
        )
    
    def benchmark_recursive_split_stress(self, text: str) -> BenchmarkResult:
        """Stress test recursive splitting with deeply nested content."""
        chunker = HierarchicalChunker(config=self.config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        # Split into small chunks to stress the algorithm
        chunks = chunker._recursive_split(
            text,
            target_tokens=100,
            max_tokens=150,
        )
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="recursive_split_stress",
            duration_seconds=duration,
            items_processed=len(chunks),
            rate_per_second=len(chunks) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={"chunk_count": len(chunks)},
        )
    
    async def benchmark_embedding_simulation(
        self, 
        num_chunks: int,
        batch_size: int = 20,
        simulated_latency_ms: float = 50,
    ) -> BenchmarkResult:
        """
        Simulate embedding throughput (without actual API calls).
        
        This measures the overhead of batching logic, not actual embedding time.
        """
        async def simulate_embedding_batch(texts: List[str]) -> List[List[float]]:
            await asyncio.sleep(simulated_latency_ms / 1000)
            return [[0.0] * 1024 for _ in texts]
        
        # Generate fake chunk texts
        chunks = [f"This is test chunk number {i} with some content." for i in range(num_chunks)]
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        # Process in batches (simulating current sequential approach)
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = await simulate_embedding_batch(batch)
            all_embeddings.extend(embeddings)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="embedding_simulation_sequential",
            duration_seconds=duration,
            items_processed=num_chunks,
            rate_per_second=num_chunks / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "batch_size": batch_size,
                "simulated_latency_ms": simulated_latency_ms,
                "num_batches": (num_chunks + batch_size - 1) // batch_size,
            },
        )
    
    def benchmark_deduplication(self, num_chunks: int, duplicate_ratio: float = 0.2) -> BenchmarkResult:
        """Benchmark chunk deduplication."""
        import hashlib
        
        # Generate chunks with some duplicates
        unique_count = int(num_chunks * (1 - duplicate_ratio))
        chunks = []
        
        for i in range(unique_count):
            chunks.append({"text": f"Unique chunk {i} with content", "id": str(uuid4())})
        
        # Add duplicates
        import random
        for _ in range(num_chunks - unique_count):
            original = random.choice(chunks[:unique_count])
            chunks.append({"text": original["text"], "id": str(uuid4())})
        
        random.shuffle(chunks)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        # Deduplicate
        seen_hashes = set()
        unique_chunks = []
        for chunk in chunks:
            text_hash = hashlib.sha256(chunk["text"].encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="deduplication",
            duration_seconds=duration,
            items_processed=num_chunks,
            rate_per_second=num_chunks / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "input_chunks": num_chunks,
                "unique_chunks": len(unique_chunks),
                "duplicates_removed": num_chunks - len(unique_chunks),
            },
        )
    
    def benchmark_rrf_fusion(self, num_results_per_channel: int = 100) -> BenchmarkResult:
        """Benchmark RRF fusion algorithm."""
        from triple_hybrid_rag.core.fusion import RRFFusion
        from triple_hybrid_rag.types import SearchResult, SearchChannel, Modality
        
        # Generate fake results
        def make_results(channel: SearchChannel, count: int) -> List[SearchResult]:
            results = []
            for i in range(count):
                r = SearchResult(
                    chunk_id=uuid4(),
                    parent_id=uuid4(),
                    document_id=uuid4(),
                    text=f"Result {i} from {channel.value}",
                    page=1,
                    modality=Modality.TEXT,
                    source_channel=channel,
                )
                if channel == SearchChannel.SEMANTIC:
                    r.semantic_score = 1.0 - (i * 0.01)
                elif channel == SearchChannel.LEXICAL:
                    r.lexical_score = 1.0 - (i * 0.01)
                results.append(r)
            return results
        
        lexical = make_results(SearchChannel.LEXICAL, num_results_per_channel)
        semantic = make_results(SearchChannel.SEMANTIC, num_results_per_channel)
        graph = make_results(SearchChannel.GRAPH, num_results_per_channel // 2)
        
        fusion = RRFFusion(config=self.config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        # Run fusion multiple times
        iterations = 100
        for _ in range(iterations):
            fused = fusion.fuse(
                lexical_results=lexical,
                semantic_results=semantic,
                graph_results=graph,
                top_k=20,
            )
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        total_results = len(lexical) + len(semantic) + len(graph)
        
        return BenchmarkResult(
            name="rrf_fusion",
            duration_seconds=duration,
            items_processed=iterations * total_results,
            rate_per_second=(iterations * total_results) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "iterations": iterations,
                "results_per_iteration": total_results,
                "fused_output_size": len(fused) if 'fused' in dir() else 0,
            },
        )
    
    def benchmark_diversity_optimization(self, num_results: int = 100) -> BenchmarkResult:
        """Benchmark diversity/MMR optimization."""
        from triple_hybrid_rag.retrieval.diversity import DiversityOptimizer, DiversityConfig
        from triple_hybrid_rag.types import SearchResult, SearchChannel, Modality
        
        # Generate fake results with embeddings
        import random
        
        results = []
        for i in range(num_results):
            r = SearchResult(
                chunk_id=uuid4(),
                parent_id=uuid4(),
                document_id=uuid4(),
                text=f"Result {i} with some content",
                page=i % 10 + 1,
                modality=Modality.TEXT,
                source_channel=SearchChannel.SEMANTIC,
            )
            r.final_score = 1.0 - (i * 0.01)
            r.metadata["embedding"] = [random.random() for _ in range(128)]  # Small embedding
            results.append(r)
        
        config = DiversityConfig(
            enabled=True,
            mmr_lambda=0.7,
            max_per_document=3,
            max_per_page=2,
        )
        optimizer = DiversityOptimizer(config=config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        iterations = 100
        for _ in range(iterations):
            optimized = optimizer.optimize(results, top_k=20)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="diversity_optimization",
            duration_seconds=duration,
            items_processed=iterations * num_results,
            rate_per_second=(iterations * num_results) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "iterations": iterations,
                "input_results": num_results,
                "mmr_lambda": config.mmr_lambda,
            },
        )
    
    def benchmark_intent_detection(self, num_queries: int = 1000) -> BenchmarkResult:
        """Benchmark query intent detection."""
        from triple_hybrid_rag.retrieval.hyde import HyDEGenerator, HyDEConfig
        
        # Sample queries with different intents
        queries = [
            "What is the refund policy?",
            "How do I reset my password?",
            "Compare product A vs product B",
            "Who is the CEO of Acme Corporation?",
            "Explain the relationship between X and Y",
            "What are the technical specifications?",
            "When was the company founded?",
            "Where is the headquarters located?",
        ]
        
        hyde_config = HyDEConfig(enabled=False)  # Just for intent detection
        hyde = HyDEGenerator(config=self.config, hyde_config=hyde_config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        import random
        for _ in range(num_queries):
            query = random.choice(queries)
            intent = hyde.detect_intent(query)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="intent_detection",
            duration_seconds=duration,
            items_processed=num_queries,
            rate_per_second=num_queries / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={"num_queries": num_queries},
        )
    
    def benchmark_keyword_extraction(self, num_queries: int = 1000) -> BenchmarkResult:
        """Benchmark keyword extraction for query expansion."""
        from triple_hybrid_rag.retrieval.query_expansion import QueryExpander, QueryExpansionConfig
        
        queries = [
            "What is the refund policy for damaged items?",
            "How do I configure the database connection settings?",
            "Explain the machine learning algorithm implementation",
            "Compare PostgreSQL performance vs MySQL for large datasets",
            "Who are the key stakeholders in this project?",
        ]
        
        exp_config = QueryExpansionConfig(enabled=False)  # Just for keyword extraction
        expander = QueryExpander(config=self.config, expansion_config=exp_config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        import random
        for _ in range(num_queries):
            query = random.choice(queries)
            keywords = expander._extract_keywords(query)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="keyword_extraction",
            duration_seconds=duration,
            items_processed=num_queries,
            rate_per_second=num_queries / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={"num_queries": num_queries},
        )
    
    def benchmark_context_compression(self, num_chunks: int = 50) -> BenchmarkResult:
        """Benchmark context compression simulation (extractive summary)."""
        import re
        
        # Generate fake chunks
        chunks = []
        for i in range(num_chunks):
            text = f"This is chunk {i}. It contains important information about topic {i % 10}. " * 5
            chunks.append(text)
        
        def extractive_summarize(text: str, max_sentences: int = 3) -> str:
            """Simple extractive summarization."""
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return '. '.join(sentences[:max_sentences]) + '.' if sentences else text
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        # Simulate compression
        iterations = 20
        for _ in range(iterations):
            for chunk in chunks:
                compressed = extractive_summarize(chunk, max_sentences=3)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="context_compression",
            duration_seconds=duration,
            items_processed=iterations * num_chunks,
            rate_per_second=(iterations * num_chunks) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "iterations": iterations,
                "chunks_per_iteration": num_chunks,
            },
        )
    
    def run_suite(self, text: str, num_embedding_chunks: int = 1000) -> BenchmarkSuite:
        """Run the full benchmark suite."""
        suite_start = time.perf_counter()
        results = []
        
        print("Running benchmarks...")
        
        # Core chunking benchmarks
        print("  [1/12] Token estimation...")
        results.append(self.benchmark_token_estimation(text))
        
        print("  [2/12] Parent chunking...")
        results.append(self.benchmark_parent_chunking(text))
        
        print("  [3/12] Full chunking (parent + child)...")
        results.append(self.benchmark_child_chunking(text))
        
        print("  [4/12] Recursive split stress test...")
        results.append(self.benchmark_recursive_split_stress(text))
        
        # Embedding simulation
        print("  [5/12] Embedding simulation...")
        results.append(asyncio.run(self.benchmark_embedding_simulation(num_embedding_chunks)))
        
        # Deduplication
        print("  [6/12] Deduplication...")
        results.append(self.benchmark_deduplication(num_embedding_chunks))
        
        # NEW: Enhanced pipeline benchmarks
        print("  [7/12] RRF Fusion...")
        results.append(self.benchmark_rrf_fusion(100))
        
        print("  [8/12] Diversity Optimization (MMR)...")
        results.append(self.benchmark_diversity_optimization(100))
        
        print("  [9/12] Intent Detection...")
        results.append(self.benchmark_intent_detection(1000))
        
        print("  [10/12] Keyword Extraction...")
        results.append(self.benchmark_keyword_extraction(1000))
        
        print("  [11/12] Context Compression...")
        results.append(self.benchmark_context_compression(50))
        
        # Placeholder for multi-stage reranking simulation
        print("  [12/12] Multi-stage reranking simulation...")
        results.append(self._benchmark_multistage_rerank_simulation(100))
        
        suite_duration = time.perf_counter() - suite_start
        
        return BenchmarkSuite(
            timestamp=datetime.now(UTC).isoformat(),
            system_info=get_system_info(),
            test_data_info={
                "text_length": len(text),
                "text_lines": text.count("\n"),
                "num_embedding_chunks": num_embedding_chunks,
            },
            results=results,
            total_duration_seconds=suite_duration,
        )
    
    def _benchmark_multistage_rerank_simulation(self, num_results: int) -> BenchmarkResult:
        """Simulate multi-stage reranking pipeline."""
        from triple_hybrid_rag.types import SearchResult, SearchChannel, Modality
        
        # Generate fake results
        results = []
        for i in range(num_results):
            r = SearchResult(
                chunk_id=uuid4(),
                parent_id=uuid4(),
                document_id=uuid4(),
                text=f"Result {i} with some content for reranking",
                page=i % 10 + 1,
                modality=Modality.TEXT,
                source_channel=SearchChannel.SEMANTIC,
            )
            r.final_score = 1.0 - (i * 0.01)
            results.append(r)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        iterations = 100
        for _ in range(iterations):
            # Stage 1: Fast bi-encoder filter (simulated)
            stage1 = results[:50]
            
            # Stage 2: Cross-encoder scoring (simulated - just sort)
            stage2 = sorted(stage1, key=lambda x: x.final_score, reverse=True)[:30]
            
            # Stage 3: MMR diversity (simulated - just slice)
            stage3 = stage2[:20]
            
            # Stage 4: Business rules (simulated - just return)
            final = stage3[:10]
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="multistage_rerank_simulation",
            duration_seconds=duration,
            items_processed=iterations * num_results,
            rate_per_second=(iterations * num_results) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "iterations": iterations,
                "input_results": num_results,
                "stages": 4,
            },
        )

def main():
    parser = argparse.ArgumentParser(description="Run Triple-Hybrid-RAG benchmarks")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Test data size: small (1K lines), medium (10K lines), large (50K lines)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--append-to-report",
        type=str,
        default=None,
        help="Append results to markdown report file",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="baseline",
        help="Label for this benchmark run (e.g., 'baseline', 'after-chunking-opt')",
    )
    
    args = parser.parse_args()
    
    # Determine test data size
    size_map = {
        "small": (1_000, 500),
        "medium": (10_000, 2_000),
        "large": (50_000, 5_000),
    }
    num_lines, num_embed_chunks = size_map[args.size]
    
    print(f"\n{'=' * 70}")
    print(f"Triple-Hybrid-RAG Performance Benchmark (Enhanced Pipeline)")
    print(f"{'=' * 70}")
    print(f"Label: {args.label}")
    print(f"Test size: {args.size} ({num_lines:,} lines, {num_embed_chunks:,} embed chunks)")
    print(f"{'=' * 70}\n")
    
    # Generate test data
    print("Generating test data...")
    test_text = generate_test_text(num_lines)
    print(f"  Generated {len(test_text):,} characters, {test_text.count(chr(10)):,} lines\n")
    
    # Run benchmarks
    config = get_settings()
    benchmarker = Benchmarker(config=config)
    suite = benchmarker.run_suite(test_text, num_embed_chunks)
    
    # Print summary
    suite.print_summary()
    
    # Save JSON output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Append to markdown report
    if args.append_to_report:
        report_path = Path(args.append_to_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create or append to report
        is_new = not report_path.exists()
        with open(report_path, "a") as f:
            if is_new:
                f.write("# Triple-Hybrid-RAG Benchmark Results\n\n")
                f.write("Tracking performance improvements across optimization phases.\n\n")
            
            f.write(f"\n## {args.label} ({suite.timestamp})\n\n")
            f.write(f"**Test Size:** {args.size} ({num_lines:,} lines)\n\n")
            f.write("| Benchmark | Rate | Duration | Memory |\n")
            f.write("|-----------|------|----------|--------|\n")
            
            for result in suite.results:
                rate = f"{result.rate_per_second:,.1f}/s"
                dur = f"{result.duration_seconds:.3f}s"
                mem = f"{result.memory_mb:.1f}MB" if result.memory_mb > 0 else "N/A"
                f.write(f"| {result.name} | {rate} | {dur} | {mem} |\n")
            
            f.write(f"\n**Total Duration:** {suite.total_duration_seconds:.2f}s\n")
        
        print(f"Results appended to: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
