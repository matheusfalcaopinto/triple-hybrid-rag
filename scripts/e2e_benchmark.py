#!/usr/bin/env python3
"""
End-to-End Performance Benchmark for Triple-Hybrid-RAG

Tests the full ingestion and retrieval pipeline with real API calls:
- Chunking (parent/child creation)
- Actual embedding via vLLM/Ollama API
- Embedding cache effectiveness
- Full pipeline throughput
- NEW: HyDE generation (with real LLM)
- NEW: Query expansion (with real LLM)
- NEW: Multi-stage reranking
- NEW: Diversity optimization
- NEW: Full retrieval pipeline

Requirements:
    - vLLM/Ollama running on RAG_EMBED_API_BASE (default: http://127.0.0.1:1234/v1)
    - OpenAI API key for HyDE/Query Expansion
    - Environment configured in .env

Usage:
    uv run python scripts/e2e_benchmark.py [--size small|medium|large] [--output results.json]
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

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.core.chunker import HierarchicalChunker
from triple_hybrid_rag.core.embedder import MultimodalEmbedder
from triple_hybrid_rag.core.embedding_cache import EmbeddingCache, CacheStats

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
class E2EBenchmarkSuite:
    """End-to-end benchmark results."""
    timestamp: str
    system_info: Dict[str, Any]
    config_info: Dict[str, Any]
    test_data_info: Dict[str, Any]
    results: List[BenchmarkResult]
    cache_stats: Optional[Dict[str, Any]] = None
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "config_info": self.config_info,
            "test_data_info": self.test_data_info,
            "results": [asdict(r) for r in self.results],
            "cache_stats": self.cache_stats,
            "total_duration_seconds": self.total_duration_seconds,
        }

    def print_summary(self):
        """Print a formatted summary of results."""
        print("\n" + "=" * 80)
        print("E2E BENCHMARK RESULTS (Enhanced Pipeline)")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Duration: {self.total_duration_seconds:.2f}s")
        print("-" * 80)
        print(f"{'Benchmark':<45} {'Rate':>12} {'Duration':>10} {'Memory':>8}")
        print("-" * 80)
        
        for result in self.results:
            rate_str = f"{result.rate_per_second:,.1f}/s"
            dur_str = f"{result.duration_seconds:.3f}s"
            mem_str = f"{result.memory_mb:.1f}MB" if result.memory_mb > 0 else "N/A"
            print(f"{result.name:<45} {rate_str:>12} {dur_str:>10} {mem_str:>8}")
        
        print("=" * 80)
        
        if self.cache_stats:
            print("\nCache Statistics:")
            print(f"  Hits: {self.cache_stats.get('hits', 0):,}")
            print(f"  Misses: {self.cache_stats.get('misses', 0):,}")
            print(f"  Hit Rate: {self.cache_stats.get('hit_rate', 0):.1%}")
        
        print()

def generate_test_text(num_lines: int, avg_words_per_line: int = 15) -> str:
    """Generate synthetic test text for benchmarking."""
    import random
    
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "document", "system", "data", "information", "process", "analysis", "report",
        "performance", "optimization", "algorithm", "function", "method", "class",
        "variable", "parameter", "configuration", "database", "query", "result",
        "retrieval", "embedding", "vector", "semantic", "lexical", "graph",
        "machine", "learning", "neural", "network", "transformer", "attention",
    ]
    
    lines = []
    for i in range(num_lines):
        line_words = random.randint(avg_words_per_line - 5, avg_words_per_line + 10)
        line = " ".join(random.choices(words, k=line_words))
        
        if i % 50 == 0 and i > 0:
            lines.append("")
        if i % 200 == 0 and i > 0:
            lines.append(f"\n## Section {i // 200}\n")
        
        lines.append(line.capitalize() + ".")
    
    return "\n".join(lines)

def get_system_info() -> Dict[str, Any]:
    """Collect system information."""
    import platform
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

def get_config_info(config: RAGConfig) -> Dict[str, Any]:
    """Extract relevant config for benchmark context."""
    return {
        "embed_api_base": config.rag_embed_api_base,
        "embed_model": config.rag_embed_model,
        "embed_batch_size": config.rag_embed_batch_size,
        "embed_concurrent_batches": config.rag_embed_concurrent_batches,
        "embed_dim_model": config.rag_embed_dim_model,
        "embed_dim_store": config.rag_embed_dim_store,
        "matryoshka_enabled": config.rag_matryoshka_embeddings,
        "cache_enabled": config.rag_embedding_cache_enabled,
        "cache_backend": config.rag_embedding_cache_backend,
        "pipeline_enabled": config.rag_pipeline_enabled,
        # Phase 1-6 enhancements
        "hyde_enabled": getattr(config, 'rag_hyde_enabled', False),
        "query_expansion_enabled": getattr(config, 'rag_query_expansion_enabled', False),
        "multistage_rerank_enabled": getattr(config, 'rag_multistage_rerank_enabled', False),
        "diversity_enabled": getattr(config, 'rag_diversity_enabled', False),
    }

def measure_memory() -> float:
    """Get current memory usage in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024

class E2EBenchmarker:
    """End-to-end benchmark runner with real API calls."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_settings()
        self.results: List[BenchmarkResult] = []
        self.embedder = MultimodalEmbedder(config=self.config)
        self.cache = EmbeddingCache(enabled=self.config.rag_embedding_cache_enabled)
    
    def benchmark_chunking(self, text: str) -> tuple[BenchmarkResult, List[str]]:
        """Benchmark chunking and return child texts for embedding."""
        chunker = HierarchicalChunker(config=self.config)
        document_id = uuid4()
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        parent_chunks, child_chunks = chunker.split_document(text, document_id, "benchmark")
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="1. Chunking",
            duration_seconds=duration,
            items_processed=len(child_chunks),
            rate_per_second=len(child_chunks) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "parent_chunks": len(parent_chunks),
                "child_chunks": len(child_chunks),
            },
        )
        
        child_texts = [c.text for c in child_chunks]
        return result, child_texts
    
    async def benchmark_embedding_real(
        self,
        texts: List[str],
        use_cache: bool = False,
    ) -> BenchmarkResult:
        """Benchmark actual embedding with real API calls."""
        await self.cache.clear()  # Start fresh
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        if use_cache:
            # Use cache-aware embedding
            cached, missing_indices = await self.cache.get_cached_embeddings(texts)
            
            if missing_indices:
                missing_texts = [texts[i] for i in missing_indices]
                new_embeddings = await self.embedder.embed_texts_concurrent(missing_texts)
                
                # Store in cache
                await self.cache.store_embeddings(
                    [texts[i] for i in missing_indices],
                    new_embeddings,
                )
                
                # Merge
                embeddings = EmbeddingCache.merge_embeddings(
                    cached, missing_indices, new_embeddings, len(texts)
                )
            else:
                embeddings = list(cached.values())
        else:
            # Direct embedding without cache
            embeddings = await self.embedder.embed_texts_concurrent(texts)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name=f"2. Embedding{'_cached' if use_cache else ''}",
            duration_seconds=duration,
            items_processed=len(texts),
            rate_per_second=len(texts) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "use_cache": use_cache,
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
            },
        )
    
    async def benchmark_embedding_cache_effectiveness(
        self,
        texts: List[str],
        duplicate_ratio: float = 0.3,
    ) -> BenchmarkResult:
        """
        Benchmark cache effectiveness with repeated content.
        
        Embeds texts twice to measure cache hit rate on second pass.
        """
        import random
        
        # Create texts with duplicates
        unique_texts = texts[:int(len(texts) * (1 - duplicate_ratio))]
        duplicate_texts = random.choices(unique_texts, k=int(len(texts) * duplicate_ratio))
        all_texts = unique_texts + duplicate_texts
        random.shuffle(all_texts)
        
        await self.cache.clear()
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        # First pass - all cache misses
        cached, missing_indices = await self.cache.get_cached_embeddings(all_texts)
        missing_texts = [all_texts[i] for i in missing_indices]
        new_embeddings = await self.embedder.embed_texts_concurrent(missing_texts)
        
        await self.cache.store_embeddings(
            [all_texts[i] for i in missing_indices],
            new_embeddings,
        )
        
        first_pass_stats = self.cache.stats
        first_hits = first_pass_stats.hits
        first_misses = first_pass_stats.misses
        
        # Second pass - should have cache hits
        cached2, missing_indices2 = await self.cache.get_cached_embeddings(all_texts)
        
        if missing_indices2:
            missing_texts2 = [all_texts[i] for i in missing_indices2]
            new_embeddings2 = await self.embedder.embed_texts_concurrent(missing_texts2)
            await self.cache.store_embeddings(
                [all_texts[i] for i in missing_indices2],
                new_embeddings2,
            )
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        final_stats = self.cache.stats
        
        return BenchmarkResult(
            name="3. Cache Effectiveness",
            duration_seconds=duration,
            items_processed=len(all_texts) * 2,  # Two passes
            rate_per_second=(len(all_texts) * 2) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "total_texts": len(all_texts),
                "unique_texts": len(set(all_texts)),
                "duplicate_ratio": duplicate_ratio,
                "first_pass_hits": first_hits,
                "first_pass_misses": first_misses,
                "total_hits": final_stats.hits,
                "total_misses": final_stats.misses,
                "final_hit_rate": final_stats.hit_rate,
            },
        )
    
    async def benchmark_concurrent_vs_sequential(
        self,
        texts: List[str],
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """Compare concurrent vs sequential embedding."""
        # Sequential (batch_size batches, one at a time)
        gc.collect()
        start_seq = time.perf_counter()
        
        embeddings_seq = await self.embedder.embed_texts(texts)
        
        duration_seq = time.perf_counter() - start_seq
        
        result_seq = BenchmarkResult(
            name="4a. Embedding (Sequential)",
            duration_seconds=duration_seq,
            items_processed=len(texts),
            rate_per_second=len(texts) / duration_seq if duration_seq > 0 else 0,
            metadata={"method": "sequential"},
        )
        
        # Concurrent (multiple batches in parallel)
        gc.collect()
        start_conc = time.perf_counter()
        
        embeddings_conc = await self.embedder.embed_texts_concurrent(texts)
        
        duration_conc = time.perf_counter() - start_conc
        
        result_conc = BenchmarkResult(
            name="4b. Embedding (Concurrent)",
            duration_seconds=duration_conc,
            items_processed=len(texts),
            rate_per_second=len(texts) / duration_conc if duration_conc > 0 else 0,
            metadata={
                "method": "concurrent",
                "speedup": f"{duration_seq / duration_conc:.2f}x" if duration_conc > 0 else "N/A",
            },
        )
        
        return result_seq, result_conc
    
    async def benchmark_hyde_generation(self, num_queries: int = 5) -> BenchmarkResult:
        """Benchmark HyDE generation with real LLM calls."""
        from triple_hybrid_rag.retrieval.hyde import HyDEGenerator, HyDEConfig
        
        queries = [
            "What is the refund policy for damaged items?",
            "How do I configure the database connection settings?",
            "Explain the machine learning algorithm implementation",
            "Who is the lead architect of this project?",
            "What are the performance optimization techniques?",
        ]
        
        hyde_enabled = getattr(self.config, 'rag_hyde_enabled', False)
        if not hyde_enabled or not self.config.openai_api_key:
            return BenchmarkResult(
                name="5. HyDE Generation",
                duration_seconds=0,
                items_processed=0,
                rate_per_second=0,
                metadata={"skipped": True, "reason": "HyDE disabled or no API key"},
            )
        
        hyde_config = HyDEConfig(
            enabled=True,
            model=getattr(self.config, 'rag_hyde_model', self.config.openai_model),
            num_hypotheticals=1,
        )
        hyde = HyDEGenerator(config=self.config, hyde_config=hyde_config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        results = []
        for query in queries[:num_queries]:
            try:
                result = await hyde.generate(query)
                results.append(result)
            except Exception as e:
                print(f"   [warn] HyDE failed for query: {e}")
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        successful = len([r for r in results if r.has_hypotheticals])
        
        return BenchmarkResult(
            name="5. HyDE Generation",
            duration_seconds=duration,
            items_processed=successful,
            rate_per_second=successful / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "queries_attempted": num_queries,
                "successful": successful,
                "avg_time_per_query": duration / num_queries if num_queries > 0 else 0,
            },
        )
    
    async def benchmark_query_expansion(self, num_queries: int = 5) -> BenchmarkResult:
        """Benchmark query expansion with real LLM calls."""
        from triple_hybrid_rag.retrieval.query_expansion import QueryExpander, QueryExpansionConfig
        
        queries = [
            "What is the refund policy?",
            "How to configure database settings?",
            "Machine learning implementation details",
            "Project architecture overview",
            "Performance benchmarks",
        ]
        
        query_exp_enabled = getattr(self.config, 'rag_query_expansion_enabled', False)
        if not query_exp_enabled or not self.config.openai_api_key:
            return BenchmarkResult(
                name="6. Query Expansion",
                duration_seconds=0,
                items_processed=0,
                rate_per_second=0,
                metadata={"skipped": True, "reason": "Query expansion disabled or no API key"},
            )
        
        exp_config = QueryExpansionConfig(
            enabled=True,
            model=getattr(self.config, 'rag_query_expansion_model', self.config.openai_model),
            num_query_variants=3,
        )
        expander = QueryExpander(config=self.config, expansion_config=exp_config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        results = []
        total_variants = 0
        for query in queries[:num_queries]:
            try:
                result = await expander.expand(query)
                results.append(result)
                total_variants += result.num_expansions
            except Exception as e:
                print(f"   [warn] Query expansion failed: {e}")
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="6. Query Expansion",
            duration_seconds=duration,
            items_processed=total_variants,
            rate_per_second=total_variants / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "queries_attempted": num_queries,
                "total_variants": total_variants,
                "avg_variants_per_query": total_variants / len(results) if results else 0,
            },
        )
    
    async def benchmark_rrf_fusion(self, num_results: int = 100) -> BenchmarkResult:
        """Benchmark RRF fusion with simulated results."""
        from triple_hybrid_rag.core.fusion import RRFFusion
        from triple_hybrid_rag.types import SearchResult, SearchChannel, Modality
        
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
        
        lexical = make_results(SearchChannel.LEXICAL, num_results)
        semantic = make_results(SearchChannel.SEMANTIC, num_results)
        graph = make_results(SearchChannel.GRAPH, num_results // 2)
        
        fusion = RRFFusion(config=self.config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        iterations = 50
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
        
        total_input = (len(lexical) + len(semantic) + len(graph)) * iterations
        
        return BenchmarkResult(
            name="7. RRF Fusion",
            duration_seconds=duration,
            items_processed=total_input,
            rate_per_second=total_input / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "iterations": iterations,
                "results_per_channel": num_results,
                "fused_output_size": len(fused),
            },
        )
    
    async def benchmark_diversity_optimization(self, num_results: int = 100) -> BenchmarkResult:
        """Benchmark diversity optimization."""
        from triple_hybrid_rag.retrieval.diversity import DiversityOptimizer, DiversityConfig
        from triple_hybrid_rag.types import SearchResult, SearchChannel, Modality
        import random
        
        results = []
        for i in range(num_results):
            r = SearchResult(
                chunk_id=uuid4(),
                parent_id=uuid4(),
                document_id=uuid4(),
                text=f"Result {i} content",
                page=i % 10 + 1,
                modality=Modality.TEXT,
                source_channel=SearchChannel.SEMANTIC,
            )
            r.final_score = 1.0 - (i * 0.01)
            r.metadata["embedding"] = [random.random() for _ in range(128)]
            results.append(r)
        
        div_config = DiversityConfig(
            enabled=True,
            mmr_lambda=getattr(self.config, 'rag_diversity_mmr_lambda', 0.7),
            max_per_document=3,
            max_per_page=2,
        )
        optimizer = DiversityOptimizer(config=div_config)
        
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        iterations = 50
        for _ in range(iterations):
            optimized = optimizer.optimize(results, top_k=20)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="8. Diversity Optimization",
            duration_seconds=duration,
            items_processed=iterations * num_results,
            rate_per_second=(iterations * num_results) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={
                "iterations": iterations,
                "input_results": num_results,
                "mmr_lambda": div_config.mmr_lambda,
            },
        )
    
    async def benchmark_full_pipeline(
        self, 
        text: str, 
        num_embed_chunks: int,
    ) -> BenchmarkResult:
        """Benchmark the complete pipeline: chunk + embed."""
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        
        chunker = HierarchicalChunker(config=self.config)
        _, child_chunks = chunker.split_document(text, uuid4(), "benchmark")
        pipeline_texts = [c.text for c in child_chunks[:num_embed_chunks]]
        await self.embedder.embed_texts_concurrent(pipeline_texts)
        
        duration = time.perf_counter() - start
        memory = measure_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            name="9. Full Pipeline (Chunk + Embed)",
            duration_seconds=duration,
            items_processed=len(pipeline_texts),
            rate_per_second=len(pipeline_texts) / duration if duration > 0 else 0,
            memory_mb=memory,
            metadata={"stages": "chunking + embedding"},
        )
    
    async def run_suite(self, text: str, num_embed_chunks: int = 500) -> E2EBenchmarkSuite:
        """Run the full e2e benchmark suite."""
        suite_start = time.perf_counter()
        results = []
        
        print("Running E2E benchmarks (Enhanced Pipeline)...\n")
        
        # 1. Chunking
        print("  [1/9] Chunking document...")
        chunk_result, child_texts = self.benchmark_chunking(text)
        results.append(chunk_result)
        print(f"        → {chunk_result.items_processed} chunks at {chunk_result.rate_per_second:.1f}/s")
        
        # Limit for embedding benchmarks
        texts_to_embed = child_texts[:num_embed_chunks]
        print(f"\n  Using {len(texts_to_embed)} chunks for embedding benchmarks\n")
        
        # 2. Direct embedding
        print("  [2/9] Direct embedding (no cache)...")
        embed_result = await self.benchmark_embedding_real(texts_to_embed, use_cache=False)
        results.append(embed_result)
        print(f"        → {embed_result.rate_per_second:.1f} texts/s")
        
        # 3. Cache effectiveness
        print("  [3/9] Cache effectiveness test...")
        cache_result = await self.benchmark_embedding_cache_effectiveness(
            texts_to_embed[:min(200, len(texts_to_embed))],
            duplicate_ratio=0.3,
        )
        results.append(cache_result)
        print(f"        → Hit rate: {cache_result.metadata.get('final_hit_rate', 0):.1%}")
        
        # 4. Sequential vs Concurrent
        print("  [4/9] Sequential vs Concurrent embedding...")
        result_seq, result_conc = await self.benchmark_concurrent_vs_sequential(
            texts_to_embed[:min(100, len(texts_to_embed))]
        )
        results.append(result_seq)
        results.append(result_conc)
        print(f"        → Seq: {result_seq.rate_per_second:.1f}/s, Conc: {result_conc.rate_per_second:.1f}/s")
        print(f"        → Speedup: {result_conc.metadata.get('speedup', 'N/A')}")
        
        # 5. HyDE Generation (NEW)
        print("  [5/9] HyDE Generation (real LLM)...")
        hyde_result = await self.benchmark_hyde_generation(num_queries=3)
        results.append(hyde_result)
        if hyde_result.metadata.get('skipped'):
            print(f"        → Skipped: {hyde_result.metadata.get('reason')}")
        else:
            print(f"        → {hyde_result.items_processed} hypotheticals generated")
        
        # 6. Query Expansion (NEW)
        print("  [6/9] Query Expansion (real LLM)...")
        qexp_result = await self.benchmark_query_expansion(num_queries=3)
        results.append(qexp_result)
        if qexp_result.metadata.get('skipped'):
            print(f"        → Skipped: {qexp_result.metadata.get('reason')}")
        else:
            print(f"        → {qexp_result.items_processed} query variants generated")
        
        # 7. RRF Fusion
        print("  [7/9] RRF Fusion...")
        fusion_result = await self.benchmark_rrf_fusion(100)
        results.append(fusion_result)
        print(f"        → {fusion_result.rate_per_second:.1f} results/s fused")
        
        # 8. Diversity Optimization
        print("  [8/9] Diversity Optimization (MMR)...")
        div_result = await self.benchmark_diversity_optimization(100)
        results.append(div_result)
        print(f"        → {div_result.rate_per_second:.1f} results/s optimized")
        
        # 9. Full pipeline
        print("  [9/9] Full pipeline (chunk + embed)...")
        pipeline_result = await self.benchmark_full_pipeline(text, num_embed_chunks)
        results.append(pipeline_result)
        print(f"        → {pipeline_result.items_processed} chunks in {pipeline_result.duration_seconds:.2f}s")
        
        suite_duration = time.perf_counter() - suite_start
        
        # Get final cache stats
        cache_stats = asdict(self.cache.stats)
        
        return E2EBenchmarkSuite(
            timestamp=datetime.now(UTC).isoformat(),
            system_info=get_system_info(),
            config_info=get_config_info(self.config),
            test_data_info={
                "text_length": len(text),
                "text_lines": text.count("\n"),
                "chunks_for_embedding": num_embed_chunks,
            },
            results=results,
            cache_stats=cache_stats,
            total_duration_seconds=suite_duration,
        )

async def main():
    parser = argparse.ArgumentParser(description="Run Triple-Hybrid-RAG E2E benchmarks")
    parser.add_argument(
        "--size",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Test data size",
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
        default="e2e-benchmark",
        help="Label for this benchmark run",
    )
    
    args = parser.parse_args()
    
    # Test data sizes
    size_map = {
        "tiny": (500, 100),
        "small": (2_000, 300),
        "medium": (5_000, 500),
        "large": (10_000, 1000),
    }
    num_lines, num_embed_chunks = size_map[args.size]
    
    print(f"\n{'=' * 80}")
    print(f"Triple-Hybrid-RAG E2E Performance Benchmark (Enhanced Pipeline)")
    print(f"{'=' * 80}")
    print(f"Label: {args.label}")
    print(f"Test size: {args.size} ({num_lines:,} lines, {num_embed_chunks:,} embed chunks)")
    
    # Show config
    config = get_settings()
    print(f"\nConfiguration:")
    print(f"  Embed API: {config.rag_embed_api_base}")
    print(f"  Model: {config.rag_embed_model}")
    print(f"  Batch Size: {config.rag_embed_batch_size}")
    print(f"  Concurrent Batches: {config.rag_embed_concurrent_batches}")
    print(f"  Cache: {config.rag_embedding_cache_backend if config.rag_embedding_cache_enabled else 'disabled'}")
    
    # Show Phase 1-6 settings
    print(f"\n  Phase 1-6 Enhancements:")
    print(f"  HyDE: {'✓' if getattr(config, 'rag_hyde_enabled', False) else '✗'}")
    print(f"  Query Expansion: {'✓' if getattr(config, 'rag_query_expansion_enabled', False) else '✗'}")
    print(f"  Multi-Stage Rerank: {'✓' if getattr(config, 'rag_multistage_rerank_enabled', False) else '✗'}")
    print(f"  Diversity Opt: {'✓' if getattr(config, 'rag_diversity_enabled', False) else '✗'}")
    
    print(f"{'=' * 80}\n")
    
    # Generate test data
    print("Generating test data...")
    test_text = generate_test_text(num_lines)
    print(f"  Generated {len(test_text):,} characters, {test_text.count(chr(10)):,} lines\n")
    
    # Run benchmarks
    benchmarker = E2EBenchmarker(config=config)
    
    try:
        suite = await benchmarker.run_suite(test_text, num_embed_chunks)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the embedding server is running:")
        print(f"  vllm serve <model> --port 1234")
        print(f"  or: ollama serve")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary
    suite.print_summary()
    
    # Save JSON output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"Results saved to: {output_path}")
    
    # Append to markdown report
    if args.append_to_report:
        report_path = Path(args.append_to_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        is_new = not report_path.exists()
        with open(report_path, "a") as f:
            if is_new:
                f.write("# Triple-Hybrid-RAG E2E Benchmark Results\n\n")
            
            f.write(f"\n## {args.label} ({suite.timestamp})\n\n")
            f.write(f"**Test Size:** {args.size} ({num_lines:,} lines, {num_embed_chunks:,} embed chunks)\n\n")
            f.write("**Config:**\n")
            f.write(f"- API: `{config.rag_embed_api_base}`\n")
            f.write(f"- Model: `{config.rag_embed_model}`\n")
            f.write(f"- Batch: {config.rag_embed_batch_size}, Concurrent: {config.rag_embed_concurrent_batches}\n")
            f.write(f"- HyDE: {'✓' if getattr(config, 'rag_hyde_enabled', False) else '✗'}")
            f.write(f", Query Expansion: {'✓' if getattr(config, 'rag_query_expansion_enabled', False) else '✗'}\n\n")
            
            f.write("| Benchmark | Rate | Duration | Memory |\n")
            f.write("|-----------|------|----------|--------|\n")
            
            for result in suite.results:
                rate = f"{result.rate_per_second:,.1f}/s"
                dur = f"{result.duration_seconds:.3f}s"
                mem = f"{result.memory_mb:.1f}MB" if result.memory_mb > 0 else "N/A"
                f.write(f"| {result.name} | {rate} | {dur} | {mem} |\n")
            
            f.write(f"\n**Total Duration:** {suite.total_duration_seconds:.2f}s\n")
            
            if suite.cache_stats:
                f.write(f"\n**Cache:** {suite.cache_stats.get('hits', 0)} hits, ")
                f.write(f"{suite.cache_stats.get('misses', 0)} misses, ")
                f.write(f"{suite.cache_stats.get('hit_rate', 0):.1%} hit rate\n")
        
        print(f"Results appended to: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
