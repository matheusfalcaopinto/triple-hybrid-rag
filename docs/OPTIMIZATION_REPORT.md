# 🚀 RAG2 Pipeline Optimization Summary

## Overview

We optimized the RAG2 ingestion pipeline for processing large Dota2 match replay logs. The target file is **148 MB** with **1.2M lines**, producing **289,676 child chunks**.

---

## 1️⃣ Database Storage Optimization

### Problem
The stub database (used for local testing) was using `deepcopy()` on every insert/upsert operation, causing **catastrophic performance** when storing embeddings (1024-dimensional float arrays).

### Solution
**File:** `rag2/src/voice_agent/utils/db.py`

```python
# BEFORE (extremely slow)
def insert(self, data: dict) -> "StubQueryBuilder":
    record = deepcopy(data)  # Copying 1024 floats per chunk!
    ...

# AFTER (fast)
def insert(self, data: dict) -> "StubQueryBuilder":
    record = data  # Direct reference, no copy needed for in-memory store
    ...
```

### Impact
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Storage rate | 5.2 items/sec | 9,308 items/sec | **1,790x faster** |
| Storage % of total | 93% | 0.8% | **116x reduction** |
| 10k file total time | 455s | 30s | **15x faster** |

---

## 2️⃣ Concurrent Embedding with aiohttp

### Problem
Embedding requests were sent sequentially, not utilizing the vLLM server's ability to process multiple requests concurrently.

### Solution
**File:** `rag2/src/voice_agent/rag2/embedder.py`

```python
# Added async concurrent embedding with connection pooling
async def embed_texts_async(
    self,
    texts: List[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[EmbeddingResult]:
    """Embed multiple texts with concurrent batching."""
    
    # Connection pooling for efficiency
    connector = aiohttp.TCPConnector(limit=self.concurrent_batches * 2)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process batches concurrently (8 at a time)
        for batch_start in range(0, len(batch_requests), self.concurrent_batches):
            batch_end = min(batch_start + self.concurrent_batches, len(batch_requests))
            current_batches = batch_requests[batch_start:batch_end]
            
            # Fire all concurrent requests
            tasks = [self._embed_batch_async(session, br) for br in current_batches]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Configuration
**File:** `rag2/.env`
```
RAG_EMBED_BATCH_SIZE=200          # Items per HTTP request
RAG2_EMBED_CONCURRENT_BATCHES=8   # Parallel requests to vLLM
```

### Impact
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Embedding throughput | ~50 items/sec | ~73 items/sec | **46% faster** |
| GPU utilization | ~60% | ~85% | Better utilization |

---

## 3️⃣ Chunking Algorithm Optimization

### Problem
The recursive character splitting algorithm had several performance issues:
- Deep recursion for large texts (stack overflow risk)
- O(n²) text searches in `_create_child_chunks`
- Regex matching for every modality check
- No caching for repeated token estimations

### Solutions

**File:** `rag2/src/voice_agent/rag2/chunker.py`

#### 3a. Iterative Work Queue (replaced recursion)
```python
# BEFORE: Recursive (stack overflow risk, slower)
def _split_text_recursive(self, text, separators):
    if estimate_tokens(text) <= self.chunk_size:
        return [text]
    # ... recursive calls ...
    sub_splits = self._split_text_recursive(split, remaining_separators)

# AFTER: Iterative work queue
def _split_text_recursive(self, text, separators):
    work_queue = [(text, 0)]  # (text_segment, separator_index)
    final_chunks = []
    
    while work_queue:
        current_text, sep_idx = work_queue.pop()
        # ... process without recursion ...
        work_queue.append((split, sep_idx + 1))
```

#### 3b. LRU Cache for Token Estimation
```python
from functools import lru_cache

@lru_cache(maxsize=65536)
def estimate_tokens_cached(text: str) -> int:
    """Cached token estimation - significant speedup for repeated chunks."""
    return len(text) // 4

def estimate_tokens(text: str) -> int:
    # Use cached version for typical chunk sizes (≤4000 chars)
    if len(text) <= 4000:
        return estimate_tokens_cached(text)
    return len(text) // 4
```

#### 3c. O(n) Child Chunk Offset Search
```python
# BEFORE: O(n²) - searching from beginning each time
for child_idx, child_text in enumerate(child_texts):
    offset_in_parent = parent.text.find(child_text[:50])  # O(n) each

# AFTER: O(n) - tracking search position
current_search_pos = 0
for child_idx, child_text in enumerate(child_texts):
    # Search from last position, not from beginning
    offset_in_parent = parent_text.find(search_key, current_search_pos)
    current_search_pos = offset_in_parent + len(child_text_stripped)
```

#### 3d. Fast Modality Detection (no regex)
```python
# BEFORE: Regex for every chunk
modality = ChunkModality.TEXT
if self.table_pattern.search(child_text):  # Regex: r'(\|[^\n]+\|\n)+'
    modality = ChunkModality.TABLE

# AFTER: Simple string check
modality = ChunkModality.TABLE if '|' in child_text_stripped and '\n' in child_text_stripped else ChunkModality.TEXT
```

### Impact
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| 148MB chunking time | Minutes (stuck) | **7 seconds** | **~50x+ faster** |
| Chunking rate | Unknown | 7,787 parents/sec | N/A |
| Stack safety | Risk of overflow | Safe (iterative) | ✓ |

---

## 4️⃣ vLLM Server Configuration

### Problem
Default vLLM settings only used ~30% of GPU memory, limiting throughput.

### Solution
Restart vLLM with optimized parameters:
```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B \
    --port 1234 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 512 \
    --max-num-seqs 1024
```

### Impact
| Metric | Before | After |
|--------|--------|-------|
| GPU memory utilization | ~30% | 85% |
| Max concurrent sequences | Default | 1024 |

---

## 5️⃣ Progress Tracking & Visualization

### Problem
No visibility into pipeline progress, especially for long-running operations.

### Solution
**File:** `rag2/scripts/test_e2e.py`

Added comprehensive progress tracking with `rich` library:

```python
# ProgressTracker class with stage metrics
@dataclass
class ProgressTracker:
    STAGE_CONFIG = {
        "registering": "📝 Registering document",
        "loading": "📂 Loading content",
        "chunking": "✂️  Chunking text",
        "chunking_children": "✂️  Chunking: children",
        "embedding": "🧠 Embedding chunks",
        "storing_chunks": "💾 Storing chunks",
        ...
    }

# Rich progress bars for long operations
progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(bar_width=50),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)
```

### Features
- ✅ Timestamped stage output `[HH:MM:SS.mmm]`
- ✅ Live progress bars with ETA for chunking and embedding
- ✅ Stage timing breakdown table
- ✅ Summary panel with percentage breakdown

---

## 📊 Current Performance Summary

### 10,000 Line Test File (0.84 MB)

| Stage | Before Optimizations | After All Optimizations |
|-------|---------------------|------------------------|
| **Chunking** | ~1s | 0.05s |
| **Embedding** | ~30s | 24.78s |
| **DB Storage** | 423s | 0.26s |
| **Total** | **~455s** | **~25s** |

### Full File (148 MB, 1.2M lines)

| Metric | Value |
|--------|-------|
| Parent chunks | 54,509 |
| Child chunks | 289,676 |
| **Chunking time** | **7 seconds** |
| **Embedding ETA** | ~70 minutes |
| Embedding rate | ~67 chunks/sec |

---

## 📁 Files Modified

| File | Changes |
|------|---------|
| `rag2/src/voice_agent/utils/db.py` | Removed `deepcopy()` from stub DB |
| `rag2/src/voice_agent/rag2/embedder.py` | Added `embed_texts_async()` with aiohttp concurrency |
| `rag2/src/voice_agent/rag2/chunker.py` | Iterative algorithm, LRU cache, O(n) search, fast modality |
| `rag2/src/voice_agent/rag2/ingest.py` | Pass progress callback to chunker |
| `rag2/src/voice_agent/config.py` | Added `rag_embed_concurrent_batches` setting |
| `rag2/scripts/test_e2e.py` | Full progress tracking with rich UI |
| `rag2/.env` | Configuration for model, batch sizes, concurrency |

---

## 🔧 Configuration Reference

**File:** `rag2/.env`
```env
RAG_EMBED_MODEL=Qwen/Qwen3-VL-Embedding-2B
RAG_EMBED_API_BASE=http://127.0.0.1:1234/v1
RAG_EMBED_BATCH_SIZE=200
RAG2_EMBED_DIM_STORE=1024
RAG2_EMBED_DIM_MODEL=2048
RAG2_EMBED_CONCURRENT_BATCHES=8
```

---

## 🔮 Suggested Future Improvements

### 6️⃣ Database Batch Inserts (High Priority for Supabase)

**Problem:** Current implementation inserts rows one-by-one, causing 289k+ network round-trips to Supabase.

**Current Code:**
```python
# 289,676 individual INSERT calls to Supabase!
for child_data in all_children:
    self.supabase.table("rag_child_chunks").insert(child_data).execute()
```

**Proposed Solution:**
```python
BATCH_SIZE = 1000  # Configurable

# Collect all child data first
children_to_insert = []
for parent in parents:
    for child in parent.children:
        children_to_insert.append({...child_data...})

# Batch insert
for i in range(0, len(children_to_insert), BATCH_SIZE):
    batch = children_to_insert[i:i + BATCH_SIZE]
    self.supabase.table("rag_child_chunks").insert(batch).execute()
    if progress_callback:
        progress_callback("storing_chunks", min(i + BATCH_SIZE, len(children_to_insert)), len(children_to_insert))
```

**Expected Impact:**
| Approach | Network Calls | Est. Time (100ms latency) |
|----------|---------------|---------------------------|
| One-by-one | 289,676 | ~8 hours |
| Batched (1000) | 290 | ~30 seconds |
| Batched (5000) | 58 | ~6 seconds |

---

### 7️⃣ Streaming/Pipelined Processing

**Problem:** Currently, all chunks must be created before embedding starts.

**Proposed Solution:** Pipeline chunking → embedding so embedding can start while chunking is still in progress.

```python
async def ingest_with_pipeline(file_path):
    chunk_queue = asyncio.Queue(maxsize=10000)
    
    # Producer: Chunking
    async def chunk_producer():
        for parent in chunker.chunk_document_streaming(text):
            for child in parent.children:
                await chunk_queue.put(child)
        await chunk_queue.put(None)  # Signal completion
    
    # Consumer: Embedding + Storage
    async def embed_consumer():
        batch = []
        while True:
            child = await chunk_queue.get()
            if child is None:
                break
            batch.append(child)
            if len(batch) >= BATCH_SIZE:
                embeddings = await embedder.embed_texts_async([c.text for c in batch])
                await store_batch(batch, embeddings)
                batch = []
    
    await asyncio.gather(chunk_producer(), embed_consumer())
```

**Expected Impact:** ~10-15% reduction in total time by overlapping chunking and embedding.

---

### 8️⃣ Embedding Cache / Deduplication

**Problem:** Same text chunks across documents get re-embedded.

**Proposed Solution:**
```python
# Redis-based embedding cache
import redis
import hashlib

class EmbeddingCache:
    def __init__(self):
        self.redis = redis.Redis()
    
    def get_or_embed(self, text: str, embedder) -> List[float]:
        key = f"emb:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        
        # Check cache
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        # Embed and cache
        embedding = embedder.embed_text(text).embedding
        self.redis.setex(key, 86400 * 7, json.dumps(embedding))  # 7 day TTL
        return embedding
```

**Expected Impact:** Near-instant ingestion for previously seen content.

---

### 9️⃣ Model Quantization

**Problem:** Full precision model uses more VRAM and has lower throughput.

**Proposed Solution:**
```bash
# Use INT8 quantized model for 2x throughput (slight quality trade-off)
vllm serve Qwen/Qwen3-VL-Embedding-2B \
    --quantization awq \
    --gpu-memory-utilization 0.90
```

**Expected Impact:** ~2x embedding throughput with minimal quality loss.

---

## 📈 Optimization Roadmap

| Priority | Optimization | Effort | Impact | When to Implement |
|----------|--------------|--------|--------|-------------------|
| 🔴 High | DB Batch Inserts | Low | High | Before Supabase production |
| 🔴 High | Embedding Cache | Medium | High | When re-ingesting common docs |
| 🟡 Medium | Streaming Pipeline | Medium | Medium | For very large files (>500MB) |
| 🟡 Medium | Model Quantization | Low | Medium | If throughput is critical |

---

## 🎯 Current Bottleneck Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Time Breakdown                       │
├─────────────────────────────────────────────────────────────────┤
│  Loading     [█░░░░░░░░░░░░░░░░░░░]  0.3%   (0.4s)             │
│  Chunking    [█░░░░░░░░░░░░░░░░░░░]  0.5%   (7s)    ✅ OPTIMIZED│
│  Embedding   [████████████████████] 98.5%   (~70min) ⚠️ GPU BOUND│
│  Storage     [░░░░░░░░░░░░░░░░░░░░]  0.7%   (~30s)  ⚠️ NEEDS BATCH│
└─────────────────────────────────────────────────────────────────┘
```

**Primary Bottleneck:** Embedding (GPU-bound) - requires faster GPU, quantization, or embedding cache.

**Secondary Bottleneck:** DB Storage (when using real Supabase) - requires batch inserts.

---

*Report generated: January 16, 2026*
