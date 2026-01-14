<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Goal: Understand DeepSeek OCR’s capabilities, especially whether it encodes visual/layout information (beyond plain text), and how it compares to OCR + image embeddings for RAG pipelines.

What to find:
Official paper or tech report: model architecture, training data, objectives (text-only vs multimodal), tokenizer/encoder details, and any visual-context encoding.
Benchmarks: OCR accuracy on scanned PDFs, multilingual performance, robustness to complex layouts (tables, multi-column, forms), and comparisons to Tesseract/TrOCR/DocTr/Donut/PaddleOCR.
Visual semantics: evidence that it captures layout/visual cues (charts, diagrams, screenshots); any mention of image-text contrastive pretraining or layout-aware modules.
Embedding output: whether it exposes embeddings for retrieval (text or image), dimensionality, pooling strategy, and recommended distance metric.
Inference behavior: input formats (PDF, images), limits (resolution, page count), latency, GPU/CPU requirements, batching guidance.
Licensing and usage: commercial terms, redistribution limits, model weights availability, and API vs self-host.
Failure modes: known weak spots (low-res scans, handwriting, heavy tables), and recommended mitigation (DPI upscaling, binarization, table recognition add-ons).
Integration patterns: example pipelines using DeepSeek OCR with RAG; best practices for chunking, provenance, and combining with image encoders (CLIP/ViT) when visuals matter.
Queries to run (mix and match):
“DeepSeek OCR paper pdf”
“DeepSeek OCR model architecture visual encoding”
“DeepSeek OCR benchmarks layout tables charts”
“DeepSeek OCR embeddings retrieval vector”
“DeepSeek OCR license commercial use”
“DeepSeek OCR vs PaddleOCR vs Donut vs TrOCR”
“DeepSeek OCR multimodal encoder CLIP”
“DeepSeek OCR inference speed GPU requirements”
“DeepSeek OCR table recognition performance”
Evidence to extract:
Citations/links to paper, GitHub, model card.
Quantitative metrics: CER/WER/accuracy on public datasets (ICDAR, FUNSD, DocVQA, ChartQA if available).
Qualitative examples: screenshots of structured/complex pages and model output.
Any statement about visual feature encoding or text-only focus.
Synthesis to produce:
Is DeepSeek OCR text-only or visually aware? What proof?
Does it expose embeddings suitable for RAG, or should you pair with an image encoder?
Recommended pipeline for PDFs with mixed text/images/tables.
Trade-offs vs adding CLIP-style image embeddings.
License/usage constraints.

Now I'll create a comprehensive report addressing all your query dimensions.

## DeepSeek OCR: Architecture, Capabilities, and RAG Integration

### Executive Summary

**DeepSeek OCR is a visually-aware, layout-preserving OCR system, not text-only.** It uses optical compression (vision tokens) to reduce context while maintaining semantic understanding of document structure, charts, and diagrams. However, **it does not expose embeddings suitable for direct RAG retrieval**—instead, it outputs structured text that must be separately embedded. For RAG pipelines, it excels at reducing token overhead and preserving layout semantics, but should be paired with a dedicated embedding model (SentenceTransformers, CLIP) for retrieval. For visually complex documents, consider hybrid approaches combining OCR text embeddings with image encoders (CLIP/ViT).

***

## Architecture \& Model Design

### Two-Stage Vision-Language System[^1][^2][^3][^4]

**DeepSeek-OCR** consists of tightly coupled encoder-decoder components designed for joint vision-text understanding:

**1. DeepEncoder (~380M parameters)**[^1]

- **Visual Perception Extractor**: SAM-base (80M params) with window attention to capture local glyph details, character boundaries, and table cell topology
- **Visual Knowledge Extractor**: CLIP-large (300M params) with dense global attention for semantic understanding of page layout, spatial relationships, and contextual information
- **2-layer convolutional downsampling module**: Reduces tokens by 16× before passing to global attention (4,096 patches → 256 tokens for a 1024×1024 image)

The key insight is **compression before expensive global attention**: window-local features stay dense until downsampled, then CLIP-large operates on just 256 tokens instead of 4,096, reducing quadratic attention cost by ~250×.[^1]

**2. DeepSeek-3B-MoE-A570M Decoder**[^2]

- Mixture-of-Experts language model: 3B total parameters, ~570M activated per token
- 6 routed experts + 2 shared experts selected per inference
- Reconstructs text from compressed vision tokens using language modeling

**Training Pipeline (2-stage)**[^5][^6]

- **Stage 1**: DeepEncoder pretrained on 30M real PDF pages + 10M synthetic charts + 5M chemical formulas + 1M geometric figures across 100+ languages
- **Stage 2**: Joint encoder-decoder training on mixed image-document inputs (OCR task) + text-only inputs (language preservation)

This two-stage approach ensures OCR capability is deeply integrated, not bolted on—the model learns to encode visual text into the same representation space as text tokens.[^5]

### Multiple Resolution Modes[^3][^1]

| Mode | Resolution | Tokens | Use Case |
| :-- | :-- | :-- | :-- |
| Tiny | 512×512 | 64 | Low-resource / simple layouts |
| Small | 640×640 | 100 | Books, slides, clean documents |
| Base | 1024×1024 | 256 | Mixed text/images, standard PDFs |
| Large | 1280×1280 | 400 | Dense tables, complex layouts |
| Gundam (tiled) | 640×640 + 1024×1024 | 100n + 256 | Ultra-high resolution, documents 2480×3508+ |
| Gundam-Master | 1024×1024 + 1280×1280 | 256n + 400 | Extended dynamic, extreme detail retention |

Dynamic resolution uses **tiling** (inspired by InternVL2.0): large pages split into tiles, each processed locally, then merged into global context.[^1]

***

## Visual Semantics: Layout and Context Encoding

### YES—DeepSeek OCR Captures Visual/Layout Information[^2][^3][^1]

**Evidence of visual awareness:**

1. **CLIP-Large Global Attention** preserves:
    - Page layout and spatial topology (multi-column, headers, footers)
    - Semantic relationships between regions (captions near images, figure labels)
    - Object grounding and visual cues[^2]
2. **SAM Window Attention** captures:
    - Local glyph shapes and character boundaries
    - Table cell alignment and row/column structure
    - Text line ordering and baseline geometry
3. **"Deep Parsing" Capability**:[^2]
    - Recognizes charts, formulas, geometric figures in context
    - Not just OCR—also high-level visual interpretation
    - Output includes SMILES strings (chemistry), geometric annotations
4. **Multimodal Training Heritage**:
    - CLIP pretraining enables vision-language contrastive alignment
    - Captions and diagrams remain aligned even after aggressive compression[^2]
    - Handles mixed text-image-table content as unified multimodal input

### Empirical Layout Preservation[^7]

On OmniDocBench, DeepSeek-OCR achieves **layout accuracy 25% better than static resolution approaches**:[^7]

- Tiny mode (64 tokens): Book paragraphs 0.147 edit distance, Slides 0.116
- Base mode (256 tokens): Financial reports 0.027 edit distance (excellent table preservation)
- Gundam mode (795 tokens): Complex newspapers 0.122 edit distance

Tables align correctly; multi-column text stays in order—a common weakness of traditional OCR that DeepSeek handles natively.[^3]

***

## OCR Accuracy \& Benchmarks

### Vision-Text Compression Study (Fox Dataset, English)[^6]

| Text Token Count | Vision Tokens (64) | Vision Tokens (100) |
| :-- | :-- | :-- |
| 600-700 | 96.5% / 10.5× | 98.5% / 6.7× |
| 700-800 | 93.8% / 11.8× | 97.3% / 7.5× |
| 800-900 | 83.8% / 13.2× | 96.8% / 8.5× |
| 900-1000 | 85.9% / 15.1× | 96.8% / 9.7× |
| 1000-1100 | 79.3% / 16.5× | 91.5% / 10.6× |
| 1100-1200 | 76.4% / 17.7× | 89.8% / 11.3× |
| 1200-1300 | **59.1%** / 19.7× | 87.1% / 12.6× |

**Key insight**: At ~10× compression (100 vision tokens), accuracy stays ~97%. Beyond 12× compression, accuracy degrades steeply. Documents with 1200+ text tokens become problematic at aggressive compression.[^6]

### OmniDocBench Comparison (Edit Distance, Lower = Better)[^6]

**DeepSeek-OCR beats competitors on efficiency:**


| Model | Tokens | English Overall | Chinese Overall | Notes |
| :-- | :-- | :-- | :-- | :-- |
| **MinerU2.0** | 6,790 | 0.133 | 0.238 | High-quality but massive token overhead |
| **GOT-OCR2.0** | 256 | 0.287 | 0.411 | Baseline end-to-end model |
| **DeepSeek Tiny** | 64 | 0.386 | 0.361 | Lightweight, acceptable on simple docs |
| **DeepSeek Small** | 100 | **0.221** | **0.284** | Best balance: beats GOT at 100 vs 256 tokens |
| **DeepSeek Base** | 256 (182 active) | **0.137** | **0.240** | Highly competitive with MinerU |
| **DeepSeek Gundam** | 795 | **0.127** | **0.181** | Matches MinerU quality at ~8.5× fewer tokens |

**By document category**:[^6]

- Slides, textbooks: Perform well at 64-100 tokens
- Books, academic papers: 256-400 tokens optimal
- Financial reports, newspapers: 400-800 tokens for high accuracy
- Training distribution gaps: Out-of-distribution layouts (watermarks, unusual fonts) degrade performance


### Multilingual Performance[^8][^7]

- Trained on 100+ languages (emphasis: Chinese, English, Cyrillic, Arabic, Persian, Urdu, Latin)
- Error reduction on multilingual: 3.2% → 1.5% CER across 100 languages (vs. previous benchmark)[^7]
- Character Error Rate at 10× compression: ~1-2%

***

## Comparisons to Other OCR Systems

### DeepSeek-OCR vs PaddleOCR[^9]

| Metric | DeepSeek | PaddleOCR |
| :-- | :-- | :-- |
| **Accuracy** | 95% (high-res) | 92% (standard) |
| **Speed** | 200 pages/min | 180 pages/min |
| **Token efficiency** | 50% reduction vs naive text | Standard text extraction |
| **Layout preservation** | Excellent (25% better) | Good, dynamic resolution |
| **Resource efficiency** | Memory-optimized (compression) | CPU-friendly option available |
| **Scalability** | Requires GPU | GPU optional |

**Decision**: DeepSeek excels for detail-retention \& token optimization; PaddleOCR better for lightweight, resource-constrained, rapid deployment.[^9]

### DeepSeek-OCR vs Tesseract/TrOCR/Donut[^1]

- **Tesseract**: Template-based, fails on noise/handwriting; DeepSeek is neural, adaptive
- **TrOCR**: Older transformer-based, weaker on complex layouts; DeepSeek has better layout awareness (CLIP)
- **Donut**: Document-specific, fewer languages; DeepSeek more general, 100+ languages
- **All three**: Output verbose text tokens, don't compress; DeepSeek reduces by 10× while preserving structure

***

## Embeddings \& RAG Retrieval

### Important Limitation: No Exposed Embeddings[^10][^11]

**DeepSeek-OCR does NOT expose embeddings for direct retrieval.** Vision tokens are internal representations used within the encoder-decoder pipeline; they are not meant for downstream vector similarity search.

**What the system outputs:**

- Structured text: JSON, Markdown (with layout tags), HTML
- Not embeddings: No dimensionality spec, no pooling strategy, no distance metric

**Why this matters for RAG:**
You cannot directly compare two DeepSeek vision tokens via cosine similarity or dot product. The vision tokens are optimized for compression + decoding, not for cross-document semantic similarity.

### Recommended RAG Integration Pattern[^11][^12][^13][^10]

**Step 1: DeepSeek OCR → Structured Output**

```
PDF pages → DeepSeek-OCR (Base/Large mode) → Markdown + HTML
Preserve section headers, table structure, list nesting
```

**Step 2: Chunk Structured Output**

```
Markdown → RecursiveCharacterTextSplitter
chunk_size=1000, chunk_overlap=200
Separators: ["\n\n", "\n", " ", ""]
(Prefer sentence/paragraph boundaries over naive character splits)
```

**Step 3: Embed Text Chunks**

```
Chunk text → SentenceTransformers ("all-mpnet-base-v2" or "bge-large-en-v1.5")
OR CLIP text encoder (for visual-semantic alignment if multi-vector)
Produces dense embeddings (384–768 dimensions)
```

**Step 4: Index \& Retrieve**

```
Embeddings → Vector DB (Milvus, Weaviate, Chroma, FAISS)
Query text → same embedding model → cosine similarity search
Retrieve top-k chunks → feed to LLM
```

**Why this works:**

- DeepSeek compresses pages to preserve structure (better chunk semantics)
- Structure preservation improves embedding quality vs flat OCR text
- Separate embedding model is explicit, debuggable, replaceable
- Compatible with standard RAG frameworks (LangChain, LlamaIndex)


### Hybrid Multimodal RAG (For Visual-Heavy Documents)[^14][^11]

If your PDFs include charts, diagrams, photographs:

```
1. Original images (from PDF) → CLIP image encoder → image embeddings
2. OCR text chunks → text encoder → text embeddings
3. Hybrid retrieval:
   - Text similarity search (BM25 or dense text)
   - + Image similarity search (CLIP images)
   - Re-rank by multimodal fusion (e.g., cross-attention or learned fusion)
4. Retrieve top chunks + associated images
5. Feed both to VLM (DeepSeek-V3, GPT-4V) for reasoning
```

**Benefit**: Captures visual semantics (charts, photos) that OCR alone misses.

***

## Input/Output Specifications

### Inference Inputs[^4][^15]

- **Format**: PNG, JPEG images only (PDF must be converted to per-page images first)
- **Resolution**: 512×512 (Tiny) to 1280×1280 (Large), or tiled dynamic (Gundam)
- **Batch**: Supported via vLLM (as of Oct 2025), NOT via Transformers library
- **No PDF input**: Users must handle PDF → image conversion separately (PyMuPDF, Pillow)


### Inference Output[^15][^3]

- **Text tokens**: 5–10× the vision token count (depends on compression mode)
- **Formats**: Plain text, JSON (structured), Markdown (with headers/lists/code blocks), HTML (with semantic tags)
- **No embedding output**: Vision tokens stay internal, not exported
- **Example**: 256 vision tokens → ~2,000–5,000 text tokens (typical doc)


### Latency \& Resource Requirements[^16][^1]

**Latency (single GPU)**:

- Transformers pipeline: ~6–8 seconds per page (RTX 3060, 12GB vRAM)
- vLLM batch (L4/A40): ~0.4–0.5 seconds per image (batch of 3–15)
- Blank/sparse pages: 30 seconds (hallucination loop) — mitigated with stricter logits processors

**GPU Requirements**:

- Minimum: 12 GB vRAM (RTX 3060)
- Recommended: 24 GB+ (L4, A40, A100)
- Throughput: 200k+ pages/day on single A100-40G in production

**Inference Software**:

- `vLLM` (nightly build, Feb 2025): Best for batch; stable release pending
- Transformers: Slower, single-image at a time
- CPU: Not recommended (very slow)

***

## Failure Modes \& Mitigation

### Known Weak Spots[^17][^18][^19][^16]

| Failure Mode | Root Cause | Mitigation |
| :-- | :-- | :-- |
| **Handwriting (cursive)** | Pen angle, stroke overlap blur boundaries | Scan at 300+ DPI, light denoise, deskew ±2° |
| **Low-resolution scans (<200 DPI)** | Insufficient pixel density for fine strokes | Upscale via Real-ESRGAN to 300 DPI (improves 60%→90%) |
| **Extreme compression (>15–20×)** | Token budget forces coarse quantization | Use Base/Large mode; reserve Tiny for simple docs only |
| **Blank/sparse pages** | Weak logits processor allows hallucination loops | Set `ngram_size=8, window_size=256` to penalize repeats |
| **Dense multi-column tables** | Token limit per column cannot capture full table | Use Gundam mode (400–800 tokens); consider table-aware chunking |
| **Mixed languages without signal** | Model mis-segments on language boundary | Add `language="en,es"` metadata hint |
| **Watermarks, overlaid text** | Spatial confusion; text OCR vs watermark interference | Pre-process to remove or reduce contrast |
| **Out-of-distribution layouts** | Training on 30M PDFs; edge cases not well-represented | Fine-tune LoRA on 50–200 domain examples |

**Production error recovery**:

- Implement error chunking: Log failures, re-queue with heavier settings (Large → Gundam)
- Validation layer: Check output against schema (e.g., expect JSON keys, table row count)
- Human review queue: Route uncertain predictions for audit[^13][^16]

***

## Licensing \& Commercial Use

### MIT License[^20][^21]

- **Completely open-source**: Code, model weights publicly available on GitHub (deepseek-ai/DeepSeek-OCR)
- **Commercial use allowed**: Freely use, modify, distribute in commercial products
- **No subscription**: No API lock-in, no per-page fees
- **Self-hosted**: Deploy on-premises, private cloud, or local machine
- **Cost**: Zero software licensing; only infrastructure (GPU, storage, engineer time)


### Data Privacy[^20]

- **On-premises**: No document data leaves your organization's network
- **Compliant**: GDPR, HIPAA, CCPA friendly (air-gappable)
- **Audit-friendly**: Transparent codebase; security teams can inspect/customize logging
- **Fine-tuning**: Can retrain on proprietary data without external service calls


### Deployment Options[^21]

- **Local development**: Zero cost (just GPU hardware)
- **Cloud GPU**: Pay for compute (A100 ~\$3–5/hr); model itself free
- **Enterprise**: Typically cost-optimized vs. SaaS OCR APIs (25–50% savings at scale due to token efficiency)

***

## Integration Best Practices for RAG Pipelines

### OCR-to-RAG Workflow (Recommended)[^13]

```python
# 1. Load PDFs, convert to images
from pdf2image import convert_from_path
images = convert_from_path("document.pdf", dpi=300)

# 2. Run DeepSeek OCR (vLLM for batch)
from vllm import LLM
llm = LLM(model="deepseek-ai/deepseek-ocr-7b-latest", ...)
outputs = llm.generate(images, sampling_params)  # Batch inference
# outputs: Markdown with structure tags

# 3. Chunk structured output
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_text(outputs)

# 4. Embed chunks
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = embedder.encode(chunks)

# 5. Store in vector DB
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_texts(chunks, embedder)

# 6. Retrieve and reason
query = "What are the payment terms?"
retrieved = vectorstore.similarity_search(query, k=5)
# Feed to LLM for final answer
```


### Chunking Strategy[^13]

- **Character vs. Semantic splits**: DeepSeek output preserves structure (headers, lists); use character-level with careful boundary detection
- **Overlap**: 200–300 tokens (~20%) helps maintain cross-chunk context for tables/multi-part statements
- **Separators order**: Prioritize paragraph boundaries (`\n\n`) > line breaks (`\n`) > words to avoid fragmenting tables
- **Parent-child retrieval**: Store both full-page context + granular chunks for structure preservation


### For Complex Layouts (Tables, Multi-Column)[^13]

- **Use Large or Gundam modes** to preserve table integrity
- **Post-process markdown**: Validate table syntax; repair if OCR misspells delimiters
- **Consider dedicated table extractor**: E.g., `TableNet` or fine-tuned document parsing layer if financial/legal tables are critical
- **Chunk at semantic boundaries**: Keep tables whole; don't split rows across chunks


### Fine-Tuning for Domain Documents[^22][^16]

If your documents have domain-specific fonts, layouts, or handwriting:

```python
# Collect 50–200 example image→target text pairs
# Target: preferred Markdown/JSON format

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ocr-3b",
    load_in_4bit=True,
)
# LoRA fine-tuning: rank 8–16, 1–2 epochs, early stopping
# Augment training data: ±2° deskew, mild blur, JPEG quality jitter

# Result: ~74% relative CER improvement on domain-specific data
# (Example: handwriting CER 23% → 6%)
```


### Validation \& Fallback[^13]

```python
# Post-process OCR output
import re

def validate_ocr(text):
    # Check for repeating patterns (blank page hallucination)
    if len(set(text.split())) < 5:  # Too few unique words
        return None
    # Check for expected keys (JSON) or structure (Markdown)
    if "{" in text and "}" in text:
        try:
            json.loads(text)  # Validate JSON
        except:
            return None
    return text

# Fallback strategy
if validate_ocr(ocr_output) is None:
    # Retry with heavier compression (Large → Gundam)
    result = heavy_ocr(page)
```


***

## Text-Only vs Multimodal Trade-Offs

### When to Use DeepSeek OCR Alone[^11][^14]

- **Text-heavy PDFs** (technical manuals, reports, contracts)
- **Token efficiency is critical** (long-context reasoning, cost-sensitive)
- **Layout preservation matters** (multi-column, tables)
- **You can add embeddings separately** (flexibility to swap embedding models)


### When to Pair with Image Encoders (CLIP, ViT)[^14][^11]

- **Visual-heavy documents** (scanned books with figures, scientific papers with plots)
- **Charts, diagrams, photographs** need semantic search
- **Multimodal queries** expected ("show me figures related to...")
- **Tolerance for higher token overhead** (CLIP embeddings ~50–100 additional tokens per image region)


### Hybrid Cost-Benefit[^14]

| Approach | Token Cost | Latency | Visual Precision | Implementation |
| :-- | :-- | :-- | :-- | :-- |
| **OCR text only** | 256–800 | ~0.5s | Low (text fallback) | Simple |
| **OCR + text embedding** | 256 + 768D vector | ~2–3s total | Medium | Standard |
| **OCR + CLIP images** | 256 + N×768D vectors | ~3–5s total | High | Multimodal DB |
| **OCR + both** | 256 + 768D + N×768D | ~5–8s total | Very high | Complex but powerful |

**Recommendation**: Start with OCR + text embeddings. Add CLIP only if visual retrieval fails or queries mention figures.

***

## Synthesis \& Recommendations

### Is DeepSeek OCR Text-Only or Visually-Aware?

**Answer: Visually-aware.** Evidence:

- CLIP-Large preserves layout, spatial relationships, semantic regions
- SAM window attention captures glyph geometry and table topology
- Output structure (Markdown, JSON, HTML) reflects visual organization
- Multimodal training on 16M synthetic visuals (charts, formulas, geometry)
- "Deep parsing" recognizes diagrams, formulas beyond plain text


### Should You Use Its Embeddings for RAG?

**No—not directly.** Vision tokens are internal, unexposed, and optimized for compression, not retrieval. **Instead:**

1. Use DeepSeek OCR to extract structured text
2. Embed that text with a dedicated model (SentenceTransformers, CLIP text encoder, BGE)
3. Store embeddings in a vector DB
4. Retrieve and rank normally

This separation is actually advantageous: you can swap embedding models (CLIP, E5, multilingual models) without reprocessing documents.

### Recommended PDF-to-RAG Pipeline

**For text-dominant documents:**

```
PDF (1024×1024 DPI)
  → DeepSeek-OCR (Base mode, 256 tokens)
  → Markdown with headers/lists/code preserved
  → Semantic chunking (1000 chars, 20% overlap)
  → SentenceTransformers embeddings (768D)
  → Milvus/Chroma vector DB
  → LLM-based retrieval + reasoning
```

**Expected savings**: 10× token reduction vs. naive text OCR, better chunk coherence from layout preservation.

**For visual-heavy documents:**

```
PDF + original images
  → DeepSeek-OCR text + CLIP image embeddings
  → Hybrid multi-vector indexing (text + image embeddings)
  → Cross-modal retrieval: text queries match text or images
  → Re-rank via fusion score
  → VLM (GPT-4V, DeepSeek-V3) interprets both
```

**Expected improvement**: Captures both text semantics and visual details; higher precision on charts/diagrams.

### Trade-Offs vs. Adding Image Encoders

| Factor | OCR Text Only | + Image Encoder |
| :-- | :-- | :-- |
| **Token overhead** | 256 + embedding | 256 + N images × 768D (high) |
| **Latency** | 0.5–1s | 2–5s (image encoding cost) |
| **Visual recall** | Low (text fallback) | High (visual similarity) |
| **Cost** | Low | Medium (extra GPU time) |
| **Complexity** | Simple (single embedding model) | High (multimodal fusion) |
| **Best for** | Text-heavy, budget-conscious | Visually rich, precision-critical |

**Bottom line**: Use image encoders only if visual retrieval is a stated requirement. Otherwise, OCR text + embeddings is simpler and more cost-effective.

***

## Key Sources

- **Official paper**: "DeepSeek-OCR: Contexts Optical Compression" (arXiv 2510.18234)[^6]
- **GitHub**: https://github.com/deepseek-ai/DeepSeek-OCR (open-source, MIT license)
- **Benchmarks**: OmniDocBench (English + Chinese), Fox dataset
- **Training data**: 30M real PDFs + 10M synthetic charts + 5M formulas + 1M geometric figures across 100+ languages
- **Inference**: vLLM (batch), Transformers (single-image)
<span style="display:none">[^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88]</span>

<div align="center">⁂</div>

[^1]: https://learnopencv.com/what-makes-deepseek-ocr-so-powerful/

[^2]: https://intuitionlabs.ai/articles/deepseek-ocr-optical-compression

[^3]: https://deepseek-ocr.io

[^4]: https://skywork.ai/blog/ai-agent/deepseek-ocr-architecture-explained/

[^5]: https://macaron.im/blog/deepseek-3b-moe-open-source-ocr

[^6]: https://arxiv.org/html/2510.18234v1

[^7]: https://sparkco.ai/blog/deepseek-ocr-accuracy-benchmark-deep-dive-2025

[^8]: https://www.digitalocean.com/community/tutorials/deepseek-ocr-optical-context-compression

[^9]: https://sparkco.ai/blog/deepseek-ocr-vs-paddle-ocr-a-performance-deep-dive

[^10]: https://aiorbitlabs.com/blog/how-deepseek-ocr-will-help-in-text-extraction/

[^11]: https://zilliz.com/blog/deepseek-ocr-explained-optical-compression-for-scalable-long-context-and-rag-systems

[^12]: https://milvus.io/ai-quick-reference/what-benefits-does-deepseekocr-bring-to-rag-and-longdocument-reasoning

[^13]: https://skywork.ai/blog/ai-agent/build-ai-document-workflow-deepseek-ocr-langchain-tutorial/

[^14]: https://milvus.io/ai-quick-reference/what-problems-does-deepseekocr-solve-for-nextgen-rag-and-multimodal-systems

[^15]: https://sparkco.ai/blog/deepseek-ocr-vs-tesseract-accuracy-comparison

[^16]: https://skywork.ai/blog/llm/deepseek-ocr-for-handwriting-recognition-accuracy-test-and-tips/

[^17]: https://labelyourdata.com/articles/deepseek-ocr

[^18]: https://skywork.ai/blog/llm/common-errors-in-deepseek-ocr-and-how-to-fix-them/

[^19]: https://ai.plainenglish.io/deepseek-ocr-redefining-document-intelligence-through-contextual-optical-compression-43caab3d56ea

[^20]: https://milvus.io/ai-quick-reference/what-are-the-licensing-deployment-and-dataprivacy-considerations-for-deepseekocr

[^21]: https://skywork.ai/blog/ai-agent/evaluation/deepseek-ocr-pricing-and-plans-explained-2025/

[^22]: https://www.youtube.com/watch?v=RqXsFfmJJew

[^23]: https://ieeexplore.ieee.org/document/10673559/

[^24]: https://ijamjournal.org/ijam/publication/index.php/ijam/article/view/163

[^25]: https://ieeexplore.ieee.org/document/11311747/

[^26]: https://www.semanticscholar.org/paper/af739f4ad5c387f9df7b8fd984e9dd070411ce58

[^27]: https://dl.acm.org/doi/10.1145/3395027.3419580

[^28]: https://ieeexplore.ieee.org/document/11209189/

[^29]: https://www.ijraset.com/best-journal/seamless-crossplatform-document-conversion-application-724

[^30]: https://ijsrem.com/download/legal-lens-ai-driven-legal-document-review-and-risk-analysis/

[^31]: https://ieeexplore.ieee.org/document/11108584/

[^32]: https://arxiv.org/abs/2505.10055

[^33]: https://arxiv.org/pdf/2405.04434.pdf

[^34]: https://arxiv.org/pdf/2503.11486.pdf

[^35]: http://arxiv.org/pdf/2401.02954v1.pdf

[^36]: https://arxiv.org/pdf/2501.12948.pdf

[^37]: https://arxiv.org/pdf/2403.05525.pdf

[^38]: https://arxiv.org/pdf/2401.14196.pdf

[^39]: https://arxiv.org/pdf/2412.19437.pdf

[^40]: https://arxiv.org/pdf/2503.00624.pdf

[^41]: https://skywork.ai/blog/ai-agent/deepseek-ocr-review-2025-speed-accuracy-use-cases/

[^42]: https://skywork.ai/blog/deepseek-ocr-training-data-model-design/

[^43]: https://www.reddit.com/r/LocalLLaMA/comments/1od1yrl/deepseekocr_observations_on_compression_ratio_and/

[^44]: https://www.iweaver.ai/blog/deepseek-ocr-vision-language-model/

[^45]: https://www.bentoml.com/blog/deepseek-ocr-contexts-optical-compression-explained

[^46]: https://ieeexplore.ieee.org/document/11280585/

[^47]: https://ieeexplore.ieee.org/document/11039481/

[^48]: https://arxiv.org/abs/2506.18902

[^49]: https://dl.acm.org/doi/10.1145/3711896.3736958

[^50]: https://ieeexplore.ieee.org/document/10810945/

[^51]: https://www.nature.com/articles/s41598-025-98185-8

[^52]: https://sol.sbc.org.br/index.php/webmedia/article/view/37975

[^53]: https://ieeexplore.ieee.org/document/11142418/

[^54]: https://arxiv.org/abs/2506.21601

[^55]: https://ieeexplore.ieee.org/document/11232834/

[^56]: http://arxiv.org/pdf/2407.08275.pdf

[^57]: https://arxiv.org/pdf/2409.17383.pdf

[^58]: https://arxiv.org/pdf/2408.04887.pdf

[^59]: https://arxiv.org/pdf/2305.12517.pdf

[^60]: https://arxiv.org/pdf/2407.15462.pdf

[^61]: https://arxiv.org/pdf/2309.13335.pdf

[^62]: https://arxiv.org/pdf/2502.03688.pdf

[^63]: https://arxiv.org/html/2504.05573v1

[^64]: https://skywork.ai/blog/ai-agent/deepseek-ocr-vs-gpt-4-vision-vs-paddleocr-2025-comparison/

[^65]: https://skywork.ai/blog/deepseek-ocr-vs-google-azure-abbyy-tesseract-paddleocr-comparison-2025/

[^66]: https://www.youtube.com/watch?v=ryRhO2qBU0k

[^67]: https://www.reddit.com/r/LocalLLaMA/comments/1obn0q7/the_innovations_in_deepseek_ocr/

[^68]: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/deepseek/deepseek-ocr

[^69]: https://www.marktechpost.com/2025/11/02/comparing-the-top-6-ocr-optical-character-recognition-models-systems-in-2025/

[^70]: https://www.semanticscholar.org/paper/6e82aaa7a4fb2e062a2b41d447a4bf82ab08dc52

[^71]: https://arxiv.org/abs/2301.13081

[^72]: https://ieeexplore.ieee.org/document/10204468/

[^73]: https://www.semanticscholar.org/paper/3124a00704a299d4013719c9d6cc221e4ef7fb20

[^74]: https://www.semanticscholar.org/paper/2f5f81bc516a6d085d39479378af1fc27104f91e

[^75]: https://ieeexplore.ieee.org/document/10688200/

[^76]: https://arxiv.org/abs/2510.18795

[^77]: https://www.mdpi.com/2504-3110/9/12/767

[^78]: https://arxiv.org/abs/2505.23004

[^79]: https://www.semanticscholar.org/paper/ae2f47b64b04ddab0b5a72196ff971465c310a0f

[^80]: https://arxiv.org/html/2408.11813v1

[^81]: https://arxiv.org/pdf/2401.09417.pdf

[^82]: https://arxiv.org/html/2502.07905v1

[^83]: https://arxiv.org/html/2310.08276v3

[^84]: http://arxiv.org/pdf/2405.00260.pdf

[^85]: https://www.mdpi.com/1424-8220/21/9/2911/pdf

[^86]: https://www.datacamp.com/tutorial/deepseek-r1-rag

[^87]: https://www.reddit.com/r/MachineLearning/comments/1oedumd/deepseek_ocr_high_compression_focus_but_is_the/

[^88]: https://www.cohorte.co/blog/the-new-ocr-by-deepseek-faster-docs-fewer-tokens-happier-engineers

