# Technical Specification: Multimodal Triple-Hybrid RAG Architecture (2026)

This document serves as the final technical foundation for the implementation of a **SOTA (State-of-the-Art) Multimodal Triple-Hybrid RAG system**. The architecture integrates visual context compression, high-reasoning orchestration through **OpenAI GPT-5**, and a unified multimodal retrieval funnel.

## 1\. System Architecture Overview

The architecture is based on a **Recall-to-Precision Funnel**, designed to bridge the gap between unstructured visual data and structured relational knowledge 1, 2\. It unifies three retrieval methodologies—**Lexical (BM25), Semantic (HNSW), and Relational (Knowledge Graph)**—into a single orchestrated workflow 3, 4\. All data resides in **Supabase (PostgreSQL)**, with **PuppyGraph** serving as a Zero-ETL graph query engine over the relational tables 5-7.

## 2\. Core Components and Technologies

### A. Visual Perception Layer: DeepSeek-OCR

* **Role:** Performs **Contexts Optical Compression**, treating document images as high-density information carriers 8, 9\.  
* **Methodology:** Uses the **Gundam** tiling strategy to extract high-fidelity **structured Markdown** from complex layouts, formulas (LaTeX), and tables (HTML) 10-12. It reduces text tokens by **7–20x** while maintaining \~97% decoding precision 13-15.

### B. Reasoning & Orchestration: OpenAI GPT-5 API

* **Named Entity Recognition (NER) & Relation Extraction:** Analyzes processed Markdown to identify domain-specific entities and their interdependencies 16, 17\.  
* **Query Decomposition:** Breaks down complex, multi-part user queries into focused sub-queries for each search channel (Semantic, Lexical, and Graph-Cypher) 18-20.

### C. Retrieval & Ranking: Qwen3 Family

* **Qwen3-VL-Embedding (4B):** Generates unified multimodal vectors for text and images 21, 22\. It supports **Matryoshka Representation Learning (MRL)**, allowing 4096-dimension vectors to be truncated to 1024 dimensions for a **75% reduction in storage** with minimal precision loss 23-25.  
* **Qwen3-VL-Reranker (4B):** A cross-encoder architecture that performs deep interaction analysis between the query and candidate documents, correcting retrieval drift and reducing hallucinations by \~35% 26-29.

## 3\. Pipeline I: Ingestion (Parse-Transform-Index)

* **Multimodal Parsing:** Raw documents are processed by **DeepSeek-OCR** to generate structured Markdown, preserving visual hierarchy and spatial relationships 30-32.  
* **Hierarchical Chunking:**  
* **Parent Chunks (800–1000 tokens):** Created via **Recursive Character Splitting** to provide full narrative context 33-35.  
* **Child Chunks (200 tokens):** Derived from parents for granular vector matching, linked via parent\_id 34, 35\.  
* **Entity-Relationship Population (GPT-5):** **GPT-5** performs NER and Relation Extraction on parent chunks 17, 36\. It resolves multimodal entity ambiguities before inserting triplets into Supabase relational tables 37-39.  
* **Vector Indexing:** **Qwen3-VL-Embedding** vectorizes child chunks into **pgvector (HNSW)** 28, 40, 41\.  
* **Graph Exposure:** **PuppyGraph** maps the Supabase relational tables as a unified graph model via JSON schema, enabling **Cypher** queries without data movement 5, 6, 42\.

## 4\. Pipeline II: Retrieval (Funnel of Precision)

* **Intent Parsing & Decomposition:** **GPT-5** analyzes the query to generate:  
* **Keywords:** For BM25 lexical search.  
* **Semantic Intent:** For HNSW vector search.  
* **Cypher Logic:** For Graph traversal 18, 20, 43\.  
* **Triple Hybrid Search:**  
* **Semantic Channel:** Top-k similarity search in **pgvector** 44, 45\.  
* **Lexical Channel:** Exact-match search for technical IDs/codes using **Postgres FTS (BM25)** 46-48.  
* **Graph Channel:** **PuppyGraph** performs multi-hop traversal to retrieve contextually connected entities 49-52.  
* **Fusion (Weighted RRF):** Rankings from the three sources are combined using **Weighted Reciprocal Rank Fusion** 53, 54\. Weights prioritize the **Graph Channel (1.0)** for relational queries over **Semantic (0.8)** and **Lexical (0.7)** 55-57.  
* **Context Re-expansion:** Child IDs are swapped for **Parent Chunks** to ensure the LLM receives semantically coherent context 34, 58, 59\.  
* **High-Precision Reranking:** **Qwen3-VL-Reranker** re-orders the top 20–50 candidates to select the final 5–10 context pieces 60-62.

## 5\. Safety, Quality, and Operational Guards

### A. Refusal Policy

* **Implementation:** The middleware monitors the **Qwen3 Reranker score**.  
* **Logic:** If the maximum relevance score is **\< 0.6**, the system returns a standard "information not found" message, preventing hallucinations from weak evidence 63-66.

### B. Context Denoising (Conformal Prediction)

* **Methodology:** Applies **Context Trimming** based on statistical guarantees 67, 68\. Snippets with non-conformity scores exceeding a calibrated threshold are discarded.  
* **Result:** Reduces context noise by **2–3x**, mitigating attention dilution and the **"Lost in the Middle"** phenomenon 69-71.

### C. Local Inference (vLLM)

* **Platform:** Qwen3 models are served via **vLLM** (version 0.14.0+) in **runner="pooling"** mode for embedding/reranking 72-75.  
* **Optimization:** Uses **FP8 quantization** and **\--mm-encoder-tp-mode data** to maximize throughput while minimizing VRAM bottlenecks 76, 77\.

## 6\. Technology Stack Summary

* **Visual Pre-processing:** DeepSeek-OCR (vLLM).  
* **Core Reasoning & NER:** OpenAI GPT-5 API.  
* **Database:** Supabase (PostgreSQL \+ pgvector \+ PuppyGraph).  
* **Graph Engine:** PuppyGraph (Zero-ETL Cypher).  
* **Embedding/Reranking:** Qwen3-VL (4B) variants.  
* **Orchestration:** Python 78-80.

