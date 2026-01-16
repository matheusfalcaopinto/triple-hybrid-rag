"""
Basic usage example for Triple-Hybrid-RAG.

This example demonstrates:
1. Initializing the configuration
2. Chunking a document
3. Embedding chunks
4. Performing retrieval (simulated)
"""

import asyncio
from uuid import uuid4

# Import from the library
from triple_hybrid_rag import RAGConfig, get_settings
from triple_hybrid_rag.core import HierarchicalChunker, MultimodalEmbedder, RRFFusion
from triple_hybrid_rag.types import SearchResult, SearchChannel


async def main():
    """Main example function."""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Get settings from environment variables / .env file
    config = get_settings()
    
    print("Triple-Hybrid-RAG Configuration")
    print("=" * 50)
    print(f"Database URL: {config.database_url}")
    print(f"Embedding Model: {config.rag_embed_model}")
    print(f"Embedding Dimension: {config.rag_embed_dim_store}d")
    print(f"Parent Chunk Tokens: {config.rag_parent_chunk_tokens}")
    print(f"Child Chunk Tokens: {config.rag_child_chunk_tokens}")
    print()
    print("Feature Flags:")
    print(f"  Lexical Search: {config.rag_lexical_enabled}")
    print(f"  Semantic Search: {config.rag_semantic_enabled}")
    print(f"  Graph Search: {config.rag_graph_enabled}")
    print(f"  Reranking: {config.rag_rerank_enabled}")
    print(f"  Multimodal: {config.rag_multimodal_embedding_enabled}")
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. CHUNKING
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Sample document text
    sample_text = """
    # Company Refund Policy
    
    ## Section 1: General Guidelines
    
    Our company provides a 30-day refund policy for all purchases. Customers may
    request a full refund within 30 days of purchase if the product is in its
    original condition with all packaging materials.
    
    For digital products, refunds are processed within 5-7 business days. Physical
    products require the item to be returned to our warehouse before the refund
    is processed.
    
    ## Section 2: Eligibility
    
    To be eligible for a refund, customers must:
    - Provide proof of purchase (receipt or order confirmation)
    - Return the product in unused condition
    - Include all original accessories and packaging
    
    Items purchased during sales or with promotional discounts may have different
    refund terms as specified at the time of purchase.
    
    ## Section 3: Contact Information
    
    For refund requests, contact our customer service team:
    - Email: support@company.com
    - Phone: 1-800-COMPANY
    - Hours: Monday-Friday, 9 AM - 5 PM EST
    
    John Smith is the Customer Service Manager and can be reached at
    john.smith@company.com for escalated issues.
    """
    
    # Initialize chunker
    chunker = HierarchicalChunker(config)
    
    # Create document ID
    document_id = uuid4()
    tenant_id = "example-tenant"
    
    # Split into hierarchical chunks
    parent_chunks, child_chunks = chunker.split_document(
        text=sample_text,
        document_id=document_id,
        tenant_id=tenant_id,
    )
    
    print("Chunking Results")
    print("=" * 50)
    print(f"Document ID: {document_id}")
    print(f"Parent Chunks: {len(parent_chunks)}")
    print(f"Child Chunks: {len(child_chunks)}")
    print()
    
    for i, parent in enumerate(parent_chunks):
        print(f"Parent {i+1}: {parent.token_count} tokens, {len(parent.children)} children")
        if parent.section_heading:
            print(f"  Section: {parent.section_heading}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. EMBEDDING (simulated - requires embedding server)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("Embedding (simulated)")
    print("=" * 50)
    print("In a real scenario, you would embed chunks like this:")
    print()
    print("  embedder = MultimodalEmbedder(config)")
    print("  children = await embedder.embed_chunks(child_chunks)")
    print()
    print("Each child chunk would have a 1024-dimensional embedding vector.")
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. RETRIEVAL FUSION (demonstration)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("RRF Fusion Demonstration")
    print("=" * 50)
    
    # Create sample search results (in real usage, these come from database)
    lexical_results = [
        SearchResult(
            chunk_id=child_chunks[0].id,
            parent_id=child_chunks[0].parent_id,
            document_id=document_id,
            text=child_chunks[0].text[:100] + "...",
            lexical_score=0.85,
            source_channel=SearchChannel.LEXICAL,
        ),
        SearchResult(
            chunk_id=child_chunks[1].id,
            parent_id=child_chunks[1].parent_id,
            document_id=document_id,
            text=child_chunks[1].text[:100] + "...",
            lexical_score=0.72,
            source_channel=SearchChannel.LEXICAL,
        ),
    ]
    
    semantic_results = [
        SearchResult(
            chunk_id=child_chunks[0].id,
            parent_id=child_chunks[0].parent_id,
            document_id=document_id,
            text=child_chunks[0].text[:100] + "...",
            semantic_score=0.91,
            source_channel=SearchChannel.SEMANTIC,
        ),
        SearchResult(
            chunk_id=child_chunks[2].id if len(child_chunks) > 2 else child_chunks[1].id,
            parent_id=child_chunks[2].parent_id if len(child_chunks) > 2 else child_chunks[1].parent_id,
            document_id=document_id,
            text=(child_chunks[2].text if len(child_chunks) > 2 else child_chunks[1].text)[:100] + "...",
            semantic_score=0.78,
            source_channel=SearchChannel.SEMANTIC,
        ),
    ]
    
    # Graph results (simulated - would come from PuppyGraph)
    graph_results = []
    
    # Initialize fusion
    fusion = RRFFusion(config)
    
    # Fuse results
    fused_results = fusion.fuse(
        lexical_results=lexical_results,
        semantic_results=semantic_results,
        graph_results=graph_results,
        top_k=5,
    )
    
    print(f"Input: {len(lexical_results)} lexical + {len(semantic_results)} semantic + {len(graph_results)} graph")
    print(f"Output: {len(fused_results)} fused results")
    print()
    
    for i, result in enumerate(fused_results):
        print(f"Result {i+1}:")
        print(f"  RRF Score: {result.rrf_score:.4f}")
        print(f"  Lexical: {result.lexical_score:.2f}, Semantic: {result.semantic_score:.2f}")
        print(f"  Sources: {result.metadata.get('source_channels', [])}")
        print(f"  Text: {result.text[:80]}...")
        print()
    
    print("=" * 50)
    print("Example complete!")
    print()
    print("Next steps:")
    print("  1. Start infrastructure: docker compose up -d")
    print("  2. Configure .env with your API keys")
    print("  3. Ingest documents into PostgreSQL")
    print("  4. Use the retrieval pipeline for your RAG application")


if __name__ == "__main__":
    asyncio.run(main())
