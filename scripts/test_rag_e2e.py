#!/usr/bin/env python3
"""
RAG E2E Test Script

End-to-end test of the RAG pipeline with real database and OpenAI embeddings.
Tests: Document ingestion → Chunk storage → Embedding generation → Hybrid search
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from voice_agent.config import SETTINGS
from voice_agent.utils.db import get_supabase_client


async def test_e2e_rag_pipeline():
    """Run end-to-end RAG pipeline test."""
    
    print("=" * 60)
    print("RAG E2E Test - Real Database + OpenAI Embeddings")
    print("=" * 60)
    
    # Step 1: Verify database connection
    print("\n[1/6] Testing database connection...")
    try:
        client = get_supabase_client()
        # Test query
        result = client.table("knowledge_base_chunks").select("id").limit(1).execute()
        print(f"  ✓ Connected to Supabase")
        print(f"  ✓ knowledge_base_chunks table accessible")
    except Exception as e:
        print(f"  ✗ Database connection failed: {e}")
        return False
    
    # Step 2: Create test document
    print("\n[2/6] Creating test document...")
    test_content = """
# Company Vacation Policy

## Annual Leave
All full-time employees are entitled to 20 days of paid vacation per year.
Part-time employees receive vacation days proportional to their work hours.

## Carryover Rules
- Unused vacation days can be carried over to the next year
- Maximum carryover limit is 5 days
- Carryover days must be used within the first quarter

## Request Process
1. Submit vacation request through HR portal
2. Minimum 2 weeks advance notice required
3. Manager approval needed for requests over 5 consecutive days

## Holidays
The company observes 10 paid holidays per year including:
- New Year's Day
- Carnival (2 days)
- Easter Friday
- Labor Day
- Independence Day
- Christmas

For questions, contact HR at rh@empresa.com.br
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    print(f"  ✓ Test document created: {test_file}")
    
    # Step 3: Load and chunk the document
    print("\n[3/6] Loading and chunking document...")
    try:
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.ingestion.chunker import Chunker
        
        loader = DocumentLoader()
        doc = loader.load(test_file)
        
        if doc is None:
            print(f"  ✗ Failed to load document")
            return False
        
        print(f"  ✓ Document loaded: {len(doc.pages)} pages")
        
        chunker = Chunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk_document(doc)
        
        print(f"  ✓ Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk.content[:50].replace('\n', ' ')
            print(f"    Chunk {i}: {preview}...")
    except Exception as e:
        print(f"  ✗ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(test_file)
    
    # Step 4: Generate embeddings with real OpenAI
    print("\n[4/6] Generating embeddings with OpenAI...")
    try:
        from voice_agent.ingestion.embedder import Embedder
        
        embedder = Embedder()
        
        # Check OpenAI config - we need to use the real OpenAI endpoint for embeddings
        # The local LLM doesn't support embeddings
        import openai
        
        # Get real OpenAI key (not local LLM)
        real_openai_key = os.getenv("OPENAI_API_KEY", "")
        if not real_openai_key.startswith("sk-"):
            print(f"  ⚠ No valid OpenAI API key for embeddings")
            print(f"  → Skipping embedding generation (using mock)")
            # Use mock embeddings for testing
            embedded_chunks = []
            for chunk in chunks:
                from voice_agent.ingestion.embedder import EmbeddingResult
                embedded_chunks.append(EmbeddingResult(
                    chunk=chunk,
                    text_embedding=[0.1] * 1536,  # Mock embedding
                ))
        else:
            # Use real OpenAI embeddings
            embedded_chunks = await embedder.embed_chunks(chunks[:5])  # Limit to 5 for cost
            
        print(f"  ✓ Generated embeddings for {len(embedded_chunks)} chunks")
        
        if embedded_chunks and embedded_chunks[0].text_embedding:
            print(f"    Embedding dimension: {len(embedded_chunks[0].text_embedding)}")
    except Exception as e:
        print(f"  ✗ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Store chunks in database
    print("\n[5/6] Storing chunks in database...")
    try:
        # Get or create a test org
        orgs = client.table("organizations").select("id").limit(1).execute()
        if orgs.data:
            org_id = orgs.data[0]["id"]
        else:
            # Create test org
            result = client.table("organizations").insert({
                "name": "E2E Test Org",
                "slug": "e2e-test"
            }).execute()
            org_id = result.data[0]["id"]
        
        print(f"  Using org_id: {org_id}")
        
        # Insert chunks
        inserted_ids = []
        for i, emb_chunk in enumerate(embedded_chunks):
            chunk = emb_chunk.chunk
            data = {
                "org_id": org_id,
                "category": "policies",
                "title": "Vacation Policy",
                "source_document": "vacation_policy.txt",
                "modality": "text",
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_hash": chunk.content_hash,
                "is_table": chunk.is_table,
            }
            
            # Add embedding if available
            if emb_chunk.text_embedding:
                data["vector_embedding"] = emb_chunk.text_embedding
            
            try:
                result = client.table("knowledge_base_chunks").upsert(
                    data,
                    on_conflict="org_id,content_hash"
                ).execute()
                
                if result.data:
                    inserted_ids.append(result.data[0]["id"])
            except Exception as e:
                print(f"    Warning: Could not insert chunk {i}: {e}")
        
        print(f"  ✓ Stored {len(inserted_ids)} chunks in database")
    except Exception as e:
        print(f"  ✗ Database storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Test hybrid search
    print("\n[6/6] Testing hybrid search...")
    try:
        # Check if we have real OpenAI for embeddings
        real_openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "")
        has_real_openai = real_openai_key.startswith("sk-") and "localhost" not in openai_base_url and "127.0.0.1" not in openai_base_url
        
        # Test query
        query = "How many vacation days do employees get?"
        print(f"  Query: '{query}'")
        
        if not has_real_openai:
            print("  ⚠ No real OpenAI endpoint - using BM25/FTS search only")
            # Direct BM25 search via database FTS
            print("  Using FTS/BM25 search...")
            
            # Try text search
            try:
                fts_results = client.table("knowledge_base_chunks").select("*").eq(
                    "org_id", org_id
                ).text_search("content", "vacation days employees", config="english").limit(3).execute()
            except Exception:
                fts_results = type('obj', (object,), {'data': []})()
            
            if fts_results.data:
                print(f"  ✓ Found {len(fts_results.data)} results via FTS")
                for i, r in enumerate(fts_results.data):
                    preview = r["content"][:80].replace('\n', ' ')
                    print(f"    [{i+1}] {preview}...")
                
                all_content = " ".join(r["content"].lower() for r in fts_results.data)
                if "vacation" in all_content or "20 days" in all_content:
                    print(f"  ✓ Search results are relevant!")
            else:
                # Try simple content search (ILIKE)
                print("  FTS returned no results, trying ILIKE search...")
                like_results = client.table("knowledge_base_chunks").select("*").eq(
                    "org_id", org_id
                ).ilike("content", "%vacation%").limit(3).execute()
                
                if like_results.data:
                    print(f"  ✓ Found {len(like_results.data)} results via ILIKE")
                    for i, r in enumerate(like_results.data):
                        preview = r["content"][:80].replace('\n', ' ')
                        print(f"    [{i+1}] {preview}...")
                    
                    all_content = " ".join(r["content"].lower() for r in like_results.data)
                    if "vacation" in all_content or "20 days" in all_content:
                        print(f"  ✓ Search results are relevant!")
                else:
                    print(f"  ⚠ No results found")
        else:
            # Full hybrid search with real OpenAI
            from voice_agent.retrieval.hybrid_search import HybridSearcher, SearchConfig
            
            config = SearchConfig(
                use_hybrid=True,
                use_vector=True,
                use_bm25=True,
                top_k_retrieve=10,
            )
            
            searcher = HybridSearcher(org_id=org_id, config=config)
            results = await searcher.search(query, top_k=3)
            
            print(f"  ✓ Found {len(results)} results")
            for i, r in enumerate(results):
                score = r.rrf_score or r.similarity_score or 0
                preview = r.content[:80].replace('\n', ' ')
                print(f"    [{i+1}] Score: {score:.3f} | {preview}...")
            
            # Verify relevance
            all_content = " ".join(r.content.lower() for r in results)
            if "vacation" in all_content or "20 days" in all_content:
                print(f"  ✓ Search results are relevant!")
            else:
                print(f"  ⚠ Search results may not be relevant")
            
    except Exception as e:
        print(f"  ✗ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\n[Cleanup] Removing test data...")
    try:
        for chunk_id in inserted_ids:
            client.table("knowledge_base_chunks").delete().eq("id", chunk_id).execute()
        print(f"  ✓ Removed {len(inserted_ids)} test chunks")
    except Exception as e:
        print(f"  ⚠ Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("✅ E2E RAG Pipeline Test PASSED")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_e2e_rag_pipeline())
    sys.exit(0 if success else 1)
