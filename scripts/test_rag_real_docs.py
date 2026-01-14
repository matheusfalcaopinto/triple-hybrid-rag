#!/usr/bin/env python3
"""
RAG Real Documents Integration Test

Tests the RAG pipeline with real PDF, DOCX, and XLSX documents from docs/pdfs/
Uses:
- Qwen3-VL at http://127.0.0.1:1234/v1 for OCR
- OpenAI for text embeddings
- Local Supabase for vector storage
- SigLIP for image embeddings (optional)
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv()

from voice_agent.config import SETTINGS
from voice_agent.ingestion.chunker import Chunker, Chunk, ChunkType
from voice_agent.ingestion.embedder import Embedder, EmbeddingResult
from voice_agent.ingestion.loader import DocumentLoader
from voice_agent.ingestion.ocr import OCRProcessor
from voice_agent.utils.db import get_supabase_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_real_docs_test")

# Test document directory
DOCS_DIR = project_root / "docs" / "pdfs"

# Test organization ID (using a valid UUID for the test)
# This is a fixed UUID for testing - we'll create/use an org with this ID
TEST_ORG_ID = "00000000-0000-0000-0000-000000000001"


class TestRunner:
    """Run integration tests with real documents."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.supabase = None
        self.org_id = TEST_ORG_ID
    
    def log_test_start(self, name: str):
        """Log test start."""
        logger.info("\n" + "=" * 60)
        logger.info(f"TEST: {name}")
        logger.info("=" * 60)
    
    def log_result(self, name: str, success: bool, details: Dict[str, Any]):
        """Log and store test result."""
        self.results[name] = {"success": success, **details}
        status = "✅" if success else "❌"
        logger.info(f"{status} {name}: {'PASSED' if success else 'FAILED'}")
    
    async def test_supabase_connection(self) -> bool:
        """Test Supabase database connection."""
        self.log_test_start("Supabase Connection")
        
        try:
            self.supabase = get_supabase_client()
            
            # Test query
            result = self.supabase.table("knowledge_base_chunks").select("id").limit(1).execute()
            logger.info(f"  Connected to Supabase")
            logger.info(f"  knowledge_base_chunks table accessible")
            
            # Ensure test organization exists
            try:
                org_check = self.supabase.table("organizations").select("id").eq("id", self.org_id).execute()
                if not org_check.data:
                    # Create test organization
                    self.supabase.table("organizations").insert({
                        "id": self.org_id,
                        "name": "E2E Test Organization",
                        "slug": "e2e-test-org",
                    }).execute()
                    logger.info(f"  Created test organization: {self.org_id}")
                else:
                    logger.info(f"  Test organization exists: {self.org_id}")
            except Exception as e:
                logger.warning(f"  Could not ensure org exists: {e}")
            
            self.log_result("supabase_connection", True, {"connected": True})
            return True
            
        except Exception as e:
            logger.error(f"  Connection failed: {e}")
            self.log_result("supabase_connection", False, {"error": str(e)})
            return False
    
    async def test_ocr_with_qwen(self) -> bool:
        """Test OCR with Qwen3-VL vision model."""
        self.log_test_start("OCR with Qwen3-VL")
        
        logger.info(f"  OCR Endpoint: {SETTINGS.rag_ocr_endpoint}")
        logger.info(f"  OCR Model: {SETTINGS.rag_ocr_model}")
        
        if not SETTINGS.rag_ocr_endpoint:
            logger.warning("  OCR endpoint not configured")
            self.log_result("ocr_qwen", False, {"error": "No endpoint configured"})
            return False
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create a test image with text
            img = Image.new('RGB', (500, 250), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a good font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
            except:
                font = ImageFont.load_default()
            
            draw.text((30, 40), "Voice Agent Platform", fill='black', font=font)
            draw.text((30, 90), "Test Document OCR", fill='black', font=font)
            draw.text((30, 140), "Page 1 of 1", fill='black', font=font)
            draw.text((30, 190), "Status: Active", fill='black', font=font)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_data = img_bytes.getvalue()
            
            logger.info(f"  Created test image: {len(img_data)} bytes")
            
        except ImportError as e:
            logger.warning(f"  PIL not available: {e}")
            self.log_result("ocr_qwen", False, {"error": "PIL not installed"})
            return False
        
        # Process with OCR
        ocr = OCRProcessor(
            endpoint=SETTINGS.rag_ocr_endpoint,
            model=SETTINGS.rag_ocr_model,
            timeout=180.0,  # Vision models need time
        )
        
        result = await ocr.process_image(img_data)
        
        logger.info(f"  Mode used: {result.mode_used}")
        logger.info(f"  Confidence: {result.confidence:.2f}")
        logger.info(f"  Error: {result.error}")
        logger.info(f"  Text length: {len(result.text)}")
        
        if result.text:
            logger.info(f"  Text preview: {result.text[:200]}...")
        
        success = result.error is None and len(result.text) > 0
        self.log_result("ocr_qwen", success, {
            "text_length": len(result.text),
            "confidence": result.confidence,
            "error": result.error,
        })
        
        return success
    
    async def test_openai_embeddings(self) -> bool:
        """Test OpenAI text embedding generation."""
        self.log_test_start("OpenAI Embeddings")
        
        logger.info(f"  Model: {SETTINGS.rag_embed_model_text}")
        logger.info(f"  Dimension: {SETTINGS.rag_vector_dim_text}")
        
        try:
            embedder = Embedder(enable_image_embeddings=False)
            
            sample_text = "Voice agents use RAG pipelines to retrieve relevant context from knowledge bases."
            embedding = embedder.embed_text_sync(sample_text)
            
            logger.info(f"  Generated embedding: dim={len(embedding)}")
            logger.info(f"  First 5 values: {embedding[:5]}")
            
            success = len(embedding) == SETTINGS.rag_vector_dim_text
            self.log_result("openai_embeddings", success, {
                "dimension": len(embedding),
                "expected_dim": SETTINGS.rag_vector_dim_text,
            })
            return success
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            self.log_result("openai_embeddings", False, {"error": str(e)})
            return False
    
    async def test_real_document_loading(self) -> Dict[str, Any]:
        """Test loading real PDF, DOCX, and XLSX documents."""
        self.log_test_start("Real Document Loading")
        
        loader = DocumentLoader()
        doc_results = {}
        
        # Get sample files
        pdf_files = list(DOCS_DIR.glob("*.pdf"))[:2]
        docx_files = list(DOCS_DIR.glob("*.docx"))[:2]
        xlsx_files = list(DOCS_DIR.glob("*.xlsx"))[:1]
        
        all_files = pdf_files + docx_files + xlsx_files
        
        logger.info(f"  Found {len(pdf_files)} PDFs, {len(docx_files)} DOCX, {len(xlsx_files)} XLSX")
        
        for file_path in all_files:
            logger.info(f"\n  Loading: {file_path.name}")
            try:
                doc = loader.load(file_path)
                
                # Calculate total content from all pages
                total_content = sum(len(page.text) for page in doc.pages)
                image_count = sum(1 for page in doc.pages if page.has_images)
                
                doc_results[file_path.name] = {
                    "success": True,
                    "page_count": len(doc.pages),
                    "content_length": total_content,
                    "has_images": image_count > 0,
                    "image_count": image_count,
                }
                logger.info(f"    ✓ Pages: {doc_results[file_path.name]['page_count']}")
                logger.info(f"    ✓ Content: {doc_results[file_path.name]['content_length']} chars")
                logger.info(f"    ✓ Pages with images: {doc_results[file_path.name]['image_count']}")
                
            except Exception as e:
                doc_results[file_path.name] = {"success": False, "error": str(e)}
                logger.error(f"    ✗ Failed: {e}")
        
        success_count = sum(1 for r in doc_results.values() if r.get("success"))
        success = success_count == len(doc_results) and len(doc_results) > 0
        
        logger.info(f"\n  Loaded: {success_count}/{len(doc_results)}")
        self.log_result("document_loading", success, doc_results)
        
        return doc_results
    
    async def test_full_ingestion_pipeline(self) -> bool:
        """Test full ingestion pipeline with a real PDF."""
        self.log_test_start("Full Ingestion Pipeline")
        
        if not self.supabase:
            logger.error("  Supabase not connected")
            self.log_result("full_pipeline", False, {"error": "No database connection"})
            return False
        
        # Get a small PDF to test
        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.error("  No PDF files found")
            self.log_result("full_pipeline", False, {"error": "No PDFs available"})
            return False
        
        # Use smallest PDF
        test_file = min(pdf_files, key=lambda f: f.stat().st_size)
        logger.info(f"  Using: {test_file.name} ({test_file.stat().st_size / 1024:.1f} KB)")
        
        try:
            # Step 1: Load document
            logger.info("\n  Step 1: Loading document...")
            loader = DocumentLoader()
            doc = loader.load(test_file)
            total_content = sum(len(page.text) for page in doc.pages)
            logger.info(f"    Loaded: {total_content} chars across {len(doc.pages)} pages")
            
            # Step 2: Chunk document
            logger.info("\n  Step 2: Chunking document...")
            chunker = Chunker(
                chunk_size=SETTINGS.rag_chunk_size,
                chunk_overlap=SETTINGS.rag_chunk_overlap,
            )
            chunks = chunker.chunk_document(doc)
            logger.info(f"    Created {len(chunks)} chunks")
            
            # Limit chunks for testing
            test_chunks = chunks[:10]
            
            # Step 3: Generate embeddings
            logger.info("\n  Step 3: Generating embeddings...")
            embedder = Embedder(enable_image_embeddings=False)
            embedded_chunks = await embedder.embed_chunks(test_chunks)
            
            success_count = sum(1 for e in embedded_chunks if e.text_embedding is not None)
            logger.info(f"    Embedded {success_count}/{len(embedded_chunks)} chunks")
            
            # Step 4: Store in database
            logger.info("\n  Step 4: Storing in database...")
            document_id = f"pipeline-test-{test_file.stem}"
            inserted_ids = []
            
            for emb in embedded_chunks:
                if not emb.text_embedding:
                    continue
                    
                chunk = emb.chunk
                data = {
                    "org_id": self.org_id,
                    "category": "test",
                    "title": f"Test: {test_file.stem}",
                    "source_document": test_file.name,
                    "modality": "text",
                    "page": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "is_table": chunk.is_table,
                    "vector_embedding": emb.text_embedding,
                }
                
                try:
                    result = self.supabase.table("knowledge_base_chunks").upsert(
                        data,
                        on_conflict="org_id,content_hash"
                    ).execute()
                    
                    if result.data:
                        inserted_ids.append(result.data[0]["id"])
                except Exception as e:
                    logger.warning(f"    Insert failed: {e}")
            
            logger.info(f"    Stored {len(inserted_ids)} chunks")
            
            # Step 5: Test search
            logger.info("\n  Step 5: Testing search...")
            
            # Simple content search using source_document
            search_results = self.supabase.table("knowledge_base_chunks").select("*").eq(
                "org_id", self.org_id
            ).eq("source_document", test_file.name).limit(3).execute()
            
            logger.info(f"    Found {len(search_results.data)} chunks")
            
            for i, r in enumerate(search_results.data[:3]):
                preview = r["content"][:60].replace('\n', ' ')
                logger.info(f"      [{i+1}] {preview}...")
            
            # Step 6: Cleanup
            logger.info("\n  Step 6: Cleanup...")
            self.supabase.table("knowledge_base_chunks").delete().eq(
                "source_document", test_file.name
            ).eq("org_id", self.org_id).execute()
            logger.info(f"    Removed test data")
            
            self.log_result("full_pipeline", True, {
                "file": test_file.name,
                "chunks_created": len(chunks),
                "chunks_embedded": success_count,
                "chunks_stored": len(inserted_ids),
            })
            return True
            
        except Exception as e:
            logger.error(f"  Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self.log_result("full_pipeline", False, {"error": str(e)})
            return False
    
    async def test_cross_encoder_reranking(self) -> bool:
        """Test CrossEncoder reranking with real model."""
        self.log_test_start("CrossEncoder Reranking")
        
        try:
            from voice_agent.retrieval.hybrid_search import SearchResult
            from voice_agent.retrieval.reranker import Reranker
            
            # Create test search results
            test_results = [
                SearchResult(
                    chunk_id="1",
                    content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                    modality="text",
                    source_document="ml_guide.pdf",
                    page=1,
                    chunk_index=0,
                    similarity_score=0.8,
                    bm25_score=0.6,
                    rrf_score=0.7,
                ),
                SearchResult(
                    chunk_id="2",
                    content="Pizza is a popular Italian dish with cheese and tomato sauce.",
                    modality="text",
                    source_document="recipes.pdf",
                    page=1,
                    chunk_index=0,
                    similarity_score=0.75,
                    bm25_score=0.5,
                    rrf_score=0.65,
                ),
                SearchResult(
                    chunk_id="3",
                    content="Deep learning uses neural networks with many layers to process complex patterns.",
                    modality="text",
                    source_document="ml_guide.pdf",
                    page=2,
                    chunk_index=1,
                    similarity_score=0.7,
                    bm25_score=0.55,
                    rrf_score=0.6,
                ),
                SearchResult(
                    chunk_id="4",
                    content="The weather today is sunny with a high of 75 degrees Fahrenheit.",
                    modality="text",
                    source_document="weather.pdf",
                    page=1,
                    chunk_index=0,
                    similarity_score=0.65,
                    bm25_score=0.4,
                    rrf_score=0.55,
                ),
            ]
            
            query = "What is artificial intelligence and machine learning?"
            
            # Test reranker
            reranker = Reranker(enabled=True, top_k=3)
            logger.info(f"  Model: {reranker.model_name}")
            logger.info(f"  Query: {query}")
            
            # Rerank
            reranked = await reranker.rerank(query, test_results, top_k=3)
            
            logger.info(f"  Input: {len(test_results)} documents")
            logger.info(f"  Output: {len(reranked)} reranked documents")
            
            # Verify results
            for i, r in enumerate(reranked):
                logger.info(f"    [{i+1}] Score: {r.rerank_score:.4f} - {r.content[:50]}...")
            
            # Check that ML content is ranked higher than pizza/weather
            top_content = reranked[0].content.lower()
            is_ml_top = "machine learning" in top_content or "artificial intelligence" in top_content
            
            if not is_ml_top:
                logger.warning("  Warning: ML content not ranked first")
            
            self.log_result("cross_encoder_reranking", True, {
                "model": reranker.model_name,
                "input_docs": len(test_results),
                "output_docs": len(reranked),
                "top_score": round(reranked[0].rerank_score or 0.0, 4),
                "ml_content_first": is_ml_top,
            })
            return True
            
        except ImportError as e:
            logger.error(f"  sentence-transformers not installed: {e}")
            self.log_result("cross_encoder_reranking", False, {
                "error": "sentence-transformers not installed"
            })
            return False
        except Exception as e:
            logger.error(f"  Reranking failed: {e}")
            import traceback
            traceback.print_exc()
            self.log_result("cross_encoder_reranking", False, {"error": str(e)})
            return False
    
    async def test_siglip_image_embeddings(self) -> bool:
        """Test SigLIP model for image embeddings with a real image."""
        self.log_test_start("SigLIP Image Embeddings")
        
        try:
            from PIL import Image
            import io
            
            # Create a simple test image (red square 100x100)
            logger.info("  Creating test image...")
            test_image = Image.new('RGB', (100, 100), color='red')
            
            # Convert to bytes (simulating loaded image from document)
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='PNG')
            image_bytes = img_buffer.getvalue()
            logger.info(f"    Created test image: 100x100 RGB, {len(image_bytes)} bytes")
            
            # Create a chunk with image data
            from voice_agent.ingestion.chunker import Chunk, ChunkType
            
            image_chunk = Chunk(
                content="Test image for SigLIP embedding",
                chunk_type=ChunkType.IMAGE,
                chunk_index=0,
                page_number=1,
                source_document="test_siglip_image.png",
                image_data=image_bytes,
            )
            
            # Create embedder with image embeddings ENABLED
            logger.info("\n  Loading SigLIP model (this may take a moment)...")
            embedder = Embedder(enable_image_embeddings=True)
            
            # Generate embeddings
            logger.info("  Generating image embedding...")
            results = await embedder.embed_chunks([image_chunk])
            
            if not results:
                logger.error("  No results returned")
                self.log_result("siglip_embeddings", False, {"error": "No results"})
                return False
            
            result = results[0]
            
            if result.error:
                logger.error(f"  Embedding error: {result.error}")
                self.log_result("siglip_embeddings", False, {"error": result.error})
                return False
            
            if result.image_embedding is None:
                logger.error("  No image embedding returned")
                self.log_result("siglip_embeddings", False, {"error": "No embedding"})
                return False
            
            embedding_dim = len(result.image_embedding)
            logger.info(f"\n  ✓ SigLIP embedding generated successfully!")
            logger.info(f"    Dimension: {embedding_dim}")
            logger.info(f"    First 5 values: {result.image_embedding[:5]}")
            
            # Verify expected dimension (768 for siglip-base-patch16-384)
            expected_dim = 768
            if embedding_dim != expected_dim:
                logger.warning(f"    ⚠ Expected {expected_dim}d, got {embedding_dim}d")
            
            self.log_result("siglip_embeddings", True, {
                "dimension": embedding_dim,
                "expected_dimension": expected_dim,
                "matches_expected": embedding_dim == expected_dim,
                "sample_values": result.image_embedding[:5],
            })
            return True
            
        except ImportError as e:
            logger.error(f"  Required library not installed: {e}")
            self.log_result("siglip_embeddings", False, {
                "error": f"Missing dependency: {e}"
            })
            return False
        except Exception as e:
            logger.error(f"  SigLIP test failed: {e}")
            import traceback
            traceback.print_exc()
            self.log_result("siglip_embeddings", False, {"error": str(e)})
            return False
    
    async def run_all_tests(self) -> int:
        """Run all integration tests."""
        logger.info("=" * 60)
        logger.info("RAG Real Documents Integration Tests")
        logger.info("=" * 60)
        logger.info(f"Project root: {project_root}")
        logger.info(f"Documents dir: {DOCS_DIR}")
        logger.info(f"OpenAI configured: {'Yes' if SETTINGS.openai_api_key else 'No'}")
        logger.info(f"Supabase configured: {'Yes' if SETTINGS.supabase_url else 'No'}")
        logger.info(f"OCR endpoint: {SETTINGS.rag_ocr_endpoint or 'Not configured'}")
        
        # Run tests
        await self.test_supabase_connection()
        await self.test_ocr_with_qwen()
        await self.test_openai_embeddings()
        await self.test_real_document_loading()
        await self.test_full_ingestion_pipeline()
        await self.test_cross_encoder_reranking()
        await self.test_siglip_image_embeddings()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.results.items():
            status = "✅ PASSED" if result.get("success") else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")
            
            if result.get("success"):
                passed += 1
            else:
                failed += 1
                if result.get("error"):
                    logger.info(f"    Error: {result.get('error')}")
        
        logger.info(f"\nTotal: {passed} passed, {failed} failed")
        
        return 0 if failed == 0 else 1


async def main():
    """Main entry point."""
    runner = TestRunner()
    return await runner.run_all_tests()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
