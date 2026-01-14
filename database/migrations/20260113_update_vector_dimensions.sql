-- ============================================================================
-- Migration: Update Vector Dimensions for Qwen3 Embedding Models
-- ============================================================================
-- Date: 2026-01-13
-- Applied: 2026-01-13 (manually via docker exec)
-- 
-- Description: Updates vector_embedding from 1536d (OpenAI) to 4096d (Qwen3-Embedding)
--              Updates image_embedding from 768d (SigLIP) to 4096d (Qwen3-VL-Embedding)
-- 
-- WARNING: This migration will drop existing embeddings! 
--          You must re-run ingestion after applying this migration.
--
-- IMPORTANT: pgvector 0.8.0 has a 2000-dimension limit for HNSW/IVFFlat indexes!
--            The index creation commands below will FAIL until pgvector is upgraded.
--            Vector columns work without indexes (uses exact scan - slower).
--
-- Applied changes:
--   ✅ Dropped old indexes
--   ✅ Changed vector_embedding to vector(4096)
--   ✅ Changed vector_image to vector(4096)  
--   ❌ Index creation skipped (pgvector 0.8.0 dimension limit)
-- ============================================================================

-- ============================================================================
-- Step 1: Drop existing indexes (they reference the old vector dimensions)
-- ============================================================================

DROP INDEX IF EXISTS idx_kb_chunks_vector_embedding;
DROP INDEX IF EXISTS idx_kb_chunks_image_embedding;

-- ============================================================================
-- Step 2: Alter vector columns to new dimensions
-- ============================================================================

-- Change text embedding from 1536d to 4096d
ALTER TABLE knowledge_base_chunks
    ALTER COLUMN vector_embedding TYPE vector(4096);

-- Change image embedding from 768d to 4096d  
ALTER TABLE knowledge_base_chunks
    ALTER COLUMN image_embedding TYPE vector(4096);

-- ============================================================================
-- Step 3: Clear existing embeddings (they are now invalid due to dimension change)
-- ============================================================================

UPDATE knowledge_base_chunks
SET vector_embedding = NULL, image_embedding = NULL;

-- ============================================================================
-- Step 4: Recreate indexes with new dimensions
-- ============================================================================

-- IVFFlat index for text embeddings (4096d)
-- Note: For 4096d vectors, lists should be ~sqrt(n) where n is expected row count
-- Using 200 lists for up to ~40K rows, adjust for larger datasets
CREATE INDEX idx_kb_chunks_vector_embedding 
    ON knowledge_base_chunks 
    USING ivfflat (vector_embedding vector_cosine_ops)
    WITH (lists = 200);

-- IVFFlat index for image embeddings (4096d)
CREATE INDEX idx_kb_chunks_image_embedding
    ON knowledge_base_chunks
    USING ivfflat (image_embedding vector_cosine_ops)
    WITH (lists = 200);

-- ============================================================================
-- Step 5: Update column comments
-- ============================================================================

COMMENT ON COLUMN knowledge_base_chunks.vector_embedding IS 'Qwen3-Embedding (4096d) for text content';
COMMENT ON COLUMN knowledge_base_chunks.image_embedding IS 'Qwen3-VL-Embedding (4096d) for image content';

-- ============================================================================
-- Step 6: Update RPC function parameter type (the function itself doesn't need 
--         changes as vector comparison operators work with any dimension)
-- ============================================================================

-- The kb_chunks_vector_search function uses vector type without dimension 
-- constraint in parameter, so no changes needed.

-- ============================================================================
-- Verification query (run after migration)
-- ============================================================================

-- SELECT 
--     column_name,
--     udt_name,
--     character_maximum_length
-- FROM information_schema.columns
-- WHERE table_name = 'knowledge_base_chunks'
--   AND column_name IN ('vector_embedding', 'image_embedding');

-- ============================================================================
-- Rollback (if needed)
-- ============================================================================

-- To rollback to OpenAI/SigLIP dimensions:
-- 
-- DROP INDEX IF EXISTS idx_kb_chunks_vector_embedding;
-- DROP INDEX IF EXISTS idx_kb_chunks_image_embedding;
-- 
-- ALTER TABLE knowledge_base_chunks
--     ALTER COLUMN vector_embedding TYPE vector(1536);
-- 
-- ALTER TABLE knowledge_base_chunks
--     ALTER COLUMN image_embedding TYPE vector(768);
-- 
-- UPDATE knowledge_base_chunks
-- SET vector_embedding = NULL, image_embedding = NULL;
-- 
-- CREATE INDEX idx_kb_chunks_vector_embedding 
--     ON knowledge_base_chunks 
--     USING ivfflat (vector_embedding vector_cosine_ops)
--     WITH (lists = 100);
-- 
-- CREATE INDEX idx_kb_chunks_image_embedding
--     ON knowledge_base_chunks
--     USING ivfflat (image_embedding vector_cosine_ops)
--     WITH (lists = 100);
