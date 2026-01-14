-- Migration: Add knowledge_base_chunks table for RAG system
-- Date: 2026-01-13
-- Description: Creates chunk-level table for semantic/hybrid search with multi-modal embeddings

-- Enable required extensions (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. Create knowledge_base_chunks table
-- =============================================================================
CREATE TABLE IF NOT EXISTS knowledge_base_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    knowledge_base_id UUID REFERENCES knowledge_base(id) ON DELETE SET NULL,
    
    -- Content & Metadata
    category TEXT,
    title TEXT,
    source_document TEXT,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'image', 'image_caption', 'table')),
    page INTEGER,
    chunk_index INTEGER,
    
    -- Chunk Content
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    
    -- OCR Quality Tracking
    ocr_confidence REAL,
    
    -- Table-specific fields
    is_table BOOLEAN DEFAULT FALSE,
    table_context TEXT,
    
    -- Image-specific fields
    alt_text TEXT,
    image_path TEXT,
    
    -- Embeddings
    vector_embedding vector(1536),  -- OpenAI text-embedding-3-small
    vector_image vector(768),       -- SigLIP base (patch16-384)
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- =============================================================================
-- 2. Create Indexes
-- =============================================================================

-- Unique constraint for deduplication per org
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_chunks_dedup 
    ON knowledge_base_chunks(org_id, content_hash);

-- B-tree index for common filters
CREATE INDEX IF NOT EXISTS idx_kb_chunks_org_source 
    ON knowledge_base_chunks(org_id, source_document, modality);

-- B-tree index for category filtering
CREATE INDEX IF NOT EXISTS idx_kb_chunks_org_category 
    ON knowledge_base_chunks(org_id, category);

-- B-tree index for knowledge_base parent lookup
CREATE INDEX IF NOT EXISTS idx_kb_chunks_parent 
    ON knowledge_base_chunks(knowledge_base_id);

-- Full-text search index (Portuguese-aware)
-- Using 'portuguese' config for PT content, 'english' as fallback
CREATE INDEX IF NOT EXISTS idx_kb_chunks_fts_pt 
    ON knowledge_base_chunks 
    USING GIN (to_tsvector('portuguese', content));

CREATE INDEX IF NOT EXISTS idx_kb_chunks_fts_en 
    ON knowledge_base_chunks 
    USING GIN (to_tsvector('english', content));

-- Vector indexes for ANN search
-- Using IVFFlat with cosine similarity (good for medium-sized datasets)
-- For larger datasets (>1M rows), consider HNSW: USING hnsw (vector_embedding vector_cosine_ops)

-- Text embedding index (1536d)
CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding 
    ON knowledge_base_chunks 
    USING ivfflat (vector_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Image embedding index (768d) - only for rows with images
CREATE INDEX IF NOT EXISTS idx_kb_chunks_image_embedding 
    ON knowledge_base_chunks 
    USING ivfflat (vector_image vector_cosine_ops)
    WITH (lists = 50)
    WHERE vector_image IS NOT NULL;

-- =============================================================================
-- 3. Enable Row-Level Security
-- =============================================================================
ALTER TABLE knowledge_base_chunks ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Org Isolation
CREATE POLICY "KB Chunks Org Isolation" ON knowledge_base_chunks
    FOR ALL USING (org_id = get_my_org_id() OR is_super_admin());

-- =============================================================================
-- 4. Add helper columns to parent knowledge_base table (if needed)
-- =============================================================================
-- Add provenance/ingestion columns if not already present
DO $$
BEGIN
    -- Add doc_checksum column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'knowledge_base' AND column_name = 'doc_checksum') THEN
        ALTER TABLE knowledge_base ADD COLUMN doc_checksum TEXT;
    END IF;
    
    -- Add ingestion_metadata column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'knowledge_base' AND column_name = 'ingestion_metadata') THEN
        ALTER TABLE knowledge_base ADD COLUMN ingestion_metadata JSONB DEFAULT '{}'::jsonb;
    END IF;
    
    -- Add is_chunked flag to track which parent docs have been chunked
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'knowledge_base' AND column_name = 'is_chunked') THEN
        ALTER TABLE knowledge_base ADD COLUMN is_chunked BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- =============================================================================
-- 5. Create updated_at trigger
-- =============================================================================
CREATE OR REPLACE FUNCTION update_kb_chunks_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_kb_chunks_updated_at ON knowledge_base_chunks;
CREATE TRIGGER trigger_kb_chunks_updated_at
    BEFORE UPDATE ON knowledge_base_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_kb_chunks_updated_at();

-- =============================================================================
-- 6. Create helper functions for hybrid search
-- =============================================================================

-- Function for Portuguese full-text search
CREATE OR REPLACE FUNCTION kb_chunks_fts_pt(
    p_org_id UUID,
    p_query TEXT,
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    modality TEXT,
    source_document TEXT,
    page INTEGER,
    chunk_index INTEGER,
    ocr_confidence REAL,
    is_table BOOLEAN,
    table_context TEXT,
    alt_text TEXT,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kbc.id,
        kbc.content,
        kbc.modality,
        kbc.source_document,
        kbc.page,
        kbc.chunk_index,
        kbc.ocr_confidence,
        kbc.is_table,
        kbc.table_context,
        kbc.alt_text,
        ts_rank(to_tsvector('portuguese', kbc.content), plainto_tsquery('portuguese', p_query)) AS rank
    FROM knowledge_base_chunks kbc
    WHERE kbc.org_id = p_org_id
      AND to_tsvector('portuguese', kbc.content) @@ plainto_tsquery('portuguese', p_query)
    ORDER BY rank DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function for vector similarity search
CREATE OR REPLACE FUNCTION kb_chunks_vector_search(
    p_org_id UUID,
    p_embedding vector(1536),
    p_limit INTEGER DEFAULT 50,
    p_category TEXT DEFAULT NULL,
    p_source_document TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    modality TEXT,
    source_document TEXT,
    page INTEGER,
    chunk_index INTEGER,
    ocr_confidence REAL,
    is_table BOOLEAN,
    table_context TEXT,
    alt_text TEXT,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kbc.id,
        kbc.content,
        kbc.modality,
        kbc.source_document,
        kbc.page,
        kbc.chunk_index,
        kbc.ocr_confidence,
        kbc.is_table,
        kbc.table_context,
        kbc.alt_text,
        1 - (kbc.vector_embedding <=> p_embedding) AS similarity
    FROM knowledge_base_chunks kbc
    WHERE kbc.org_id = p_org_id
      AND kbc.vector_embedding IS NOT NULL
      AND (p_category IS NULL OR kbc.category = p_category)
      AND (p_source_document IS NULL OR kbc.source_document = p_source_document)
    ORDER BY kbc.vector_embedding <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function for image vector similarity search
CREATE OR REPLACE FUNCTION kb_chunks_image_search(
    p_org_id UUID,
    p_image_embedding vector(768),
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    modality TEXT,
    source_document TEXT,
    page INTEGER,
    alt_text TEXT,
    image_path TEXT,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kbc.id,
        kbc.content,
        kbc.modality,
        kbc.source_document,
        kbc.page,
        kbc.alt_text,
        kbc.image_path,
        1 - (kbc.vector_image <=> p_image_embedding) AS similarity
    FROM knowledge_base_chunks kbc
    WHERE kbc.org_id = p_org_id
      AND kbc.vector_image IS NOT NULL
    ORDER BY kbc.vector_image <=> p_image_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- 7. Comments for documentation
-- =============================================================================
COMMENT ON TABLE knowledge_base_chunks IS 'Chunk-level storage for RAG system with multi-modal embeddings and hybrid search support';
COMMENT ON COLUMN knowledge_base_chunks.modality IS 'Content type: text, image, image_caption, or table';
COMMENT ON COLUMN knowledge_base_chunks.content_hash IS 'SHA-256 hash of normalized content for deduplication';
COMMENT ON COLUMN knowledge_base_chunks.ocr_confidence IS 'OCR confidence score (0-1) for scanned/image content';
COMMENT ON COLUMN knowledge_base_chunks.vector_embedding IS 'OpenAI text-embedding-3-small (1536d) for text content';
COMMENT ON COLUMN knowledge_base_chunks.vector_image IS 'SigLIP base (768d) for image content';
COMMENT ON COLUMN knowledge_base_chunks.is_table IS 'True if this chunk contains a complete table';
COMMENT ON COLUMN knowledge_base_chunks.table_context IS 'Description of table contents (what/when/units)';
COMMENT ON COLUMN knowledge_base_chunks.alt_text IS 'Caption or alt text for images/charts/diagrams';

-- =============================================================================
-- Migration complete
-- =============================================================================
