-- =============================================================================
-- Migration: RAG 2.0 Schema - Triple-Hybrid Architecture
-- Date: 2026-01-14
-- Description: Creates new tables for RAG 2.0 with parent/child hierarchy,
--              knowledge graph support, and Matryoshka 1024d embeddings
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. DOCUMENTS TABLE (top-level document registry)
-- =============================================================================
CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Identification & Idempotency
    external_id TEXT,                    -- Optional external system reference
    hash_sha256 TEXT NOT NULL,           -- SHA-256 of raw file bytes for dedup
    
    -- Document Metadata
    file_name TEXT NOT NULL,
    mime_type TEXT,
    language TEXT DEFAULT 'pt-BR',
    source_type TEXT CHECK (source_type IN ('upload', 'url', 'api')),
    source_url TEXT,
    storage_path TEXT,                   -- Path in storage bucket (if applicable)
    
    -- Organization / Classification
    collection TEXT,                     -- e.g., 'contracts', 'policies', 'products'
    tags TEXT[] DEFAULT '{}',
    title TEXT,
    
    -- Processing State
    ingestion_status TEXT DEFAULT 'pending' CHECK (ingestion_status IN ('pending', 'processing', 'completed', 'failed')),
    ingestion_error TEXT,
    
    -- Extended Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Unique constraint: one document per org + hash (idempotency)
CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_docs_org_hash 
    ON rag_documents(org_id, hash_sha256);

-- Index for collection filtering
CREATE INDEX IF NOT EXISTS idx_rag_docs_org_collection 
    ON rag_documents(org_id, collection);

-- Index for status queries
CREATE INDEX IF NOT EXISTS idx_rag_docs_status 
    ON rag_documents(org_id, ingestion_status);

-- =============================================================================
-- 2. PARENT_CHUNKS TABLE (800-1000 token context units)
-- =============================================================================
CREATE TABLE IF NOT EXISTS rag_parent_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Position in Document
    index_in_document INT NOT NULL,
    
    -- Content
    text TEXT NOT NULL,
    token_count INT,
    
    -- Provenance
    page_start INT,
    page_end INT,
    section_heading TEXT,
    
    -- OCR Quality (if applicable)
    ocr_confidence REAL,
    
    -- Classification inherited from document
    tags TEXT[] DEFAULT '{}',
    
    -- Extended Metadata (table_context, alt_text, etc.)
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Index for document lookup
CREATE INDEX IF NOT EXISTS idx_rag_parents_doc 
    ON rag_parent_chunks(document_id, index_in_document);

-- Index for org queries
CREATE INDEX IF NOT EXISTS idx_rag_parents_org 
    ON rag_parent_chunks(org_id);

-- =============================================================================
-- 3. CHILD_CHUNKS TABLE (~200 token retrieval units)
-- =============================================================================
CREATE TABLE IF NOT EXISTS rag_child_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_id UUID NOT NULL REFERENCES rag_parent_chunks(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Position
    index_in_parent INT NOT NULL,
    
    -- Content
    text TEXT NOT NULL,
    token_count INT,
    
    -- Character Offsets (within parent)
    start_char_offset INT,
    end_char_offset INT,
    
    -- Provenance
    page INT,
    modality TEXT DEFAULT 'text' CHECK (modality IN ('text', 'table', 'image', 'image_caption')),
    
    -- Deduplication
    content_hash TEXT NOT NULL,
    
    -- Tags (inherited or specific)
    tags TEXT[] DEFAULT '{}',
    
    -- Extended Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- =========================================================================
    -- EMBEDDINGS (Matryoshka 4096 → 1024)
    -- =========================================================================
    -- Primary: 1024d truncated (MRL) - used for ANN search
    embedding_1024 vector(1024),
    
    -- Optional: Full 4096d for offline analysis / future use
    -- Uncomment if needed: embedding_4096 vector(4096),
    
    -- =========================================================================
    -- Full-Text Search (generated column)
    -- =========================================================================
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('portuguese', coalesce(text, ''))
    ) STORED,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Unique constraint for deduplication per org
CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_children_dedup 
    ON rag_child_chunks(org_id, content_hash);

-- Index for parent lookup
CREATE INDEX IF NOT EXISTS idx_rag_children_parent 
    ON rag_child_chunks(parent_id, index_in_parent);

-- Index for document lookup
CREATE INDEX IF NOT EXISTS idx_rag_children_doc 
    ON rag_child_chunks(document_id);

-- Index for org queries
CREATE INDEX IF NOT EXISTS idx_rag_children_org 
    ON rag_child_chunks(org_id);

-- Full-text search index (Portuguese)
CREATE INDEX IF NOT EXISTS idx_rag_children_fts 
    ON rag_child_chunks USING GIN (tsv);

-- HNSW vector index for ANN search (1024d)
-- Using cosine similarity (vector_cosine_ops)
CREATE INDEX IF NOT EXISTS idx_rag_children_embedding_hnsw 
    ON rag_child_chunks 
    USING hnsw (embedding_1024 vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- 4. ENTITIES TABLE (Knowledge Graph nodes)
-- =============================================================================
CREATE TABLE IF NOT EXISTS rag_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
    
    -- Entity Type & Name
    entity_type TEXT NOT NULL,           -- e.g., PERSON, ORG, PRODUCT, CLAUSE, DATE
    name TEXT NOT NULL,
    canonical_name TEXT,                 -- Normalized/deduplicated name
    
    -- Description / Context
    description TEXT,
    
    -- Extended Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Index for entity lookup by type and name
CREATE INDEX IF NOT EXISTS idx_rag_entities_type_name 
    ON rag_entities(org_id, entity_type, canonical_name);

-- Index for document lookup
CREATE INDEX IF NOT EXISTS idx_rag_entities_doc 
    ON rag_entities(document_id);

-- =============================================================================
-- 5. ENTITY_MENTIONS TABLE (links entities to chunks)
-- =============================================================================
CREATE TABLE IF NOT EXISTS rag_entity_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES rag_entities(id) ON DELETE CASCADE,
    parent_chunk_id UUID REFERENCES rag_parent_chunks(id) ON DELETE CASCADE,
    child_chunk_id UUID REFERENCES rag_child_chunks(id) ON DELETE CASCADE,
    document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
    
    -- Position in text
    char_start INT,
    char_end INT,
    mention_text TEXT,
    
    -- Confidence
    confidence REAL DEFAULT 1.0,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Index for entity lookup
CREATE INDEX IF NOT EXISTS idx_rag_mentions_entity 
    ON rag_entity_mentions(entity_id);

-- Index for chunk lookup
CREATE INDEX IF NOT EXISTS idx_rag_mentions_parent 
    ON rag_entity_mentions(parent_chunk_id);

CREATE INDEX IF NOT EXISTS idx_rag_mentions_child 
    ON rag_entity_mentions(child_chunk_id);

-- =============================================================================
-- 6. RELATIONS TABLE (Knowledge Graph edges / triplets)
-- =============================================================================
CREATE TABLE IF NOT EXISTS rag_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
    
    -- Triplet: (subject, relation_type, object)
    relation_type TEXT NOT NULL,          -- e.g., OWNS, EMPLOYS, REFERENCES, DEPENDS_ON
    subject_entity_id UUID NOT NULL REFERENCES rag_entities(id) ON DELETE CASCADE,
    object_entity_id UUID NOT NULL REFERENCES rag_entities(id) ON DELETE CASCADE,
    
    -- Extraction Confidence
    confidence REAL DEFAULT 1.0,
    
    -- Source chunk where this relation was extracted
    source_parent_id UUID REFERENCES rag_parent_chunks(id) ON DELETE SET NULL,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Index for relation queries
CREATE INDEX IF NOT EXISTS idx_rag_relations_type 
    ON rag_relations(org_id, relation_type);

-- Index for subject/object traversal
CREATE INDEX IF NOT EXISTS idx_rag_relations_subject 
    ON rag_relations(subject_entity_id);

CREATE INDEX IF NOT EXISTS idx_rag_relations_object 
    ON rag_relations(object_entity_id);

-- =============================================================================
-- 7. ROW-LEVEL SECURITY
-- =============================================================================
ALTER TABLE rag_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_parent_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_child_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_entity_mentions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_relations ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Org Isolation
CREATE POLICY "RAG Documents Org Isolation" ON rag_documents
    FOR ALL USING (org_id = get_my_org_id() OR is_super_admin());

CREATE POLICY "RAG Parent Chunks Org Isolation" ON rag_parent_chunks
    FOR ALL USING (org_id = get_my_org_id() OR is_super_admin());

CREATE POLICY "RAG Child Chunks Org Isolation" ON rag_child_chunks
    FOR ALL USING (org_id = get_my_org_id() OR is_super_admin());

CREATE POLICY "RAG Entities Org Isolation" ON rag_entities
    FOR ALL USING (org_id = get_my_org_id() OR is_super_admin());

CREATE POLICY "RAG Entity Mentions Org Isolation" ON rag_entity_mentions
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM rag_entities e 
            WHERE e.id = entity_id AND (e.org_id = get_my_org_id() OR is_super_admin())
        )
    );

CREATE POLICY "RAG Relations Org Isolation" ON rag_relations
    FOR ALL USING (org_id = get_my_org_id() OR is_super_admin());

-- =============================================================================
-- 8. UPDATED_AT TRIGGERS
-- =============================================================================
CREATE OR REPLACE FUNCTION update_rag_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_rag_docs_updated_at ON rag_documents;
CREATE TRIGGER trigger_rag_docs_updated_at
    BEFORE UPDATE ON rag_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_rag_updated_at();

-- =============================================================================
-- 9. HELPER FUNCTIONS FOR RAG 2.0 RETRIEVAL
-- =============================================================================

-- Function for lexical (FTS) search on child chunks
CREATE OR REPLACE FUNCTION rag2_lexical_search(
    p_org_id UUID,
    p_query TEXT,
    p_limit INTEGER DEFAULT 50,
    p_collection TEXT DEFAULT NULL
)
RETURNS TABLE (
    child_id UUID,
    parent_id UUID,
    document_id UUID,
    text TEXT,
    page INT,
    modality TEXT,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS child_id,
        c.parent_id,
        c.document_id,
        c.text,
        c.page,
        c.modality,
        ts_rank_cd(c.tsv, plainto_tsquery('portuguese', p_query)) AS rank
    FROM rag_child_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.org_id = p_org_id
      AND c.tsv @@ plainto_tsquery('portuguese', p_query)
      AND (p_collection IS NULL OR d.collection = p_collection)
    ORDER BY rank DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function for semantic (vector) search on child chunks
CREATE OR REPLACE FUNCTION rag2_semantic_search(
    p_org_id UUID,
    p_embedding vector(1024),
    p_limit INTEGER DEFAULT 100,
    p_collection TEXT DEFAULT NULL
)
RETURNS TABLE (
    child_id UUID,
    parent_id UUID,
    document_id UUID,
    text TEXT,
    page INT,
    modality TEXT,
    similarity DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS child_id,
        c.parent_id,
        c.document_id,
        c.text,
        c.page,
        c.modality,
        (1 - (c.embedding_1024 <=> p_embedding))::DOUBLE PRECISION AS similarity
    FROM rag_child_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.org_id = p_org_id
      AND c.embedding_1024 IS NOT NULL
      AND (p_collection IS NULL OR d.collection = p_collection)
    ORDER BY c.embedding_1024 <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function for hybrid RRF search
CREATE OR REPLACE FUNCTION rag2_hybrid_rrf_search(
    p_org_id UUID,
    p_embedding vector(1024),
    p_query TEXT,
    p_limit INTEGER DEFAULT 50,
    p_collection TEXT DEFAULT NULL,
    p_rrf_k INTEGER DEFAULT 60,
    p_lexical_weight REAL DEFAULT 0.7,
    p_semantic_weight REAL DEFAULT 0.8
)
RETURNS TABLE (
    child_id UUID,
    parent_id UUID,
    document_id UUID,
    text TEXT,
    page INT,
    modality TEXT,
    rrf_score REAL,
    lexical_rank INT,
    semantic_rank INT
) AS $$
WITH lexical AS (
    SELECT 
        c.id AS child_id,
        c.parent_id,
        c.document_id,
        c.text,
        c.page,
        c.modality,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, plainto_tsquery('portuguese', p_query)) DESC) AS rank
    FROM rag_child_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.org_id = p_org_id
      AND c.tsv @@ plainto_tsquery('portuguese', p_query)
      AND (p_collection IS NULL OR d.collection = p_collection)
    LIMIT p_limit * 2
),
semantic AS (
    SELECT 
        c.id AS child_id,
        c.parent_id,
        c.document_id,
        c.text,
        c.page,
        c.modality,
        ROW_NUMBER() OVER (ORDER BY c.embedding_1024 <=> p_embedding) AS rank
    FROM rag_child_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.org_id = p_org_id
      AND c.embedding_1024 IS NOT NULL
      AND (p_collection IS NULL OR d.collection = p_collection)
    LIMIT p_limit * 2
),
combined AS (
    SELECT 
        COALESCE(l.child_id, s.child_id) AS child_id,
        COALESCE(l.parent_id, s.parent_id) AS parent_id,
        COALESCE(l.document_id, s.document_id) AS document_id,
        COALESCE(l.text, s.text) AS text,
        COALESCE(l.page, s.page) AS page,
        COALESCE(l.modality, s.modality) AS modality,
        l.rank AS lexical_rank,
        s.rank AS semantic_rank,
        (
            COALESCE(p_lexical_weight / (p_rrf_k + l.rank), 0) +
            COALESCE(p_semantic_weight / (p_rrf_k + s.rank), 0)
        ) AS rrf_score
    FROM lexical l
    FULL OUTER JOIN semantic s ON l.child_id = s.child_id
)
SELECT 
    combined.child_id,
    combined.parent_id,
    combined.document_id,
    combined.text,
    combined.page,
    combined.modality,
    combined.rrf_score,
    combined.lexical_rank::INT,
    combined.semantic_rank::INT
FROM combined
ORDER BY combined.rrf_score DESC
LIMIT p_limit;
$$ LANGUAGE sql STABLE;

-- Function to expand child IDs to parent chunks
CREATE OR REPLACE FUNCTION rag2_expand_to_parents(
    p_child_ids UUID[],
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    parent_id UUID,
    document_id UUID,
    text TEXT,
    page_start INT,
    page_end INT,
    section_heading TEXT,
    aggregated_score REAL
) AS $$
BEGIN
    -- This is a simple expansion; in practice you'd pass scores
    RETURN QUERY
    SELECT DISTINCT ON (pc.id)
        pc.id AS parent_id,
        pc.document_id,
        pc.text,
        pc.page_start,
        pc.page_end,
        pc.section_heading,
        1.0::REAL AS aggregated_score  -- Placeholder; actual scoring done in Python
    FROM rag_parent_chunks pc
    JOIN rag_child_chunks cc ON cc.parent_id = pc.id
    WHERE cc.id = ANY(p_child_ids)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- 10. COMMENTS
-- =============================================================================
COMMENT ON TABLE rag_documents IS 'RAG 2.0: Top-level document registry with idempotency';
COMMENT ON TABLE rag_parent_chunks IS 'RAG 2.0: Parent chunks (800-1000 tokens) for LLM context';
COMMENT ON TABLE rag_child_chunks IS 'RAG 2.0: Child chunks (~200 tokens) for retrieval with 1024d embeddings';
COMMENT ON TABLE rag_entities IS 'RAG 2.0: Knowledge graph entities extracted via NER';
COMMENT ON TABLE rag_entity_mentions IS 'RAG 2.0: Links entities to chunks where they appear';
COMMENT ON TABLE rag_relations IS 'RAG 2.0: Knowledge graph triplets (subject, relation, object)';

COMMENT ON COLUMN rag_child_chunks.embedding_1024 IS 'Matryoshka 4096→1024 truncated embedding (L2 normalized)';
COMMENT ON COLUMN rag_child_chunks.tsv IS 'Full-text search vector (Portuguese)';
COMMENT ON COLUMN rag_documents.hash_sha256 IS 'SHA-256 of raw file bytes for idempotent ingestion';

-- =============================================================================
-- Migration Complete
-- =============================================================================
