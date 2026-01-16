-- ═══════════════════════════════════════════════════════════════════════════════
-- TRIPLE-HYBRID-RAG DATABASE SCHEMA
-- PostgreSQL + pgvector for the Ultimate RAG System
-- ═══════════════════════════════════════════════════════════════════════════════

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text matching

-- ═══════════════════════════════════════════════════════════════════════════════
-- DOCUMENTS
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    hash_sha256 TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT,
    mime_type TEXT,
    collection TEXT DEFAULT 'general',
    title TEXT,
    tags TEXT[] DEFAULT '{}'::TEXT[],
    ingestion_status TEXT DEFAULT 'pending',
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc', NOW()) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Unique constraint: one document per tenant+hash
CREATE UNIQUE INDEX IF NOT EXISTS rag_documents_tenant_hash_uq
    ON rag_documents (tenant_id, hash_sha256);

-- Index for collection filtering
CREATE INDEX IF NOT EXISTS rag_documents_collection_idx
    ON rag_documents (tenant_id, collection);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS rag_documents_status_idx
    ON rag_documents (tenant_id, ingestion_status);

-- ═══════════════════════════════════════════════════════════════════════════════
-- PARENT CHUNKS (Context Units)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS rag_parent_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    index_in_document INT NOT NULL,
    text TEXT NOT NULL,
    token_count INT,
    page_start INT,
    page_end INT,
    section_heading TEXT,
    ocr_confidence REAL,
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Index for document ordering
CREATE INDEX IF NOT EXISTS rag_parent_chunks_doc_idx
    ON rag_parent_chunks (document_id, index_in_document);

-- Index for tenant filtering
CREATE INDEX IF NOT EXISTS rag_parent_chunks_tenant_idx
    ON rag_parent_chunks (tenant_id);

-- ═══════════════════════════════════════════════════════════════════════════════
-- CHILD CHUNKS (Retrieval Units)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS rag_child_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_id UUID NOT NULL REFERENCES rag_parent_chunks(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    index_in_parent INT NOT NULL,
    text TEXT NOT NULL,
    token_count INT,
    start_char_offset INT,
    end_char_offset INT,
    page INT,
    modality TEXT DEFAULT 'text',
    content_hash TEXT NOT NULL,
    
    -- Text embedding (1024d, Matryoshka truncated from 4096d)
    embedding_1024 VECTOR(1024),
    
    -- Image embedding for multimodal (1024d)
    image_embedding_1024 VECTOR(1024),
    
    -- Raw image data for multimodal chunks
    image_data BYTEA,
    
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Unique constraint: prevent duplicate chunks
CREATE UNIQUE INDEX IF NOT EXISTS rag_child_chunks_tenant_hash_uq
    ON rag_child_chunks (tenant_id, content_hash);

-- Index for parent/document ordering
CREATE INDEX IF NOT EXISTS rag_child_chunks_doc_idx
    ON rag_child_chunks (document_id, parent_id, index_in_parent);

-- Index for tenant+document filtering
CREATE INDEX IF NOT EXISTS rag_child_chunks_tenant_doc_idx
    ON rag_child_chunks (tenant_id, document_id);

-- Index for page filtering
CREATE INDEX IF NOT EXISTS rag_child_chunks_page_idx
    ON rag_child_chunks (tenant_id, page);

-- Full-text search index (GIN for fast FTS)
CREATE INDEX IF NOT EXISTS rag_child_chunks_fts_idx
    ON rag_child_chunks USING GIN (to_tsvector('english', text));

-- HNSW index for text vector search (cosine distance)
CREATE INDEX IF NOT EXISTS rag_child_chunks_embedding_hnsw_idx
    ON rag_child_chunks USING hnsw (embedding_1024 vector_cosine_ops)
    WITH (m = 16, ef_construction = 100);

-- HNSW index for image vector search (cosine distance)
CREATE INDEX IF NOT EXISTS rag_child_chunks_image_embedding_hnsw_idx
    ON rag_child_chunks USING hnsw (image_embedding_1024 vector_cosine_ops)
    WITH (m = 16, ef_construction = 100)
    WHERE image_embedding_1024 IS NOT NULL;

-- ═══════════════════════════════════════════════════════════════════════════════
-- ENTITIES (Knowledge Graph Nodes)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS rag_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Unique constraint: one canonical entity per tenant
CREATE UNIQUE INDEX IF NOT EXISTS rag_entities_tenant_canonical_uq
    ON rag_entities (tenant_id, canonical_name);

-- Index for type filtering
CREATE INDEX IF NOT EXISTS rag_entities_type_idx
    ON rag_entities (tenant_id, entity_type);

-- Trigram index for fuzzy name search
CREATE INDEX IF NOT EXISTS rag_entities_name_trgm_idx
    ON rag_entities USING GIN (name gin_trgm_ops);

-- ═══════════════════════════════════════════════════════════════════════════════
-- RELATIONS (Knowledge Graph Edges)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS rag_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    subject_entity_id UUID NOT NULL REFERENCES rag_entities(id) ON DELETE CASCADE,
    object_entity_id UUID NOT NULL REFERENCES rag_entities(id) ON DELETE CASCADE,
    confidence REAL DEFAULT 1.0,
    source_parent_id UUID REFERENCES rag_parent_chunks(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Index for entity lookup
CREATE INDEX IF NOT EXISTS rag_relations_subject_idx
    ON rag_relations (subject_entity_id);

CREATE INDEX IF NOT EXISTS rag_relations_object_idx
    ON rag_relations (object_entity_id);

-- Index for relation type filtering
CREATE INDEX IF NOT EXISTS rag_relations_type_idx
    ON rag_relations (tenant_id, relation_type);

-- ═══════════════════════════════════════════════════════════════════════════════
-- ENTITY MENTIONS (Links Entities to Chunks)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS rag_entity_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES rag_entities(id) ON DELETE CASCADE,
    parent_chunk_id UUID REFERENCES rag_parent_chunks(id) ON DELETE SET NULL,
    child_chunk_id UUID REFERENCES rag_child_chunks(id) ON DELETE SET NULL,
    document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
    mention_text TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    char_start INT,
    char_end INT,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Index for entity lookup
CREATE INDEX IF NOT EXISTS rag_entity_mentions_entity_idx
    ON rag_entity_mentions (entity_id);

-- Index for chunk lookup
CREATE INDEX IF NOT EXISTS rag_entity_mentions_chunk_idx
    ON rag_entity_mentions (child_chunk_id);

-- ═══════════════════════════════════════════════════════════════════════════════
-- RETRIEVAL FUNCTIONS (RPCs)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Lexical (BM25-style) search using ts_rank_cd
CREATE OR REPLACE FUNCTION rag_lexical_search(
    p_tenant_id TEXT,
    p_query TEXT,
    p_limit INT DEFAULT 50,
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
)
LANGUAGE SQL
STABLE
AS $$
    SELECT
        c.id AS child_id,
        c.parent_id,
        c.document_id,
        c.text,
        c.page,
        c.modality,
        ts_rank_cd(to_tsvector('english', c.text), plainto_tsquery('english', p_query)) AS rank
    FROM rag_child_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.tenant_id = p_tenant_id
      AND (p_collection IS NULL OR d.collection = p_collection)
      AND to_tsvector('english', c.text) @@ plainto_tsquery('english', p_query)
    ORDER BY rank DESC
    LIMIT p_limit;
$$;

-- Semantic (vector) search using HNSW
CREATE OR REPLACE FUNCTION rag_semantic_search(
    p_tenant_id TEXT,
    p_embedding VECTOR(1024),
    p_limit INT DEFAULT 100,
    p_collection TEXT DEFAULT NULL
)
RETURNS TABLE (
    child_id UUID,
    parent_id UUID,
    document_id UUID,
    text TEXT,
    page INT,
    modality TEXT,
    similarity REAL
)
LANGUAGE SQL
STABLE
AS $$
    SELECT
        c.id AS child_id,
        c.parent_id,
        c.document_id,
        c.text,
        c.page,
        c.modality,
        (1 - (c.embedding_1024 <=> p_embedding))::REAL AS similarity
    FROM rag_child_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.tenant_id = p_tenant_id
      AND c.embedding_1024 IS NOT NULL
      AND (p_collection IS NULL OR d.collection = p_collection)
    ORDER BY c.embedding_1024 <=> p_embedding
    LIMIT p_limit;
$$;

-- Image semantic search (multimodal)
CREATE OR REPLACE FUNCTION rag_image_semantic_search(
    p_tenant_id TEXT,
    p_embedding VECTOR(1024),
    p_limit INT DEFAULT 50,
    p_collection TEXT DEFAULT NULL
)
RETURNS TABLE (
    child_id UUID,
    parent_id UUID,
    document_id UUID,
    text TEXT,
    page INT,
    modality TEXT,
    similarity REAL
)
LANGUAGE SQL
STABLE
AS $$
    SELECT
        c.id AS child_id,
        c.parent_id,
        c.document_id,
        c.text,
        c.page,
        c.modality,
        (1 - (c.image_embedding_1024 <=> p_embedding))::REAL AS similarity
    FROM rag_child_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    WHERE c.tenant_id = p_tenant_id
      AND c.image_embedding_1024 IS NOT NULL
      AND (p_collection IS NULL OR d.collection = p_collection)
    ORDER BY c.image_embedding_1024 <=> p_embedding
    LIMIT p_limit;
$$;

-- Get parent chunk with all children
CREATE OR REPLACE FUNCTION rag_get_parent_with_children(
    p_parent_id UUID
)
RETURNS TABLE (
    parent_id UUID,
    parent_text TEXT,
    parent_page_start INT,
    parent_page_end INT,
    parent_section TEXT,
    child_id UUID,
    child_text TEXT,
    child_page INT,
    child_modality TEXT
)
LANGUAGE SQL
STABLE
AS $$
    SELECT
        p.id AS parent_id,
        p.text AS parent_text,
        p.page_start AS parent_page_start,
        p.page_end AS parent_page_end,
        p.section_heading AS parent_section,
        c.id AS child_id,
        c.text AS child_text,
        c.page AS child_page,
        c.modality AS child_modality
    FROM rag_parent_chunks p
    LEFT JOIN rag_child_chunks c ON c.parent_id = p.id
    WHERE p.id = p_parent_id
    ORDER BY c.index_in_parent;
$$;

-- ═══════════════════════════════════════════════════════════════════════════════
-- TRIGGERS (Auto-update timestamps)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc', NOW());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_rag_documents_updated_at
    BEFORE UPDATE ON rag_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
