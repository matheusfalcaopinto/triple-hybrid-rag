-- Migration: Fix vector search function return types
-- Date: 2026-01-13
-- Description: Cast similarity to REAL to match return type

-- Fix vector similarity search function
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
        (1 - (kbc.vector_embedding <=> p_embedding))::REAL AS similarity
    FROM knowledge_base_chunks kbc
    WHERE kbc.org_id = p_org_id
      AND kbc.vector_embedding IS NOT NULL
      AND (p_category IS NULL OR kbc.category = p_category)
      AND (p_source_document IS NULL OR kbc.source_document = p_source_document)
    ORDER BY kbc.vector_embedding <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Fix image search function similarly
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
        (1 - (kbc.vector_image <=> p_image_embedding))::REAL AS similarity
    FROM knowledge_base_chunks kbc
    WHERE kbc.org_id = p_org_id
      AND kbc.vector_image IS NOT NULL
      AND kbc.modality = 'image'
    ORDER BY kbc.vector_image <=> p_image_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Fix hybrid RRF search function
CREATE OR REPLACE FUNCTION kb_chunks_hybrid_rrf_search(
    p_org_id UUID,
    p_embedding vector(1536),
    p_query TEXT,
    p_limit INTEGER DEFAULT 50,
    p_category TEXT DEFAULT NULL,
    p_rrf_k INTEGER DEFAULT 60,
    p_fts_config TEXT DEFAULT 'portuguese'
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
    category TEXT,
    title TEXT,
    vector_rank INTEGER,
    bm25_rank INTEGER,
    rrf_score REAL
) AS $$
DECLARE
    v_ts_query tsquery;
BEGIN
    -- Create tsquery from search text
    v_ts_query := plainto_tsquery(p_fts_config, p_query);
    
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            kbc.id,
            ROW_NUMBER() OVER (ORDER BY kbc.vector_embedding <=> p_embedding) AS rank
        FROM knowledge_base_chunks kbc
        WHERE kbc.org_id = p_org_id
          AND kbc.vector_embedding IS NOT NULL
          AND (p_category IS NULL OR kbc.category = p_category)
        LIMIT p_limit * 2
    ),
    bm25_results AS (
        SELECT 
            kbc.id,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(kbc.fts_vector, v_ts_query) DESC) AS rank
        FROM knowledge_base_chunks kbc
        WHERE kbc.org_id = p_org_id
          AND kbc.fts_vector @@ v_ts_query
          AND (p_category IS NULL OR kbc.category = p_category)
        LIMIT p_limit * 2
    ),
    combined AS (
        SELECT 
            COALESCE(v.id, b.id) AS chunk_id,
            v.rank AS v_rank,
            b.rank AS b_rank
        FROM vector_results v
        FULL OUTER JOIN bm25_results b ON v.id = b.id
    ),
    scored AS (
        SELECT 
            c.chunk_id,
            c.v_rank::INTEGER,
            c.b_rank::INTEGER,
            (
                COALESCE(1.0 / (p_rrf_k + c.v_rank), 0) +
                COALESCE(1.0 / (p_rrf_k + c.b_rank), 0)
            )::REAL AS score
        FROM combined c
    )
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
        kbc.category,
        kbc.title,
        s.v_rank,
        s.b_rank,
        s.score
    FROM scored s
    JOIN knowledge_base_chunks kbc ON kbc.id = s.chunk_id
    ORDER BY s.score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;
