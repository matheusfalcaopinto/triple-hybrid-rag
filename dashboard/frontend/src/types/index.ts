export interface ConfigParameter {
    value: any;
    category: string;
    type: 'boolean' | 'string' | 'integer' | 'float';
    description: string;
    min?: number;
    max?: number;
}

export interface ConfigResponse {
    config: Record<string, ConfigParameter>;
    categories: string[];
}

export interface IngestionJob {
    job_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    file_name: string;
    progress: number;
    stages: Array<{
        name: string;
        status: string;
    }>;
    result?: {
        document_id: string;
        parent_chunks: number;
        child_chunks: number;
        entities: number;
        relations: number;
    };
    error?: string;
    created_at: string;
    updated_at: string;
}

export interface SearchResult {
    chunk_id: string;
    document_id: string;
    text: string;
    parent_text?: string;
    lexical_score: number;
    semantic_score: number;
    graph_score: number;
    rrf_score: number;
    rerank_score?: number;
    final_score: number;
    metadata: Record<string, unknown>;
}

export interface RetrievalResponse {
    query: string;
    results: SearchResult[];
    total_results: number;
    query_plan?: any;
}

export interface DatabaseStats {
    documents: number;
    parent_chunks: number;
    child_chunks: number;
    entities: number;
    relations: number;
    error?: string;
}

export interface Document {
    id: string;
    tenant_id: string;
    file_name: string;
    collection: string;
    title?: string;
    status: string;
    chunk_count: number;
    created_at?: string;
    has_file?: boolean;
    download_url?: string;
}

export interface Entity {
    id: string;
    name: string;
    entity_type: string;
    mention_count: number;
}

export interface MetricsResponse {
    database: DatabaseStats;
    config: {
        rag_enabled: boolean;
        lexical_enabled: boolean;
        semantic_enabled: boolean;
        graph_enabled: boolean;
        rerank_enabled: boolean;
        hyde_enabled: boolean;
        query_expansion_enabled: boolean;
        diversity_enabled: boolean;
        ocr_enabled: boolean;
        ocr_mode: string;
    };
    ingestion_jobs: {
        total: number;
        pending: number;
        processing: number;
        completed: number;
        failed: number;
    };
}

export type Page =
    | 'dashboard'
    | 'config'
    | 'ingestion'
    | 'retrieval'
    | 'database'
    | 'graph';
