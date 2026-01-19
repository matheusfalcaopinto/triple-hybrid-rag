/**
 * API Hooks for Triple-Hybrid-RAG Dashboard
 */

import { useState, useCallback } from 'react';

// Dynamic API base URL - uses the same host as the frontend
// When using Vite proxy, use relative path; otherwise use the full URL
const getApiBase = () => {
  // In development with Vite proxy, use relative path
  if (import.meta.env.DEV) {
    return '/api';
  }
  // In production, use HTTPS (same protocol as frontend)
  const hostname = window.location.hostname;
  return `https://${hostname}:8009/api`;
};

const API_BASE = getApiBase();

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface IngestionStage {
    name: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    items_processed: number;
    error?: string;
}

export interface IngestionJob {
    job_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    file_name: string;
    file_type: string;
    progress: number;
    stages: IngestionStage[];
    result?: {
        document_id: string;
        file_type: string;
        pages: number;
        pages_ocr: number;
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
    channels: {
        semantic: number;
        lexical: number;
        graph: number;
    };
    query_plan?: unknown;
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
    title: string;
    status: string;
    chunk_count: number;
    created_at: string;
    has_file?: boolean;
    download_url?: string;
}

export interface Entity {
    id: string;
    name: string;
    entity_type: string;
    mention_count: number;
}

export interface ChunkDetail {
    id: string;
    parent_id?: string;
    index: number;
    text: string;
    token_count: number;
    page?: number;
    modality?: string;
}

export interface ParentChunkDetail {
    id: string;
    index: number;
    text: string;
    token_count: number;
    page_start?: number;
    section_heading?: string;
}

export interface EntityMention {
    chunk_id: string;
    start_char?: number;
    end_char?: number;
    confidence?: number;
}

export interface EntityDetail {
    id: string;
    name: string;
    entity_type: string;
    mentions: EntityMention[];
}

export interface RelationDetail {
    id: string;
    source: {
        id: string;
        name: string;
        type: string;
    };
    target: {
        id: string;
        name: string;
        type: string;
    };
    relation_type: string;
    confidence?: number;
}

export interface DocumentDetails {
    document: {
        id: string;
        tenant_id: string;
        file_name: string;
        collection: string;
        title: string;
        status: string;
        created_at: string;
    };
    parent_chunks: ParentChunkDetail[];
    child_chunks: ChunkDetail[];
    entities: EntityDetail[];
    relations: RelationDetail[];
    stats: {
        parent_chunks: number;
        child_chunks: number;
        entities: number;
        relations: number;
    };
}

export interface ConfigItem {
    value: unknown;
    category: string;
    type: string;
    description: string;
    min?: number;
    max?: number;
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

// ═══════════════════════════════════════════════════════════════════════════════
// GENERIC FETCH HOOK
// ═══════════════════════════════════════════════════════════════════════════════

interface UseFetchState<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
}

async function fetchApi<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
        ...options,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    return response.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// INGESTION HOOKS
// ═══════════════════════════════════════════════════════════════════════════════

export function useIngestion() {
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const uploadFile = useCallback(async (file: File): Promise<{ job_id: string } | null> => {
        setUploading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE}/ingest/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Upload failed: ${response.status}`);
            }

            return await response.json();
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Upload failed';
            setError(message);
            return null;
        } finally {
            setUploading(false);
        }
    }, []);

    const getJobStatus = useCallback(async (jobId: string): Promise<IngestionJob | null> => {
        try {
            return await fetchApi<IngestionJob>(`/ingest/status/${jobId}`);
        } catch {
            return null;
        }
    }, []);

    const listJobs = useCallback(async (): Promise<IngestionJob[]> => {
        try {
            const response = await fetchApi<{ jobs: IngestionJob[] }>('/ingest/jobs');
            return response.jobs;
        } catch {
            return [];
        }
    }, []);

    return {
        uploading,
        error,
        uploadFile,
        getJobStatus,
        listJobs,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// RETRIEVAL HOOK
// ═══════════════════════════════════════════════════════════════════════════════

export function useRetrieval() {
    const [state, setState] = useState<UseFetchState<RetrievalResponse>>({
        data: null,
        loading: false,
        error: null,
    });

    const query = useCallback(async (
        queryText: string,
        options?: { tenant_id?: string; collection?: string; top_k?: number }
    ) => {
        setState({ data: null, loading: true, error: null });

        try {
            const data = await fetchApi<RetrievalResponse>('/retrieve', {
                method: 'POST',
                body: JSON.stringify({
                    query: queryText,
                    tenant_id: options?.tenant_id || 'default',
                    collection: options?.collection || null,
                    top_k: options?.top_k || null,
                }),
            });
            setState({ data, loading: false, error: null });
            return data;
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Query failed';
            setState({ data: null, loading: false, error: message });
            return null;
        }
    }, []);

    return {
        ...state,
        query,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATABASE HOOK
// ═══════════════════════════════════════════════════════════════════════════════

export function useDatabase() {
    const [loading, setLoading] = useState(false);

    const getStats = useCallback(async (): Promise<DatabaseStats | null> => {
        setLoading(true);
        try {
            return await fetchApi<DatabaseStats>('/database/stats');
        } catch {
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    const listDocuments = useCallback(async (
        limit = 50,
        offset = 0
    ): Promise<{ documents: Document[] } | null> => {
        setLoading(true);
        try {
            return await fetchApi<{ documents: Document[] }>(
                `/database/documents?limit=${limit}&offset=${offset}`
            );
        } catch {
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    const listEntities = useCallback(async (
        limit = 50,
        offset = 0,
        entityType?: string
    ): Promise<{ entities: Entity[] } | null> => {
        setLoading(true);
        try {
            const typeParam = entityType ? `&entity_type=${entityType}` : '';
            return await fetchApi<{ entities: Entity[] }>(
                `/database/entities?limit=${limit}&offset=${offset}${typeParam}`
            );
        } catch {
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    const deleteDocument = useCallback(async (documentId: string): Promise<{ status: string; error?: string } | null> => {
        try {
            return await fetchApi<{ status: string; error?: string }>(
                `/database/documents/${documentId}`,
                { method: 'DELETE' }
            );
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Delete failed';
            return { status: 'error', error: message };
        }
    }, []);

    const getDocumentDetails = useCallback(async (documentId: string): Promise<DocumentDetails | null> => {
        setLoading(true);
        try {
            return await fetchApi<DocumentDetails>(`/database/documents/${documentId}/details`);
        } catch {
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    return {
        loading,
        getStats,
        listDocuments,
        listEntities,
        deleteDocument,
        getDocumentDetails,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIG HOOK
// ═══════════════════════════════════════════════════════════════════════════════

export function useConfig() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const getConfig = useCallback(async (): Promise<{
        config: Record<string, ConfigItem>;
        categories: string[];
    } | null> => {
        setLoading(true);
        setError(null);
        try {
            return await fetchApi<{
                config: Record<string, ConfigItem>;
                categories: string[];
            }>('/config');
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load config';
            setError(message);
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    const updateConfig = useCallback(async (
        updates: Record<string, unknown>
    ): Promise<boolean> => {
        setLoading(true);
        setError(null);
        try {
            await fetchApi('/config', {
                method: 'POST',
                body: JSON.stringify({ updates }),
            });
            return true;
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to update config';
            setError(message);
            return false;
        } finally {
            setLoading(false);
        }
    }, []);

    const reloadConfig = useCallback(async (): Promise<boolean> => {
        try {
            await fetchApi('/config/reload', { method: 'POST' });
            return true;
        } catch {
            return false;
        }
    }, []);

    return {
        loading,
        error,
        getConfig,
        updateConfig,
        reloadConfig,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRICS HOOK
// ═══════════════════════════════════════════════════════════════════════════════

export function useMetrics() {
    const [state, setState] = useState<UseFetchState<MetricsResponse>>({
        data: null,
        loading: false,
        error: null,
    });

    const fetchMetrics = useCallback(async () => {
        setState(s => ({ ...s, loading: true, error: null }));
        try {
            const data = await fetchApi<MetricsResponse>('/metrics');
            setState({ data, loading: false, error: null });
            return data;
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load metrics';
            setState({ data: null, loading: false, error: message });
            return null;
        }
    }, []);

    return {
        ...state,
        fetchMetrics,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEALTH CHECK
// ═══════════════════════════════════════════════════════════════════════════════

export async function checkHealth(): Promise<boolean> {
    try {
        await fetchApi<{ status: string }>('/health');
        return true;
    } catch {
        return false;
    }
}
