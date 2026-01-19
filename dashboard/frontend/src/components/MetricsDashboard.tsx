import type { MetricsResponse } from '../types';

interface MetricsDashboardProps {
    metrics: MetricsResponse | null;
    onRefresh: () => void;
}

export function MetricsDashboard({ metrics, onRefresh }: MetricsDashboardProps) {
    if (!metrics) {
        return (
            <div className="animate-fadeIn">
                <div className="page-header">
                    <h2>Dashboard</h2>
                    <p>Loading metrics...</p>
                </div>
                <div className="spinner" style={{ margin: '2rem auto' }} />
            </div>
        );
    }

    const { database, config, ingestion_jobs } = metrics;

    return (
        <div className="animate-fadeIn">
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                    <h2>Dashboard</h2>
                    <p>Overview of your Triple-Hybrid-RAG system</p>
                </div>
                <button className="btn btn-secondary" onClick={onRefresh}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="23 4 23 10 17 10" />
                        <polyline points="1 20 1 14 7 14" />
                        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                    </svg>
                    Refresh
                </button>
            </div>

            {/* Database Stats */}
            <h3 style={{ marginBottom: 'var(--space-4)' }}>Database Statistics</h3>
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                        </svg>
                    </div>
                    <div className="stat-label">Documents</div>
                    <div className="stat-value">{database.documents.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                            <line x1="9" y1="3" x2="9" y2="21" />
                        </svg>
                    </div>
                    <div className="stat-label">Parent Chunks</div>
                    <div className="stat-value">{database.parent_chunks.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                            <line x1="9" y1="3" x2="9" y2="21" />
                            <line x1="15" y1="3" x2="15" y2="21" />
                        </svg>
                    </div>
                    <div className="stat-label">Child Chunks</div>
                    <div className="stat-value">{database.child_chunks.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <circle cx="12" cy="12" r="3" />
                        </svg>
                    </div>
                    <div className="stat-label">Entities</div>
                    <div className="stat-value">{database.entities.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="20" x2="18" y2="10" />
                            <line x1="12" y1="20" x2="12" y2="4" />
                            <line x1="6" y1="20" x2="6" y2="14" />
                        </svg>
                    </div>
                    <div className="stat-label">Relations</div>
                    <div className="stat-value">{database.relations.toLocaleString()}</div>
                </div>
            </div>

            {/* Feature Status */}
            <h3 style={{ marginBottom: 'var(--space-4)', marginTop: 'var(--space-8)' }}>Feature Status</h3>
            <div className="card">
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 'var(--space-4)' }}>
                    {[
                        { label: 'RAG Enabled', active: config.rag_enabled },
                        { label: 'Lexical Search', active: config.lexical_enabled },
                        { label: 'Semantic Search', active: config.semantic_enabled },
                        { label: 'Graph Search', active: config.graph_enabled },
                        { label: 'Reranking', active: config.rerank_enabled },
                        { label: 'HyDE', active: config.hyde_enabled },
                        { label: 'Query Expansion', active: config.query_expansion_enabled },
                        { label: 'Diversity', active: config.diversity_enabled },
                        { label: 'OCR', active: config.ocr_enabled },
                    ].map((feature) => (
                        <div
                            key={feature.label}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 'var(--space-3)',
                                padding: 'var(--space-3)',
                                background: feature.active ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                                borderRadius: 'var(--radius-md)',
                            }}
                        >
                            <span
                                style={{
                                    width: 10,
                                    height: 10,
                                    borderRadius: 'var(--radius-full)',
                                    background: feature.active ? 'var(--color-success)' : 'var(--color-error)',
                                }}
                            />
                            <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-primary)' }}>
                                {feature.label}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Ingestion Jobs Summary */}
            <h3 style={{ marginBottom: 'var(--space-4)', marginTop: 'var(--space-8)' }}>Ingestion Jobs</h3>
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-label">Total</div>
                    <div className="stat-value">{ingestion_jobs.total}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Pending</div>
                    <div className="stat-value" style={{ color: 'var(--color-warning)' }}>{ingestion_jobs.pending}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Processing</div>
                    <div className="stat-value" style={{ color: 'var(--color-info)' }}>{ingestion_jobs.processing}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Completed</div>
                    <div className="stat-value" style={{ color: 'var(--color-success)' }}>{ingestion_jobs.completed}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Failed</div>
                    <div className="stat-value" style={{ color: 'var(--color-error)' }}>{ingestion_jobs.failed}</div>
                </div>
            </div>

            {/* Quick Stats */}
            <h3 style={{ marginBottom: 'var(--space-4)', marginTop: 'var(--space-8)' }}>System Info</h3>
            <div className="card">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                    <div>
                        <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)', marginBottom: 'var(--space-1)' }}>
                            Average Chunks per Document
                        </div>
                        <div style={{ fontSize: 'var(--text-lg)', fontWeight: 600 }}>
                            {database.documents > 0
                                ? Math.round(database.child_chunks / database.documents).toLocaleString()
                                : '—'}
                        </div>
                    </div>
                    <div>
                        <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)', marginBottom: 'var(--space-1)' }}>
                            Average Entities per Document
                        </div>
                        <div style={{ fontSize: 'var(--text-lg)', fontWeight: 600 }}>
                            {database.documents > 0
                                ? Math.round(database.entities / database.documents).toLocaleString()
                                : '—'}
                        </div>
                    </div>
                    <div>
                        <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)', marginBottom: 'var(--space-1)' }}>
                            Child/Parent Ratio
                        </div>
                        <div style={{ fontSize: 'var(--text-lg)', fontWeight: 600 }}>
                            {database.parent_chunks > 0
                                ? (database.child_chunks / database.parent_chunks).toFixed(2)
                                : '—'}
                        </div>
                    </div>
                    <div>
                        <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)', marginBottom: 'var(--space-1)' }}>
                            Relations per Entity
                        </div>
                        <div style={{ fontSize: 'var(--text-lg)', fontWeight: 600 }}>
                            {database.entities > 0
                                ? (database.relations / database.entities).toFixed(2)
                                : '—'}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
