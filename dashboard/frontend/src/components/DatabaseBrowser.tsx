import { useState, useEffect } from 'react';
import { useDatabase } from '../hooks/useApi';
import type { Document, Entity } from '../types';

type Tab = 'documents' | 'entities';

export function DatabaseBrowser() {
    const [activeTab, setActiveTab] = useState<Tab>('documents');
    const { loading, getStats, listDocuments, listEntities } = useDatabase();
    const [documents, setDocuments] = useState<Document[]>([]);
    const [entities, setEntities] = useState<Entity[]>([]);
    const [stats, setStats] = useState<any>(null);
    const [entityTypeFilter, setEntityTypeFilter] = useState<string>('');

    useEffect(() => {
        loadData();
    }, [activeTab, entityTypeFilter]);

    const loadData = async () => {
        const statsData = await getStats();
        if (statsData) setStats(statsData);

        if (activeTab === 'documents') {
            const docsData = await listDocuments(50, 0);
            if (docsData?.documents) setDocuments(docsData.documents);
        } else {
            const entitiesData = await listEntities(50, 0, entityTypeFilter || undefined);
            if (entitiesData?.entities) setEntities(entitiesData.entities);
        }
    };

    const entityTypes = [
        'PERSON',
        'ORGANIZATION',
        'PRODUCT',
        'CLAUSE',
        'DATE',
        'MONEY',
        'LOCATION',
        'TECHNICAL_TERM',
    ];

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h2>Database Browser</h2>
                <p>Explore documents, chunks, and entities stored in the database</p>
            </div>

            {/* Stats Summary */}
            {stats && (
                <div className="stats-grid" style={{ marginBottom: 'var(--space-6)' }}>
                    <div className="stat-card" style={{ padding: 'var(--space-3)' }}>
                        <div className="stat-label">Documents</div>
                        <div className="stat-value" style={{ fontSize: 'var(--text-xl)' }}>{stats.documents}</div>
                    </div>
                    <div className="stat-card" style={{ padding: 'var(--space-3)' }}>
                        <div className="stat-label">Chunks</div>
                        <div className="stat-value" style={{ fontSize: 'var(--text-xl)' }}>{stats.child_chunks}</div>
                    </div>
                    <div className="stat-card" style={{ padding: 'var(--space-3)' }}>
                        <div className="stat-label">Entities</div>
                        <div className="stat-value" style={{ fontSize: 'var(--text-xl)' }}>{stats.entities}</div>
                    </div>
                    <div className="stat-card" style={{ padding: 'var(--space-3)' }}>
                        <div className="stat-label">Relations</div>
                        <div className="stat-value" style={{ fontSize: 'var(--text-xl)' }}>{stats.relations}</div>
                    </div>
                </div>
            )}

            {/* Tabs */}
            <div className="tabs">
                <button
                    className={`tab ${activeTab === 'documents' ? 'active' : ''}`}
                    onClick={() => setActiveTab('documents')}
                >
                    Documents
                </button>
                <button
                    className={`tab ${activeTab === 'entities' ? 'active' : ''}`}
                    onClick={() => setActiveTab('entities')}
                >
                    Entities
                </button>
            </div>

            {/* Documents Tab */}
            {activeTab === 'documents' && (
                <div className="card">
                    {loading ? (
                        <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                            <div className="spinner" style={{ margin: '0 auto' }} />
                        </div>
                    ) : documents.length === 0 ? (
                        <div className="empty-state">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                <polyline points="14 2 14 8 20 8" />
                            </svg>
                            <p>No documents found</p>
                        </div>
                    ) : (
                        <div className="table-container">
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>File Name</th>
                                        <th>Collection</th>
                                        <th>Status</th>
                                        <th>Chunks</th>
                                        <th>Created</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {documents.map((doc) => (
                                        <tr key={doc.id}>
                                            <td>
                                                <div style={{ fontWeight: 500 }}>{doc.file_name}</div>
                                                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>
                                                    {doc.id.slice(0, 8)}...
                                                </div>
                                            </td>
                                            <td>
                                                <span className="badge badge-info">{doc.collection}</span>
                                            </td>
                                            <td>
                                                <span className={`badge ${doc.status === 'completed' ? 'badge-success' :
                                                    doc.status === 'failed' ? 'badge-error' :
                                                        'badge-warning'
                                                    }`}>
                                                    {doc.status}
                                                </span>
                                            </td>
                                            <td>{doc.chunk_count}</td>
                                            <td style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)' }}>
                                                {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '—'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}

            {/* Entities Tab */}
            {activeTab === 'entities' && (
                <div>
                    {/* Filter */}
                    <div style={{ marginBottom: 'var(--space-4)' }}>
                        <select
                            className="form-input"
                            value={entityTypeFilter}
                            onChange={(e) => setEntityTypeFilter(e.target.value)}
                            style={{ width: 'auto', minWidth: '200px' }}
                        >
                            <option value="">All Entity Types</option>
                            {entityTypes.map((type) => (
                                <option key={type} value={type}>{type}</option>
                            ))}
                        </select>
                    </div>

                    <div className="card">
                        {loading ? (
                            <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                                <div className="spinner" style={{ margin: '0 auto' }} />
                            </div>
                        ) : entities.length === 0 ? (
                            <div className="empty-state">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="10" />
                                    <circle cx="12" cy="12" r="3" />
                                </svg>
                                <p>No entities found</p>
                            </div>
                        ) : (
                            <div className="table-container">
                                <table className="table">
                                    <thead>
                                        <tr>
                                            <th>Name</th>
                                            <th>Type</th>
                                            <th>Mentions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {entities.map((entity) => (
                                            <tr key={entity.id}>
                                                <td style={{ fontWeight: 500 }}>{entity.name}</td>
                                                <td>
                                                    <span className={`badge ${entity.entity_type === 'PERSON' ? 'badge-info' :
                                                        entity.entity_type === 'ORGANIZATION' ? 'badge-success' :
                                                            entity.entity_type === 'LOCATION' ? 'badge-warning' :
                                                                ''
                                                        }`}>
                                                        {entity.entity_type}
                                                    </span>
                                                </td>
                                                <td>{entity.mention_count}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
