import { useState, useEffect } from 'react';
import { useDatabase } from '../hooks/useApi';
import type { Document, Entity, DocumentDetails, ChunkDetail, EntityDetail, RelationDetail } from '../hooks/useApi';

type Tab = 'documents' | 'entities';
type DetailTab = 'chunks' | 'entities' | 'relations';

export function DatabaseBrowser() {
    const [activeTab, setActiveTab] = useState<Tab>('documents');
    const { loading, getStats, listDocuments, listEntities, deleteDocument, getDocumentDetails } = useDatabase();
    const [documents, setDocuments] = useState<Document[]>([]);
    const [entities, setEntities] = useState<Entity[]>([]);
    const [stats, setStats] = useState<any>(null);
    const [entityTypeFilter, setEntityTypeFilter] = useState<string>('');
    const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
    const [deleting, setDeleting] = useState<string | null>(null);
    
    // Document detail view state
    const [expandedDocId, setExpandedDocId] = useState<string | null>(null);
    const [documentDetails, setDocumentDetails] = useState<DocumentDetails | null>(null);
    const [detailsLoading, setDetailsLoading] = useState(false);
    const [activeDetailTab, setActiveDetailTab] = useState<DetailTab>('chunks');

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
        setSelectedDocs(new Set());
    };

    const handleSelectDoc = (docId: string) => {
        const newSelection = new Set(selectedDocs);
        if (newSelection.has(docId)) {
            newSelection.delete(docId);
        } else {
            newSelection.add(docId);
        }
        setSelectedDocs(newSelection);
    };

    const handleSelectAll = () => {
        if (selectedDocs.size === documents.length) {
            setSelectedDocs(new Set());
        } else {
            setSelectedDocs(new Set(documents.map(d => d.id)));
        }
    };

    const handleDeleteSingle = async (docId: string) => {
        if (!confirm('Are you sure you want to delete this document and all related data?')) {
            return;
        }
        setDeleting(docId);
        try {
            const result = await deleteDocument(docId);
            if (result?.status === 'deleted') {
                if (expandedDocId === docId) {
                    setExpandedDocId(null);
                    setDocumentDetails(null);
                }
                await loadData();
            } else {
                alert(`Failed to delete document: ${result?.error || 'Unknown error'}`);
            }
        } catch (e: any) {
            alert(`Error: ${e.message}`);
        } finally {
            setDeleting(null);
        }
    };

    const handleDeleteSelected = async () => {
        if (selectedDocs.size === 0) return;
        if (!confirm(`Are you sure you want to delete ${selectedDocs.size} document(s) and all related data?`)) {
            return;
        }
        setDeleting('batch');
        try {
            for (const docId of selectedDocs) {
                await deleteDocument(docId);
            }
            setExpandedDocId(null);
            setDocumentDetails(null);
            await loadData();
        } catch (e: any) {
            alert(`Error: ${e.message}`);
        } finally {
            setDeleting(null);
        }
    };

    const handleExpandDocument = async (docId: string) => {
        if (expandedDocId === docId) {
            // Collapse
            setExpandedDocId(null);
            setDocumentDetails(null);
        } else {
            // Expand and load details
            setExpandedDocId(docId);
            setDetailsLoading(true);
            setActiveDetailTab('chunks');
            try {
                const details = await getDocumentDetails(docId);
                setDocumentDetails(details);
            } catch (e) {
                console.error('Failed to load document details:', e);
                setDocumentDetails(null);
            } finally {
                setDetailsLoading(false);
            }
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

    const getEntityTypeBadgeClass = (type: string) => {
        switch (type) {
            case 'PERSON': return 'badge-info';
            case 'ORGANIZATION': return 'badge-success';
            case 'LOCATION': return 'badge-warning';
            case 'DATE': return 'badge-secondary';
            case 'MONEY': return 'badge-accent';
            default: return '';
        }
    };

    // Render document detail card
    const renderDocumentDetails = () => {
        if (!expandedDocId) return null;

        return (
            <div className="card" style={{ 
                marginTop: 'var(--space-4)', 
                borderLeft: '4px solid var(--color-accent)',
                animation: 'fadeIn 0.2s ease-out'
            }}>
                {detailsLoading ? (
                    <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                        <div className="spinner" style={{ margin: '0 auto' }} />
                        <p style={{ marginTop: 'var(--space-3)', color: 'var(--color-text-secondary)' }}>
                            Loading document details...
                        </p>
                    </div>
                ) : documentDetails ? (
                    <>
                        {/* Header */}
                        <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            marginBottom: 'var(--space-4)',
                            paddingBottom: 'var(--space-3)',
                            borderBottom: '1px solid var(--color-border)'
                        }}>
                            <div>
                                <h3 style={{ margin: 0, fontSize: 'var(--text-lg)' }}>
                                    📄 {documentDetails.document.file_name}
                                </h3>
                                <p style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)' }}>
                                    {documentDetails.document.title} • {documentDetails.document.collection}
                                </p>
                            </div>
                            <button 
                                className="btn btn-secondary btn-sm"
                                onClick={() => { setExpandedDocId(null); setDocumentDetails(null); }}
                            >
                                ✕ Close
                            </button>
                        </div>

                        {/* Stats */}
                        <div className="stats-grid" style={{ marginBottom: 'var(--space-4)', gridTemplateColumns: 'repeat(4, 1fr)' }}>
                            <div className="stat-card" style={{ padding: 'var(--space-2)' }}>
                                <div className="stat-label" style={{ fontSize: 'var(--text-xs)' }}>Parent Chunks</div>
                                <div className="stat-value" style={{ fontSize: 'var(--text-lg)' }}>{documentDetails.stats.parent_chunks}</div>
                            </div>
                            <div className="stat-card" style={{ padding: 'var(--space-2)' }}>
                                <div className="stat-label" style={{ fontSize: 'var(--text-xs)' }}>Child Chunks</div>
                                <div className="stat-value" style={{ fontSize: 'var(--text-lg)' }}>{documentDetails.stats.child_chunks}</div>
                            </div>
                            <div className="stat-card" style={{ padding: 'var(--space-2)' }}>
                                <div className="stat-label" style={{ fontSize: 'var(--text-xs)' }}>Entities</div>
                                <div className="stat-value" style={{ fontSize: 'var(--text-lg)' }}>{documentDetails.stats.entities}</div>
                            </div>
                            <div className="stat-card" style={{ padding: 'var(--space-2)' }}>
                                <div className="stat-label" style={{ fontSize: 'var(--text-xs)' }}>Relations</div>
                                <div className="stat-value" style={{ fontSize: 'var(--text-lg)' }}>{documentDetails.stats.relations}</div>
                            </div>
                        </div>

                        {/* Detail Tabs */}
                        <div className="tabs" style={{ marginBottom: 'var(--space-3)' }}>
                            <button
                                className={`tab ${activeDetailTab === 'chunks' ? 'active' : ''}`}
                                onClick={() => setActiveDetailTab('chunks')}
                            >
                                📝 Chunks ({documentDetails.stats.child_chunks})
                            </button>
                            <button
                                className={`tab ${activeDetailTab === 'entities' ? 'active' : ''}`}
                                onClick={() => setActiveDetailTab('entities')}
                            >
                                🏷️ Entities ({documentDetails.stats.entities})
                            </button>
                            <button
                                className={`tab ${activeDetailTab === 'relations' ? 'active' : ''}`}
                                onClick={() => setActiveDetailTab('relations')}
                            >
                                🔗 Relations ({documentDetails.stats.relations})
                            </button>
                        </div>

                        {/* Chunks Tab */}
                        {activeDetailTab === 'chunks' && (
                            <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
                                {documentDetails.child_chunks.length === 0 ? (
                                    <p style={{ color: 'var(--color-text-tertiary)', textAlign: 'center', padding: 'var(--space-4)' }}>
                                        No chunks found
                                    </p>
                                ) : (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                                        {documentDetails.child_chunks.map((chunk: ChunkDetail, index: number) => (
                                            <div 
                                                key={chunk.id} 
                                                style={{ 
                                                    padding: 'var(--space-3)',
                                                    backgroundColor: 'var(--color-bg-tertiary)',
                                                    borderRadius: 'var(--radius-md)',
                                                    border: '1px solid var(--color-border)'
                                                }}
                                            >
                                                <div style={{ 
                                                    display: 'flex', 
                                                    justifyContent: 'space-between', 
                                                    alignItems: 'center',
                                                    marginBottom: 'var(--space-2)'
                                                }}>
                                                    <span style={{ 
                                                        fontWeight: 600, 
                                                        fontSize: 'var(--text-sm)',
                                                        color: 'var(--color-accent)'
                                                    }}>
                                                        Chunk #{index + 1}
                                                    </span>
                                                    <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                                                        {chunk.page && (
                                                            <span className="badge badge-secondary" style={{ fontSize: 'var(--text-xs)' }}>
                                                                Page {chunk.page}
                                                            </span>
                                                        )}
                                                        <span className="badge" style={{ fontSize: 'var(--text-xs)' }}>
                                                            {chunk.token_count} tokens
                                                        </span>
                                                        {chunk.modality && (
                                                            <span className="badge badge-info" style={{ fontSize: 'var(--text-xs)' }}>
                                                                {chunk.modality}
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                                <pre style={{ 
                                                    margin: 0,
                                                    padding: 'var(--space-2)',
                                                    backgroundColor: 'var(--color-bg-primary)',
                                                    borderRadius: 'var(--radius-sm)',
                                                    fontSize: 'var(--text-sm)',
                                                    whiteSpace: 'pre-wrap',
                                                    wordBreak: 'break-word',
                                                    maxHeight: '200px',
                                                    overflowY: 'auto',
                                                    fontFamily: 'inherit'
                                                }}>
                                                    {chunk.text}
                                                </pre>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Entities Tab */}
                        {activeDetailTab === 'entities' && (
                            <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
                                {documentDetails.entities.length === 0 ? (
                                    <p style={{ color: 'var(--color-text-tertiary)', textAlign: 'center', padding: 'var(--space-4)' }}>
                                        No entities extracted. Enable entity extraction in configuration.
                                    </p>
                                ) : (
                                    <div className="table-container">
                                        <table className="table">
                                            <thead>
                                                <tr>
                                                    <th>Entity</th>
                                                    <th>Type</th>
                                                    <th>Mentions</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {documentDetails.entities.map((entity: EntityDetail) => (
                                                    <tr key={entity.id}>
                                                        <td style={{ fontWeight: 500 }}>{entity.name}</td>
                                                        <td>
                                                            <span className={`badge ${getEntityTypeBadgeClass(entity.entity_type)}`}>
                                                                {entity.entity_type}
                                                            </span>
                                                        </td>
                                                        <td>
                                                            <span className="badge badge-secondary">
                                                                {entity.mentions.length} mention{entity.mentions.length !== 1 ? 's' : ''}
                                                            </span>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Relations Tab */}
                        {activeDetailTab === 'relations' && (
                            <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
                                {documentDetails.relations.length === 0 ? (
                                    <p style={{ color: 'var(--color-text-tertiary)', textAlign: 'center', padding: 'var(--space-4)' }}>
                                        No relations extracted. Enable entity extraction in configuration.
                                    </p>
                                ) : (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                                        {documentDetails.relations.map((relation: RelationDetail) => (
                                            <div 
                                                key={relation.id}
                                                style={{ 
                                                    padding: 'var(--space-3)',
                                                    backgroundColor: 'var(--color-bg-tertiary)',
                                                    borderRadius: 'var(--radius-md)',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: 'var(--space-3)',
                                                    flexWrap: 'wrap'
                                                }}
                                            >
                                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                                    <span className={`badge ${getEntityTypeBadgeClass(relation.source.type)}`} style={{ fontSize: 'var(--text-xs)' }}>
                                                        {relation.source.type}
                                                    </span>
                                                    <strong>{relation.source.name}</strong>
                                                </div>
                                                <span style={{ 
                                                    color: 'var(--color-accent)', 
                                                    fontWeight: 600,
                                                    padding: '2px 8px',
                                                    backgroundColor: 'var(--color-bg-primary)',
                                                    borderRadius: 'var(--radius-sm)',
                                                    fontSize: 'var(--text-sm)'
                                                }}>
                                                    → {relation.relation_type} →
                                                </span>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                                    <span className={`badge ${getEntityTypeBadgeClass(relation.target.type)}`} style={{ fontSize: 'var(--text-xs)' }}>
                                                        {relation.target.type}
                                                    </span>
                                                    <strong>{relation.target.name}</strong>
                                                </div>
                                                {relation.confidence && (
                                                    <span style={{ 
                                                        marginLeft: 'auto', 
                                                        fontSize: 'var(--text-xs)', 
                                                        color: 'var(--color-text-tertiary)' 
                                                    }}>
                                                        {(relation.confidence * 100).toFixed(0)}% confidence
                                                    </span>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </>
                ) : (
                    <p style={{ color: 'var(--color-error)', textAlign: 'center', padding: 'var(--space-4)' }}>
                        Failed to load document details
                    </p>
                )}
            </div>
        );
    };

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
                    {/* Bulk Actions */}
                    {documents.length > 0 && (
                        <div style={{ 
                            display: 'flex', 
                            gap: 'var(--space-3)', 
                            marginBottom: 'var(--space-4)',
                            alignItems: 'center',
                            padding: 'var(--space-3)',
                            backgroundColor: 'var(--color-bg-tertiary)',
                            borderRadius: 'var(--radius-md)'
                        }}>
                            <button 
                                className="btn btn-secondary btn-sm"
                                onClick={loadData}
                                disabled={loading}
                            >
                                ↻ Refresh
                            </button>
                            {selectedDocs.size > 0 && (
                                <button 
                                    className="btn btn-sm"
                                    style={{ backgroundColor: 'var(--color-error)', color: 'white' }}
                                    onClick={handleDeleteSelected}
                                    disabled={deleting !== null}
                                >
                                    {deleting === 'batch' ? '⏳ Deleting...' : `🗑 Delete ${selectedDocs.size} selected`}
                                </button>
                            )}
                            <span style={{ marginLeft: 'auto', fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)' }}>
                                {selectedDocs.size} of {documents.length} selected
                            </span>
                        </div>
                    )}

                    {loading && !detailsLoading ? (
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
                                        <th style={{ width: '40px' }}>
                                            <input 
                                                type="checkbox" 
                                                checked={selectedDocs.size === documents.length && documents.length > 0}
                                                onChange={handleSelectAll}
                                            />
                                        </th>
                                        <th>File Name</th>
                                        <th>Collection</th>
                                        <th>Status</th>
                                        <th>Chunks</th>
                                        <th>Created</th>
                                        <th style={{ width: '180px' }}>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {documents.map((doc) => (
                                        <tr 
                                            key={doc.id}
                                            style={{ 
                                                backgroundColor: expandedDocId === doc.id ? 'var(--color-bg-tertiary)' : undefined,
                                                cursor: 'pointer'
                                            }}
                                        >
                                            <td onClick={(e) => e.stopPropagation()}>
                                                <input 
                                                    type="checkbox" 
                                                    checked={selectedDocs.has(doc.id)}
                                                    onChange={() => handleSelectDoc(doc.id)}
                                                />
                                            </td>
                                            <td onClick={() => handleExpandDocument(doc.id)}>
                                                <div style={{ fontWeight: 500 }}>
                                                    {expandedDocId === doc.id ? '▼' : '▶'} {doc.file_name}
                                                </div>
                                                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>
                                                    {doc.id.slice(0, 8)}...
                                                </div>
                                            </td>
                                            <td onClick={() => handleExpandDocument(doc.id)}>
                                                <span className="badge badge-info">{doc.collection}</span>
                                            </td>
                                            <td onClick={() => handleExpandDocument(doc.id)}>
                                                <span className={`badge ${doc.status === 'completed' ? 'badge-success' :
                                                    doc.status === 'failed' ? 'badge-error' :
                                                        'badge-warning'
                                                    }`}>
                                                    {doc.status}
                                                </span>
                                            </td>
                                            <td onClick={() => handleExpandDocument(doc.id)}>{doc.chunk_count}</td>
                                            <td 
                                                onClick={() => handleExpandDocument(doc.id)}
                                                style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)' }}
                                            >
                                                {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '—'}
                                            </td>
                                            <td onClick={(e) => e.stopPropagation()}>
                                                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                                                    <button 
                                                        className={`btn btn-sm ${expandedDocId === doc.id ? 'btn-primary' : 'btn-secondary'}`}
                                                        onClick={() => handleExpandDocument(doc.id)}
                                                        title="View document details"
                                                    >
                                                        👁
                                                    </button>
                                                    {doc.has_file && (
                                                        <button 
                                                            className="btn btn-sm btn-secondary"
                                                            onClick={() => window.open(`/api/documents/${doc.id}/download`, '_blank')}
                                                            title="Download original file"
                                                        >
                                                            ⬇
                                                        </button>
                                                    )}
                                                    <button 
                                                        className="btn btn-sm"
                                                        style={{ 
                                                            backgroundColor: 'var(--color-error)', 
                                                            color: 'white',
                                                            opacity: deleting === doc.id ? 0.6 : 1
                                                        }}
                                                        onClick={() => handleDeleteSingle(doc.id)}
                                                        disabled={deleting !== null}
                                                        title="Delete document and all related data"
                                                    >
                                                        {deleting === doc.id ? '⏳' : '🗑'}
                                                    </button>
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {/* Document Details Card */}
                    {renderDocumentDetails()}
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
                                                    <span className={`badge ${getEntityTypeBadgeClass(entity.entity_type)}`}>
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
