import { useState, useCallback } from 'react';
import { useRetrieval } from '../hooks/useApi';
import type { SearchResult } from '../types';

export function QueryInterface() {
    const [query, setQuery] = useState('');
    const { data, loading, error, query: executeQuery } = useRetrieval();

    const handleSubmit = useCallback(
        async (e: React.FormEvent) => {
            e.preventDefault();
            if (!query.trim()) return;
            await executeQuery(query.trim());
        },
        [query, executeQuery]
    );

    const results: SearchResult[] = data?.results ?? [];

    const formatScore = (score: number | undefined | null) => {
        if (score === undefined || score === null) return '—';
        return score.toFixed(4);
    };

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h2>Query Interface</h2>
                <p>Execute retrieval queries against the RAG pipeline</p>
            </div>

            {/* Query Input */}
            <form onSubmit={handleSubmit}>
                <div className="query-input-container">
                    <input
                        type="text"
                        className="form-input query-input"
                        placeholder="Enter your query..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                    />
                    <button type="submit" className="btn btn-primary" disabled={loading || !query.trim()}>
                        {loading ? (
                            <>
                                <span className="spinner" style={{ width: 16, height: 16 }} />
                                Searching...
                            </>
                        ) : (
                            <>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="11" cy="11" r="8" />
                                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                                </svg>
                                Search
                            </>
                        )}
                    </button>
                </div>
            </form>

            {/* Error */}
            {error && (
                <div style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    padding: 'var(--space-4)',
                    borderRadius: 'var(--radius-lg)',
                    color: 'var(--color-error)',
                    marginBottom: 'var(--space-6)',
                }}>
                    Error: {error}
                </div>
            )}

            {/* Results */}
            {data && (
                <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
                        <h3>Results ({data.total_results})</h3>
                        {data.query_plan && (
                            <span className="badge badge-info">Query Plan Available</span>
                        )}
                    </div>

                    {results.length === 0 ? (
                        <div className="empty-state">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="11" cy="11" r="8" />
                                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                            </svg>
                            <p>No results found for your query.</p>
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                            {results.map((result, index) => (
                                <div key={result.chunk_id} className="result-card">
                                    <div className="result-header">
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                                            <span style={{
                                                width: 28,
                                                height: 28,
                                                background: 'var(--color-accent-gradient)',
                                                borderRadius: 'var(--radius-full)',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                fontSize: 'var(--text-sm)',
                                                fontWeight: 600,
                                                color: 'white',
                                            }}>
                                                {index + 1}
                                            </span>
                                            <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)' }}>
                                                Doc: {result.document_id.slice(0, 8)}...
                                            </span>
                                        </div>
                                        <div className="result-scores">
                                            <div className="result-score">
                                                <span className="result-score-label">Final</span>
                                                <span className="result-score-value" style={{ color: 'var(--color-success)' }}>
                                                    {formatScore(result.final_score)}
                                                </span>
                                            </div>
                                            <div className="result-score">
                                                <span className="result-score-label">RRF</span>
                                                <span className="result-score-value">{formatScore(result.rrf_score)}</span>
                                            </div>
                                            <div className="result-score">
                                                <span className="result-score-label">Lexical</span>
                                                <span className="result-score-value">{formatScore(result.lexical_score)}</span>
                                            </div>
                                            <div className="result-score">
                                                <span className="result-score-label">Semantic</span>
                                                <span className="result-score-value">{formatScore(result.semantic_score)}</span>
                                            </div>
                                            <div className="result-score">
                                                <span className="result-score-label">Graph</span>
                                                <span className="result-score-value">{formatScore(result.graph_score)}</span>
                                            </div>
                                            <div className="result-score">
                                                <span className="result-score-label">Rerank</span>
                                                <span className="result-score-value">{formatScore(result.rerank_score)}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="result-text">{result.text}</div>
                                    {result.parent_text && result.parent_text !== result.text && (
                                        <details style={{ marginTop: 'var(--space-3)' }}>
                                            <summary style={{ cursor: 'pointer', fontSize: 'var(--text-sm)', color: 'var(--color-accent-primary)' }}>
                                                Show parent context
                                            </summary>
                                            <div style={{
                                                marginTop: 'var(--space-2)',
                                                padding: 'var(--space-3)',
                                                background: 'var(--color-bg-secondary)',
                                                borderRadius: 'var(--radius-md)',
                                                fontSize: 'var(--text-sm)',
                                                color: 'var(--color-text-tertiary)',
                                                whiteSpace: 'pre-wrap',
                                            }}>
                                                {result.parent_text}
                                            </div>
                                        </details>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Empty State */}
            {!data && !loading && (
                <div className="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                    <p>Enter a query to search your documents</p>
                </div>
            )}
        </div>
    );
}
