import { useState, useEffect } from 'react';

export function GraphViewer() {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [puppyGraphUrl, setPuppyGraphUrl] = useState('http://localhost:8091');

    useEffect(() => {
        // Fetch PuppyGraph URL from config
        fetch('http://localhost:8009/api/config')
            .then((res) => res.json())
            .then((data) => {
                if (data.config?.puppygraph_web_ui_url?.value) {
                    setPuppyGraphUrl(data.config.puppygraph_web_ui_url.value);
                }
            })
            .catch((err) => {
                console.log('Could not fetch config:', err);
            });
    }, []);

    const handleIframeLoad = () => {
        setLoading(false);
    };

    const handleIframeError = () => {
        setLoading(false);
        setError('Could not connect to PuppyGraph. Make sure PuppyGraph is running on port 8091.');
    };

    return (
        <div className="animate-fadeIn">
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                    <h2>Graph Visualization</h2>
                    <p>Explore the knowledge graph using PuppyGraph's Web UI</p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                    <a
                        href={puppyGraphUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn btn-secondary"
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                            <polyline points="15 3 21 3 21 9" />
                            <line x1="10" y1="14" x2="21" y2="3" />
                        </svg>
                        Open in New Tab
                    </a>
                </div>
            </div>

            {/* Connection Status */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-2)',
                marginBottom: 'var(--space-4)',
            }}>
                <span style={{
                    width: 8,
                    height: 8,
                    borderRadius: 'var(--radius-full)',
                    background: error ? 'var(--color-error)' : (loading ? 'var(--color-warning)' : 'var(--color-success)'),
                }} />
                <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-secondary)' }}>
                    {error ? 'Disconnected' : (loading ? 'Connecting...' : 'Connected to PuppyGraph')}
                </span>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>
                    ({puppyGraphUrl})
                </span>
            </div>

            {/* Error State */}
            {error && (
                <div className="card" style={{ textAlign: 'center', padding: 'var(--space-12)' }}>
                    <svg
                        width="64"
                        height="64"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="var(--color-error)"
                        strokeWidth="2"
                        style={{ margin: '0 auto var(--space-4)' }}
                    >
                        <circle cx="12" cy="12" r="10" />
                        <line x1="15" y1="9" x2="9" y2="15" />
                        <line x1="9" y1="9" x2="15" y2="15" />
                    </svg>
                    <h3 style={{ marginBottom: 'var(--space-2)' }}>Cannot Connect to PuppyGraph</h3>
                    <p style={{ marginBottom: 'var(--space-4)' }}>
                        Make sure PuppyGraph is running and accessible at <code>{puppyGraphUrl}</code>
                    </p>
                    <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)' }}>
                        <p style={{ marginBottom: 'var(--space-2)' }}>To start PuppyGraph:</p>
                        <code style={{
                            display: 'block',
                            background: 'var(--color-bg-tertiary)',
                            padding: 'var(--space-3)',
                            borderRadius: 'var(--radius-md)',
                        }}>
                            docker compose up -d
                        </code>
                    </div>
                </div>
            )}

            {/* Loading State */}
            {loading && !error && (
                <div className="graph-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                        <div className="spinner" style={{ margin: '0 auto var(--space-4)' }} />
                        <p>Loading PuppyGraph UI...</p>
                    </div>
                </div>
            )}

            {/* Graph Iframe */}
            {!error && (
                <div className="graph-container" style={{ display: loading ? 'none' : 'block' }}>
                    <iframe
                        className="graph-iframe"
                        src={puppyGraphUrl}
                        title="PuppyGraph Web UI"
                        onLoad={handleIframeLoad}
                        onError={handleIframeError}
                    />
                </div>
            )}

            {/* Tips */}
            <div style={{ marginTop: 'var(--space-6)' }}>
                <h4 style={{ marginBottom: 'var(--space-3)', fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)' }}>
                    Quick Tips
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 'var(--space-4)' }}>
                    <div className="card" style={{ padding: 'var(--space-4)' }}>
                        <div style={{ fontSize: 'var(--text-sm)', fontWeight: 500, marginBottom: 'var(--space-2)' }}>
                            Find Entities
                        </div>
                        <code style={{ fontSize: 'var(--text-xs)' }}>
                            MATCH (e:Entity) RETURN e LIMIT 20
                        </code>
                    </div>
                    <div className="card" style={{ padding: 'var(--space-4)' }}>
                        <div style={{ fontSize: 'var(--text-sm)', fontWeight: 500, marginBottom: 'var(--space-2)' }}>
                            Find Relations
                        </div>
                        <code style={{ fontSize: 'var(--text-xs)' }}>
                            MATCH (a)-[r]-&gt;(b) RETURN a, r, b LIMIT 50
                        </code>
                    </div>
                    <div className="card" style={{ padding: 'var(--space-4)' }}>
                        <div style={{ fontSize: 'var(--text-sm)', fontWeight: 500, marginBottom: 'var(--space-2)' }}>
                            Entity by Name
                        </div>
                        <code style={{ fontSize: 'var(--text-xs)' }}>
                            MATCH (e:Entity &#123;name: 'John'&#125;) RETURN e
                        </code>
                    </div>
                </div>
            </div>
        </div>
    );
}
