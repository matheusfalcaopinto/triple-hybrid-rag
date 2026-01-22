import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d';
import { useGraph } from '../hooks/useApi';
import type { GraphData, EntityDetails } from '../hooks/useApi';

// =============================================================================
// Types & Interfaces
// =============================================================================

interface ForceNode {
    id: string;
    name: string;
    type: string;
    mentions: number;
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
    fx?: number;
    fy?: number;
    [key: string]: unknown;
}

interface ForceLink {
    id: string;
    source: string;
    target: string;
    type: string;
    confidence: number | null;
    [key: string]: unknown;
}

interface ForceGraphData {
    nodes: ForceNode[];
    links: ForceLink[];
}

// =============================================================================
// Design System & Theme Configuration
// =============================================================================

const ENTITY_COLORS: Record<string, string> = {
    PERSON: '#3B82F6',
    ORGANIZATION: '#10B981',
    LOCATION: '#F59E0B',
    CONCEPT: '#8B5CF6',
    EVENT: '#EF4444',
    PRODUCT: '#EC4899',
    TECHNOLOGY: '#06B6D4',
    TECHNICAL_TERM: '#14B8A6',
    DATE: '#84CC16',
    DEFAULT: '#6B7280',
};

const ENTITY_COLORS_BRIGHT: Record<string, string> = {
    PERSON: '#60A5FA',
    ORGANIZATION: '#34D399',
    LOCATION: '#FBBF24',
    CONCEPT: '#A78BFA',
    EVENT: '#F87171',
    PRODUCT: '#F472B6',
    TECHNOLOGY: '#22D3EE',
    TECHNICAL_TERM: '#2DD4BF',
    DATE: '#A3E635',
    DEFAULT: '#9CA3AF',
};

const getNodeColor = (type: string): string => ENTITY_COLORS[type] || ENTITY_COLORS.DEFAULT;
const getNodeColorBright = (type: string): string => ENTITY_COLORS_BRIGHT[type] || ENTITY_COLORS_BRIGHT.DEFAULT;

const getNodeRadius = (mentions: number): number => {
    const base = 6;
    const scale = Math.log2((mentions || 1) + 1) * 4;
    return Math.min(28, Math.max(base, base + scale));
};

// =============================================================================
// Component
// =============================================================================

export function GraphViewer() {
    const { loading, error, getGraphData, getEntityDetails } = useGraph();
    
    const [graphData, setGraphData] = useState<GraphData | null>(null);
    const [selectedNode, setSelectedNode] = useState<ForceNode | null>(null);
    const [hoveredNode, setHoveredNode] = useState<ForceNode | null>(null);
    const [entityDetails, setEntityDetails] = useState<EntityDetails | null>(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [filterType, setFilterType] = useState<string>('all');
    const [searchTerm, setSearchTerm] = useState('');
    const [isSimulationRunning, setIsSimulationRunning] = useState(true);
    const [hasInitialized, setHasInitialized] = useState(false);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const graphRef = useRef<ForceGraphMethods<any, any> | undefined>(undefined);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    // -------------------------------------------------------------------------
    // Data Fetching
    // -------------------------------------------------------------------------

    const loadGraph = useCallback(async () => {
        setHasInitialized(false);
        const data = await getGraphData(1000);
        if (data) {
            setGraphData(data);
        }
    }, [getGraphData]);

    const loadEntityDetails = useCallback(async (node: ForceNode) => {
        setLoadingDetails(true);
        try {
            const details = await getEntityDetails(node.id);
            setEntityDetails(details);
        } catch (err) {
            console.error('Failed to load entity details:', err);
        } finally {
            setLoadingDetails(false);
        }
    }, [getEntityDetails]);

    useEffect(() => {
        loadGraph();
    }, [loadGraph]);

    // -------------------------------------------------------------------------
    // Memoized Graph Data (CRITICAL for stability)
    // -------------------------------------------------------------------------

    const filteredGraphData: ForceGraphData = useMemo(() => {
        if (!graphData) return { nodes: [], links: [] };

        // Deduplicate nodes by ID (keep first occurrence)
        const seenNodeIds = new Set<string>();
        const filteredNodes: ForceNode[] = graphData.nodes
            .filter(node => {
                // Skip duplicates
                if (seenNodeIds.has(node.id)) return false;
                seenNodeIds.add(node.id);
                
                const matchesType = filterType === 'all' || node.type === filterType;
                const matchesSearch = !searchTerm || 
                    node.name.toLowerCase().includes(searchTerm.toLowerCase());
                return matchesType && matchesSearch;
            })
            .map(node => ({
                ...node,
                x: undefined,
                y: undefined,
            }));

        const validNodeIds = new Set(filteredNodes.map(n => n.id));

        // Deduplicate links by ID
        const seenLinkIds = new Set<string>();
        const filteredLinks: ForceLink[] = graphData.edges
            .filter(edge => {
                if (seenLinkIds.has(edge.id)) return false;
                seenLinkIds.add(edge.id);
                
                const sourceId = edge.source;
                const targetId = edge.target;
                return validNodeIds.has(sourceId) && validNodeIds.has(targetId);
            })
            .map(edge => ({ ...edge }));

        return { nodes: filteredNodes, links: filteredLinks };
    }, [graphData, filterType, searchTerm]);

    // -------------------------------------------------------------------------
    // Layout & Resizing
    // -------------------------------------------------------------------------

    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect();
                setDimensions({ 
                    width: Math.max(400, Math.floor(rect.width) || 800), 
                    height: Math.max(400, Math.floor(rect.height) || 600) 
                });
            }
        };

        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        
        const timer = setTimeout(updateDimensions, 100);
        
        return () => {
            window.removeEventListener('resize', updateDimensions);
            clearTimeout(timer);
        };
    }, [selectedNode]);

    // -------------------------------------------------------------------------
    // Auto-center on load
    // -------------------------------------------------------------------------

    const handleEngineStop = useCallback(() => {
        if (!hasInitialized && graphRef.current && filteredGraphData.nodes.length > 0) {
            setTimeout(() => {
                graphRef.current?.zoomToFit(400, 60);
                setHasInitialized(true);
                setIsSimulationRunning(false);
            }, 100);
        }
    }, [hasInitialized, filteredGraphData.nodes.length]);

    // -------------------------------------------------------------------------
    // Interaction Handlers
    // -------------------------------------------------------------------------

    const handleNodeClick = useCallback((node: ForceNode) => {
        setSelectedNode(node);
        loadEntityDetails(node);
        
        if (graphRef.current) {
            graphRef.current.centerAt(node.x, node.y, 800);
            setTimeout(() => {
                graphRef.current?.zoom(3, 800);
            }, 200);
        }
    }, [loadEntityDetails]);

    const handleNodeHover = useCallback((node: ForceNode | null) => {
        setHoveredNode(node);
    }, []);

    const handleZoomIn = useCallback(() => {
        if (graphRef.current) {
            graphRef.current.zoom(graphRef.current.zoom() * 1.5, 400);
        }
    }, []);

    const handleZoomOut = useCallback(() => {
        if (graphRef.current) {
            graphRef.current.zoom(graphRef.current.zoom() / 1.5, 400);
        }
    }, []);

    const handleZoomToFit = useCallback(() => {
        if (graphRef.current) {
            graphRef.current.zoomToFit(400, 60);
        }
    }, []);

    const handleToggleSimulation = useCallback(() => {
        if (graphRef.current) {
            if (isSimulationRunning) {
                graphRef.current.pauseAnimation();
            } else {
                graphRef.current.resumeAnimation();
            }
            setIsSimulationRunning(!isSimulationRunning);
        }
    }, [isSimulationRunning]);

    const handleCloseDetails = useCallback(() => {
        setSelectedNode(null);
        setEntityDetails(null);
    }, []);

    // -------------------------------------------------------------------------
    // Custom Canvas Rendering
    // -------------------------------------------------------------------------

    const drawNode = useCallback((node: ForceNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const radius = getNodeRadius(node.mentions);
        const isSelected = selectedNode?.id === node.id;
        const isHovered = hoveredNode?.id === node.id;
        const x = node.x ?? 0;
        const y = node.y ?? 0;

        // Outer Glow
        if (isSelected || isHovered) {
            const glowRadius = radius + (isSelected ? 8 : 5);
            const gradient = ctx.createRadialGradient(x, y, radius, x, y, glowRadius);
            gradient.addColorStop(0, isSelected ? 'rgba(255, 255, 255, 0.4)' : 'rgba(255, 255, 255, 0.2)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
            ctx.beginPath();
            ctx.arc(x, y, glowRadius, 0, 2 * Math.PI);
            ctx.fillStyle = gradient;
            ctx.fill();
        }

        // Main Node
        const nodeGradient = ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.3, 0, x, y, radius);
        nodeGradient.addColorStop(0, getNodeColorBright(node.type));
        nodeGradient.addColorStop(1, getNodeColor(node.type));
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = nodeGradient;
        ctx.fill();

        // Border
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = isSelected ? '#FFFFFF' : 'rgba(0, 0, 0, 0.3)';
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.stroke();

        // Inner Highlight
        const highlightGradient = ctx.createRadialGradient(x - radius * 0.4, y - radius * 0.4, 0, x - radius * 0.2, y - radius * 0.2, radius * 0.5);
        highlightGradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
        highlightGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.beginPath();
        ctx.arc(x, y, radius * 0.8, 0, 2 * Math.PI);
        ctx.fillStyle = highlightGradient;
        ctx.fill();

        // Label
        const showLabel = isSelected || isHovered || globalScale > 0.8;
        if (showLabel) {
            const label = node.name.length > 20 ? node.name.substring(0, 18) + '...' : node.name;
            const fontSize = Math.max(10, Math.min(14, 12 / globalScale));
            ctx.font = `${isSelected || isHovered ? '600' : '500'} ${fontSize}px Inter, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            const textWidth = ctx.measureText(label).width;
            const padding = 4;
            const labelY = y + radius + 4;

            // Background
            ctx.fillStyle = 'rgba(17, 24, 39, 0.9)';
            ctx.fillRect(x - textWidth / 2 - padding, labelY - padding, textWidth + padding * 2, fontSize + padding * 2);
            
            // Text
            ctx.fillStyle = '#F9FAFB';
            ctx.fillText(label, x, labelY);
        }
    }, [selectedNode, hoveredNode]);

    const nodePointerAreaPaint = useCallback((node: ForceNode, color: string, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const radius = getNodeRadius(node.mentions);
        const x = node.x ?? 0;
        const y = node.y ?? 0;
        
        // Draw circular hit area for the node itself (with padding for easier clicking)
        const hitPadding = Math.max(8, 12 / globalScale); // Larger hit area when zoomed out
        ctx.beginPath();
        ctx.arc(x, y, radius + hitPadding, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        
        // Also include the label area in the hit zone
        // Labels are shown when: selected, hovered, or globalScale > 0.8
        const showLabel = selectedNode?.id === node.id || hoveredNode?.id === node.id || globalScale > 0.8;
        if (showLabel) {
            const label = node.name.length > 20 ? node.name.substring(0, 18) + '...' : node.name;
            const fontSize = Math.max(10, Math.min(14, 12 / globalScale));
            ctx.font = `500 ${fontSize}px Inter, sans-serif`;
            const textWidth = ctx.measureText(label).width;
            const padding = 4;
            const labelY = y + radius + 4;
            
            // Draw rectangle hit area for the label
            ctx.fillRect(
                x - textWidth / 2 - padding - 2,
                labelY - padding - 2,
                textWidth + padding * 2 + 4,
                fontSize + padding * 2 + 4
            );
        }
    }, [selectedNode, hoveredNode]);

    // -------------------------------------------------------------------------
    // Render
    // -------------------------------------------------------------------------

    const nodeCount = filteredGraphData.nodes.length;
    const linkCount = filteredGraphData.links.length;
    const graphWidth = selectedNode ? Math.floor(dimensions.width * 0.65) : dimensions.width;

    return (
        <div className="animate-fadeIn">
            {/* Header */}
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                    <h2>Knowledge Graph</h2>
                    <p>Interactive visualization of entities and their relationships</p>
                </div>
                <button onClick={loadGraph} className="btn btn-secondary" disabled={loading}>
                    {loading ? 'Loading...' : 'Refresh Data'}
                </button>
            </div>

            {/* Controls Bar */}
            {graphData && (
                <div style={{ display: 'flex', gap: 'var(--space-3)', marginBottom: 'var(--space-4)', flexWrap: 'wrap', alignItems: 'center' }}>
                    <div className="card" style={{ padding: 'var(--space-2) var(--space-3)', display: 'flex', gap: 'var(--space-4)', background: 'var(--color-bg-tertiary)' }}>
                        <span style={{ fontSize: 'var(--text-sm)' }}>
                            <strong style={{ color: 'var(--color-accent-primary)' }}>{nodeCount}</strong> nodes
                        </span>
                        <span style={{ fontSize: 'var(--text-sm)' }}>
                            <strong style={{ color: 'var(--color-accent-primary)' }}>{linkCount}</strong> edges
                        </span>
                    </div>

                    <div style={{ position: 'relative' }}>
                        <input
                            type="text"
                            placeholder="Search entities..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            style={{
                                padding: 'var(--space-2) var(--space-3)',
                                paddingLeft: 'var(--space-8)',
                                borderRadius: 'var(--radius-md)',
                                border: '1px solid var(--color-border)',
                                background: 'var(--color-bg-secondary)',
                                color: 'var(--color-text-primary)',
                                minWidth: 220,
                                fontSize: 'var(--text-sm)',
                            }}
                        />
                        <svg style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', opacity: 0.5 }} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                        </svg>
                    </div>

                    <select
                        value={filterType}
                        onChange={(e) => setFilterType(e.target.value)}
                        style={{
                            padding: 'var(--space-2) var(--space-3)',
                            borderRadius: 'var(--radius-md)',
                            border: '1px solid var(--color-border)',
                            background: 'var(--color-bg-secondary)',
                            color: 'var(--color-text-primary)',
                            fontSize: 'var(--text-sm)',
                        }}
                    >
                        <option value="all">All Types</option>
                        {graphData.stats.entity_types.map(type => (
                            <option key={type} value={type}>{type}</option>
                        ))}
                    </select>

                    <div style={{ flex: 1 }} />

                    <div style={{ display: 'flex', gap: 2, background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 2 }}>
                        <button onClick={handleZoomOut} className="btn btn-secondary" style={{ padding: 'var(--space-2)', minWidth: 36 }} title="Zoom Out">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/>
                            </svg>
                        </button>
                        <button onClick={handleZoomToFit} className="btn btn-secondary" style={{ padding: 'var(--space-2)', minWidth: 36 }} title="Fit to View">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>
                            </svg>
                        </button>
                        <button onClick={handleZoomIn} className="btn btn-secondary" style={{ padding: 'var(--space-2)', minWidth: 36 }} title="Zoom In">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
                            </svg>
                        </button>
                        <div style={{ width: 1, background: 'var(--color-border)', margin: '4px 2px' }} />
                        <button onClick={handleToggleSimulation} className="btn btn-secondary" style={{ padding: 'var(--space-2)', minWidth: 36 }} title={isSimulationRunning ? 'Pause Physics' : 'Resume Physics'}>
                            {isSimulationRunning ? (
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
                                </svg>
                            ) : (
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <polygon points="5 3 19 12 5 21 5 3"/>
                                </svg>
                            )}
                        </button>
                    </div>
                </div>
            )}

            {/* Legend */}
            {graphData && graphData.stats.entity_types.length > 0 && (
                <div style={{ display: 'flex', gap: 'var(--space-4)', marginBottom: 'var(--space-4)', flexWrap: 'wrap', padding: 'var(--space-2) var(--space-3)', background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
                    {graphData.stats.entity_types.map(type => (
                        <button
                            key={type}
                            onClick={() => setFilterType(filterType === type ? 'all' : type)}
                            style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px', borderRadius: 'var(--radius-sm)', border: 'none', background: filterType === type ? getNodeColor(type) + '30' : 'transparent', cursor: 'pointer' }}
                        >
                            <span style={{ width: 10, height: 10, borderRadius: '50%', background: getNodeColor(type), boxShadow: `0 0 6px ${getNodeColor(type)}60` }} />
                            <span style={{ fontSize: 'var(--text-xs)', color: filterType === type ? 'var(--color-text-primary)' : 'var(--color-text-secondary)', fontWeight: filterType === type ? 600 : 400 }}>
                                {type}
                            </span>
                        </button>
                    ))}
                </div>
            )}

            {/* Error State */}
            {error && (
                <div className="card" style={{ textAlign: 'center', padding: 'var(--space-8)', background: 'rgba(239, 68, 68, 0.1)' }}>
                    <p style={{ color: '#ef4444' }}>{error}</p>
                    <button onClick={loadGraph} className="btn btn-primary" style={{ marginTop: 'var(--space-4)' }}>Retry</button>
                </div>
            )}

            {/* Loading State */}
            {loading && !graphData && (
                <div className="card" style={{ textAlign: 'center', padding: 'var(--space-12)' }}>
                    <div className="spinner" style={{ margin: '0 auto var(--space-4)' }} />
                    <p>Loading knowledge graph...</p>
                </div>
            )}

            {/* Main Graph Container */}
            {!error && graphData && (
                <div style={{ display: 'flex', gap: 'var(--space-4)' }}>
                    {/* Graph Canvas */}
                    <div 
                        ref={containerRef}
                        className="card" 
                        style={{ 
                            flex: selectedNode ? '1 1 65%' : '1 1 100%',
                            padding: 0, 
                            overflow: 'hidden',
                            background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)',
                            position: 'relative',
                            minHeight: 600,
                            borderRadius: 'var(--radius-lg)',
                        }}
                    >
                        {filteredGraphData.nodes.length > 0 ? (
                            <ForceGraph2D
                                ref={graphRef}
                                graphData={filteredGraphData}
                                width={graphWidth}
                                height={Math.max(600, dimensions.height)}
                                
                                nodeId="id"
                                nodeLabel={() => ''}
                                nodeCanvasObject={drawNode}
                                nodePointerAreaPaint={nodePointerAreaPaint}
                                
                                linkSource="source"
                                linkTarget="target"
                                linkColor={() => 'rgba(148, 163, 184, 0.35)'}
                                linkWidth={1.5}
                                linkDirectionalArrowLength={8}
                                linkDirectionalArrowRelPos={0.85}
                                linkDirectionalArrowColor={() => 'rgba(226, 232, 240, 0.8)'}
                                linkCurvature={0.15}
                                
                                d3AlphaDecay={0.02}
                                d3VelocityDecay={0.35}
                                cooldownTicks={120}
                                warmupTicks={30}
                                onEngineStop={handleEngineStop}
                                
                                enableNodeDrag={true}
                                enableZoomInteraction={true}
                                enablePanInteraction={true}
                                minZoom={0.2}
                                maxZoom={12}
                                
                                onNodeClick={handleNodeClick}
                                onNodeHover={handleNodeHover}
                                onBackgroundClick={() => { if (selectedNode) handleCloseDetails(); }}
                            />
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 600, color: 'var(--color-text-tertiary)' }}>
                                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ marginBottom: 16, opacity: 0.5 }}>
                                    <circle cx="12" cy="12" r="3" />
                                    <circle cx="19" cy="5" r="2" />
                                    <circle cx="5" cy="19" r="2" />
                                    <line x1="14" y1="10" x2="17" y2="7" />
                                    <line x1="10" y1="14" x2="7" y2="17" />
                                </svg>
                                <p>No entities match your filter criteria</p>
                            </div>
                        )}

                        {loading && (
                            <div style={{ position: 'absolute', top: 12, left: 12, background: 'rgba(0,0,0,0.7)', padding: '8px 12px', borderRadius: 'var(--radius-md)', fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                                <div className="spinner" style={{ width: 12, height: 12 }} />
                                Updating...
                            </div>
                        )}
                    </div>

                    {/* Detail Panel */}
                    {selectedNode && (
                        <div className="card" style={{ flex: '0 0 33%', maxHeight: 700, overflow: 'auto' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 'var(--space-4)', paddingBottom: 'var(--space-3)', borderBottom: '1px solid var(--color-border)' }}>
                                <div>
                                    <h3 style={{ margin: 0, fontSize: 'var(--text-lg)', marginBottom: 8 }}>{selectedNode.name}</h3>
                                    <span style={{ display: 'inline-block', padding: '4px 10px', borderRadius: 'var(--radius-full)', background: getNodeColor(selectedNode.type) + '25', color: getNodeColor(selectedNode.type), fontSize: 'var(--text-xs)', fontWeight: 600, border: `1px solid ${getNodeColor(selectedNode.type)}40` }}>
                                        {selectedNode.type}
                                    </span>
                                </div>
                                <button onClick={handleCloseDetails} style={{ background: 'var(--color-bg-tertiary)', border: 'none', cursor: 'pointer', padding: 8, borderRadius: 'var(--radius-sm)', color: 'var(--color-text-secondary)' }}>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <line x1="18" y1="6" x2="6" y2="18" />
                                        <line x1="6" y1="6" x2="18" y2="18" />
                                    </svg>
                                </button>
                            </div>

                            {loadingDetails && (
                                <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                                    <div className="spinner" style={{ margin: '0 auto' }} />
                                </div>
                            )}

                            {!loadingDetails && entityDetails && (
                                <div>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 20 }}>
                                        {[
                                            { label: 'Mentions', value: entityDetails.stats.mention_count, color: '#3B82F6' },
                                            { label: 'Outgoing', value: entityDetails.stats.outgoing_relations, color: '#10B981' },
                                            { label: 'Incoming', value: entityDetails.stats.incoming_relations, color: '#F59E0B' },
                                        ].map(stat => (
                                            <div key={stat.label} style={{ textAlign: 'center', padding: 12, background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
                                                <div style={{ fontSize: 'var(--text-xl)', fontWeight: 700, color: stat.color, marginBottom: 4 }}>{stat.value}</div>
                                                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{stat.label}</div>
                                            </div>
                                        ))}
                                    </div>

                                    {(entityDetails.relations.outgoing.length > 0 || entityDetails.relations.incoming.length > 0) && (
                                        <div style={{ marginBottom: 20 }}>
                                            <h4 style={{ fontSize: 'var(--text-sm)', marginBottom: 12, color: 'var(--color-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Relationships</h4>
                                            <div style={{ maxHeight: 200, overflow: 'auto' }}>
                                                {entityDetails.relations.outgoing.map(rel => (
                                                    <div key={rel.id} style={{ fontSize: 'var(--text-sm)', padding: '8px 12px', marginBottom: 4, background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', gap: 8 }}>
                                                        <span style={{ color: '#10B981' }}>→</span>
                                                        <span style={{ flex: 1, color: getNodeColor(rel.target?.type || 'DEFAULT'), fontWeight: 500 }}>{rel.target?.name || 'Unknown'}</span>
                                                        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)', background: 'var(--color-bg-secondary)', padding: '2px 6px', borderRadius: 'var(--radius-sm)' }}>{rel.type}</span>
                                                    </div>
                                                ))}
                                                {entityDetails.relations.incoming.map(rel => (
                                                    <div key={rel.id} style={{ fontSize: 'var(--text-sm)', padding: '8px 12px', marginBottom: 4, background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', gap: 8 }}>
                                                        <span style={{ color: '#F59E0B' }}>←</span>
                                                        <span style={{ flex: 1, color: getNodeColor(rel.source?.type || 'DEFAULT'), fontWeight: 500 }}>{rel.source?.name || 'Unknown'}</span>
                                                        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)', background: 'var(--color-bg-secondary)', padding: '2px 6px', borderRadius: 'var(--radius-sm)' }}>{rel.type}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {entityDetails.mentions.length > 0 && (
                                        <div>
                                            <h4 style={{ fontSize: 'var(--text-sm)', marginBottom: 12, color: 'var(--color-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Document Mentions</h4>
                                            <div style={{ maxHeight: 250, overflow: 'auto' }}>
                                                {entityDetails.mentions.map((mention, idx) => (
                                                    <div key={idx} style={{ fontSize: 'var(--text-sm)', padding: 12, marginBottom: 8, background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-md)', borderLeft: '3px solid var(--color-accent-primary)' }}>
                                                        <div style={{ color: 'var(--color-text-tertiary)', marginBottom: 8, fontSize: 'var(--text-xs)', fontWeight: 500 }}>
                                                            {mention.document}
                                                        </div>
                                                        <div style={{ color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
                                                            "{mention.text_snippet}"
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {!loadingDetails && !entityDetails && (
                                <p style={{ color: 'var(--color-text-tertiary)', textAlign: 'center', padding: 16 }}>
                                    No details available for this entity.
                                </p>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Empty State */}
            {!loading && !error && graphData && graphData.nodes.length === 0 && (
                <div className="card" style={{ textAlign: 'center', padding: 'var(--space-12)' }}>
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-tertiary)" strokeWidth="1" style={{ margin: '0 auto 16px' }}>
                        <circle cx="12" cy="12" r="3" />
                        <circle cx="19" cy="5" r="2" />
                        <circle cx="5" cy="19" r="2" />
                        <line x1="14" y1="10" x2="17" y2="7" />
                        <line x1="10" y1="14" x2="7" y2="17" />
                    </svg>
                    <h3>No Graph Data Yet</h3>
                    <p style={{ color: 'var(--color-text-secondary)', maxWidth: 400, margin: '0 auto' }}>
                        Upload documents and enable entity extraction to build your knowledge graph visualization.
                    </p>
                </div>
            )}
        </div>
    );
}
