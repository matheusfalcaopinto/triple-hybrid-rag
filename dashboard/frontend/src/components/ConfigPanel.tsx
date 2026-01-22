import { useState, useEffect, useCallback } from 'react';
import { useConfig } from '../hooks/useApi';
import type { ConfigItem } from '../hooks/useApi';

interface ConfigCategory {
    name: string;
    items: Array<{
        key: string;
        item: ConfigItem;
    }>;
}

// Icons for each category
const categoryIcons: Record<string, React.ReactNode> = {
    'Feature Flags': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
            <line x1="4" y1="22" x2="4" y2="15" />
        </svg>
    ),
    'OCR': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
        </svg>
    ),
    'HyDE': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
        </svg>
    ),
    'Query Expansion': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
            <line x1="11" y1="8" x2="11" y2="14" />
            <line x1="8" y1="11" x2="14" y2="11" />
        </svg>
    ),
    'Retrieval Weights': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="4" y1="21" x2="4" y2="14" />
            <line x1="4" y1="10" x2="4" y2="3" />
            <line x1="12" y1="21" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12" y2="3" />
            <line x1="20" y1="21" x2="20" y2="16" />
            <line x1="20" y1="12" x2="20" y2="3" />
            <line x1="1" y1="14" x2="7" y2="14" />
            <line x1="9" y1="8" x2="15" y2="8" />
            <line x1="17" y1="16" x2="23" y2="16" />
        </svg>
    ),
    'Top-K Settings': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
    ),
    'Multi-Stage Reranking': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="8" y1="6" x2="21" y2="6" />
            <line x1="8" y1="12" x2="21" y2="12" />
            <line x1="8" y1="18" x2="21" y2="18" />
            <line x1="3" y1="6" x2="3.01" y2="6" />
            <line x1="3" y1="12" x2="3.01" y2="12" />
            <line x1="3" y1="18" x2="3.01" y2="18" />
        </svg>
    ),
    'Diversity': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M8 14s1.5 2 4 2 4-2 4-2" />
            <line x1="9" y1="9" x2="9.01" y2="9" />
            <line x1="15" y1="9" x2="15.01" y2="9" />
        </svg>
    ),
    'Chunking': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="7" height="7" />
            <rect x="14" y="3" width="7" height="7" />
            <rect x="14" y="14" width="7" height="7" />
            <rect x="3" y="14" width="7" height="7" />
        </svg>
    ),
    'Embedding': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v4" />
            <path d="M12 18v4" />
            <path d="m4.93 4.93 2.83 2.83" />
            <path d="m16.24 16.24 2.83 2.83" />
            <path d="M2 12h4" />
            <path d="M18 12h4" />
            <path d="m4.93 19.07 2.83-2.83" />
            <path d="m16.24 7.76 2.83-2.83" />
        </svg>
    ),
    'Jina AI': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
        </svg>
    ),
    'Database': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <ellipse cx="12" cy="5" rx="9" ry="3" />
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
        </svg>
    ),
    'Entity Extraction': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="18" cy="5" r="3" />
            <circle cx="6" cy="12" r="3" />
            <circle cx="18" cy="19" r="3" />
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
        </svg>
    ),
    'Observability': (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
        </svg>
    ),
};

const defaultIcon = (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
);

export function ConfigPanel() {
    const { loading, error, getConfig, updateConfig, reloadConfig } = useConfig();
    const [config, setConfig] = useState<Record<string, ConfigItem>>({});
    const [categories, setCategories] = useState<string[]>([]);
    const [activeCategory, setActiveCategory] = useState<string>('');
    const [pendingChanges, setPendingChanges] = useState<Record<string, unknown>>({});
    const [saving, setSaving] = useState(false);
    const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    useEffect(() => {
        loadConfig();
    }, []);

    const loadConfig = async () => {
        const data = await getConfig();
        if (data) {
            setConfig(data.config);
            setCategories(data.categories);
            if (data.categories.length > 0 && !activeCategory) {
                setActiveCategory(data.categories[0]);
            }
        }
    };

    const handleValueChange = useCallback((key: string, value: unknown) => {
        setPendingChanges(prev => ({ ...prev, [key]: value }));
        setSaveMessage(null);
    }, []);

    const handleSave = async () => {
        if (Object.keys(pendingChanges).length === 0) return;
        
        setSaving(true);
        setSaveMessage(null);
        
        const success = await updateConfig(pendingChanges);
        
        if (success) {
            setPendingChanges({});
            setSaveMessage({ type: 'success', text: 'Configuration saved successfully' });
            await loadConfig();
        } else {
            setSaveMessage({ type: 'error', text: 'Failed to save configuration' });
        }
        
        setSaving(false);
    };

    const handleReload = async () => {
        await reloadConfig();
        await loadConfig();
        setPendingChanges({});
        setSaveMessage({ type: 'success', text: 'Configuration reloaded from .env' });
    };

    const handleReset = () => {
        setPendingChanges({});
        setSaveMessage(null);
    };

    const getCurrentValue = (key: string, item: ConfigItem): unknown => {
        if (key in pendingChanges) {
            return pendingChanges[key];
        }
        return item.value;
    };

    const getCategorizedItems = (): ConfigCategory[] => {
        const grouped: Record<string, ConfigCategory> = {};
        
        for (const [key, item] of Object.entries(config)) {
            const category = item.category;
            if (!grouped[category]) {
                grouped[category] = { name: category, items: [] };
            }
            grouped[category].items.push({ key, item });
        }
        
        return categories.map(cat => grouped[cat]).filter(Boolean);
    };

    const formatLabel = (key: string): string => {
        return key
            .replace(/^rag_/, '')
            .replace(/^jina_/, '')
            .replace(/^local_vrag_/, '')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    };

    const renderConfigInput = (key: string, item: ConfigItem) => {
        const value = getCurrentValue(key, item);
        const hasChange = key in pendingChanges;

        switch (item.type) {
            case 'boolean':
                return (
                    <label className="config-toggle">
                        <input
                            type="checkbox"
                            className="config-toggle-input"
                            checked={Boolean(value)}
                            onChange={(e) => handleValueChange(key, e.target.checked)}
                        />
                        <span className="config-toggle-track">
                            <span className="config-toggle-thumb" />
                        </span>
                        <span className="config-toggle-label">
                            {Boolean(value) ? 'Enabled' : 'Disabled'}
                        </span>
                    </label>
                );

            case 'integer':
            case 'float':
                return (
                    <div className="config-slider-group">
                        <input
                            type="range"
                            className="config-slider"
                            min={item.min ?? 0}
                            max={item.max ?? 100}
                            step={item.type === 'float' ? 0.01 : 1}
                            value={Number(value) || 0}
                            onChange={(e) => handleValueChange(key, item.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value))}
                        />
                        <input
                            type="number"
                            className={`config-number-input ${hasChange ? 'modified' : ''}`}
                            min={item.min}
                            max={item.max}
                            step={item.type === 'float' ? 0.01 : 1}
                            value={Number(value) || 0}
                            onChange={(e) => handleValueChange(key, item.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value))}
                        />
                    </div>
                );

            case 'string':
            default:
                return (
                    <input
                        type="text"
                        className={`config-text-input ${hasChange ? 'modified' : ''}`}
                        value={String(value ?? '')}
                        onChange={(e) => handleValueChange(key, e.target.value)}
                        placeholder="Enter value..."
                    />
                );
        }
    };

    const categorizedItems = getCategorizedItems();
    const currentCategoryItems = categorizedItems.find(c => c.name === activeCategory)?.items || [];
    const hasChanges = Object.keys(pendingChanges).length > 0;

    return (
        <div className="config-page animate-fadeIn">
            {/* Header with Actions */}
            <div className="config-header">
                <div className="config-header-info">
                    <h2>Configuration</h2>
                    <p>Manage RAG pipeline settings, feature flags, and API configurations</p>
                </div>
                <div className="config-header-actions">
                    <button
                        className="config-btn config-btn-ghost"
                        onClick={handleReload}
                        disabled={loading}
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="23 4 23 10 17 10" />
                            <polyline points="1 20 1 14 7 14" />
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                        </svg>
                        Reload from .env
                    </button>
                </div>
            </div>

            {/* Status Messages */}
            {(error || saveMessage) && (
                <div className={`config-alert ${error || saveMessage?.type === 'error' ? 'config-alert-error' : 'config-alert-success'}`}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        {error || saveMessage?.type === 'error' ? (
                            <><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></>
                        ) : (
                            <><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></>
                        )}
                    </svg>
                    {error || saveMessage?.text}
                </div>
            )}

            {loading && !Object.keys(config).length ? (
                <div className="config-loading">
                    <div className="spinner" />
                    <span>Loading configuration...</span>
                </div>
            ) : (
                <div className="config-layout">
                    {/* Categories Sidebar */}
                    <aside className="config-sidebar">
                        <div className="config-sidebar-header">
                            <span>Categories</span>
                            <span className="config-sidebar-count">{categories.length}</span>
                        </div>
                        <nav className="config-sidebar-nav">
                            {categories.map(category => {
                                const categoryItems = categorizedItems.find(c => c.name === category)?.items || [];
                                const modifiedCount = categoryItems.filter(({ key }) => key in pendingChanges).length;
                                
                                return (
                                    <button
                                        key={category}
                                        className={`config-sidebar-item ${activeCategory === category ? 'active' : ''}`}
                                        onClick={() => setActiveCategory(category)}
                                    >
                                        <span className="config-sidebar-item-icon">
                                            {categoryIcons[category] || defaultIcon}
                                        </span>
                                        <span className="config-sidebar-item-label">{category}</span>
                                        {modifiedCount > 0 && (
                                            <span className="config-sidebar-item-badge">{modifiedCount}</span>
                                        )}
                                    </button>
                                );
                            })}
                        </nav>
                    </aside>

                    {/* Config Items */}
                    <main className="config-main">
                        <div className="config-main-header">
                            <div className="config-main-title">
                                <span className="config-main-title-icon">
                                    {categoryIcons[activeCategory] || defaultIcon}
                                </span>
                                <h3>{activeCategory}</h3>
                            </div>
                            <span className="config-main-count">{currentCategoryItems.length} settings</span>
                        </div>
                        
                        <div className="config-items-grid">
                            {currentCategoryItems.map(({ key, item }) => {
                                const isModified = key in pendingChanges;
                                return (
                                    <div
                                        key={key}
                                        className={`config-item-card ${isModified ? 'modified' : ''}`}
                                    >
                                        <div className="config-item-header">
                                            <span className="config-item-label">
                                                {formatLabel(key)}
                                            </span>
                                            {isModified && (
                                                <span className="config-item-modified-badge">Modified</span>
                                            )}
                                        </div>
                                        <p className="config-item-description">{item.description}</p>
                                        <div className="config-item-control">
                                            {renderConfigInput(key, item)}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </main>
                </div>
            )}

            {/* Floating Action Bar */}
            {hasChanges && (
                <div className="config-action-bar">
                    <div className="config-action-bar-info">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        <span><strong>{Object.keys(pendingChanges).length}</strong> unsaved change{Object.keys(pendingChanges).length > 1 ? 's' : ''}</span>
                    </div>
                    <div className="config-action-bar-buttons">
                        <button className="config-btn config-btn-ghost" onClick={handleReset}>
                            Discard
                        </button>
                        <button
                            className="config-btn config-btn-primary"
                            onClick={handleSave}
                            disabled={saving}
                        >
                            {saving ? (
                                <>
                                    <span className="spinner" style={{ width: 14, height: 14 }} />
                                    Saving...
                                </>
                            ) : (
                                <>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                                        <polyline points="17 21 17 13 7 13 7 21" />
                                        <polyline points="7 3 7 8 15 8" />
                                    </svg>
                                    Save Changes
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
