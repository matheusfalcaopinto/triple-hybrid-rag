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

    const renderConfigInput = (key: string, item: ConfigItem) => {
        const value = getCurrentValue(key, item);
        const hasChange = key in pendingChanges;

        switch (item.type) {
            case 'boolean':
                return (
                    <label className="toggle">
                        <input
                            type="checkbox"
                            className="toggle-input"
                            checked={Boolean(value)}
                            onChange={(e) => handleValueChange(key, e.target.checked)}
                        />
                        <span className="toggle-slider" />
                    </label>
                );

            case 'integer':
            case 'float':
                return (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                        <input
                            type="range"
                            className="slider"
                            min={item.min ?? 0}
                            max={item.max ?? 100}
                            step={item.type === 'float' ? 0.1 : 1}
                            value={Number(value) || 0}
                            onChange={(e) => handleValueChange(key, item.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value))}
                            style={{ flex: 1, minWidth: 100 }}
                        />
                        <input
                            type="number"
                            className="form-input"
                            min={item.min}
                            max={item.max}
                            step={item.type === 'float' ? 0.1 : 1}
                            value={Number(value) || 0}
                            onChange={(e) => handleValueChange(key, item.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value))}
                            style={{ width: 80 }}
                        />
                    </div>
                );

            case 'string':
            default:
                return (
                    <input
                        type="text"
                        className="form-input"
                        value={String(value ?? '')}
                        onChange={(e) => handleValueChange(key, e.target.value)}
                        style={{
                            width: '100%',
                            background: hasChange ? 'rgba(99, 102, 241, 0.1)' : undefined,
                            borderColor: hasChange ? 'var(--color-accent-primary)' : undefined,
                        }}
                    />
                );
        }
    };

    const categorizedItems = getCategorizedItems();
    const currentCategoryItems = categorizedItems.find(c => c.name === activeCategory)?.items || [];
    const hasChanges = Object.keys(pendingChanges).length > 0;

    return (
        <div className="animate-fadeIn">
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                    <h2>Configuration</h2>
                    <p>Manage RAG pipeline settings and feature flags</p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                    <button
                        className="btn btn-secondary"
                        onClick={handleReload}
                        disabled={loading}
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="23 4 23 10 17 10" />
                            <polyline points="1 20 1 14 7 14" />
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                        </svg>
                        Reload
                    </button>
                    {hasChanges && (
                        <>
                            <button className="btn btn-secondary" onClick={handleReset}>
                                Reset
                            </button>
                            <button
                                className="btn btn-primary"
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
                                        Save Changes ({Object.keys(pendingChanges).length})
                                    </>
                                )}
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* Error/Success Message */}
            {(error || saveMessage) && (
                <div
                    style={{
                        padding: 'var(--space-3)',
                        borderRadius: 'var(--radius-md)',
                        marginBottom: 'var(--space-4)',
                        background: (error || saveMessage?.type === 'error') 
                            ? 'rgba(239, 68, 68, 0.1)' 
                            : 'rgba(34, 197, 94, 0.1)',
                        color: (error || saveMessage?.type === 'error')
                            ? 'var(--color-error)'
                            : 'var(--color-success)',
                    }}
                >
                    {error || saveMessage?.text}
                </div>
            )}

            {loading && !Object.keys(config).length ? (
                <div style={{ textAlign: 'center', padding: 'var(--space-12)' }}>
                    <div className="spinner" style={{ margin: '0 auto' }} />
                </div>
            ) : (
                <div style={{ display: 'flex', gap: 'var(--space-6)' }}>
                    {/* Category Sidebar */}
                    <div style={{ width: 220, flexShrink: 0 }}>
                        <div
                            style={{
                                background: 'var(--color-bg-secondary)',
                                borderRadius: 'var(--radius-xl)',
                                padding: 'var(--space-2)',
                            }}
                        >
                            {categories.map(category => (
                                <button
                                    key={category}
                                    className={`nav-item ${activeCategory === category ? 'active' : ''}`}
                                    onClick={() => setActiveCategory(category)}
                                    style={{ width: '100%', textAlign: 'left' }}
                                >
                                    {category}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Config Items */}
                    <div style={{ flex: 1 }}>
                        <div className="card">
                            <h3 style={{ marginBottom: 'var(--space-6)' }}>{activeCategory}</h3>
                            
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                                {currentCategoryItems.map(({ key, item }) => (
                                    <div
                                        key={key}
                                        className="config-item"
                                        style={{
                                            background: key in pendingChanges 
                                                ? 'rgba(99, 102, 241, 0.05)' 
                                                : 'var(--color-bg-tertiary)',
                                            border: key in pendingChanges 
                                                ? '1px solid rgba(99, 102, 241, 0.3)' 
                                                : '1px solid transparent',
                                        }}
                                    >
                                        <div className="config-item-info">
                                            <div className="config-item-label">
                                                {key.replace(/^rag_/, '').replace(/_/g, ' ')}
                                                {key in pendingChanges && (
                                                    <span
                                                        style={{
                                                            marginLeft: 'var(--space-2)',
                                                            color: 'var(--color-accent-primary)',
                                                            fontSize: 'var(--text-xs)',
                                                        }}
                                                    >
                                                        (modified)
                                                    </span>
                                                )}
                                            </div>
                                            <div className="config-item-description">
                                                {item.description}
                                            </div>
                                        </div>
                                        <div style={{ minWidth: item.type === 'string' ? 250 : 180 }}>
                                            {renderConfigInput(key, item)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
