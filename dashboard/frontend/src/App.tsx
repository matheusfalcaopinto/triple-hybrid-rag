import { useState, useEffect, useCallback } from 'react';
import './index.css';
import { Sidebar } from './components/Sidebar';
import { ConfigPanel } from './components/ConfigPanel';
import { FileUpload } from './components/FileUpload';
import { QueryInterface } from './components/QueryInterface';
import { MetricsDashboard } from './components/MetricsDashboard';
import { GraphViewer } from './components/GraphViewer';
import { DatabaseBrowser } from './components/DatabaseBrowser';
import type { Page } from './types';
import type { MetricsResponse } from './hooks/useApi';
import { useMetrics } from './hooks/useApi';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('dashboard');
  const { data: metrics, fetchMetrics } = useMetrics();

  const loadMetrics = useCallback(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  useEffect(() => {
    loadMetrics();
    // Refresh metrics every 30 seconds
    const interval = setInterval(loadMetrics, 30000);
    return () => clearInterval(interval);
  }, [loadMetrics]);

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <MetricsDashboard metrics={metrics as MetricsResponse | null} onRefresh={loadMetrics} />;
      case 'config':
        return <ConfigPanel />;
      case 'ingestion':
        return <FileUpload />;
      case 'retrieval':
        return <QueryInterface />;
      case 'database':
        return <DatabaseBrowser />;
      case 'graph':
        return <GraphViewer />;
      default:
        return <MetricsDashboard metrics={metrics as MetricsResponse | null} onRefresh={loadMetrics} />;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  );
}

export default App;
