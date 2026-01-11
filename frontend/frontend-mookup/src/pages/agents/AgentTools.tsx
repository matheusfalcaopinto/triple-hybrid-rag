import { integrations } from '../../data/mockData';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const AgentTools = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="section-title">
        <h2>Connected tools</h2>
        <Button onClick={() => openPreview('agent-integration-add')}>Add integration</Button>
      </div>
      <div className="card-grid">
        {integrations.map((tool) => (
          <div key={tool.name} className="card" style={{ borderRadius: 16 }}>
            <h3>{tool.name}</h3>
            <p style={{ color: 'var(--muted)' }}>{tool.description}</p>
            <Button variant="outline" onClick={() => openPreview('agent-integration-configure')}>
              Configure
            </Button>
          </div>
        ))}
        <div className="card" style={{ borderRadius: 16 }}>
          <h3>MCP Custom</h3>
          <p style={{ color: 'var(--muted)' }}>
            Build bespoke automations via MCP commands and secure tunnels.
          </p>
          <Button variant="secondary" onClick={() => openPreview('agent-mcp-builder')}>
            Launch builder
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AgentTools;
