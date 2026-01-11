import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const commands = ['sync-knowledge', 'rotate-keys', 'deploy-agent', 'fetch-metrics'];

const SettingsMCP = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <div className="section-title">
          <h2>MCP control plane</h2>
          <Button onClick={() => openPreview('settings-open-terminal')}>Open terminal</Button>
        </div>
        <div className="grid-2">
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Server status</h3>
            <p style={{ color: 'var(--muted)' }}>Online · 42 commands processed today</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Automations</h3>
            <p style={{ color: 'var(--muted)' }}>3 scheduled syncs · 1 paused workflow</p>
          </div>
        </div>
        <div className="card" style={{ borderRadius: 12 }}>
          <h3>Available commands</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {commands.map((command) => (
              <span key={command} className="tag">
                {command}
              </span>
            ))}
          </div>
        </div>
        <div className="card" style={{ borderRadius: 12 }}>
          <h3>Terminal</h3>
          <pre style={{ background: 'rgba(15,23,42,0.7)', padding: 16, borderRadius: 12, color: 'var(--info)' }}>
            {`$ deploy-agent --establishment="Helios" --agent="Nova"
> Agent deployment scheduled`}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default SettingsMCP;
