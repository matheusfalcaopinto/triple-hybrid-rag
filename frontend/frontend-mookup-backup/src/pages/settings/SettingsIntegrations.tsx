import { integrations } from '../../data/mockData';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const SettingsIntegrations = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="section-title">
        <h2>Integrations</h2>
        <Button onClick={() => openPreview('settings-connect-integration')}>Connect new</Button>
      </div>
      <div className="card-grid">
        {integrations.map((integration) => (
          <div key={integration.name} className="card" style={{ borderRadius: 16 }}>
            <h3>{integration.name}</h3>
            <p style={{ color: 'var(--muted)' }}>{integration.description}</p>
            <Button variant="outline" onClick={() => openPreview('settings-manage-integration')}>
              Manage
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SettingsIntegrations;
