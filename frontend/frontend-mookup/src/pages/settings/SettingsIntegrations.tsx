import { useState } from 'react';
import { 
  useAvailableIntegrations, 
  useIntegrationConnections, 
  useCreateConnection,
  useDeleteConnection 
} from '../../api/hooks/useIntegrations';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Loader2, AlertCircle, CheckCircle, Link, Unlink, Calendar, Phone, MessageSquare, Mail, Cloud, Bot } from 'lucide-react';
import type { Integration, IntegrationConnection } from '../../api/types';

const getIntegrationIcon = (type: string) => {
  switch (type) {
    case 'calendar': return <Calendar size={24} />;
    case 'crm': return <Cloud size={24} />;
    case 'voice': return <Phone size={24} />;
    case 'llm': return <Bot size={24} />;
    case 'sms': return <MessageSquare size={24} />;
    case 'email': return <Mail size={24} />;
    default: return <Link size={24} />;
  }
};

const SettingsIntegrations = () => {
  const openPreview = usePreviewStore((state) => state.open);
  const [selectedIntegration, setSelectedIntegration] = useState<Integration | null>(null);

  const { data: integrations, isLoading: integrationsLoading, error: integrationsError } = useAvailableIntegrations();
  const { data: connections, isLoading: connectionsLoading, refetch: refetchConnections } = useIntegrationConnections();
  const createConnection = useCreateConnection();
  const deleteConnection = useDeleteConnection();

  const isLoading = integrationsLoading || connectionsLoading;

  // Find connection for an integration
  const getConnection = (integrationId: string): IntegrationConnection | undefined => {
    return connections?.find((c) => c.integration_id === integrationId);
  };

  const handleConnect = (integration: Integration) => {
    if (integration.requires_oauth) {
      openPreview('settings-connect-integration');
    } else {
      createConnection.mutate(
        { integration_id: integration.id },
        { onSuccess: () => refetchConnections() }
      );
    }
  };

  const handleDisconnect = (connectionId: string) => {
    if (confirm('Are you sure you want to disconnect this integration?')) {
      deleteConnection.mutate(connectionId, {
        onSuccess: () => refetchConnections(),
      });
    }
  };

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px' }}>
          <Loader2 className="animate-spin" size={32} style={{ color: 'var(--primary)' }} />
          <span style={{ marginLeft: 12 }}>Loading integrations...</span>
        </div>
      </div>
    );
  }

  if (integrationsError) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px', flexDirection: 'column', gap: 16 }}>
          <AlertCircle size={32} style={{ color: 'var(--danger)' }} />
          <span>Failed to load integrations</span>
          <Button onClick={() => window.location.reload()}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="section-title">
        <h2>Integrations</h2>
        <Button onClick={() => openPreview('settings-connect-integration')}>Connect new</Button>
      </div>

      {/* Connected Integrations */}
      {connections && connections.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ marginBottom: 16 }}>Connected</h3>
          <div className="card-grid">
            {connections.map((connection) => (
              <div key={connection.id} className="card" style={{ borderRadius: 16, border: '2px solid var(--success)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <CheckCircle size={20} style={{ color: 'var(--success)' }} />
                    <h3 style={{ margin: 0 }}>{connection.integration_name}</h3>
                  </div>
                  <span className={`tag ${connection.is_active ? 'tag-success' : 'tag-warning'}`}>
                    {connection.is_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
                {connection.last_sync_at && (
                  <p style={{ color: 'var(--muted)', fontSize: 13, margin: '8px 0' }}>
                    Last synced: {new Date(connection.last_sync_at).toLocaleString()}
                  </p>
                )}
                <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                  <Button variant="outline" size="sm" onClick={() => openPreview('settings-manage-integration')}>
                    Configure
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDisconnect(connection.id)}
                    disabled={deleteConnection.isPending}
                  >
                    <Unlink size={14} style={{ marginRight: 4 }} />
                    Disconnect
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Available Integrations */}
      <div>
        <h3 style={{ marginBottom: 16 }}>Available integrations</h3>
        <div className="card-grid">
          {integrations?.filter((integration) => !getConnection(integration.id)).map((integration) => (
            <div
              key={integration.id}
              className="card"
              style={{
                borderRadius: 16,
                cursor: 'pointer',
                opacity: integration.is_available ? 1 : 0.6,
              }}
              onClick={() => setSelectedIntegration(integration)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                <div style={{ 
                  width: 48, 
                  height: 48, 
                  borderRadius: 12, 
                  background: 'var(--primary-light)', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  color: 'var(--primary)' 
                }}>
                  {getIntegrationIcon(integration.type)}
                </div>
                <div>
                  <h3 style={{ margin: 0 }}>{integration.name}</h3>
                  <span className="tag" style={{ fontSize: 10, marginTop: 4 }}>{integration.type}</span>
                </div>
              </div>
              <p style={{ color: 'var(--muted)', margin: '8px 0 12px' }}>{integration.description}</p>
              <Button
                variant="outline"
                onClick={(e) => {
                  e.stopPropagation();
                  handleConnect(integration);
                }}
                disabled={!integration.is_available || createConnection.isPending}
              >
                {createConnection.isPending ? (
                  <Loader2 className="animate-spin" size={14} style={{ marginRight: 4 }} />
                ) : (
                  <Link size={14} style={{ marginRight: 4 }} />
                )}
                {integration.is_available ? 'Connect' : 'Coming soon'}
              </Button>
            </div>
          ))}
        </div>
      </div>

      {/* Integration Details Modal/Panel could go here */}
      {selectedIntegration && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="section-title">
            <h3>Integration Details: {selectedIntegration.name}</h3>
            <Button variant="ghost" size="sm" onClick={() => setSelectedIntegration(null)}>
              Close
            </Button>
          </div>
          <p style={{ color: 'var(--muted)', marginBottom: 16 }}>{selectedIntegration.description}</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
            <div>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>Type</label>
              <p>{selectedIntegration.type}</p>
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>OAuth Required</label>
              <p>{selectedIntegration.requires_oauth ? 'Yes' : 'No'}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SettingsIntegrations;
