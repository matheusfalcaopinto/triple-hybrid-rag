import { Filter, Bot, Crown, DollarSign } from 'lucide-react';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const agents = [
  {
    name: 'Nova GPT',
    establishment: 'Helios Energy',
    status: 'Online',
    tier: 'Premium',
    metrics: '98 calls • 4.8 CSAT',
  },
  {
    name: 'Hera Concierge',
    establishment: 'Andromeda Hotels',
    status: 'Calling',
    tier: 'Premium',
    metrics: '76 calls • 92% success',
  },
  {
    name: 'Atlas Billing',
    establishment: 'Poseidon Marine',
    status: 'Paused',
    tier: 'Economic',
    metrics: '41 calls • 88% success',
  },
];

const AgentsList = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="section-title">
        <h2>AI Agents</h2>
        <Button onClick={() => openPreview('agents-new')}>New agent</Button>
      </div>
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', gap: 12 }}>
          <Button variant="outline" onClick={() => openPreview('agents-filters')}>
            <Filter size={14} /> Filters
          </Button>
          <span className="tag" style={{ background: 'rgba(99,102,241,0.15)' }}>Voice tiers</span>
        </div>
        <div className="card-grid">
          {agents.map((agent) => (
            <div className="card" key={agent.name} style={{ borderRadius: 16 }}>
              <div style={{ display: 'flex', gap: 12 }}>
                <div style={{ width: 48, height: 48, borderRadius: 16, background: 'rgba(99,102,241,0.2)', display: 'grid', placeItems: 'center' }}>
                  <Bot size={20} />
                </div>
                <div>
                  <strong>{agent.name}</strong>
                  <div style={{ fontSize: 12, color: 'var(--muted)' }}>{agent.establishment}</div>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <span className="tag">{agent.status}</span>
                <span className="tag" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <Crown size={12} /> {agent.tier}
                </span>
                <span className="tag" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <DollarSign size={12} /> {agent.metrics}
                </span>
              </div>
              <Button
                variant="outline"
                onClick={() => openPreview('agents-configure')}
              >
                Configure
              </Button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AgentsList;
