import { Filter, Bot, Crown, DollarSign, Loader2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { useAgents } from '../../api/hooks/useAgents';
import { config } from '../../config';
import { Button } from '../../components/ui/button';

// Mock data for fallback
const mockAgents = [
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
  const navigate = useNavigate();
  const openPreview = usePreviewStore((state) => state.open);
  
  // API hook
  const { data: agentsData, isLoading } = useAgents();
  
  // Use mock data if enabled or API data not available
  const useMock = config.enableMockData;
  const agents = useMock 
    ? mockAgents 
    : (agentsData?.items.map(a => ({
        id: a.id,
        name: a.name,
        establishment: 'Current', // Would need lookup
        status: a.status === 'active' ? 'Online' : a.status,
        tier: 'Standard',
        metrics: `${a.agent_type} agent`,
      })) || mockAgents);

  return (
    <div className="section">
      <div className="section-title">
        <h2>AI Agents</h2>
        <Button onClick={() => navigate('/agents/new')}>New agent</Button>
      </div>
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', gap: 12 }}>
          <Button variant="outline" onClick={() => openPreview('agents-filters')}>
            <Filter size={14} /> Filters
          </Button>
          <span className="tag" style={{ background: 'rgba(99,102,241,0.15)' }}>Voice tiers</span>
        </div>
        {isLoading && !useMock ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
            <Loader2 className="animate-spin" size={32} />
          </div>
        ) : (
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
                  onClick={() => navigate(`/agents/${'id' in agent ? agent.id : agent.name}`)}
                >
                  Configure
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentsList;
