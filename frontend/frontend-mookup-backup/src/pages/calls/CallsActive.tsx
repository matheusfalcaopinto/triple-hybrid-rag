import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const activeCalls = [
  { agent: 'Nova GPT', number: '+1 555-010-2244', duration: '03:22', status: 'In conversation' },
  { agent: 'Hera Concierge', number: '+1 555-018-4481', duration: '07:10', status: 'Escalating' },
  { agent: 'Atlas Billing', number: '+1 555-021-3378', duration: '01:58', status: 'On hold' },
];

const CallsActive = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="section-title">
        <h2>Live calls</h2>
        <Button variant="secondary" size="sm" onClick={() => openPreview('calls-intervene')}>
          Intervene
        </Button>
      </div>
      <div className="call-grid">
        {activeCalls.map((call) => (
          <div key={call.number} className="card" style={{ borderRadius: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div>
                <strong>{call.agent}</strong>
                <div style={{ color: 'var(--muted)', fontSize: 12 }}>{call.number}</div>
              </div>
              <span className="tag">{call.status}</span>
            </div>
            <div style={{ fontSize: 32, fontWeight: 700 }}>{call.duration}</div>
            <div style={{ height: 60, background: 'linear-gradient(90deg, rgba(99,102,241,0.2), transparent)', borderRadius: 12 }} />
          </div>
        ))}
      </div>
      <div className="card" style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
        <div>
          <h3>Live transcript</h3>
          <div className="transcript">
            <div><strong>Customer:</strong> I need to update my service plan.</div>
            <div><strong>Nova GPT:</strong> Happy to help! Let me confirm your service address.</div>
            <div><strong>Customer:</strong> 52 Ocean View Rd.</div>
          </div>
        </div>
        <div>
          <h3>Sentiment & customer</h3>
          <div className="card" style={{ borderRadius: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Sentiment</span>
              <span className="tag" style={{ background: 'rgba(16,185,129,0.16)', color: 'var(--success)' }}>Positive</span>
            </div>
            <div style={{ marginTop: 12, color: 'var(--muted)', fontSize: 13 }}>
              Customer tier: Gold<br />
              Account health: Stable<br />
              Recent issues: Billing dispute
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CallsActive;
