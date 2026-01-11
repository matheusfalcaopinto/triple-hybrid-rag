import { Sliders } from 'lucide-react';

const AgentConfig = () => {
  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Agent configuration</h2>
          <span className="tag">Nova GPT · Premium tier</span>
        </div>
        <div className="grid-2">
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Prompt</h3>
            <p style={{ color: 'var(--muted)' }}>
              You are Nova GPT, a friendly concierge for Helios Energy. Confirm service addresses and energy usage before offering upgrades.
            </p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>LLM parameters</h3>
            <div style={{ display: 'grid', gap: 8, fontSize: 13 }}>
              <span>Temperature: 0.7</span>
              <span>Max tokens: 1024</span>
              <span>Timeout: 45s</span>
            </div>
          </div>
        </div>
        <div className="grid-2">
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Call settings</h3>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: 'var(--muted)', fontSize: 13, display: 'grid', gap: 6 }}>
              <li>Max duration: 15 minutes</li>
              <li>Schedule: Mon-Sat 08:00-22:00</li>
              <li>Blacklist: Finance escalation numbers</li>
            </ul>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Twilio number</h3>
            <p style={{ color: 'var(--muted)' }}>Provisioned: +1 (555) 301-2211</p>
            <span className="tag" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <Sliders size={12} /> SIP routing optimized
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentConfig;
