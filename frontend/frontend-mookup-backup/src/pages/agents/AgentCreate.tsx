import { Crown, DollarSign } from 'lucide-react';

const AgentCreate = () => {
  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Create AI agent</h2>
          <span className="tag">6 sections</span>
        </div>
        <div className="section">
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>1. Basic info</h3>
            <p style={{ color: 'var(--muted)' }}>Name, description, agent type, avatar</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>2. Voice config</h3>
            <p style={{ color: 'var(--muted)' }}>Tier selector, preview, language, speed</p>
            <div style={{ display: 'flex', gap: 12 }}>
              <span className="tag" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <Crown size={12} /> Premium $0.13-0.17
              </span>
              <span className="tag" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <DollarSign size={12} /> Economic $0.11-0.14
              </span>
            </div>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>3. System prompt</h3>
            <p style={{ color: 'var(--muted)' }}>4000 chars with templates and guardrails</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>4. LLM parameters</h3>
            <p style={{ color: 'var(--muted)' }}>Temperature, max tokens, timeout</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>5. Call settings</h3>
            <p style={{ color: 'var(--muted)' }}>Duration limits, schedule, blacklist</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>6. Twilio number</h3>
            <p style={{ color: 'var(--muted)' }}>Number provisioning and SIP routing</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentCreate;
