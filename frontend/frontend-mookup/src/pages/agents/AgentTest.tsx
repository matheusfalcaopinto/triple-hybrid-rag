import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Textarea } from '../../components/ui/textarea';

const messages = [
  { sender: 'Customer', text: 'Hi, I received a notification about my energy usage.' },
  { sender: 'Agent', text: 'Hello! I can help you optimize your usage. Could you confirm your service address?' },
  { sender: 'Customer', text: 'Sure, 782 Market Street.' },
  { sender: 'Agent', text: 'Thanks! I see you are eligible for the Helios Smart Saver plan.' },
];

const AgentTest = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <div className="section-title">
          <h2>Test conversation</h2>
          <span className="tag">Simulator</span>
        </div>
        <div className="transcript">
          {messages.map((message, index) => (
            <div key={index} style={{
              alignSelf: message.sender === 'Agent' ? 'flex-end' : 'flex-start',
              background: message.sender === 'Agent' ? 'rgba(99,102,241,0.18)' : 'rgba(148,163,184,0.12)',
              padding: '12px 16px',
              borderRadius: message.sender === 'Agent' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
            }}>
              <strong style={{ display: 'block', fontSize: 12, color: 'var(--muted)' }}>{message.sender}</strong>
              {message.text}
            </div>
          ))}
        </div>
        <Textarea placeholder="Type a test message" style={{ minHeight: 80 }} />
      </div>
      <div className="card">
        <div className="section-title">
          <h3>Debug info</h3>
          <Button variant="secondary" size="sm" onClick={() => openPreview('agent-test-download')}>
            Download JSON
          </Button>
        </div>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: 'var(--muted)', fontSize: 13, display: 'grid', gap: 6 }}>
          <li>Latency: 1.2s</li>
          <li>Model: Claude 3.5</li>
          <li>Voice: Nova Indigo (Premium)</li>
          <li>Sentiment: Positive</li>
        </ul>
      </div>
    </div>
  );
};

export default AgentTest;
