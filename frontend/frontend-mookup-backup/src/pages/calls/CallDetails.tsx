import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const transcript = [
  { speaker: 'Agent', text: 'Hello! Thanks for calling Helios Energy.' },
  { speaker: 'Customer', text: 'I received an alert about high usage.' },
  { speaker: 'Agent', text: 'I can help. Let me review your recent consumption.' },
];

const CallDetails = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section" style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
      <div className="card">
        <h2>Call metadata</h2>
        <div style={{ display: 'grid', gap: 8, marginTop: 12, color: 'var(--muted)', fontSize: 13 }}>
          <span>Date: 2024-08-16 14:32 UTC</span>
          <span>Agent: Nova GPT</span>
          <span>Customer: Maria Costa</span>
          <span>Duration: 08:44</span>
          <span>Tier: Premium</span>
          <span>Cost: $3.12</span>
        </div>
      </div>
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <div className="section-title">
          <h2>Transcript</h2>
          <Button variant="secondary" size="sm" onClick={() => openPreview('call-download-audio')}>
            Download audio
          </Button>
        </div>
        <div className="transcript">
          {transcript.map((item, index) => (
            <div key={index}>
              <strong style={{ display: 'block', fontSize: 12, color: 'var(--muted)' }}>{item.speaker}</strong>
              {item.text}
            </div>
          ))}
        </div>
        <audio controls style={{ width: '100%' }}>
          <source src="" />
        </audio>
      </div>
    </div>
  );
};

export default CallDetails;
