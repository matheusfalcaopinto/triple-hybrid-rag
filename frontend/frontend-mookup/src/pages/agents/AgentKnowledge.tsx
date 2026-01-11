import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Textarea } from '../../components/ui/textarea';

const knowledgeLevels = ['General', 'Agent', 'Customer'];
const documents = ['Onboarding.pdf', 'Pricing.xlsx', 'Troubleshooting.md', 'VIP Accounts.csv'];

const AgentKnowledge = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card" style={{ display: 'grid', gridTemplateColumns: '240px 280px 1fr', gap: 16 }}>
        <div>
          <h3>Knowledge levels</h3>
          <div style={{ display: 'grid', gap: '12px', marginTop: 12 }}>
            {knowledgeLevels.map((level, index) => (
              <Button
                key={level}
                variant={index === 0 ? 'secondary' : 'ghost'}
                className="tab"
                style={{ width: '100%', justifyContent: 'flex-start' }}
                onClick={() => openPreview('knowledge-levels')}
              >
                {level}
              </Button>
            ))}
          </div>
        </div>
        <div>
          <h3>Documents</h3>
          <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
            {documents.map((doc) => (
              <div key={doc} className="card" style={{ borderRadius: 12 }}>
                {doc}
              </div>
            ))}
            <Button variant="outline" onClick={() => openPreview('knowledge-upload')}>
              Upload
            </Button>
          </div>
        </div>
        <div>
          <h3>Editor</h3>
          <div className="card" style={{ borderRadius: 12, minHeight: 240 }}>
            <p style={{ color: 'var(--muted)' }}>
              Combine global policies with agent-specific FAQs. Highlight
              critical compliance statements and auto-sync with CRM records.
            </p>
            <Textarea
              defaultValue={`## Verification steps
- Confirm caller identity
- Validate service address
- Note utility plan preference`}
              style={{ minHeight: 120, resize: 'vertical' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentKnowledge;
