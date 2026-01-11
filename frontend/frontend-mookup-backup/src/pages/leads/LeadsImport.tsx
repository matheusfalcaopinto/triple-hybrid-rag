import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const steps = ['Upload', 'Map fields', 'Configure rules', 'Confirm'];

const LeadsImport = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Import leads</h2>
          <span className="tag">4-step wizard</span>
        </div>
        <div className="wizard-steps">
          {steps.map((step, index) => (
            <div key={step} className="wizard-step">
              <h4>{index + 1}. {step}</h4>
              <p style={{ color: 'var(--muted)', fontSize: 13 }}>
                {index === 0 && 'Upload CSV or connect CRM'}
                {index === 1 && 'Match source columns to platform fields'}
                {index === 2 && 'Deduplicate, assign agents, schedule windows'}
                {index === 3 && 'Validate sample data then import'}
              </p>
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <div className="section-title">
          <h3>Validation summary</h3>
          <Button variant="secondary" size="sm" onClick={() => openPreview('leads-download-report')}>
            Download report
          </Button>
        </div>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: 8, color: 'var(--muted)', fontSize: 13 }}>
          <li>• 1,248 leads detected</li>
          <li>• 32 duplicates flagged</li>
          <li>• 12 invalid numbers repaired</li>
          <li>• Assignment: Energy Growth Pod</li>
        </ul>
      </div>
    </div>
  );
};

export default LeadsImport;
