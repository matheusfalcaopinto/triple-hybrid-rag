import { CheckCircle2 } from 'lucide-react';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const steps = [
  {
    title: 'Basic data',
    description: 'CNPJ, name, address, operating region',
  },
  {
    title: 'Contacts',
    description: 'Owners, finance, escalation paths',
  },
  {
    title: 'Configuration',
    description: 'Hours, timezone, call routing, budget',
  },
  {
    title: 'Review',
    description: 'Verify SLAs and activation requirements',
  },
];

const EstablishmentNew = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Create establishment</h2>
          <span className="tag">4-step wizard</span>
        </div>
        <p style={{ color: 'var(--muted)', margin: '8px 0 24px' }}>
          Capture company metadata, configure operational hours, and align
          budgets before activating new agents.
        </p>
        <div className="wizard-steps">
          {steps.map((step, index) => (
            <div className="wizard-step" key={step.title}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h4>{index + 1}. {step.title}</h4>
                {index === 0 && <CheckCircle2 size={16} color="var(--success)" />}
              </div>
              <p style={{ color: 'var(--muted)', fontSize: 13 }}>{step.description}</p>
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <div className="section-title">
          <h2>Activation checklist</h2>
          <Button variant="secondary" size="sm" onClick={() => openPreview('establishment-activation-pdf')}>
            Download PDF
          </Button>
        </div>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: 12 }}>
          <li>• Verify legal entity and payment method</li>
          <li>• Configure voice tier defaults</li>
          <li>• Upload compliance recordings policy</li>
          <li>• Invite establishment admins</li>
        </ul>
      </div>
    </div>
  );
};

export default EstablishmentNew;
