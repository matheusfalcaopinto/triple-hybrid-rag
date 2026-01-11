import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const EstablishmentBilling = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Billing overview</h2>
          <Button variant="secondary" size="sm" onClick={() => openPreview('establishment-billing-download')}>
            Download invoice
          </Button>
        </div>
        <div className="grid-3">
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Current cycle</h3>
            <div className="metric">$4,287</div>
            <span className="kpi-trend">Usage 72% of budget</span>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Projected</h3>
            <div className="metric">$5,840</div>
            <span className="kpi-trend" style={{ color: 'var(--warning)' }}>
              Forecast +12%
            </span>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Invoices</h3>
            <div className="metric">3</div>
            <span className="kpi-trend">Last paid Aug 15</span>
          </div>
        </div>
      </div>
      <div className="card">
        <div className="section-title">
          <h3>Cost breakdown</h3>
          <span className="tag">Premium vs Economic</span>
        </div>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Tier</th>
                <th>Minutes</th>
                <th>Rate</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Premium</td>
                <td>3,240</td>
                <td>$0.16</td>
                <td>$518.40</td>
              </tr>
              <tr>
                <td>Economic</td>
                <td>6,232</td>
                <td>$0.12</td>
                <td>$747.84</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div className="card">
        <div className="section-title">
          <h3>Budget alerts</h3>
          <Button variant="secondary" size="sm" onClick={() => openPreview('establishment-budget-config')}>
            Configure
          </Button>
        </div>
        <div className="timeline">
          <div className="timeline-item">Notify finance when 80% consumed</div>
          <div className="timeline-item">Escalate to ops at 95% usage</div>
        </div>
      </div>
    </div>
  );
};

export default EstablishmentBilling;
