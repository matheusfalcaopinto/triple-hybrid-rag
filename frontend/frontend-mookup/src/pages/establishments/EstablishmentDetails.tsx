import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const tabs = ['Overview', 'Agents', 'Billing', 'Settings'];

const tabKeyMap: Record<string, string> = {
  Overview: 'establishment-tab-overview',
  Agents: 'establishment-tab-agents',
  Billing: 'establishment-tab-billing',
  Settings: 'establishment-tab-settings',
};

const EstablishmentDetails = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Helios Energy HQ</h2>
          <span className="tag">Enterprise</span>
        </div>
        <p style={{ color: 'var(--muted)', margin: 0 }}>
          CNPJ 12.345.678/0001-90 · São Paulo · Operating 24/7 · Budget $12k
        </p>
      </div>

      <div className="card">
        <div className="tabs">
          {tabs.map((tab, index) => (
            <Button
              key={tab}
              variant={index === 0 ? 'secondary' : 'ghost'}
              className={`tab${index === 0 ? ' active' : ''}`}
              onClick={() => openPreview(tabKeyMap[tab])}
            >
              {tab}
            </Button>
          ))}
        </div>
        <div className="section" style={{ paddingTop: 12 }}>
          <div className="grid-3">
            <div className="card" style={{ borderRadius: 12 }}>
              <h3>Calls (7d)</h3>
              <div className="metric">1,284</div>
              <span className="kpi-trend">+6.2% vs prior</span>
            </div>
            <div className="card" style={{ borderRadius: 12 }}>
              <h3>Minutes</h3>
              <div className="metric">9,472</div>
              <span className="kpi-trend">Budget at 64%</span>
            </div>
            <div className="card" style={{ borderRadius: 12 }}>
              <h3>CSAT</h3>
              <div className="metric">4.7</div>
              <span className="kpi-trend">Top performer</span>
            </div>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <div className="section-title">
              <h3>Agents</h3>
              <Button variant="secondary" size="sm" onClick={() => openPreview('establishment-add-agent')}>
                Add agent
              </Button>
            </div>
            <div style={{ display: 'grid', gap: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <div>
                  <strong>Nova GPT</strong>
                  <div style={{ color: 'var(--muted)', fontSize: 12 }}>Premium · 98 calls</div>
                </div>
                <span className="tag">Calling</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <div>
                  <strong>Atlas Billing</strong>
                  <div style={{ color: 'var(--muted)', fontSize: 12 }}>Economic · 64 calls</div>
                </div>
                <span className="tag" style={{ background: 'rgba(59,130,246,0.18)', color: 'var(--info)' }}>Online</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EstablishmentDetails;
