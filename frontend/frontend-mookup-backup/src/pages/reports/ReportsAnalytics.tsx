import { ResponsiveContainer, BarChart, Bar, CartesianGrid, XAxis, Tooltip } from 'recharts';

const performance = [
  { name: 'Nova', calls: 124 },
  { name: 'Hera', calls: 98 },
  { name: 'Atlas', calls: 74 },
  { name: 'Echo', calls: 56 },
];

const ReportsAnalytics = () => {
  return (
    <div className="section">
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <div className="section-title">
          <h2>Analytics</h2>
          <div style={{ display: 'flex', gap: 8 }}>
            <span className="tag">Last 30 days</span>
            <span className="tag">Compare establishments</span>
          </div>
        </div>
        <div style={{ height: 260 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={performance}>
              <CartesianGrid vertical={false} stroke="rgba(148,163,184,0.2)" />
              <XAxis dataKey="name" stroke="rgba(148,163,184,0.5)" />
              <Tooltip cursor={{ fill: 'rgba(99,102,241,0.1)' }} />
              <Bar dataKey="calls" fill="rgba(99,102,241,0.8)" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="grid-3">
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Performance</h3>
            <p style={{ color: 'var(--muted)' }}>Top agents vs SLA targets</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Customer analysis</h3>
            <p style={{ color: 'var(--muted)' }}>Segments trending positive sentiment</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Trends</h3>
            <p style={{ color: 'var(--muted)' }}>Voice tier adoption by establishment</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportsAnalytics;
