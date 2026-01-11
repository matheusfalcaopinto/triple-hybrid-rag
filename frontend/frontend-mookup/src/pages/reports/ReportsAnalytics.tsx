import { useState, useMemo } from 'react';
import { ResponsiveContainer, BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { useCallVolumeReport, useAgentComparisonReport, useSentimentReport } from '../../api/hooks/useReports';
import { Loader2, AlertCircle, TrendingUp, Users, Smile } from 'lucide-react';
import { Button } from '../../components/ui/button';

const ReportsAnalytics = () => {
  const [dateRange, setDateRange] = useState<'7d' | '30d' | '90d'>('30d');

  const filters = useMemo(() => {
    const now = new Date();
    const days = dateRange === '7d' ? 7 : dateRange === '30d' ? 30 : 90;
    const from = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);
    return {
      date_from: from.toISOString().split('T')[0],
      date_to: now.toISOString().split('T')[0],
    };
  }, [dateRange]);

  const { data: volumeData, isLoading: volumeLoading, error: volumeError } = useCallVolumeReport(filters);
  const { data: agentData, isLoading: agentLoading, error: agentError } = useAgentComparisonReport(filters);
  const { data: sentimentData, isLoading: sentimentLoading, error: sentimentError } = useSentimentReport(filters);

  const isLoading = volumeLoading || agentLoading || sentimentLoading;
  const error = volumeError || agentError || sentimentError;

  // Transform agent data for bar chart
  const agentChartData = useMemo(() => {
    if (!agentData?.agents) return [];
    return agentData.agents.map((item) => ({
      name: item.agent_name || 'Unknown',
      calls: item.total_calls || 0,
    }));
  }, [agentData]);

  // Transform sentiment data for pie chart
  const sentimentChartData = useMemo(() => {
    if (!sentimentData?.breakdown) return [];
    return [
      { name: 'Positive', value: sentimentData.breakdown.positive || 0, color: '#22c55e' },
      { name: 'Neutral', value: sentimentData.breakdown.neutral || 0, color: '#6366f1' },
      { name: 'Negative', value: sentimentData.breakdown.negative || 0, color: '#ef4444' },
    ];
  }, [sentimentData]);

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px' }}>
          <Loader2 className="animate-spin" size={32} style={{ color: 'var(--primary)' }} />
          <span style={{ marginLeft: 12 }}>Loading analytics...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px', flexDirection: 'column', gap: 16 }}>
          <AlertCircle size={32} style={{ color: 'var(--danger)' }} />
          <span>Failed to load analytics data</span>
          <Button onClick={() => window.location.reload()}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <div className="section-title">
          <h2>Analytics</h2>
          <div style={{ display: 'flex', gap: 8 }}>
            {(['7d', '30d', '90d'] as const).map((range) => (
              <button
                key={range}
                className={`tag ${dateRange === range ? 'tag-primary' : ''}`}
                onClick={() => setDateRange(range)}
                style={{ cursor: 'pointer', border: 'none' }}
              >
                {range === '7d' ? 'Last 7 days' : range === '30d' ? 'Last 30 days' : 'Last 90 days'}
              </button>
            ))}
          </div>
        </div>

        {/* Agent Performance Chart */}
        <div>
          <h3 style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Users size={18} />
            Agent Performance
          </h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={agentChartData}>
                <CartesianGrid vertical={false} stroke="rgba(148,163,184,0.2)" />
                <XAxis dataKey="name" stroke="rgba(148,163,184,0.5)" />
                <YAxis stroke="rgba(148,163,184,0.5)" />
                <Tooltip cursor={{ fill: 'rgba(99,102,241,0.1)' }} />
                <Bar dataKey="calls" fill="rgba(99,102,241,0.8)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Metrics Cards */}
        <div className="grid-3">
          <div className="card" style={{ borderRadius: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <TrendingUp size={18} style={{ color: 'var(--primary)' }} />
              <h3 style={{ margin: 0 }}>Call Volume</h3>
            </div>
            <div style={{ fontSize: 32, fontWeight: 700 }}>
              {volumeData?.total_calls?.toLocaleString() || '0'}
            </div>
            <p style={{ color: 'var(--muted)', margin: '8px 0 0' }}>
              Total calls in period
            </p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <Smile size={18} style={{ color: 'var(--success)' }} />
              <h3 style={{ margin: 0 }}>Sentiment</h3>
            </div>
            {sentimentChartData.length > 0 ? (
              <div style={{ height: 120 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sentimentChartData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={30}
                      outerRadius={50}
                    >
                      {sentimentChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p style={{ color: 'var(--muted)' }}>No sentiment data available</p>
            )}
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Trends</h3>
            <p style={{ color: 'var(--muted)' }}>Voice tier adoption by establishment</p>
            <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
              <span className="tag tag-success">+12% inbound</span>
              <span className="tag tag-warning">-3% outbound</span>
            </div>
          </div>
        </div>

        {/* Call Volume Over Time */}
        {volumeData?.data && volumeData.data.length > 0 && (
          <div>
            <h3 style={{ marginBottom: 16 }}>Daily Call Volume</h3>
            <div style={{ height: 200 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={volumeData.data}>
                  <CartesianGrid vertical={false} stroke="rgba(148,163,184,0.2)" />
                  <XAxis dataKey="timestamp" stroke="rgba(148,163,184,0.5)" />
                  <YAxis stroke="rgba(148,163,184,0.5)" />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportsAnalytics;
