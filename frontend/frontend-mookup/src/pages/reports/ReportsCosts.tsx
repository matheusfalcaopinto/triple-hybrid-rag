import { useState, useMemo } from 'react';
import { useCostReport } from '../../api/hooks/useReports';
import { usePreviewStore } from '../../stores/usePreviewStore';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from '../../components/ui/table';
import { Loader2, AlertCircle, DollarSign, TrendingUp, TrendingDown } from 'lucide-react';

const ReportsCosts = () => {
  const openPreview = usePreviewStore((state) => state.open);
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

  const { data: costData, isLoading, error, refetch } = useCostReport(filters);

  // Calculate budget percentage (assuming a default monthly budget)
  const monthlyBudget = 7500; // This would come from settings
  const budgetPercentage = costData ? Math.round((costData.total_cost / monthlyBudget) * 100) : 0;
  const forecastedCost = costData ? costData.total_cost * 1.12 : 0; // Simple forecast

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px' }}>
          <Loader2 className="animate-spin" size={32} style={{ color: 'var(--primary)' }} />
          <span style={{ marginLeft: 12 }}>Loading cost data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px', flexDirection: 'column', gap: 16 }}>
          <AlertCircle size={32} style={{ color: 'var(--danger)' }} />
          <span>Failed to load cost data</span>
          <Button onClick={() => refetch()}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      {/* Date Range Selector */}
      <div style={{ marginBottom: 16, display: 'flex', gap: 8 }}>
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

      {/* KPI Cards */}
      <div className="card-grid">
        <Card className="report-card">
          <CardHeader>
            <CardTitle style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <DollarSign size={18} />
              Total Cost
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="metric">${costData?.total_cost?.toFixed(2) || '0.00'}</div>
            <span className="kpi-trend">Budget at {budgetPercentage}%</span>
          </CardContent>
        </Card>
        <Card className="report-card">
          <CardHeader>
            <CardTitle style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <TrendingUp size={18} />
              Forecast
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="metric">${forecastedCost.toFixed(2)}</div>
            <span className="kpi-trend" style={{ color: forecastedCost > monthlyBudget ? 'var(--warning)' : 'var(--success)' }}>
              {forecastedCost > monthlyBudget ? (
                <><TrendingUp size={14} /> Over budget</>
              ) : (
                <><TrendingDown size={14} /> Within budget</>
              )}
            </span>
          </CardContent>
        </Card>
        <Card className="report-card">
          <CardHeader>
            <CardTitle>Cost per Call</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="metric">${costData?.cost_per_call?.toFixed(2) || '0.00'}</div>
            <span className="kpi-trend">Average across all agents</span>
          </CardContent>
        </Card>
      </div>

      {/* Cost by Agent Table */}
      <Card className="report-card">
        <CardHeader className="section-title">
          <CardTitle>Cost by Agent</CardTitle>
          <Button variant="secondary" size="sm" onClick={() => openPreview('reports-view-invoices')}>
            View invoices
          </Button>
        </CardHeader>
        <CardContent className="table-scroll">
          {costData?.cost_by_agent && costData.cost_by_agent.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Agent</TableHead>
                  <TableHead>Cost</TableHead>
                  <TableHead>% of Total</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {costData.cost_by_agent.map((agent) => (
                  <TableRow key={agent.agent_id}>
                    <TableCell>{agent.agent_name}</TableCell>
                    <TableCell>${agent.cost.toFixed(2)}</TableCell>
                    <TableCell>
                      {costData.total_cost > 0 
                        ? `${((agent.cost / costData.total_cost) * 100).toFixed(1)}%`
                        : '0%'
                      }
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div style={{ textAlign: 'center', padding: 32, color: 'var(--muted)' }}>
              No cost data available for this period
            </div>
          )}
        </CardContent>
      </Card>

      {/* Cost Breakdown */}
      <Card className="report-card">
        <CardHeader className="section-title">
          <CardTitle>Cost Breakdown</CardTitle>
          <Badge variant="outline">By Service Provider</Badge>
        </CardHeader>
        <CardContent className="grid-2">
          <Card className="report-tier">
            <CardHeader>
              <CardTitle>Telephony (Twilio)</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Estimated: ${((costData?.total_cost || 0) * 0.45).toFixed(2)}</p>
              <p style={{ color: 'var(--muted)', fontSize: 13 }}>~45% of total</p>
            </CardContent>
          </Card>
          <Card className="report-tier">
            <CardHeader>
              <CardTitle>Voice AI (Cartesia)</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Estimated: ${((costData?.total_cost || 0) * 0.35).toFixed(2)}</p>
              <p style={{ color: 'var(--muted)', fontSize: 13 }}>~35% of total</p>
            </CardContent>
          </Card>
          <Card className="report-tier">
            <CardHeader>
              <CardTitle>LLM (OpenAI/Anthropic)</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Estimated: ${((costData?.total_cost || 0) * 0.15).toFixed(2)}</p>
              <p style={{ color: 'var(--muted)', fontSize: 13 }}>~15% of total</p>
            </CardContent>
          </Card>
          <Card className="report-tier">
            <CardHeader>
              <CardTitle>Other Services</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Estimated: ${((costData?.total_cost || 0) * 0.05).toFixed(2)}</p>
              <p style={{ color: 'var(--muted)', fontSize: 13 }}>~5% of total</p>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};

export default ReportsCosts;
