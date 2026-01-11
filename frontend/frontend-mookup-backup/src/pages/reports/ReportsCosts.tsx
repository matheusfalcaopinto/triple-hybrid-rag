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

const tiers = [
  { name: 'Premium', rate: '$0.16', usage: '3,240 min', cost: '$518' },
  { name: 'Economic', rate: '$0.12', usage: '6,232 min', cost: '$748' },
];

const providers = [
  { service: 'Twilio', cost: '$2,840', trend: '+6%' },
  { service: 'Cartesia Voice', cost: '$1,220', trend: '+3%' },
  { service: 'Anthropic', cost: '$980', trend: '-2%' },
];

const ReportsCosts = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card-grid">
        <Card className="report-card">
          <CardHeader>
            <CardTitle>This month</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="metric">$5,040</div>
            <span className="kpi-trend">Budget at 68%</span>
          </CardContent>
        </Card>
        <Card className="report-card">
          <CardHeader>
            <CardTitle>Forecast</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="metric">$6,120</div>
            <span className="kpi-trend" style={{ color: 'var(--warning)' }}>+12%</span>
          </CardContent>
        </Card>
        <Card className="report-card">
          <CardHeader>
            <CardTitle>Budget alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="metric">3</div>
            <span className="kpi-trend">2 active, 1 snoozed</span>
          </CardContent>
        </Card>
      </div>
      <Card className="report-card">
        <CardHeader className="section-title">
          <CardTitle>API costs</CardTitle>
          <Button variant="secondary" size="sm" onClick={() => openPreview('reports-view-invoices')}>
            View invoices
          </Button>
        </CardHeader>
        <CardContent className="table-scroll">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Service</TableHead>
                <TableHead>Cost</TableHead>
                <TableHead>Trend</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {providers.map((provider) => (
                <TableRow key={provider.service}>
                  <TableCell>{provider.service}</TableCell>
                  <TableCell>{provider.cost}</TableCell>
                  <TableCell>{provider.trend}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <Card className="report-card">
        <CardHeader className="section-title">
          <CardTitle>Tier comparison</CardTitle>
          <Badge variant="outline">Premium vs Economic</Badge>
        </CardHeader>
        <CardContent className="grid-2">
          {tiers.map((tier) => (
            <Card key={tier.name} className="report-tier">
              <CardHeader>
                <CardTitle>{tier.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Rate {tier.rate}</p>
                <p>Usage {tier.usage}</p>
                <p>Cost {tier.cost}</p>
              </CardContent>
            </Card>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};

export default ReportsCosts;
