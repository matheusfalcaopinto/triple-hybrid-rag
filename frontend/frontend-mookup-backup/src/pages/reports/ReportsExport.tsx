import { usePreviewStore } from '../../stores/usePreviewStore';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';

const templates = ['Daily summary', 'Weekly executive', 'Monthly compliance', 'Custom CSV'];

const ReportsExport = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <Card className="report-card">
        <CardHeader className="section-title">
          <CardTitle>Export center</CardTitle>
          <Button variant="secondary" size="sm" onClick={() => openPreview('reports-create-template')}>
            Create template
          </Button>
        </CardHeader>
        <CardContent className="card-grid">
          {templates.map((template) => (
            <Card key={template} className="report-template">
              <CardHeader>
                <CardTitle>{template}</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Configure destinations, cadence, and filters for recurring exports.</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => openPreview('reports-schedule-export')}
                >
                  Schedule
                </Button>
              </CardContent>
            </Card>
          ))}
        </CardContent>
      </Card>
      <Card className="report-card">
        <CardHeader className="section-title">
          <CardTitle>Custom builder</CardTitle>
          <Badge variant="outline">Advanced</Badge>
        </CardHeader>
        <CardContent className="grid-2">
          <Card className="report-template">
            <CardHeader>
              <CardTitle>Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Select KPIs, date range, tiers</p>
            </CardContent>
          </Card>
          <Card className="report-template">
            <CardHeader>
              <CardTitle>Destinations</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Email, S3, Data Warehouse</p>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};

export default ReportsExport;
