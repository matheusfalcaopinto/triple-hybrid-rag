import { useState } from 'react';
import { useReportTemplates } from '../../api/hooks/useReports';
import { usePreviewStore } from '../../stores/usePreviewStore';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';
import { Loader2, Download, Calendar, FileSpreadsheet, Mail, Cloud } from 'lucide-react';

interface ExportTemplate {
  id: string;
  name: string;
  description: string;
  format: 'csv' | 'json' | 'xlsx';
  schedule?: string;
}

const defaultTemplates: ExportTemplate[] = [
  { id: '1', name: 'Daily summary', description: 'Daily call metrics and performance KPIs', format: 'csv' },
  { id: '2', name: 'Weekly executive', description: 'Executive summary with trends and insights', format: 'xlsx' },
  { id: '3', name: 'Monthly compliance', description: 'Compliance and audit trail export', format: 'csv' },
  { id: '4', name: 'Custom CSV', description: 'Configure your own export with custom fields', format: 'csv' },
];

const ReportsExport = () => {
  const openPreview = usePreviewStore((state) => state.open);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);

  const { data: templatesData, isLoading } = useReportTemplates();
  
  // Use API templates if available, otherwise fall back to defaults
  const templates: ExportTemplate[] = templatesData 
    ? templatesData.map((t: { id: string; name: string; description: string }) => ({
        id: t.id,
        name: t.name,
        description: t.description,
        format: 'csv' as const,
      }))
    : defaultTemplates;

  const handleExport = (template: ExportTemplate) => {
    setSelectedTemplate(template.id);
    openPreview('reports-schedule-export');
  };

  return (
    <div className="section">
      <Card className="report-card">
        <CardHeader className="section-title">
          <CardTitle style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <FileSpreadsheet size={20} />
            Export center
          </CardTitle>
          <Button variant="secondary" size="sm" onClick={() => openPreview('reports-create-template')}>
            Create template
          </Button>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 32 }}>
              <Loader2 className="animate-spin" size={24} style={{ color: 'var(--primary)' }} />
              <span style={{ marginLeft: 12 }}>Loading templates...</span>
            </div>
          ) : (
            <div className="card-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
              {(templates as ExportTemplate[]).map((template) => (
                <Card
                  key={template.id}
                  className="report-template"
                  style={{
                    cursor: 'pointer',
                    border: selectedTemplate === template.id ? '2px solid var(--primary)' : undefined,
                  }}
                  onClick={() => setSelectedTemplate(template.id)}
                >
                  <CardHeader>
                    <CardTitle style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      {template.name}
                      <Badge variant="outline">{template.format.toUpperCase()}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p style={{ color: 'var(--muted)', marginBottom: 16 }}>{template.description}</p>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleExport(template);
                        }}
                      >
                        <Calendar size={14} style={{ marginRight: 4 }} />
                        Schedule
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          openPreview('reports-download-now');
                        }}
                      >
                        <Download size={14} style={{ marginRight: 4 }} />
                        Download
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
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
              <CardTitle style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <FileSpreadsheet size={18} />
                Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p style={{ color: 'var(--muted)' }}>Select KPIs, date range, agents, and data fields to include in your export.</p>
              <Button
                variant="outline"
                size="sm"
                style={{ marginTop: 12 }}
                onClick={() => openPreview('reports-select-metrics')}
              >
                Configure
              </Button>
            </CardContent>
          </Card>
          <Card className="report-template">
            <CardHeader>
              <CardTitle style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Cloud size={18} />
                Destinations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
                <Badge><Mail size={12} style={{ marginRight: 4 }} /> Email</Badge>
                <Badge><Cloud size={12} style={{ marginRight: 4 }} /> S3</Badge>
                <Badge>Data Warehouse</Badge>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => openPreview('reports-configure-destinations')}
              >
                Configure
              </Button>
            </CardContent>
          </Card>
        </CardContent>
      </Card>

      {/* Scheduled Exports */}
      <Card className="report-card">
        <CardHeader className="section-title">
          <CardTitle>Scheduled exports</CardTitle>
          <Badge variant="outline">3 active</Badge>
        </CardHeader>
        <CardContent>
          <div style={{ display: 'grid', gap: 12 }}>
            <div className="card" style={{ padding: 16, borderRadius: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong>Daily summary</strong>
                <p style={{ color: 'var(--muted)', fontSize: 13, margin: '4px 0 0' }}>
                  Every day at 8:00 AM → team@company.com
                </p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <Badge variant="outline" className="tag-success">Active</Badge>
                <Button variant="ghost" size="sm">Edit</Button>
              </div>
            </div>
            <div className="card" style={{ padding: 16, borderRadius: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong>Weekly executive</strong>
                <p style={{ color: 'var(--muted)', fontSize: 13, margin: '4px 0 0' }}>
                  Every Monday at 9:00 AM → executives@company.com
                </p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <Badge variant="outline" className="tag-success">Active</Badge>
                <Button variant="ghost" size="sm">Edit</Button>
              </div>
            </div>
            <div className="card" style={{ padding: 16, borderRadius: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong>Monthly compliance</strong>
                <p style={{ color: 'var(--muted)', fontSize: 13, margin: '4px 0 0' }}>
                  1st of each month → S3 bucket
                </p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <Badge variant="outline" className="tag-success">Active</Badge>
                <Button variant="ghost" size="sm">Edit</Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ReportsExport;
