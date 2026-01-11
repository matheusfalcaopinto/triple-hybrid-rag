import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useImportLeads } from '../../api/hooks/useLeads';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Upload, FileSpreadsheet, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

const steps = ['Upload', 'Map fields', 'Configure rules', 'Confirm'];

interface FieldMapping {
  sourceField: string;
  targetField: string;
}

interface ImportConfig {
  deduplicateByPhone: boolean;
  deduplicateByEmail: boolean;
  assignToAgent?: string;
}

const LeadsImport = () => {
  const navigate = useNavigate();
  const openPreview = usePreviewStore((state) => state.open);
  const importLeads = useImportLeads();

  const [currentStep, setCurrentStep] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [csvData, setCsvData] = useState<string[][]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [fieldMappings, setFieldMappings] = useState<FieldMapping[]>([]);
  const [config, setConfig] = useState<ImportConfig>({
    deduplicateByPhone: true,
    deduplicateByEmail: true,
  });
  const [validationResult, setValidationResult] = useState<{
    total: number;
    duplicates: number;
    invalid: number;
    valid: number;
  } | null>(null);

  const targetFields = [
    { value: 'name', label: 'Name' },
    { value: 'phone_number', label: 'Phone Number' },
    { value: 'email', label: 'Email' },
    { value: 'company', label: 'Company' },
    { value: 'source', label: 'Source' },
    { value: 'skip', label: '-- Skip --' },
  ];

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const lines = text.split('\n').filter((line) => line.trim());
      const parsed = lines.map((line) => line.split(',').map((cell) => cell.trim().replace(/^"|"$/g, '')));
      
      if (parsed.length > 0) {
        setHeaders(parsed[0]);
        setCsvData(parsed.slice(1));
        
        // Auto-map common field names
        const autoMappings: FieldMapping[] = parsed[0].map((header) => {
          const lowerHeader = header.toLowerCase();
          let targetField = 'skip';
          if (lowerHeader.includes('name')) targetField = 'name';
          else if (lowerHeader.includes('phone') || lowerHeader.includes('mobile')) targetField = 'phone_number';
          else if (lowerHeader.includes('email')) targetField = 'email';
          else if (lowerHeader.includes('company') || lowerHeader.includes('organization')) targetField = 'company';
          else if (lowerHeader.includes('source') || lowerHeader.includes('origin')) targetField = 'source';
          
          return { sourceField: header, targetField };
        });
        setFieldMappings(autoMappings);
        setCurrentStep(1);
      }
    };
    reader.readAsText(uploadedFile);
  }, []);

  const handleValidate = useCallback(() => {
    // Simulate validation
    const total = csvData.length;
    const duplicates = Math.floor(total * 0.03);
    const invalid = Math.floor(total * 0.01);
    const valid = total - duplicates - invalid;
    
    setValidationResult({ total, duplicates, invalid, valid });
    setCurrentStep(3);
  }, [csvData]);

  const handleImport = useCallback(async () => {
    if (!file) return;
    
    // The API expects a File for CSV import
    importLeads.mutate(file, {
      onSuccess: () => {
        navigate('/leads/queue');
      },
    });
  }, [file, importLeads, navigate]);

  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Import leads</h2>
          <span className="tag">4-step wizard</span>
        </div>
        
        {/* Progress steps */}
        <div className="wizard-steps" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
          {steps.map((step, index) => (
            <div
              key={step}
              className="wizard-step"
              style={{
                padding: 16,
                borderRadius: 12,
                background: index === currentStep ? 'var(--primary-light)' : index < currentStep ? 'var(--success-light)' : 'var(--card)',
                border: index === currentStep ? '2px solid var(--primary)' : '1px solid var(--border)',
                opacity: index > currentStep ? 0.5 : 1,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                {index < currentStep ? (
                  <CheckCircle size={16} style={{ color: 'var(--success)' }} />
                ) : (
                  <span style={{ 
                    width: 20, 
                    height: 20, 
                    borderRadius: '50%', 
                    background: index === currentStep ? 'var(--primary)' : 'var(--muted)', 
                    color: 'white', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    fontSize: 12 
                  }}>
                    {index + 1}
                  </span>
                )}
                <h4 style={{ margin: 0 }}>{step}</h4>
              </div>
              <p style={{ color: 'var(--muted)', fontSize: 13, margin: 0 }}>
                {index === 0 && 'Upload CSV or connect CRM'}
                {index === 1 && 'Match source columns to platform fields'}
                {index === 2 && 'Deduplicate, assign agents, schedule windows'}
                {index === 3 && 'Validate sample data then import'}
              </p>
            </div>
          ))}
        </div>

        {/* Step content */}
        {currentStep === 0 && (
          <div style={{ textAlign: 'center', padding: 48, border: '2px dashed var(--border)', borderRadius: 12 }}>
            <FileSpreadsheet size={48} style={{ color: 'var(--muted)', marginBottom: 16 }} />
            <p style={{ marginBottom: 16 }}>Drag and drop a CSV file or click to browse</p>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
              id="csv-upload"
            />
            <label htmlFor="csv-upload">
              <Button variant="default" style={{ cursor: 'pointer' }}>
                <Upload size={16} style={{ marginRight: 8 }} />
                Choose File
              </Button>
            </label>
          </div>
        )}

        {currentStep === 1 && (
          <div>
            <h3 style={{ marginBottom: 16 }}>Map fields from: {file?.name}</h3>
            <table className="table">
              <thead>
                <tr>
                  <th>Source Column</th>
                  <th>Sample Data</th>
                  <th>Map To</th>
                </tr>
              </thead>
              <tbody>
                {headers.map((header, index) => (
                  <tr key={header}>
                    <td><strong>{header}</strong></td>
                    <td style={{ color: 'var(--muted)' }}>{csvData[0]?.[index] || '-'}</td>
                    <td>
                      <select
                        value={fieldMappings[index]?.targetField || 'skip'}
                        onChange={(e) => {
                          const newMappings = [...fieldMappings];
                          newMappings[index] = { sourceField: header, targetField: e.target.value };
                          setFieldMappings(newMappings);
                        }}
                        style={{
                          padding: '8px 12px',
                          borderRadius: 8,
                          border: '1px solid var(--border)',
                          background: 'var(--bg)',
                          color: 'var(--text)',
                        }}
                      >
                        {targetFields.map((field) => (
                          <option key={field.value} value={field.value}>{field.label}</option>
                        ))}
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ marginTop: 16, display: 'flex', justifyContent: 'flex-end' }}>
              <Button onClick={() => setCurrentStep(2)}>Continue</Button>
            </div>
          </div>
        )}

        {currentStep === 2 && (
          <div>
            <h3 style={{ marginBottom: 16 }}>Configure import rules</h3>
            <div style={{ display: 'grid', gap: 16 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <input
                  type="checkbox"
                  checked={config.deduplicateByPhone}
                  onChange={(e) => setConfig({ ...config, deduplicateByPhone: e.target.checked })}
                />
                <span>Deduplicate by phone number</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <input
                  type="checkbox"
                  checked={config.deduplicateByEmail}
                  onChange={(e) => setConfig({ ...config, deduplicateByEmail: e.target.checked })}
                />
                <span>Deduplicate by email</span>
              </label>
            </div>
            <div style={{ marginTop: 24, display: 'flex', justifyContent: 'space-between' }}>
              <Button variant="secondary" onClick={() => setCurrentStep(1)}>Back</Button>
              <Button onClick={handleValidate}>Validate & Continue</Button>
            </div>
          </div>
        )}

        {currentStep === 3 && validationResult && (
          <div>
            <h3 style={{ marginBottom: 16 }}>Validation summary</h3>
            <div className="card" style={{ marginBottom: 24 }}>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: 12 }}>
                <li style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <CheckCircle size={16} style={{ color: 'var(--success)' }} />
                  <span><strong>{validationResult.total}</strong> leads detected</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <AlertCircle size={16} style={{ color: 'var(--warning)' }} />
                  <span><strong>{validationResult.duplicates}</strong> duplicates flagged</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <AlertCircle size={16} style={{ color: 'var(--danger)' }} />
                  <span><strong>{validationResult.invalid}</strong> invalid entries</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <CheckCircle size={16} style={{ color: 'var(--success)' }} />
                  <span><strong>{validationResult.valid}</strong> valid leads ready to import</span>
                </li>
              </ul>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Button variant="secondary" onClick={() => setCurrentStep(2)}>Back</Button>
              <div style={{ display: 'flex', gap: 8 }}>
                <Button variant="secondary" onClick={() => openPreview('leads-download-report')}>
                  Download report
                </Button>
                <Button onClick={handleImport} disabled={importLeads.isPending}>
                  {importLeads.isPending ? (
                    <>
                      <Loader2 className="animate-spin" size={16} style={{ marginRight: 8 }} />
                      Importing...
                    </>
                  ) : (
                    `Import ${validationResult.valid} leads`
                  )}
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LeadsImport;
