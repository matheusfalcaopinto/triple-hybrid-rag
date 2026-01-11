import { useState, useMemo } from 'react';
import { useLeads, useUpdateLead, useDeleteLead } from '../../api/hooks/useLeads';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Loader2, AlertCircle, Search, Users, Trash2, Phone } from 'lucide-react';
import type { LeadStatus, Lead } from '../../api/types';

const statusColumns: { status: LeadStatus; title: string; color: string }[] = [
  { status: 'new', title: 'New', color: 'var(--primary)' },
  { status: 'trying', title: 'Trying', color: 'var(--warning)' },
  { status: 'connected', title: 'Connected', color: 'var(--info)' },
  { status: 'qualified', title: 'Qualified', color: 'var(--success)' },
  { status: 'converted', title: 'Converted', color: 'var(--success)' },
  { status: 'discarded', title: 'Discarded', color: 'var(--danger)' },
];

const LeadsQueue = () => {
  const openPreview = usePreviewStore((state) => state.open);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLead, setSelectedLead] = useState<Lead | null>(null);

  // Fetch all leads
  const { data, isLoading, error, refetch } = useLeads({
    page: 1,
    page_size: 100,
  });

  const updateLead = useUpdateLead();
  const deleteLead = useDeleteLead();

  const leads = data?.items ?? [];

  // Group leads by status
  const leadsByStatus = useMemo(() => {
    const grouped: Record<LeadStatus, Lead[]> = {
      new: [],
      trying: [],
      connected: [],
      qualified: [],
      converted: [],
      discarded: [],
    };

    leads.forEach((lead) => {
      if (grouped[lead.status]) {
        // Apply search filter
        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          const matchesName = lead.name?.toLowerCase().includes(query);
          const matchesPhone = lead.phone_number?.toLowerCase().includes(query);
          const matchesEmail = lead.email?.toLowerCase().includes(query);
          if (!matchesName && !matchesPhone && !matchesEmail) {
            return;
          }
        }
        grouped[lead.status].push(lead);
      }
    });

    return grouped;
  }, [leads, searchQuery]);

  const handleStatusChange = (lead: Lead, newStatus: LeadStatus) => {
    updateLead.mutate(
      { id: lead.id, data: { status: newStatus } },
      { onSuccess: () => refetch() }
    );
  };

  const handleDelete = (lead: Lead) => {
    if (confirm(`Delete lead ${lead.name || lead.phone_number}?`)) {
      deleteLead.mutate(lead.id, { onSuccess: () => refetch() });
    }
  };

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px' }}>
          <Loader2 className="animate-spin" size={32} style={{ color: 'var(--primary)' }} />
          <span style={{ marginLeft: 12 }}>Loading leads...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px', flexDirection: 'column', gap: 16 }}>
          <AlertCircle size={32} style={{ color: 'var(--danger)' }} />
          <span>Failed to load leads</span>
          <Button onClick={() => refetch()}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="section-title">
        <h2>
          <Users size={20} style={{ marginRight: 8 }} />
          Leads pipeline
        </h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <div style={{ position: 'relative', width: 240 }}>
            <Search size={16} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted)' }} />
            <Input
              placeholder="Search leads..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ paddingLeft: 32 }}
            />
          </div>
          <Button variant="secondary" size="sm" onClick={() => openPreview('leads-import-csv')}>
            Import CSV
          </Button>
        </div>
      </div>

      <div className="kanban" style={{ display: 'grid', gridTemplateColumns: `repeat(${statusColumns.length}, 1fr)`, gap: 16, overflowX: 'auto' }}>
        {statusColumns.map((column) => (
          <div key={column.status} className="kanban-column" style={{ minWidth: 200 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <strong style={{ color: column.color }}>{column.title}</strong>
              <span className="tag">{leadsByStatus[column.status]?.length || 0}</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {leadsByStatus[column.status]?.map((lead) => (
                <div
                  key={lead.id}
                  className="card"
                  style={{
                    borderRadius: 12,
                    padding: 12,
                    cursor: 'pointer',
                    border: selectedLead?.id === lead.id ? '2px solid var(--primary)' : undefined,
                  }}
                  onClick={() => setSelectedLead(lead)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <strong style={{ fontSize: 14 }}>
                        {lead.name || 'Unknown'}
                      </strong>
                      <div style={{ color: 'var(--muted)', fontSize: 12, marginTop: 4 }}>
                        <Phone size={10} style={{ marginRight: 4 }} />
                        {lead.phone_number}
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(lead);
                      }}
                      style={{ padding: 4 }}
                    >
                      <Trash2 size={14} style={{ color: 'var(--danger)' }} />
                    </Button>
                  </div>
                  {lead.source && (
                    <span className="tag" style={{ marginTop: 8, fontSize: 10 }}>
                      {lead.source}
                    </span>
                  )}
                  {lead.tags.length > 0 && (
                    <div style={{ marginTop: 8, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {lead.tags.slice(0, 2).map((tag) => (
                        <span key={tag} className="tag" style={{ fontSize: 10 }}>{tag}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {(!leadsByStatus[column.status] || leadsByStatus[column.status].length === 0) && (
                <div style={{ padding: 16, textAlign: 'center', color: 'var(--muted)', fontSize: 12 }}>
                  No leads
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Lead details sidebar */}
      {selectedLead && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="section-title">
            <h3>Lead Details: {selectedLead.name || selectedLead.phone_number}</h3>
            <Button variant="ghost" size="sm" onClick={() => setSelectedLead(null)}>
              Close
            </Button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
            <div>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>Phone</label>
              <p>{selectedLead.phone_number}</p>
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>Email</label>
              <p>{selectedLead.email || '-'}</p>
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>Status</label>
              <select
                value={selectedLead.status}
                onChange={(e) => handleStatusChange(selectedLead, e.target.value as LeadStatus)}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  borderRadius: 8,
                  border: '1px solid var(--border)',
                  background: 'var(--bg)',
                  color: 'var(--text)',
                }}
              >
                {statusColumns.map((col) => (
                  <option key={col.status} value={col.status}>{col.title}</option>
                ))}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>Company</label>
              <p>{selectedLead.company || '-'}</p>
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>Contact Attempts</label>
              <p>{selectedLead.contact_attempts || 0}</p>
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>Last Contacted</label>
              <p>{selectedLead.last_contacted_at ? new Date(selectedLead.last_contacted_at).toLocaleString() : '-'}</p>
            </div>
          </div>
          {selectedLead.tags.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <label style={{ fontSize: 12, color: 'var(--muted)' }}>Tags</label>
              <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                {selectedLead.tags.map((tag) => (
                  <span key={tag} className="tag">{tag}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LeadsQueue;
