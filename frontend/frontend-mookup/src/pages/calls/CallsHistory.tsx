import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCalls } from '../../api/hooks/useCalls';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Search, ChevronLeft, ChevronRight, Loader2, AlertCircle, Phone } from 'lucide-react';
import type { CallStatus } from '../../api/types';

const CallsHistory = () => {
  const navigate = useNavigate();
  const openPreview = usePreviewStore((state) => state.open);
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<CallStatus | ''>('');

  const { data, isLoading, error, refetch } = useCalls({
    status: statusFilter || undefined,
    page,
    page_size: 20,
  });

  const calls = data?.items ?? [];
  const totalPages = data?.pages ?? 1;

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px' }}>
          <Loader2 className="animate-spin" size={32} style={{ color: 'var(--primary)' }} />
          <span style={{ marginLeft: 12 }}>Loading call history...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px', flexDirection: 'column', gap: 16 }}>
          <AlertCircle size={32} style={{ color: 'var(--danger)' }} />
          <span>Failed to load call history</span>
          <Button onClick={() => refetch()}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Call history</h2>
          <div style={{ display: 'flex', gap: 8 }}>
            <div style={{ position: 'relative', width: 240 }}>
              <Search size={16} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted)' }} />
              <Input
                placeholder="Search calls..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ paddingLeft: 32 }}
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as CallStatus | '')}
              style={{
                padding: '8px 12px',
                borderRadius: 8,
                border: '1px solid var(--border)',
                background: 'var(--bg)',
                color: 'var(--text)',
              }}
            >
              <option value="">All statuses</option>
              <option value="completed">Completed</option>
              <option value="in_progress">In Progress</option>
              <option value="failed">Failed</option>
              <option value="busy">Busy</option>
              <option value="no_answer">No Answer</option>
            </select>
            <Button variant="secondary" size="sm" onClick={() => openPreview('calls-export')}>
              Export CSV
            </Button>
          </div>
        </div>

        {calls.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px', color: 'var(--muted)' }}>
            <Phone size={48} style={{ opacity: 0.3, marginBottom: 16 }} />
            <p>No calls found</p>
          </div>
        ) : (
          <>
            <div className="table-scroll">
              <table className="table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Direction</th>
                    <th>Number</th>
                    <th>Duration</th>
                    <th>Status</th>
                    <th>Sentiment</th>
                    <th>Cost</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {calls.map((call) => (
                    <tr key={call.id} style={{ cursor: 'pointer' }} onClick={() => navigate(`/calls/${call.id}`)}>
                      <td>{formatDate(call.started_at || '')}</td>
                      <td>
                        <span className={`tag ${call.direction === 'inbound' ? '' : 'tag-warning'}`}>
                          {call.direction}
                        </span>
                      </td>
                      <td>{call.to_number || call.from_number}</td>
                      <td>{call.duration_seconds ? formatDuration(call.duration_seconds) : '-'}</td>
                      <td>
                        <span className={`tag ${
                          call.status === 'completed' ? 'tag-success' :
                          call.status === 'failed' ? 'tag-danger' :
                          call.status === 'in_progress' ? 'tag-warning' : ''
                        }`}>
                          {call.status}
                        </span>
                      </td>
                      <td>
                        {call.sentiment_score !== undefined ? (
                          <span className={`tag ${
                            call.sentiment_score > 0.3 ? 'tag-success' :
                            call.sentiment_score < -0.3 ? 'tag-danger' : ''
                          }`}>
                            {call.sentiment_score > 0.3 ? 'Positive' :
                             call.sentiment_score < -0.3 ? 'Negative' : 'Neutral'}
                          </span>
                        ) : '-'}
                      </td>
                      <td>${call.cost ? call.cost.toFixed(2) : '0.00'}</td>
                      <td>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/calls/${call.id}`);
                          }}
                        >
                          View
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 16 }}>
              <span style={{ color: 'var(--muted)', fontSize: 13 }}>
                Page {page} of {totalPages} ({data?.total ?? 0} total calls)
              </span>
              <div style={{ display: 'flex', gap: 8 }}>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  <ChevronLeft size={16} />
                  Previous
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                >
                  Next
                  <ChevronRight size={16} />
                </Button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default CallsHistory;
