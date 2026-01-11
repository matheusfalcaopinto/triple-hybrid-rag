import { useParams, useNavigate } from 'react-router-dom';
import { useCall, useCallTranscript } from '../../api/hooks/useCalls';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Loader2, AlertCircle, ArrowLeft, Phone, Clock, DollarSign, User, Bot } from 'lucide-react';

const CallDetails = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const openPreview = usePreviewStore((state) => state.open);

  const { data: call, isLoading, error } = useCall(id!);
  const { data: transcriptData, isLoading: transcriptLoading } = useCallTranscript(id!);

  const transcript = transcriptData ?? [];

  const formatDuration = (seconds?: number) => {
    if (!seconds) return '-';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px' }}>
          <Loader2 className="animate-spin" size={32} style={{ color: 'var(--primary)' }} />
          <span style={{ marginLeft: 12 }}>Loading call details...</span>
        </div>
      </div>
    );
  }

  if (error || !call) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px', flexDirection: 'column', gap: 16 }}>
          <AlertCircle size={32} style={{ color: 'var(--danger)' }} />
          <span>Failed to load call details</span>
          <Button onClick={() => navigate('/calls/history')}>Back to History</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
          <ArrowLeft size={16} />
          Back
        </Button>
        <h1 style={{ margin: 0 }}>Call Details</h1>
        <span className={`tag ${
          call.status === 'completed' ? 'tag-success' :
          call.status === 'failed' ? 'tag-danger' :
          call.status === 'in_progress' ? 'tag-warning' : ''
        }`}>
          {call.status}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
        {/* Metadata sidebar */}
        <div className="card">
          <h2>Call metadata</h2>
          <div style={{ display: 'grid', gap: 12, marginTop: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)', fontSize: 13 }}>
              <Clock size={14} />
              <span>Started: {formatDate(call.started_at)}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)', fontSize: 13 }}>
              <Clock size={14} />
              <span>Duration: {formatDuration(call.duration_seconds)}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)', fontSize: 13 }}>
              <Phone size={14} />
              <span>Direction: {call.direction}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)', fontSize: 13 }}>
              <Phone size={14} />
              <span>From: {call.from_number}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)', fontSize: 13 }}>
              <Phone size={14} />
              <span>To: {call.to_number}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)', fontSize: 13 }}>
              <DollarSign size={14} />
              <span>Cost: ${call.cost?.toFixed(2) ?? '0.00'}</span>
            </div>
            {call.agent_name && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)', fontSize: 13 }}>
                <Bot size={14} />
                <span>Agent: {call.agent_name}</span>
              </div>
            )}
          </div>

          {/* Sentiment */}
          {call.sentiment_score !== undefined && (
            <div style={{ marginTop: 24 }}>
              <h3 style={{ fontSize: 14, marginBottom: 8 }}>Sentiment</h3>
              <span className={`tag ${
                call.sentiment_score > 0.3 ? 'tag-success' :
                call.sentiment_score < -0.3 ? 'tag-danger' : ''
              }`}>
                {call.sentiment_label || (
                  call.sentiment_score > 0.3 ? 'Positive' :
                  call.sentiment_score < -0.3 ? 'Negative' : 'Neutral'
                )}
              </span>
              <div style={{ marginTop: 8, fontSize: 12, color: 'var(--muted)' }}>
                Score: {call.sentiment_score.toFixed(2)}
              </div>
            </div>
          )}

          {/* Summary */}
          {call.summary && (
            <div style={{ marginTop: 24 }}>
              <h3 style={{ fontSize: 14, marginBottom: 8 }}>Summary</h3>
              <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.5 }}>
                {call.summary}
              </p>
            </div>
          )}

          {/* Outcome */}
          {call.outcome && (
            <div style={{ marginTop: 24 }}>
              <h3 style={{ fontSize: 14, marginBottom: 8 }}>Outcome</h3>
              <span className="tag">{call.outcome}</span>
            </div>
          )}
        </div>

        {/* Transcript panel */}
        <div className="card" style={{ display: 'grid', gap: 16 }}>
          <div className="section-title">
            <h2>Transcript</h2>
            <div style={{ display: 'flex', gap: 8 }}>
              {call.has_recording && (
                <Button variant="secondary" size="sm" onClick={() => openPreview('call-download-audio')}>
                  Download audio
                </Button>
              )}
            </div>
          </div>

          {transcriptLoading ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 32 }}>
              <Loader2 className="animate-spin" size={24} style={{ color: 'var(--primary)' }} />
              <span style={{ marginLeft: 12, color: 'var(--muted)' }}>Loading transcript...</span>
            </div>
          ) : transcript.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 32, color: 'var(--muted)' }}>
              <p>No transcript available for this call</p>
            </div>
          ) : (
            <div className="transcript" style={{ maxHeight: 500, overflowY: 'auto' }}>
              {transcript.map((segment) => (
                <div key={segment.id} style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    {segment.speaker === 'agent' ? (
                      <Bot size={14} style={{ color: 'var(--primary)' }} />
                    ) : (
                      <User size={14} style={{ color: 'var(--muted)' }} />
                    )}
                    <strong style={{ fontSize: 12, color: 'var(--muted)' }}>
                      {segment.speaker === 'agent' ? 'Agent' : 'Caller'}
                    </strong>
                    <span style={{ fontSize: 11, color: 'var(--muted)' }}>
                      {segment.start_time.toFixed(1)}s
                    </span>
                  </div>
                  <p style={{ margin: 0, paddingLeft: 22 }}>{segment.text}</p>
                </div>
              ))}
            </div>
          )}

          {/* Audio player */}
          {call.has_recording && (
            <div style={{ marginTop: 16 }}>
              <audio controls style={{ width: '100%' }}>
                <source src={`/api/v1/calls/${call.id}/recording`} type="audio/mpeg" />
                Your browser does not support the audio element.
              </audio>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CallDetails;
