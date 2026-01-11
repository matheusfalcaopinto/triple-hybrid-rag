import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCalls } from '../../api/hooks/useCalls';
import { useSSE } from '../../api/hooks/useSSE';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Loader2, AlertCircle, Phone, Radio } from 'lucide-react';
import type { Call, CallStartedEvent, CallEndedEvent, TranscriptEvent } from '../../api/types';

const CallsActive = () => {
  const navigate = useNavigate();
  const openPreview = usePreviewStore((state) => state.open);
  const [selectedCall, setSelectedCall] = useState<Call | null>(null);
  const [liveTranscript, setLiveTranscript] = useState<Array<{ role: string; text: string }>>([]);

  // Fetch active calls (in_progress status)
  const { data, isLoading, error, refetch } = useCalls({
    status: 'in_progress',
    page: 1,
    page_size: 50,
  });

  const activeCalls = data?.items ?? [];

  // Handle SSE callbacks
  const handleCallStarted = (event: CallStartedEvent) => {
    refetch();
  };

  const handleCallEnded = (event: CallEndedEvent) => {
    refetch();
  };

  const handleTranscript = (event: TranscriptEvent) => {
    if (selectedCall && event.call_id === selectedCall.id) {
      setLiveTranscript((prev) => [
        ...prev,
        { role: event.segment.speaker, text: event.segment.text },
      ]);
    }
  };

  // Subscribe to SSE for real-time updates
  const { status: sseStatus } = useSSE({
    channels: ['calls'],
    onCallStarted: handleCallStarted,
    onCallEnded: handleCallEnded,
    onTranscript: handleTranscript,
    enabled: true,
  });

  const isConnected = sseStatus === 'connected';

  // Auto-select first call
  useEffect(() => {
    if (activeCalls.length > 0 && !selectedCall) {
      setSelectedCall(activeCalls[0]);
    }
  }, [activeCalls, selectedCall]);

  // Format duration from seconds
  const formatDuration = (startedAt: string) => {
    const start = new Date(startedAt).getTime();
    const now = Date.now();
    const seconds = Math.floor((now - start) / 1000);
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Live duration ticker
  const [, setTick] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px' }}>
          <Loader2 className="animate-spin" size={32} style={{ color: 'var(--primary)' }} />
          <span style={{ marginLeft: 12 }}>Loading active calls...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px', flexDirection: 'column', gap: 16 }}>
          <AlertCircle size={32} style={{ color: 'var(--danger)' }} />
          <span>Failed to load active calls</span>
          <Button onClick={() => refetch()}>Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="section-title">
        <h2>
          <Radio size={20} style={{ marginRight: 8, color: isConnected ? 'var(--success)' : 'var(--muted)' }} />
          Live calls
          {activeCalls.length > 0 && (
            <span className="tag tag-success" style={{ marginLeft: 8 }}>{activeCalls.length} active</span>
          )}
        </h2>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => openPreview('calls-intervene')}
          disabled={!selectedCall}
        >
          Intervene
        </Button>
      </div>

      {activeCalls.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '48px', color: 'var(--muted)' }}>
          <Phone size={48} style={{ opacity: 0.3, marginBottom: 16 }} />
          <p>No active calls at the moment</p>
          <p style={{ fontSize: 13 }}>Calls will appear here when agents are on live calls</p>
        </div>
      ) : (
        <>
          <div className="call-grid">
            {activeCalls.map((call) => (
              <div
                key={call.id}
                className="card"
                style={{
                  borderRadius: 16,
                  cursor: 'pointer',
                  border: selectedCall?.id === call.id ? '2px solid var(--primary)' : undefined,
                }}
                onClick={() => {
                  setSelectedCall(call);
                  setLiveTranscript([]);
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <div>
                    <strong>{call.direction === 'inbound' ? 'Inbound' : 'Outbound'}</strong>
                    <div style={{ color: 'var(--muted)', fontSize: 12 }}>
                      {call.to_number || call.from_number}
                    </div>
                  </div>
                  <span className="tag tag-success">
                    <Radio size={10} style={{ marginRight: 4 }} />
                    Live
                  </span>
                </div>
                <div style={{ fontSize: 32, fontWeight: 700 }}>
                  {call.started_at ? formatDuration(call.started_at) : '00:00'}
                </div>
                <div
                  style={{
                    height: 60,
                    background: 'linear-gradient(90deg, rgba(99,102,241,0.2), transparent)',
                    borderRadius: 12,
                  }}
                />
              </div>
            ))}
          </div>

          {selectedCall && (
            <div className="card" style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
              <div>
                <h3>Live transcript</h3>
                <div className="transcript" style={{ maxHeight: 300, overflowY: 'auto' }}>
                  {liveTranscript.length === 0 ? (
                    <div style={{ color: 'var(--muted)', fontStyle: 'italic' }}>
                      Waiting for transcript data...
                    </div>
                  ) : (
                    liveTranscript.map((entry, idx) => (
                      <div key={idx}>
                        <strong>{entry.role === 'agent' ? 'Agent' : 'Customer'}:</strong> {entry.text}
                      </div>
                    ))
                  )}
                </div>
              </div>
              <div>
                <h3>Call info</h3>
                <div className="card" style={{ borderRadius: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                    <span>Sentiment</span>
                    <span
                      className={`tag ${
                        selectedCall.sentiment_score !== undefined
                          ? selectedCall.sentiment_score > 0.3
                            ? 'tag-success'
                            : selectedCall.sentiment_score < -0.3
                            ? 'tag-danger'
                            : ''
                          : ''
                      }`}
                    >
                      {selectedCall.sentiment_score !== undefined
                        ? selectedCall.sentiment_score > 0.3
                          ? 'Positive'
                          : selectedCall.sentiment_score < -0.3
                          ? 'Negative'
                          : 'Neutral'
                        : 'Analyzing...'}
                    </span>
                  </div>
                  <div style={{ color: 'var(--muted)', fontSize: 13 }}>
                    Direction: {selectedCall.direction}<br />
                    From: {selectedCall.from_number}<br />
                    To: {selectedCall.to_number}<br />
                    Started: {selectedCall.started_at ? new Date(selectedCall.started_at).toLocaleTimeString() : '-'}
                  </div>
                  <div style={{ marginTop: 16 }}>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => navigate(`/calls/${selectedCall.id}`)}
                    >
                      View details
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default CallsActive;
