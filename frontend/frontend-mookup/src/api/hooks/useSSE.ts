/**
 * Server-Sent Events hook for real-time updates
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { config, API_URL } from '../../config';
import type { SSEEvent, CallStartedEvent, CallEndedEvent, TranscriptEvent } from '../types';

// Query key imports for invalidation
import { callKeys } from './useCalls';
import { dashboardKeys } from './useDashboard';
import { agentKeys } from './useAgents';

type SSEChannel = 'calls' | 'dashboard' | 'agents' | 'all';
type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface UseSSEOptions {
  channels?: SSEChannel[];
  onCallStarted?: (event: CallStartedEvent) => void;
  onCallEnded?: (event: CallEndedEvent) => void;
  onTranscript?: (event: TranscriptEvent) => void;
  onMetricUpdated?: (metric: string, value: unknown) => void;
  onEvent?: (event: SSEEvent) => void;
  enabled?: boolean;
}

interface UseSSEReturn {
  status: ConnectionStatus;
  lastEvent: SSEEvent | null;
  reconnect: () => void;
}

/**
 * Hook for subscribing to real-time events via SSE
 */
export function useSSE(options: UseSSEOptions = {}): UseSSEReturn {
  const {
    channels = ['all'],
    onCallStarted,
    onCallEnded,
    onTranscript,
    onMetricUpdated,
    onEvent,
    enabled = true,
  } = options;

  const queryClient = useQueryClient();
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [lastEvent, setLastEvent] = useState<SSEEvent | null>(null);

  // Build SSE URL with channels
  const buildUrl = useCallback(() => {
    const token = localStorage.getItem(config.accessTokenKey);
    const params = new URLSearchParams();
    channels.forEach((ch) => params.append('channels', ch));
    // Note: SSE doesn't support headers, so we pass token as query param
    // Backend should accept this for SSE connections
    if (token) {
      params.append('token', token);
    }
    return `${API_URL}/events/stream?${params.toString()}`;
  }, [channels]);

  // Handle incoming events
  const handleEvent = useCallback(
    (eventType: string, data: unknown) => {
      const event: SSEEvent = {
        type: eventType,
        data,
        timestamp: new Date().toISOString(),
      };

      setLastEvent(event);
      onEvent?.(event);

      // Handle specific event types
      switch (eventType) {
        case 'call.started':
          onCallStarted?.(data as CallStartedEvent);
          // Invalidate calls list
          queryClient.invalidateQueries({ queryKey: callKeys.lists() });
          queryClient.invalidateQueries({ queryKey: dashboardKeys.kpis() });
          break;

        case 'call.ended':
        case 'call.completed':
          onCallEnded?.(data as CallEndedEvent);
          queryClient.invalidateQueries({ queryKey: callKeys.lists() });
          queryClient.invalidateQueries({ queryKey: dashboardKeys.metrics() });
          break;

        case 'call.transcript':
          onTranscript?.(data as TranscriptEvent);
          // Could update specific call transcript cache
          break;

        case 'metric.updated':
          const { metric, value } = data as { metric: string; value: unknown };
          onMetricUpdated?.(metric, value);
          queryClient.invalidateQueries({ queryKey: dashboardKeys.kpis() });
          break;

        case 'agent.status_changed':
          queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
          queryClient.invalidateQueries({ queryKey: dashboardKeys.agentUtilization() });
          break;
      }
    },
    [queryClient, onEvent, onCallStarted, onCallEnded, onTranscript, onMetricUpdated]
  );

  // Connect to SSE
  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setStatus('connecting');

    try {
      const eventSource = new EventSource(buildUrl());
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        setStatus('connected');
        console.log('[SSE] Connected');
      };

      eventSource.onerror = (error) => {
        console.error('[SSE] Error:', error);
        setStatus('error');
        eventSource.close();

        // Attempt reconnect
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        reconnectTimeoutRef.current = setTimeout(() => {
          if (enabled) {
            console.log('[SSE] Reconnecting...');
            connect();
          }
        }, config.sseReconnectDelay);
      };

      // Listen for specific event types
      eventSource.addEventListener('connected', () => {
        console.log('[SSE] Handshake complete');
      });

      eventSource.addEventListener('heartbeat', () => {
        // Keep-alive, no action needed
      });

      // Dynamic event listener for all events
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleEvent(data.type || 'message', data.data || data);
        } catch (e) {
          console.warn('[SSE] Failed to parse event:', e);
        }
      };

      // Specific event listeners
      const eventTypes = [
        'call.started',
        'call.ended',
        'call.completed',
        'call.transcript',
        'metric.updated',
        'agent.status_changed',
      ];

      eventTypes.forEach((type) => {
        eventSource.addEventListener(type, (event: MessageEvent) => {
          try {
            const data = JSON.parse(event.data);
            handleEvent(type, data.data || data);
          } catch (e) {
            console.warn(`[SSE] Failed to parse ${type} event:`, e);
          }
        });
      });
    } catch (error) {
      console.error('[SSE] Failed to connect:', error);
      setStatus('error');
    }
  }, [buildUrl, enabled, handleEvent]);

  // Reconnect function
  const reconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    connect();
  }, [connect]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      setStatus('disconnected');
    };
  }, [enabled, connect]);

  return {
    status,
    lastEvent,
    reconnect,
  };
}

/**
 * Hook specifically for call transcript streaming
 */
export function useCallTranscriptStream(callId: string, onSegment?: (segment: TranscriptEvent['segment']) => void) {
  const [segments, setSegments] = useState<TranscriptEvent['segment'][]>([]);

  const { status } = useSSE({
    channels: ['calls'],
    enabled: !!callId,
    onTranscript: (event) => {
      if (event.call_id === callId) {
        setSegments((prev) => [...prev, event.segment]);
        onSegment?.(event.segment);
      }
    },
  });

  return {
    status,
    segments,
    clearSegments: () => setSegments([]),
  };
}
