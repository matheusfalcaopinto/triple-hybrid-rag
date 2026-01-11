/**
 * Calls hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { callsApi } from '../services/calls';
import type { CallFilters } from '../types';

export const callKeys = {
  all: ['calls'] as const,
  lists: () => [...callKeys.all, 'list'] as const,
  list: (filters?: CallFilters) => [...callKeys.lists(), filters] as const,
  details: () => [...callKeys.all, 'detail'] as const,
  detail: (id: string) => [...callKeys.details(), id] as const,
  transcript: (id: string) => [...callKeys.detail(id), 'transcript'] as const,
  recording: (id: string) => [...callKeys.detail(id), 'recording'] as const,
  stats: (params?: Record<string, unknown>) => [...callKeys.all, 'stats', params] as const,
};

/**
 * Hook to list calls
 */
export function useCalls(filters?: CallFilters & { page?: number; page_size?: number }) {
  return useQuery({
    queryKey: callKeys.list(filters),
    queryFn: () => callsApi.list(filters),
  });
}

/**
 * Hook to get call by ID
 */
export function useCall(id: string) {
  return useQuery({
    queryKey: callKeys.detail(id),
    queryFn: () => callsApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook to get call transcript
 */
export function useCallTranscript(id: string) {
  return useQuery({
    queryKey: callKeys.transcript(id),
    queryFn: () => callsApi.getTranscript(id),
    enabled: !!id,
  });
}

/**
 * Hook to get call recording
 */
export function useCallRecording(id: string) {
  return useQuery({
    queryKey: callKeys.recording(id),
    queryFn: () => callsApi.getRecording(id),
    enabled: !!id,
  });
}

/**
 * Hook to get call stats
 */
export function useCallStats(params?: { date_from?: string; date_to?: string; agent_id?: string }) {
  return useQuery({
    queryKey: callKeys.stats(params),
    queryFn: () => callsApi.getStats(params),
  });
}

/**
 * Hook for call intervention (whisper/barge/transfer/hangup)
 */
export function useCallIntervention() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      callId,
      command,
      data,
    }: {
      callId: string;
      command: 'whisper' | 'barge' | 'transfer' | 'hangup';
      data?: { message?: string; transfer_to?: string };
    }) => callsApi.intervene(callId, command, data),
    onSuccess: (_, { callId }) => {
      queryClient.invalidateQueries({ queryKey: callKeys.detail(callId) });
    },
  });
}

/**
 * Hook for whisper specifically
 */
export function useWhisper() {
  return useMutation({
    mutationFn: ({ callId, message }: { callId: string; message: string }) =>
      callsApi.whisper(callId, message),
  });
}

/**
 * Hook for hangup
 */
export function useHangup() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (callId: string) => callsApi.hangup(callId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: callKeys.lists() });
    },
  });
}
