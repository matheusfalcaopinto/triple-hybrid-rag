/**
 * Calls API service
 */

import { apiClient } from '../client';
import { API_URL } from '../../config';
import type {
  Call,
  CallWithDetails,
  CallTranscriptSegment,
  CallRecording,
  CallFilters,
  PaginatedResponse,
  MessageResponse,
} from '../types';

export interface CallExportOptions {
  format: 'csv' | 'json';
  include_transcripts?: boolean;
  include_recordings?: boolean;
}

// Helper to serialize filters to query params
function serializeFilters(
  filters?: CallFilters & { page?: number; page_size?: number }
): Record<string, string | number | boolean | undefined> | undefined {
  if (!filters) return undefined;
  const result: Record<string, string | number | boolean | undefined> = {};
  if (filters.status) result.status = filters.status;
  if (filters.direction) result.direction = filters.direction;
  if (filters.agent_id) result.agent_id = filters.agent_id;
  if (filters.date_from) result.date_from = filters.date_from;
  if (filters.date_to) result.date_to = filters.date_to;
  if (filters.search) result.search = filters.search;
  if (filters.page !== undefined) result.page = filters.page;
  if (filters.page_size !== undefined) result.page_size = filters.page_size;
  return result;
}

export const callsApi = {
  /**
   * List calls with filters
   */
  list: async (
    params?: CallFilters & { page?: number; page_size?: number }
  ): Promise<PaginatedResponse<CallWithDetails>> => {
    return apiClient.get<PaginatedResponse<CallWithDetails>>('/calls', { params: serializeFilters(params) });
  },

  /**
   * Get call by ID
   */
  get: async (id: string): Promise<CallWithDetails> => {
    return apiClient.get<CallWithDetails>(`/calls/${id}`);
  },

  /**
   * Get call transcript
   */
  getTranscript: async (id: string): Promise<CallTranscriptSegment[]> => {
    return apiClient.get<CallTranscriptSegment[]>(`/calls/${id}/transcript`);
  },

  /**
   * Get call recording
   */
  getRecording: async (id: string): Promise<CallRecording | null> => {
    return apiClient.get<CallRecording | null>(`/calls/${id}/recording`);
  },

  /**
   * Export calls
   */
  export: async (
    filters: CallFilters,
    options: CallExportOptions
  ): Promise<Blob> => {
    // Build params object with proper string conversion
    const params: Record<string, string> = {
      format: options.format,
    };
    if (options.include_transcripts !== undefined) params.include_transcripts = String(options.include_transcripts);
    if (options.include_recordings !== undefined) params.include_recordings = String(options.include_recordings);
    if (filters.status) params.status = filters.status;
    if (filters.direction) params.direction = filters.direction;
    if (filters.agent_id) params.agent_id = filters.agent_id;
    if (filters.date_from) params.date_from = filters.date_from;
    if (filters.date_to) params.date_to = filters.date_to;
    if (filters.search) params.search = filters.search;

    const response = await fetch(
      `${API_URL}/calls/export?${new URLSearchParams(params)}`,
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('access_token')}`,
        },
      }
    );
    return response.blob();
  },

  // ============ Live Call Actions ============

  /**
   * Send intervention command to active call
   */
  intervene: async (
    callId: string,
    command: 'whisper' | 'barge' | 'transfer' | 'hangup',
    data?: { message?: string; transfer_to?: string }
  ): Promise<MessageResponse> => {
    return apiClient.post<MessageResponse>(`/calls/${callId}/intervene`, {
      command,
      ...data,
    });
  },

  /**
   * Whisper message to agent (caller can't hear)
   */
  whisper: async (callId: string, message: string): Promise<MessageResponse> => {
    return callsApi.intervene(callId, 'whisper', { message });
  },

  /**
   * Barge into call (both parties can hear)
   */
  barge: async (callId: string): Promise<MessageResponse> => {
    return callsApi.intervene(callId, 'barge');
  },

  /**
   * Transfer call to another number
   */
  transfer: async (callId: string, transferTo: string): Promise<MessageResponse> => {
    return callsApi.intervene(callId, 'transfer', { transfer_to: transferTo });
  },

  /**
   * Hang up the call
   */
  hangup: async (callId: string): Promise<MessageResponse> => {
    return callsApi.intervene(callId, 'hangup');
  },

  // ============ Statistics ============

  /**
   * Get call statistics
   */
  getStats: async (params?: {
    date_from?: string;
    date_to?: string;
    agent_id?: string;
  }): Promise<{
    total_calls: number;
    completed_calls: number;
    average_duration: number;
    success_rate: number;
  }> => {
    return apiClient.get('/calls/stats', { params });
  },
};
