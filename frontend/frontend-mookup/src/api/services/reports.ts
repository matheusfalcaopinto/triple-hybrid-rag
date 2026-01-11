/**
 * Reports API service
 */

import { apiClient } from '../client';
import { API_URL } from '../../config';
import type {
  ReportFilters,
  CallVolumeReport,
  CostReport,
  PerformanceReport,
  TimeSeriesPoint,
} from '../types';

export interface ReportExportOptions {
  format: 'csv' | 'pdf' | 'xlsx';
  report_type: 'calls' | 'costs' | 'performance' | 'agents';
}

// Helper to serialize report filters
function serializeFilters(
  filters: ReportFilters
): Record<string, string | number | boolean | undefined> {
  const result: Record<string, string | number | boolean | undefined> = {
    date_from: filters.date_from,
    date_to: filters.date_to,
  };
  if (filters.agent_ids && filters.agent_ids.length > 0) {
    result.agent_ids = filters.agent_ids.join(',');
  }
  if (filters.granularity) result.granularity = filters.granularity;
  return result;
}

export const reportsApi = {
  /**
   * Get call volume report
   */
  getCallVolume: async (filters: ReportFilters): Promise<CallVolumeReport> => {
    return apiClient.get<CallVolumeReport>('/reports/call-volume', {
      params: serializeFilters(filters),
    });
  },

  /**
   * Get call duration report
   */
  getCallDuration: async (filters: ReportFilters): Promise<{
    data: TimeSeriesPoint[];
    average_duration: number;
    total_minutes: number;
  }> => {
    return apiClient.get('/reports/call-duration', {
      params: serializeFilters(filters),
    });
  },

  /**
   * Get cost report
   */
  getCosts: async (filters: ReportFilters): Promise<CostReport> => {
    return apiClient.get<CostReport>('/reports/costs', {
      params: serializeFilters(filters),
    });
  },

  /**
   * Get performance report
   */
  getPerformance: async (filters: ReportFilters): Promise<PerformanceReport> => {
    return apiClient.get<PerformanceReport>('/reports/performance', {
      params: serializeFilters(filters),
    });
  },

  /**
   * Get agent performance comparison
   */
  getAgentComparison: async (
    filters: ReportFilters
  ): Promise<{
    agents: {
      agent_id: string;
      agent_name: string;
      total_calls: number;
      success_rate: number;
      average_duration: number;
      total_cost: number;
    }[];
  }> => {
    return apiClient.get('/reports/agents', {
      params: serializeFilters(filters),
    });
  },

  /**
   * Get sentiment analysis report
   */
  getSentiment: async (filters: ReportFilters): Promise<{
    data: TimeSeriesPoint[];
    breakdown: { positive: number; neutral: number; negative: number };
    average_score: number;
  }> => {
    return apiClient.get('/reports/sentiment', {
      params: serializeFilters(filters),
    });
  },

  /**
   * Export report
   */
  export: async (
    filters: ReportFilters,
    options: ReportExportOptions
  ): Promise<Blob> => {
    // Build params object with proper string conversion
    const params: Record<string, string> = {
      format: options.format,
      report_type: options.report_type,
      date_from: filters.date_from,
      date_to: filters.date_to,
    };
    if (filters.agent_ids && filters.agent_ids.length > 0) {
      params.agent_ids = filters.agent_ids.join(',');
    }
    if (filters.granularity) params.granularity = filters.granularity;

    const response = await fetch(
      `${API_URL}/reports/export?${new URLSearchParams(params)}`,
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('access_token')}`,
        },
      }
    );
    return response.blob();
  },

  /**
   * Get available report templates
   */
  getTemplates: async (): Promise<{
    id: string;
    name: string;
    description: string;
    filters: Partial<ReportFilters>;
  }[]> => {
    return apiClient.get('/reports/templates');
  },

  /**
   * Save report template
   */
  saveTemplate: async (
    name: string,
    filters: ReportFilters
  ): Promise<{ id: string; name: string }> => {
    return apiClient.post('/reports/templates', { name, filters });
  },
};
