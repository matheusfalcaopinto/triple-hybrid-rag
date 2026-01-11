/**
 * Reports hooks
 */

import { useQuery } from '@tanstack/react-query';
import { reportsApi } from '../services/reports';
import type { ReportFilters } from '../types';

export const reportKeys = {
  all: ['reports'] as const,
  callVolume: (filters: ReportFilters) => [...reportKeys.all, 'call-volume', filters] as const,
  callDuration: (filters: ReportFilters) => [...reportKeys.all, 'call-duration', filters] as const,
  costs: (filters: ReportFilters) => [...reportKeys.all, 'costs', filters] as const,
  performance: (filters: ReportFilters) => [...reportKeys.all, 'performance', filters] as const,
  agents: (filters: ReportFilters) => [...reportKeys.all, 'agents', filters] as const,
  sentiment: (filters: ReportFilters) => [...reportKeys.all, 'sentiment', filters] as const,
};

/**
 * Hook to get call volume report
 */
export function useCallVolumeReport(filters: ReportFilters) {
  return useQuery({
    queryKey: reportKeys.callVolume(filters),
    queryFn: () => reportsApi.getCallVolume(filters),
    enabled: !!filters.date_from && !!filters.date_to,
  });
}

/**
 * Hook to get call duration report
 */
export function useCallDurationReport(filters: ReportFilters) {
  return useQuery({
    queryKey: reportKeys.callDuration(filters),
    queryFn: () => reportsApi.getCallDuration(filters),
    enabled: !!filters.date_from && !!filters.date_to,
  });
}

/**
 * Hook to get cost report
 */
export function useCostReport(filters: ReportFilters) {
  return useQuery({
    queryKey: reportKeys.costs(filters),
    queryFn: () => reportsApi.getCosts(filters),
    enabled: !!filters.date_from && !!filters.date_to,
  });
}

/**
 * Hook to get performance report
 */
export function usePerformanceReport(filters: ReportFilters) {
  return useQuery({
    queryKey: reportKeys.performance(filters),
    queryFn: () => reportsApi.getPerformance(filters),
    enabled: !!filters.date_from && !!filters.date_to,
  });
}

/**
 * Hook to get agent comparison report
 */
export function useAgentComparisonReport(filters: ReportFilters) {
  return useQuery({
    queryKey: reportKeys.agents(filters),
    queryFn: () => reportsApi.getAgentComparison(filters),
    enabled: !!filters.date_from && !!filters.date_to,
  });
}

/**
 * Hook to get sentiment report
 */
export function useSentimentReport(filters: ReportFilters) {
  return useQuery({
    queryKey: reportKeys.sentiment(filters),
    queryFn: () => reportsApi.getSentiment(filters),
    enabled: !!filters.date_from && !!filters.date_to,
  });
}

/**
 * Hook to get report templates
 */
export function useReportTemplates() {
  return useQuery({
    queryKey: [...reportKeys.all, 'templates'] as const,
    queryFn: () => reportsApi.getTemplates(),
  });
}
