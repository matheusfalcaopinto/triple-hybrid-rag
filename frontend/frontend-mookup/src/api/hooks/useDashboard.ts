/**
 * Dashboard hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { dashboardApi } from '../services/dashboard';

export const dashboardKeys = {
  all: ['dashboard'] as const,
  metrics: () => [...dashboardKeys.all, 'metrics'] as const,
  kpis: () => [...dashboardKeys.all, 'kpis'] as const,
  callVolume: (range: string) => [...dashboardKeys.all, 'call-volume', range] as const,
  agentUtilization: () => [...dashboardKeys.all, 'agent-utilization'] as const,
  activity: () => [...dashboardKeys.all, 'activity'] as const,
  alerts: () => [...dashboardKeys.all, 'alerts'] as const,
};

/**
 * Hook to get all dashboard metrics
 */
export function useDashboardMetrics() {
  return useQuery({
    queryKey: dashboardKeys.metrics(),
    queryFn: () => dashboardApi.getMetrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });
}

/**
 * Hook to get KPIs only
 */
export function useKPIs() {
  return useQuery({
    queryKey: dashboardKeys.kpis(),
    queryFn: () => dashboardApi.getKPIs(),
    refetchInterval: 30000,
  });
}

/**
 * Hook to get call volume chart data
 */
export function useCallVolume(range: '24h' | '7d' | '30d' = '24h') {
  return useQuery({
    queryKey: dashboardKeys.callVolume(range),
    queryFn: () => dashboardApi.getCallVolume(range),
  });
}

/**
 * Hook to get agent utilization
 */
export function useAgentUtilization() {
  return useQuery({
    queryKey: dashboardKeys.agentUtilization(),
    queryFn: () => dashboardApi.getAgentUtilization(),
    refetchInterval: 15000, // More frequent for real-time status
  });
}

/**
 * Hook to get recent activity
 */
export function useRecentActivity(limit: number = 10) {
  return useQuery({
    queryKey: dashboardKeys.activity(),
    queryFn: () => dashboardApi.getRecentActivity(limit),
    refetchInterval: 10000,
  });
}

/**
 * Hook to get alerts
 */
export function useAlerts(unreadOnly: boolean = false) {
  return useQuery({
    queryKey: dashboardKeys.alerts(),
    queryFn: () => dashboardApi.getAlerts(unreadOnly),
    refetchInterval: 30000,
  });
}

/**
 * Hook to mark alert as read
 */
export function useMarkAlertRead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (alertId: string) => dashboardApi.markAlertRead(alertId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: dashboardKeys.alerts() });
    },
  });
}

/**
 * Hook to mark all alerts as read
 */
export function useMarkAllAlertsRead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => dashboardApi.markAllAlertsRead(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: dashboardKeys.alerts() });
    },
  });
}

/**
 * Hook to dismiss alert
 */
export function useDismissAlert() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (alertId: string) => dashboardApi.dismissAlert(alertId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: dashboardKeys.alerts() });
    },
  });
}
