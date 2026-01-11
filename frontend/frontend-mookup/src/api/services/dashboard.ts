/**
 * Dashboard API service
 */

import { apiClient } from '../client';
import type {
  DashboardKPI,
  DashboardMetrics,
  TimeSeriesPoint,
  AgentUtilization,
  ActivityItem,
  Alert,
} from '../types';

export const dashboardApi = {
  /**
   * Get all dashboard metrics
   */
  getMetrics: async (): Promise<DashboardMetrics> => {
    return apiClient.get<DashboardMetrics>('/dashboard/metrics');
  },

  /**
   * Get KPIs only
   */
  getKPIs: async (): Promise<DashboardKPI[]> => {
    return apiClient.get<DashboardKPI[]>('/dashboard/kpis');
  },

  /**
   * Get call volume chart data
   */
  getCallVolume: async (range: '24h' | '7d' | '30d' = '24h'): Promise<TimeSeriesPoint[]> => {
    return apiClient.get<TimeSeriesPoint[]>('/dashboard/call-volume', {
      params: { range },
    });
  },

  /**
   * Get agent utilization
   */
  getAgentUtilization: async (): Promise<AgentUtilization[]> => {
    return apiClient.get<AgentUtilization[]>('/dashboard/agent-utilization');
  },

  /**
   * Get recent activity
   */
  getRecentActivity: async (limit: number = 10): Promise<ActivityItem[]> => {
    return apiClient.get<ActivityItem[]>('/dashboard/activity', {
      params: { limit },
    });
  },

  /**
   * Get alerts
   */
  getAlerts: async (unreadOnly: boolean = false): Promise<Alert[]> => {
    return apiClient.get<Alert[]>('/dashboard/alerts', {
      params: { unread_only: unreadOnly },
    });
  },

  /**
   * Mark alert as read
   */
  markAlertRead: async (alertId: string): Promise<Alert> => {
    return apiClient.patch<Alert>(`/dashboard/alerts/${alertId}`, { is_read: true });
  },

  /**
   * Mark all alerts as read
   */
  markAllAlertsRead: async (): Promise<{ updated: number }> => {
    return apiClient.post<{ updated: number }>('/dashboard/alerts/mark-all-read');
  },

  /**
   * Dismiss alert
   */
  dismissAlert: async (alertId: string): Promise<void> => {
    return apiClient.delete(`/dashboard/alerts/${alertId}`);
  },
};
