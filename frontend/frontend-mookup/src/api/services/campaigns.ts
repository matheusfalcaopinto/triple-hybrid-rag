/**
 * Campaigns API service
 */

import { apiClient } from '../client';
import type {
  Campaign,
  CampaignCreate,
  CampaignUpdate,
  CampaignStatus,
  CampaignStats,
  PaginatedResponse,
  MessageResponse,
} from '../types';

export interface CampaignEnrollment {
  id: string;
  campaign_id: string;
  lead_id: string;
  lead_name?: string;
  lead_phone: string;
  status: 'pending' | 'calling' | 'completed' | 'failed' | 'skipped';
  call_id?: string;
  attempted_at?: string;
  completed_at?: string;
}

export interface CampaignRun {
  id: string;
  campaign_id: string;
  status: 'running' | 'paused' | 'completed' | 'cancelled';
  started_at: string;
  ended_at?: string;
  leads_processed: number;
  leads_connected: number;
}

export const campaignsApi = {
  /**
   * List campaigns
   */
  list: async (params?: {
    status?: CampaignStatus;
    page?: number;
    page_size?: number;
  }): Promise<PaginatedResponse<Campaign>> => {
    return apiClient.get<PaginatedResponse<Campaign>>('/campaigns', { params });
  },

  /**
   * Get campaign by ID
   */
  get: async (id: string): Promise<Campaign> => {
    return apiClient.get<Campaign>(`/campaigns/${id}`);
  },

  /**
   * Create new campaign
   */
  create: async (data: CampaignCreate): Promise<Campaign> => {
    return apiClient.post<Campaign>('/campaigns', data);
  },

  /**
   * Update campaign
   */
  update: async (id: string, data: CampaignUpdate): Promise<Campaign> => {
    return apiClient.patch<Campaign>(`/campaigns/${id}`, data);
  },

  /**
   * Delete campaign
   */
  delete: async (id: string): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(`/campaigns/${id}`);
  },

  // ============ Campaign Actions ============

  /**
   * Start campaign
   */
  start: async (id: string): Promise<Campaign> => {
    return apiClient.post<Campaign>(`/campaigns/${id}/start`);
  },

  /**
   * Pause campaign
   */
  pause: async (id: string): Promise<Campaign> => {
    return apiClient.post<Campaign>(`/campaigns/${id}/pause`);
  },

  /**
   * Resume campaign
   */
  resume: async (id: string): Promise<Campaign> => {
    return apiClient.post<Campaign>(`/campaigns/${id}/resume`);
  },

  /**
   * Cancel campaign
   */
  cancel: async (id: string): Promise<Campaign> => {
    return apiClient.post<Campaign>(`/campaigns/${id}/cancel`);
  },

  // ============ Enrollments ============

  /**
   * Get campaign enrollments
   */
  getEnrollments: async (
    id: string,
    params?: { status?: string; page?: number; page_size?: number }
  ): Promise<PaginatedResponse<CampaignEnrollment>> => {
    return apiClient.get<PaginatedResponse<CampaignEnrollment>>(
      `/campaigns/${id}/enrollments`,
      { params }
    );
  },

  /**
   * Add leads to campaign
   */
  enrollLeads: async (id: string, leadIds: string[]): Promise<{ enrolled: number }> => {
    return apiClient.post<{ enrolled: number }>(`/campaigns/${id}/enroll`, {
      lead_ids: leadIds,
    });
  },

  /**
   * Remove lead from campaign
   */
  unenrollLead: async (campaignId: string, enrollmentId: string): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(
      `/campaigns/${campaignId}/enrollments/${enrollmentId}`
    );
  },

  // ============ Statistics ============

  /**
   * Get campaign statistics
   */
  getStats: async (id: string): Promise<CampaignStats> => {
    return apiClient.get<CampaignStats>(`/campaigns/${id}/stats`);
  },

  /**
   * Get campaign runs history
   */
  getRuns: async (id: string): Promise<CampaignRun[]> => {
    return apiClient.get<CampaignRun[]>(`/campaigns/${id}/runs`);
  },
};
