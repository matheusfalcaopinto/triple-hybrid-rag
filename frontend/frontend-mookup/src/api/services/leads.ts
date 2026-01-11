/**
 * Leads API service
 */

import { apiClient } from '../client';
import type {
  Lead,
  LeadCreate,
  LeadUpdate,
  LeadStatus,
  LeadImportResult,
  PaginatedResponse,
  MessageResponse,
} from '../types';

export interface LeadFilters {
  status?: LeadStatus;
  source?: string;
  tags?: string[];
  search?: string;
}

export interface LeadsByStatus {
  new: number;
  trying: number;
  connected: number;
  qualified: number;
  converted: number;
  discarded: number;
}

// Helper to serialize filters
function serializeFilters(
  filters?: LeadFilters & { page?: number; page_size?: number }
): Record<string, string | number | boolean | undefined> | undefined {
  if (!filters) return undefined;
  const result: Record<string, string | number | boolean | undefined> = {};
  if (filters.status) result.status = filters.status;
  if (filters.source) result.source = filters.source;
  if (filters.tags && filters.tags.length > 0) result.tags = filters.tags.join(',');
  if (filters.search) result.search = filters.search;
  if (filters.page !== undefined) result.page = filters.page;
  if (filters.page_size !== undefined) result.page_size = filters.page_size;
  return result;
}

export const leadsApi = {
  /**
   * List leads with filters
   */
  list: async (
    params?: LeadFilters & { page?: number; page_size?: number }
  ): Promise<PaginatedResponse<Lead>> => {
    return apiClient.get<PaginatedResponse<Lead>>('/leads', { params: serializeFilters(params) });
  },

  /**
   * Get leads grouped by status (for Kanban view)
   */
  getByStatus: async (): Promise<LeadsByStatus> => {
    return apiClient.get<LeadsByStatus>('/leads/by-status');
  },

  /**
   * Get lead by ID
   */
  get: async (id: string): Promise<Lead> => {
    return apiClient.get<Lead>(`/leads/${id}`);
  },

  /**
   * Create new lead
   */
  create: async (data: LeadCreate): Promise<Lead> => {
    return apiClient.post<Lead>('/leads', data);
  },

  /**
   * Update lead
   */
  update: async (id: string, data: LeadUpdate): Promise<Lead> => {
    return apiClient.patch<Lead>(`/leads/${id}`, data);
  },

  /**
   * Update lead status
   */
  updateStatus: async (id: string, status: LeadStatus): Promise<Lead> => {
    return apiClient.patch<Lead>(`/leads/${id}`, { status });
  },

  /**
   * Delete lead
   */
  delete: async (id: string): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(`/leads/${id}`);
  },

  /**
   * Bulk delete leads
   */
  bulkDelete: async (ids: string[]): Promise<MessageResponse> => {
    return apiClient.post<MessageResponse>('/leads/bulk-delete', { ids });
  },

  /**
   * Import leads from CSV
   */
  importCsv: async (file: File): Promise<LeadImportResult> => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.upload<LeadImportResult>('/leads/import', formData);
  },

  /**
   * Add tags to lead
   */
  addTags: async (id: string, tags: string[]): Promise<Lead> => {
    return apiClient.post<Lead>(`/leads/${id}/tags`, { tags });
  },

  /**
   * Remove tags from lead
   */
  removeTags: async (id: string, tags: string[]): Promise<Lead> => {
    return apiClient.delete<Lead>(`/leads/${id}/tags`, {
      body: JSON.stringify({ tags }),
    } as RequestInit);
  },

  /**
   * Get lead call history
   */
  getCallHistory: async (id: string): Promise<{ calls: { id: string; timestamp: string; status: string; duration?: number }[] }> => {
    return apiClient.get(`/leads/${id}/calls`);
  },
};
