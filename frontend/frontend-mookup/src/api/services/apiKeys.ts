/**
 * API Keys service
 */

import { apiClient } from '../client';
import type {
  APIKey,
  APIKeyCreate,
  APIKeyCreated,
  MessageResponse,
} from '../types';

export const apiKeysApi = {
  /**
   * List all API keys
   */
  list: async (): Promise<APIKey[]> => {
    return apiClient.get<APIKey[]>('/api-keys');
  },

  /**
   * Get single API key
   */
  get: async (id: string): Promise<APIKey> => {
    return apiClient.get<APIKey>(`/api-keys/${id}`);
  },

  /**
   * Create new API key
   */
  create: async (data: APIKeyCreate): Promise<APIKeyCreated> => {
    return apiClient.post<APIKeyCreated>('/api-keys', data);
  },

  /**
   * Revoke/delete API key
   */
  revoke: async (id: string): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(`/api-keys/${id}`);
  },

  /**
   * Toggle API key active status
   */
  toggle: async (id: string, isActive: boolean): Promise<APIKey> => {
    return apiClient.patch<APIKey>(`/api-keys/${id}`, { is_active: isActive });
  },

  /**
   * Rotate API key (regenerate with same settings)
   */
  rotate: async (id: string): Promise<APIKeyCreated> => {
    return apiClient.post<APIKeyCreated>(`/api-keys/${id}/rotate`);
  },
};
