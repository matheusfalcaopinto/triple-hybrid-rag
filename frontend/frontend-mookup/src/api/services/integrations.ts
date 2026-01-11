/**
 * Integrations API service
 */

import { apiClient } from '../client';
import type {
  Integration,
  IntegrationConnection,
  IntegrationConnectionCreate,
  MessageResponse,
} from '../types';

export interface OAuthStartResponse {
  auth_url: string;
  state: string;
}

export interface ToolProxyRequest {
  tool_name: string;
  parameters: Record<string, unknown>;
}

export interface ToolProxyResponse {
  success: boolean;
  result?: unknown;
  error?: string;
}

export const integrationsApi = {
  /**
   * List available integrations
   */
  listAvailable: async (): Promise<Integration[]> => {
    return apiClient.get<Integration[]>('/integrations');
  },

  /**
   * Get integration details
   */
  get: async (id: string): Promise<Integration> => {
    return apiClient.get<Integration>(`/integrations/${id}`);
  },

  // ============ Connections ============

  /**
   * List connected integrations for establishment
   */
  listConnections: async (): Promise<IntegrationConnection[]> => {
    return apiClient.get<IntegrationConnection[]>('/integrations/connections');
  },

  /**
   * Get connection details
   */
  getConnection: async (id: string): Promise<IntegrationConnection> => {
    return apiClient.get<IntegrationConnection>(`/integrations/connections/${id}`);
  },

  /**
   * Create connection (for non-OAuth integrations)
   */
  createConnection: async (data: IntegrationConnectionCreate): Promise<IntegrationConnection> => {
    return apiClient.post<IntegrationConnection>('/integrations/connections', data);
  },

  /**
   * Update connection config
   */
  updateConnection: async (
    id: string,
    config: Record<string, unknown>
  ): Promise<IntegrationConnection> => {
    return apiClient.patch<IntegrationConnection>(
      `/integrations/connections/${id}`,
      { config }
    );
  },

  /**
   * Delete connection
   */
  deleteConnection: async (id: string): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(`/integrations/connections/${id}`);
  },

  /**
   * Toggle connection active state
   */
  toggleConnection: async (id: string, isActive: boolean): Promise<IntegrationConnection> => {
    return apiClient.patch<IntegrationConnection>(
      `/integrations/connections/${id}`,
      { is_active: isActive }
    );
  },

  // ============ OAuth ============

  /**
   * Start OAuth flow
   */
  startOAuth: async (integrationId: string): Promise<OAuthStartResponse> => {
    return apiClient.post<OAuthStartResponse>(
      `/integrations/${integrationId}/oauth/start`
    );
  },

  /**
   * Complete OAuth flow (handle callback)
   */
  completeOAuth: async (
    integrationId: string,
    code: string,
    state: string
  ): Promise<IntegrationConnection> => {
    return apiClient.post<IntegrationConnection>(
      `/integrations/${integrationId}/oauth/callback`,
      { code, state }
    );
  },

  // ============ Sync ============

  /**
   * Trigger sync for connection
   */
  syncConnection: async (id: string): Promise<MessageResponse> => {
    return apiClient.post<MessageResponse>(`/integrations/connections/${id}/sync`);
  },

  /**
   * Get sync status
   */
  getSyncStatus: async (id: string): Promise<{
    status: 'syncing' | 'synced' | 'error';
    last_sync_at?: string;
    error?: string;
  }> => {
    return apiClient.get(`/integrations/connections/${id}/sync-status`);
  },

  // ============ Tool Proxy ============

  /**
   * Execute tool through integration
   */
  executeTool: async (
    connectionId: string,
    request: ToolProxyRequest
  ): Promise<ToolProxyResponse> => {
    return apiClient.post<ToolProxyResponse>(
      `/integrations/connections/${connectionId}/execute`,
      request
    );
  },
};
