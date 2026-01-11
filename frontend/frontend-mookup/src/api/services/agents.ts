/**
 * Agents API service
 */

import { apiClient } from '../client';
import type {
  Agent,
  AgentCreate,
  AgentUpdate,
  AgentVersion,
  AgentVersionCreate,
  AgentWithVersion,
  PaginatedResponse,
  MessageResponse,
} from '../types';

export interface AgentDeployment {
  id: string;
  agent_id: string;
  agent_version_id: string;
  runtime_id: string;
  runtime_name: string;
  deployed_at: string;
  deployed_by_id: string;
  is_active: boolean;
}

export const agentsApi = {
  /**
   * List all agents for current establishment
   */
  list: async (params?: {
    status?: string;
    type?: string;
    page?: number;
    page_size?: number;
  }): Promise<PaginatedResponse<AgentWithVersion>> => {
    return apiClient.get<PaginatedResponse<AgentWithVersion>>('/agents', { params });
  },

  /**
   * Get agent by ID
   */
  get: async (id: string): Promise<AgentWithVersion> => {
    return apiClient.get<AgentWithVersion>(`/agents/${id}`);
  },

  /**
   * Create new agent
   */
  create: async (data: AgentCreate): Promise<Agent> => {
    return apiClient.post<Agent>('/agents', data);
  },

  /**
   * Update agent
   */
  update: async (id: string, data: AgentUpdate): Promise<Agent> => {
    return apiClient.patch<Agent>(`/agents/${id}`, data);
  },

  /**
   * Delete agent
   */
  delete: async (id: string): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(`/agents/${id}`);
  },

  // ============ Versions ============

  /**
   * List agent versions
   */
  listVersions: async (agentId: string): Promise<AgentVersion[]> => {
    return apiClient.get<AgentVersion[]>(`/agents/${agentId}/versions`);
  },

  /**
   * Get specific version
   */
  getVersion: async (agentId: string, versionId: string): Promise<AgentVersion> => {
    return apiClient.get<AgentVersion>(`/agents/${agentId}/versions/${versionId}`);
  },

  /**
   * Create new version
   */
  createVersion: async (agentId: string, data: AgentVersionCreate): Promise<AgentVersion> => {
    return apiClient.post<AgentVersion>(`/agents/${agentId}/versions`, data);
  },

  /**
   * Set active version
   */
  setActiveVersion: async (agentId: string, versionId: string): Promise<Agent> => {
    return apiClient.post<Agent>(`/agents/${agentId}/versions/${versionId}/activate`);
  },

  // ============ Deployments ============

  /**
   * List agent deployments
   */
  listDeployments: async (agentId: string): Promise<AgentDeployment[]> => {
    return apiClient.get<AgentDeployment[]>(`/agents/${agentId}/deployments`);
  },

  /**
   * Deploy agent to runtime
   */
  deploy: async (
    agentId: string,
    runtimeId: string,
    versionId?: string
  ): Promise<AgentDeployment> => {
    return apiClient.post<AgentDeployment>(`/agents/${agentId}/deploy`, {
      runtime_id: runtimeId,
      version_id: versionId,
    });
  },

  /**
   * Undeploy agent from runtime
   */
  undeploy: async (agentId: string, runtimeId: string): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(`/agents/${agentId}/deployments/${runtimeId}`);
  },

  // ============ Testing ============

  /**
   * Test agent with sample input
   */
  test: async (
    agentId: string,
    input: string,
    versionId?: string
  ): Promise<{ response: string; latency_ms: number }> => {
    return apiClient.post<{ response: string; latency_ms: number }>(
      `/agents/${agentId}/test`,
      { input, version_id: versionId }
    );
  },
};
