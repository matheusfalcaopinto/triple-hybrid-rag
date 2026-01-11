/**
 * Agents hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { agentsApi } from '../services/agents';
import type { AgentCreate, AgentUpdate, AgentVersionCreate } from '../types';

export const agentKeys = {
  all: ['agents'] as const,
  lists: () => [...agentKeys.all, 'list'] as const,
  list: (filters?: Record<string, unknown>) => [...agentKeys.lists(), filters] as const,
  details: () => [...agentKeys.all, 'detail'] as const,
  detail: (id: string) => [...agentKeys.details(), id] as const,
  versions: (id: string) => [...agentKeys.detail(id), 'versions'] as const,
  deployments: (id: string) => [...agentKeys.detail(id), 'deployments'] as const,
};

/**
 * Hook to list agents
 */
export function useAgents(filters?: { status?: string; type?: string }) {
  return useQuery({
    queryKey: agentKeys.list(filters),
    queryFn: () => agentsApi.list(filters),
  });
}

/**
 * Hook to get agent by ID
 */
export function useAgent(id: string) {
  return useQuery({
    queryKey: agentKeys.detail(id),
    queryFn: () => agentsApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook to create agent
 */
export function useCreateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: AgentCreate) => agentsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
    },
  });
}

/**
 * Hook to update agent
 */
export function useUpdateAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: AgentUpdate }) =>
      agentsApi.update(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: agentKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
    },
  });
}

/**
 * Hook to delete agent
 */
export function useDeleteAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => agentsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
    },
  });
}

/**
 * Hook to list agent versions
 */
export function useAgentVersions(agentId: string) {
  return useQuery({
    queryKey: agentKeys.versions(agentId),
    queryFn: () => agentsApi.listVersions(agentId),
    enabled: !!agentId,
  });
}

/**
 * Hook to create agent version
 */
export function useCreateAgentVersion() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: AgentVersionCreate }) =>
      agentsApi.createVersion(agentId, data),
    onSuccess: (_, { agentId }) => {
      queryClient.invalidateQueries({ queryKey: agentKeys.versions(agentId) });
      queryClient.invalidateQueries({ queryKey: agentKeys.detail(agentId) });
    },
  });
}

/**
 * Hook to set active version
 */
export function useSetActiveVersion() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agentId, versionId }: { agentId: string; versionId: string }) =>
      agentsApi.setActiveVersion(agentId, versionId),
    onSuccess: (_, { agentId }) => {
      queryClient.invalidateQueries({ queryKey: agentKeys.detail(agentId) });
    },
  });
}

/**
 * Hook to deploy agent
 */
export function useDeployAgent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      agentId,
      runtimeId,
      versionId,
    }: {
      agentId: string;
      runtimeId: string;
      versionId?: string;
    }) => agentsApi.deploy(agentId, runtimeId, versionId),
    onSuccess: (_, { agentId }) => {
      queryClient.invalidateQueries({ queryKey: agentKeys.deployments(agentId) });
    },
  });
}

/**
 * Hook to test agent
 */
export function useTestAgent() {
  return useMutation({
    mutationFn: ({
      agentId,
      input,
      versionId,
    }: {
      agentId: string;
      input: string;
      versionId?: string;
    }) => agentsApi.test(agentId, input, versionId),
  });
}
