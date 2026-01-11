/**
 * API Keys hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiKeysApi } from '../services/apiKeys';
import type { APIKeyCreate } from '../types';

export const apiKeyKeys = {
  all: ['api-keys'] as const,
  list: () => [...apiKeyKeys.all, 'list'] as const,
  detail: (id: string) => [...apiKeyKeys.all, 'detail', id] as const,
};

/**
 * Hook to list all API keys
 */
export function useAPIKeys() {
  return useQuery({
    queryKey: apiKeyKeys.list(),
    queryFn: () => apiKeysApi.list(),
  });
}

/**
 * Hook to get single API key
 */
export function useAPIKey(id: string) {
  return useQuery({
    queryKey: apiKeyKeys.detail(id),
    queryFn: () => apiKeysApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook to create API key
 */
export function useCreateAPIKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: APIKeyCreate) => apiKeysApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.all });
    },
  });
}

/**
 * Hook to revoke API key
 */
export function useRevokeAPIKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => apiKeysApi.revoke(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.all });
    },
  });
}

/**
 * Hook to toggle API key status
 */
export function useToggleAPIKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, isActive }: { id: string; isActive: boolean }) =>
      apiKeysApi.toggle(id, isActive),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.all });
    },
  });
}

/**
 * Hook to rotate API key
 */
export function useRotateAPIKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => apiKeysApi.rotate(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.all });
    },
  });
}
