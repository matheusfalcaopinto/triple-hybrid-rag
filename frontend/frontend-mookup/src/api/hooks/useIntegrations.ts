/**
 * Integrations hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { integrationsApi, type ToolProxyRequest } from '../services/integrations';
import type { IntegrationConnectionCreate } from '../types';

export const integrationKeys = {
  all: ['integrations'] as const,
  available: () => [...integrationKeys.all, 'available'] as const,
  connections: () => [...integrationKeys.all, 'connections'] as const,
  connection: (id: string) => [...integrationKeys.connections(), id] as const,
};

/**
 * Hook to list available integrations
 */
export function useAvailableIntegrations() {
  return useQuery({
    queryKey: integrationKeys.available(),
    queryFn: () => integrationsApi.listAvailable(),
  });
}

/**
 * Hook to list connected integrations
 */
export function useIntegrationConnections() {
  return useQuery({
    queryKey: integrationKeys.connections(),
    queryFn: () => integrationsApi.listConnections(),
  });
}

/**
 * Hook to get connection details
 */
export function useIntegrationConnection(id: string) {
  return useQuery({
    queryKey: integrationKeys.connection(id),
    queryFn: () => integrationsApi.getConnection(id),
    enabled: !!id,
  });
}

/**
 * Hook to create connection
 */
export function useCreateConnection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: IntegrationConnectionCreate) =>
      integrationsApi.createConnection(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: integrationKeys.connections() });
    },
  });
}

/**
 * Hook to delete connection
 */
export function useDeleteConnection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => integrationsApi.deleteConnection(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: integrationKeys.connections() });
    },
  });
}

/**
 * Hook to toggle connection active state
 */
export function useToggleConnection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, isActive }: { id: string; isActive: boolean }) =>
      integrationsApi.toggleConnection(id, isActive),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: integrationKeys.connection(id) });
      queryClient.invalidateQueries({ queryKey: integrationKeys.connections() });
    },
  });
}

/**
 * Hook to start OAuth flow
 */
export function useStartOAuth() {
  return useMutation({
    mutationFn: (integrationId: string) => integrationsApi.startOAuth(integrationId),
    onSuccess: (data) => {
      // Redirect to OAuth URL
      window.location.href = data.auth_url;
    },
  });
}

/**
 * Hook to sync connection
 */
export function useSyncConnection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => integrationsApi.syncConnection(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: integrationKeys.connection(id) });
    },
  });
}

/**
 * Hook to execute tool through integration
 */
export function useExecuteTool() {
  return useMutation({
    mutationFn: ({
      connectionId,
      request,
    }: {
      connectionId: string;
      request: ToolProxyRequest;
    }) => integrationsApi.executeTool(connectionId, request),
  });
}
