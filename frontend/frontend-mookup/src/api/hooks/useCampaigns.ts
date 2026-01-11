/**
 * Campaigns hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { campaignsApi } from '../services/campaigns';
import type { CampaignCreate, CampaignUpdate, CampaignStatus } from '../types';

export const campaignKeys = {
  all: ['campaigns'] as const,
  lists: () => [...campaignKeys.all, 'list'] as const,
  list: (status?: CampaignStatus) => [...campaignKeys.lists(), status] as const,
  details: () => [...campaignKeys.all, 'detail'] as const,
  detail: (id: string) => [...campaignKeys.details(), id] as const,
  enrollments: (id: string) => [...campaignKeys.detail(id), 'enrollments'] as const,
  stats: (id: string) => [...campaignKeys.detail(id), 'stats'] as const,
  runs: (id: string) => [...campaignKeys.detail(id), 'runs'] as const,
};

/**
 * Hook to list campaigns
 */
export function useCampaigns(status?: CampaignStatus) {
  return useQuery({
    queryKey: campaignKeys.list(status),
    queryFn: () => campaignsApi.list({ status }),
  });
}

/**
 * Hook to get campaign by ID
 */
export function useCampaign(id: string) {
  return useQuery({
    queryKey: campaignKeys.detail(id),
    queryFn: () => campaignsApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook to create campaign
 */
export function useCreateCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CampaignCreate) => campaignsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.lists() });
    },
  });
}

/**
 * Hook to update campaign
 */
export function useUpdateCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: CampaignUpdate }) =>
      campaignsApi.update(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: campaignKeys.lists() });
    },
  });
}

/**
 * Hook to start campaign
 */
export function useStartCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => campaignsApi.start(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: campaignKeys.lists() });
    },
  });
}

/**
 * Hook to pause campaign
 */
export function usePauseCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => campaignsApi.pause(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.detail(id) });
    },
  });
}

/**
 * Hook to resume campaign
 */
export function useResumeCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => campaignsApi.resume(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.detail(id) });
    },
  });
}

/**
 * Hook to cancel campaign
 */
export function useCancelCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => campaignsApi.cancel(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: campaignKeys.lists() });
    },
  });
}

/**
 * Hook to get campaign enrollments
 */
export function useCampaignEnrollments(id: string, status?: string) {
  return useQuery({
    queryKey: campaignKeys.enrollments(id),
    queryFn: () => campaignsApi.getEnrollments(id, { status }),
    enabled: !!id,
  });
}

/**
 * Hook to enroll leads in campaign
 */
export function useEnrollLeads() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ campaignId, leadIds }: { campaignId: string; leadIds: string[] }) =>
      campaignsApi.enrollLeads(campaignId, leadIds),
    onSuccess: (_, { campaignId }) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.enrollments(campaignId) });
      queryClient.invalidateQueries({ queryKey: campaignKeys.detail(campaignId) });
    },
  });
}

/**
 * Hook to get campaign stats
 */
export function useCampaignStats(id: string) {
  return useQuery({
    queryKey: campaignKeys.stats(id),
    queryFn: () => campaignsApi.getStats(id),
    enabled: !!id,
  });
}
