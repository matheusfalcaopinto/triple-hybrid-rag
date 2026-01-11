/**
 * Leads hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { leadsApi, type LeadFilters } from '../services/leads';
import type { LeadCreate, LeadUpdate, LeadStatus } from '../types';

export const leadKeys = {
  all: ['leads'] as const,
  lists: () => [...leadKeys.all, 'list'] as const,
  list: (filters?: LeadFilters) => [...leadKeys.lists(), filters] as const,
  byStatus: () => [...leadKeys.all, 'by-status'] as const,
  details: () => [...leadKeys.all, 'detail'] as const,
  detail: (id: string) => [...leadKeys.details(), id] as const,
  callHistory: (id: string) => [...leadKeys.detail(id), 'calls'] as const,
};

/**
 * Hook to list leads
 */
export function useLeads(filters?: LeadFilters & { page?: number; page_size?: number }) {
  return useQuery({
    queryKey: leadKeys.list(filters),
    queryFn: () => leadsApi.list(filters),
  });
}

/**
 * Hook to get leads grouped by status (Kanban)
 */
export function useLeadsByStatus() {
  return useQuery({
    queryKey: leadKeys.byStatus(),
    queryFn: () => leadsApi.getByStatus(),
  });
}

/**
 * Hook to get lead by ID
 */
export function useLead(id: string) {
  return useQuery({
    queryKey: leadKeys.detail(id),
    queryFn: () => leadsApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook to create lead
 */
export function useCreateLead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: LeadCreate) => leadsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leadKeys.lists() });
      queryClient.invalidateQueries({ queryKey: leadKeys.byStatus() });
    },
  });
}

/**
 * Hook to update lead
 */
export function useUpdateLead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: LeadUpdate }) =>
      leadsApi.update(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: leadKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: leadKeys.lists() });
      queryClient.invalidateQueries({ queryKey: leadKeys.byStatus() });
    },
  });
}

/**
 * Hook to update lead status
 */
export function useUpdateLeadStatus() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: LeadStatus }) =>
      leadsApi.updateStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leadKeys.lists() });
      queryClient.invalidateQueries({ queryKey: leadKeys.byStatus() });
    },
  });
}

/**
 * Hook to delete lead
 */
export function useDeleteLead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => leadsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leadKeys.lists() });
      queryClient.invalidateQueries({ queryKey: leadKeys.byStatus() });
    },
  });
}

/**
 * Hook to import leads from CSV
 */
export function useImportLeads() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => leadsApi.importCsv(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leadKeys.lists() });
      queryClient.invalidateQueries({ queryKey: leadKeys.byStatus() });
    },
  });
}

/**
 * Hook to get lead call history
 */
export function useLeadCallHistory(id: string) {
  return useQuery({
    queryKey: leadKeys.callHistory(id),
    queryFn: () => leadsApi.getCallHistory(id),
    enabled: !!id,
  });
}
