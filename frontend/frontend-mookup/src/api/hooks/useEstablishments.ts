/**
 * Establishments hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { establishmentsApi } from '../services/establishments';
import type { EstablishmentCreate, EstablishmentUpdate } from '../types';

export const establishmentKeys = {
  all: ['establishments'] as const,
  lists: () => [...establishmentKeys.all, 'list'] as const,
  list: (filters?: Record<string, unknown>) => [...establishmentKeys.lists(), filters] as const,
  details: () => [...establishmentKeys.all, 'detail'] as const,
  detail: (id: string) => [...establishmentKeys.details(), id] as const,
  members: (id: string) => [...establishmentKeys.detail(id), 'members'] as const,
  phoneNumbers: (id: string) => [...establishmentKeys.detail(id), 'phone-numbers'] as const,
};

/**
 * Hook to list establishments
 */
export function useEstablishments() {
  return useQuery({
    queryKey: establishmentKeys.lists(),
    queryFn: () => establishmentsApi.list(),
  });
}

/**
 * Hook to get establishment by ID
 */
export function useEstablishment(id: string) {
  return useQuery({
    queryKey: establishmentKeys.detail(id),
    queryFn: () => establishmentsApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook to create establishment
 */
export function useCreateEstablishment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: EstablishmentCreate) => establishmentsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: establishmentKeys.lists() });
    },
  });
}

/**
 * Hook to update establishment
 */
export function useUpdateEstablishment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: EstablishmentUpdate }) =>
      establishmentsApi.update(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: establishmentKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: establishmentKeys.lists() });
    },
  });
}

/**
 * Hook to delete establishment
 */
export function useDeleteEstablishment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => establishmentsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: establishmentKeys.lists() });
    },
  });
}

/**
 * Hook to list establishment members
 */
export function useEstablishmentMembers(establishmentId: string) {
  return useQuery({
    queryKey: establishmentKeys.members(establishmentId),
    queryFn: () => establishmentsApi.listMembers(establishmentId),
    enabled: !!establishmentId,
  });
}

/**
 * Hook to invite member
 */
export function useInviteMember() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      establishmentId,
      email,
      role,
    }: {
      establishmentId: string;
      email: string;
      role: 'admin' | 'member' | 'viewer';
    }) => establishmentsApi.inviteMember(establishmentId, email, role),
    onSuccess: (_, { establishmentId }) => {
      queryClient.invalidateQueries({ queryKey: establishmentKeys.members(establishmentId) });
    },
  });
}

/**
 * Hook to list phone numbers
 */
export function usePhoneNumbers(establishmentId: string) {
  return useQuery({
    queryKey: establishmentKeys.phoneNumbers(establishmentId),
    queryFn: () => establishmentsApi.listPhoneNumbers(establishmentId),
    enabled: !!establishmentId,
  });
}
