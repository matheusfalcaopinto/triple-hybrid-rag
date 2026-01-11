/**
 * Authentication hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { authApi } from '../services/auth';
import type { UserLogin, UserCreate } from '../types';

export const authKeys = {
  all: ['auth'] as const,
  me: () => [...authKeys.all, 'me'] as const,
};

/**
 * Hook to get current user
 */
export function useCurrentUser() {
  return useQuery({
    queryKey: authKeys.me(),
    queryFn: () => authApi.getMe(),
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook for login
 */
export function useLogin() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (credentials: UserLogin) => authApi.login(credentials),
    onSuccess: () => {
      // Invalidate user query to refetch
      queryClient.invalidateQueries({ queryKey: authKeys.me() });
    },
  });
}

/**
 * Hook for registration
 */
export function useRegister() {
  return useMutation({
    mutationFn: (data: UserCreate) => authApi.register(data),
  });
}

/**
 * Hook for logout
 */
export function useLogout() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      authApi.logout();
    },
    onSuccess: () => {
      // Clear all queries
      queryClient.clear();
    },
  });
}

/**
 * Hook for forgot password
 */
export function useForgotPassword() {
  return useMutation({
    mutationFn: (email: string) => authApi.forgotPassword(email),
  });
}

/**
 * Hook for password change
 */
export function useChangePassword() {
  return useMutation({
    mutationFn: ({ currentPassword, newPassword }: { currentPassword: string; newPassword: string }) =>
      authApi.changePassword(currentPassword, newPassword),
  });
}

/**
 * Hook to update user profile
 */
export function useUpdateProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: authApi.updateMe,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: authKeys.me() });
    },
  });
}
