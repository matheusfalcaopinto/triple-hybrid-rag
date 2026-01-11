/**
 * Authentication API service
 */

import { apiClient, tokenStorage } from '../client';
import type {
  UserLogin,
  UserCreate,
  TokenResponse,
  User,
  UserWithEstablishments,
  MessageResponse,
} from '../types';

export const authApi = {
  /**
   * Login with email and password
   */
  login: async (credentials: UserLogin): Promise<TokenResponse> => {
    const response = await apiClient.post<TokenResponse>(
      '/auth/login',
      credentials,
      { skipAuth: true }
    );
    
    // Store tokens
    tokenStorage.setTokens(response.access_token, response.refresh_token);
    
    return response;
  },

  /**
   * Register a new user
   */
  register: async (data: UserCreate): Promise<User> => {
    return apiClient.post<User>('/auth/register', data, { skipAuth: true });
  },

  /**
   * Request password reset
   */
  forgotPassword: async (email: string): Promise<MessageResponse> => {
    return apiClient.post<MessageResponse>(
      '/auth/forgot-password',
      { email },
      { skipAuth: true }
    );
  },

  /**
   * Reset password with token
   */
  resetPassword: async (token: string, newPassword: string): Promise<MessageResponse> => {
    return apiClient.post<MessageResponse>(
      '/auth/reset-password',
      { token, new_password: newPassword },
      { skipAuth: true }
    );
  },

  /**
   * Refresh access token
   */
  refreshToken: async (refreshToken: string): Promise<TokenResponse> => {
    const response = await apiClient.post<TokenResponse>(
      '/auth/refresh',
      { refresh_token: refreshToken },
      { skipAuth: true }
    );
    
    tokenStorage.setTokens(response.access_token, response.refresh_token);
    
    return response;
  },

  /**
   * Logout (clear tokens)
   */
  logout: (): void => {
    tokenStorage.clearTokens();
    window.dispatchEvent(new CustomEvent('auth:logout'));
  },

  /**
   * Get current user profile
   */
  getMe: async (): Promise<UserWithEstablishments> => {
    return apiClient.get<UserWithEstablishments>('/users/me');
  },

  /**
   * Update current user profile
   */
  updateMe: async (data: Partial<User>): Promise<User> => {
    return apiClient.patch<User>('/users/me', data);
  },

  /**
   * Change password
   */
  changePassword: async (currentPassword: string, newPassword: string): Promise<MessageResponse> => {
    return apiClient.post<MessageResponse>('/users/me/password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  },

  /**
   * Check if user is authenticated (has valid token)
   */
  isAuthenticated: (): boolean => {
    return !!tokenStorage.getAccessToken();
  },
};
