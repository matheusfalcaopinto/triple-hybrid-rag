/**
 * Authentication Context and Provider
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { authApi } from '../api/services/auth';
import { tokenStorage } from '../api/client';
import type { User, UserWithEstablishments, EstablishmentMembership } from '../api/types';

interface AuthContextType {
  user: UserWithEstablishments | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  currentEstablishment: EstablishmentMembership | null;
  setCurrentEstablishment: (establishment: EstablishmentMembership) => void;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<UserWithEstablishments | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [currentEstablishment, setCurrentEstablishment] = useState<EstablishmentMembership | null>(null);

  // Load user on mount
  useEffect(() => {
    const loadUser = async () => {
      if (!tokenStorage.getAccessToken()) {
        setIsLoading(false);
        return;
      }

      try {
        const userData = await authApi.getMe();
        setUser(userData);
        
        // Set default establishment
        if (userData.establishments.length > 0) {
          const savedEstId = localStorage.getItem('current_establishment_id');
          const savedEst = userData.establishments.find(e => e.establishment_id === savedEstId);
          setCurrentEstablishment(savedEst || userData.establishments[0]);
        }
      } catch (error) {
        console.error('Failed to load user:', error);
        tokenStorage.clearTokens();
      } finally {
        setIsLoading(false);
      }
    };

    loadUser();

    // Listen for logout events from API client
    const handleLogout = () => {
      setUser(null);
      setCurrentEstablishment(null);
      queryClient.clear();
    };

    window.addEventListener('auth:logout', handleLogout);
    return () => window.removeEventListener('auth:logout', handleLogout);
  }, [queryClient]);

  // Save establishment selection
  useEffect(() => {
    if (currentEstablishment) {
      localStorage.setItem('current_establishment_id', currentEstablishment.establishment_id);
    }
  }, [currentEstablishment]);

  const login = useCallback(async (email: string, password: string) => {
    await authApi.login({ email, password });
    const userData = await authApi.getMe();
    setUser(userData);
    
    if (userData.establishments.length > 0) {
      setCurrentEstablishment(userData.establishments[0]);
    }
  }, []);

  const logout = useCallback(() => {
    authApi.logout();
    setUser(null);
    setCurrentEstablishment(null);
    queryClient.clear();
  }, [queryClient]);

  const refreshUser = useCallback(async () => {
    if (!tokenStorage.getAccessToken()) return;
    
    try {
      const userData = await authApi.getMe();
      setUser(userData);
    } catch (error) {
      console.error('Failed to refresh user:', error);
    }
  }, []);

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated: !!user,
    currentEstablishment,
    setCurrentEstablishment,
    login,
    logout,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

/**
 * Hook to get current establishment ID for API calls
 */
export function useEstablishmentId(): string | null {
  const { currentEstablishment } = useAuth();
  return currentEstablishment?.establishment_id ?? null;
}

/**
 * Higher-order component for protected routes
 */
export function withAuth<P extends object>(
  Component: React.ComponentType<P>
): React.FC<P> {
  return function AuthenticatedComponent(props: P) {
    const { isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      );
    }

    if (!isAuthenticated) {
      // Redirect to login
      window.location.href = '/login';
      return null;
    }

    return <Component {...props} />;
  };
}
