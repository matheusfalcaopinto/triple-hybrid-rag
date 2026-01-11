/**
 * API Client with authentication and error handling
 */

import { config, API_URL } from '../config';

// Error types
export class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public data?: unknown
  ) {
    super(`API Error: ${status} ${statusText}`);
    this.name = 'ApiError';
  }
}

export class AuthError extends ApiError {
  constructor(message: string = 'Authentication required') {
    super(401, message);
    this.name = 'AuthError';
  }
}

// Token management
const tokenStorage = {
  getAccessToken: (): string | null => {
    return localStorage.getItem(config.accessTokenKey);
  },
  
  getRefreshToken: (): string | null => {
    return localStorage.getItem(config.refreshTokenKey);
  },
  
  setTokens: (accessToken: string, refreshToken?: string): void => {
    localStorage.setItem(config.accessTokenKey, accessToken);
    if (refreshToken) {
      localStorage.setItem(config.refreshTokenKey, refreshToken);
    }
  },
  
  clearTokens: (): void => {
    localStorage.removeItem(config.accessTokenKey);
    localStorage.removeItem(config.refreshTokenKey);
  },
};

// Request options type
interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
  skipAuth?: boolean;
}

// Build URL with query params
function buildUrl(endpoint: string, params?: Record<string, string | number | boolean | undefined>): string {
  const url = new URL(endpoint.startsWith('http') ? endpoint : `${API_URL}${endpoint}`);
  
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        url.searchParams.append(key, String(value));
      }
    });
  }
  
  return url.toString();
}

// Refresh token logic
let refreshPromise: Promise<boolean> | null = null;

async function refreshAccessToken(): Promise<boolean> {
  const refreshToken = tokenStorage.getRefreshToken();
  
  if (!refreshToken) {
    return false;
  }
  
  try {
    const response = await fetch(`${API_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    
    if (!response.ok) {
      return false;
    }
    
    const data = await response.json();
    tokenStorage.setTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

// Main fetch function
async function apiFetch<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { params, skipAuth = false, ...fetchOptions } = options;
  
  const url = buildUrl(endpoint, params);
  
  // Setup headers
  const headers = new Headers(fetchOptions.headers);
  
  if (!headers.has('Content-Type') && fetchOptions.body) {
    headers.set('Content-Type', 'application/json');
  }
  
  // Add auth header if not skipped
  if (!skipAuth) {
    const token = tokenStorage.getAccessToken();
    if (token) {
      headers.set('Authorization', `${config.tokenType} ${token}`);
    }
  }
  
  // Make request
  let response = await fetch(url, {
    ...fetchOptions,
    headers,
  });
  
  // Handle 401 - try refresh
  if (response.status === 401 && !skipAuth) {
    // Deduplicate refresh requests
    if (!refreshPromise) {
      refreshPromise = refreshAccessToken().finally(() => {
        refreshPromise = null;
      });
    }
    
    const refreshed = await refreshPromise;
    
    if (refreshed) {
      // Retry with new token
      const newToken = tokenStorage.getAccessToken();
      headers.set('Authorization', `${config.tokenType} ${newToken}`);
      
      response = await fetch(url, {
        ...fetchOptions,
        headers,
      });
    } else {
      // Clear tokens and redirect to login
      tokenStorage.clearTokens();
      window.dispatchEvent(new CustomEvent('auth:logout'));
      throw new AuthError();
    }
  }
  
  // Handle errors
  if (!response.ok) {
    let errorData: unknown;
    try {
      errorData = await response.json();
    } catch {
      errorData = await response.text();
    }
    throw new ApiError(response.status, response.statusText, errorData);
  }
  
  // Handle empty responses
  if (response.status === 204) {
    return undefined as T;
  }
  
  // Parse JSON
  return response.json();
}

// HTTP method helpers
export const apiClient = {
  get: <T>(endpoint: string, options?: RequestOptions) =>
    apiFetch<T>(endpoint, { ...options, method: 'GET' }),
    
  post: <T>(endpoint: string, data?: unknown, options?: RequestOptions) =>
    apiFetch<T>(endpoint, {
      ...options,
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    }),
    
  put: <T>(endpoint: string, data?: unknown, options?: RequestOptions) =>
    apiFetch<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    }),
    
  patch: <T>(endpoint: string, data?: unknown, options?: RequestOptions) =>
    apiFetch<T>(endpoint, {
      ...options,
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    }),
    
  delete: <T>(endpoint: string, options?: RequestOptions) =>
    apiFetch<T>(endpoint, { ...options, method: 'DELETE' }),
    
  // Upload with FormData
  upload: <T>(endpoint: string, formData: FormData, options?: RequestOptions) =>
    apiFetch<T>(endpoint, {
      ...options,
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type with boundary
    }),
};

// Export token management for auth context
export { tokenStorage };
