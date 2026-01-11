/**
 * Application configuration
 */

export const config = {
  // API Configuration
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  apiVersion: 'v1',
  
  // Auth Configuration
  accessTokenKey: 'access_token',
  refreshTokenKey: 'refresh_token',
  tokenType: 'Bearer',
  
  // SSE Configuration
  sseReconnectDelay: 3000, // ms
  sseHeartbeatTimeout: 45000, // ms
  
  // Feature flags
  enableMockData: import.meta.env.VITE_ENABLE_MOCK_DATA === 'true',
  enableDevTools: import.meta.env.DEV,
} as const;

// Computed API URL with version
export const API_URL = `${config.apiBaseUrl}/api/${config.apiVersion}`;

// Environment checks
export const isDevelopment = import.meta.env.DEV;
export const isProduction = import.meta.env.PROD;
