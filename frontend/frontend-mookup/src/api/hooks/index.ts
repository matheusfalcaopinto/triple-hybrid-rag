/**
 * React Query hooks for data fetching
 */

// Auth
export { authKeys, useCurrentUser, useLogin, useRegister, useLogout, useForgotPassword, useChangePassword, useUpdateProfile } from './useAuth';

// Establishments
export { establishmentKeys, useEstablishments, useEstablishment, useCreateEstablishment, useUpdateEstablishment, useDeleteEstablishment, useEstablishmentMembers, useInviteMember, usePhoneNumbers } from './useEstablishments';

// Agents
export { agentKeys, useAgents, useAgent, useCreateAgent, useUpdateAgent, useDeleteAgent, useAgentVersions, useCreateAgentVersion, useSetActiveVersion, useDeployAgent, useTestAgent } from './useAgents';

// Calls
export { callKeys, useCalls, useCall, useCallTranscript, useCallRecording, useCallStats, useCallIntervention, useWhisper, useHangup } from './useCalls';

// Leads
export { leadKeys, useLeads, useLeadsByStatus, useLead, useCreateLead, useUpdateLead, useUpdateLeadStatus, useDeleteLead, useImportLeads, useLeadCallHistory } from './useLeads';

// Campaigns
export { campaignKeys, useCampaigns, useCampaign, useCreateCampaign, useUpdateCampaign, useStartCampaign, usePauseCampaign, useResumeCampaign, useCancelCampaign, useCampaignEnrollments, useEnrollLeads, useCampaignStats } from './useCampaigns';

// Dashboard
export { dashboardKeys, useDashboardMetrics, useKPIs, useCallVolume, useAgentUtilization, useRecentActivity, useAlerts, useMarkAlertRead, useMarkAllAlertsRead, useDismissAlert } from './useDashboard';

// Reports
export { reportKeys, useCallVolumeReport, useCallDurationReport, useCostReport, usePerformanceReport, useAgentComparisonReport, useSentimentReport, useReportTemplates } from './useReports';

// Integrations
export { integrationKeys, useAvailableIntegrations, useIntegrationConnections, useIntegrationConnection, useCreateConnection, useDeleteConnection, useToggleConnection, useStartOAuth, useSyncConnection, useExecuteTool } from './useIntegrations';

// API Keys
export { apiKeyKeys, useAPIKeys, useAPIKey, useCreateAPIKey, useRevokeAPIKey, useToggleAPIKey, useRotateAPIKey } from './useAPIKeys';

// SSE
export { useSSE, useCallTranscriptStream } from './useSSE';
