/**
 * TypeScript types matching backend Pydantic schemas
 */

// ============ Common ============

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface MessageResponse {
  message: string;
  detail?: string;
}

// ============ Auth ============

export interface UserLogin {
  email: string;
  password: string;
}

export interface UserCreate {
  email: string;
  password: string;
  full_name?: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserWithEstablishments extends User {
  establishments: EstablishmentMembership[];
}

export interface EstablishmentMembership {
  establishment_id: string;
  establishment_name: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
}

// ============ Establishments ============

export interface Establishment {
  id: string;
  name: string;
  slug: string;
  timezone: string;
  settings: Record<string, unknown>;
  is_active: boolean;
  subscription_tier: string;
  subscription_status: string;
  created_at: string;
  updated_at: string;
}

export interface EstablishmentCreate {
  name: string;
  slug?: string;
  timezone?: string;
  settings?: Record<string, unknown>;
}

export interface EstablishmentUpdate {
  name?: string;
  timezone?: string;
  settings?: Record<string, unknown>;
}

export interface EstablishmentWithStats extends Establishment {
  member_count: number;
  agent_count: number;
  monthly_cost: number;
}

export interface PhoneNumber {
  id: string;
  e164: string;
  display_name?: string;
  is_active: boolean;
  routing_agent_id?: string;
  created_at: string;
}

// ============ Agents ============

export type AgentType = 'inbound' | 'outbound' | 'hybrid';
export type AgentStatus = 'draft' | 'active' | 'paused' | 'archived';

export interface Agent {
  id: string;
  name: string;
  description?: string;
  agent_type: AgentType;
  status: AgentStatus;
  active_version_id?: string;
  created_at: string;
  updated_at: string;
}

export interface AgentCreate {
  name: string;
  description?: string;
  agent_type?: AgentType;
}

export interface AgentUpdate {
  name?: string;
  description?: string;
  status?: AgentStatus;
}

export interface AgentVersion {
  id: string;
  agent_id: string;
  version_number: number;
  system_prompt: string;
  voice_id: string;
  llm_model: string;
  tools: AgentTool[];
  config: Record<string, unknown>;
  created_at: string;
  created_by_id: string;
}

export interface AgentTool {
  name: string;
  description: string;
  integration_id?: string;
  parameters: Record<string, unknown>;
}

export interface AgentVersionCreate {
  system_prompt: string;
  voice_id: string;
  llm_model: string;
  tools?: AgentTool[];
  config?: Record<string, unknown>;
}

export interface AgentWithVersion extends Agent {
  active_version?: AgentVersion;
}

// ============ Runtimes ============

export type RuntimeStatus = 'pending' | 'starting' | 'running' | 'draining' | 'stopped' | 'error';

export interface Runtime {
  id: string;
  name: string;
  establishment_id: string;
  status: RuntimeStatus;
  container_name: string;
  base_url?: string;
  is_ready: boolean;
  ready_issues?: string[];
  active_calls: number;
  max_concurrent_calls: number;
  current_version?: string;
  last_health_check_at?: string;
  started_at?: string;
  stopped_at?: string;
  created_at: string;
  updated_at: string;
}

export interface RuntimeCreate {
  name: string;
  max_concurrent_calls?: number;
}

export interface RuntimeUpdate {
  name?: string;
  max_concurrent_calls?: number;
}

// ============ Calls ============

export type CallDirection = 'inbound' | 'outbound';
export type CallStatus = 'queued' | 'ringing' | 'in_progress' | 'completed' | 'failed' | 'busy' | 'no_answer';

export interface Call {
  id: string;
  establishment_id: string;
  agent_id?: string;
  runtime_id?: string;
  external_id?: string;
  direction: CallDirection;
  status: CallStatus;
  from_number: string;
  to_number: string;
  started_at?: string;
  answered_at?: string;
  ended_at?: string;
  duration_seconds?: number;
  cost?: number;
  sentiment_score?: number;
  sentiment_label?: 'positive' | 'neutral' | 'negative';
  summary?: string;
  outcome?: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface CallWithDetails extends Call {
  agent_name?: string;
  transcript_count: number;
  has_recording: boolean;
}

export interface CallTranscriptSegment {
  id: string;
  call_id: string;
  speaker: 'agent' | 'caller';
  text: string;
  start_time: number;
  end_time: number;
  confidence?: number;
  created_at: string;
}

export interface CallRecording {
  id: string;
  call_id: string;
  url: string;
  duration_seconds: number;
  format: string;
  size_bytes: number;
  created_at: string;
}

export interface CallFilters {
  status?: CallStatus;
  direction?: CallDirection;
  agent_id?: string;
  date_from?: string;
  date_to?: string;
  search?: string;
}

// ============ Leads ============

export type LeadStatus = 'new' | 'trying' | 'connected' | 'qualified' | 'converted' | 'discarded';

export interface Lead {
  id: string;
  establishment_id: string;
  phone_number: string;
  name?: string;
  email?: string;
  company?: string;
  status: LeadStatus;
  source?: string;
  tags: string[];
  custom_fields: Record<string, unknown>;
  last_contacted_at?: string;
  contact_attempts: number;
  created_at: string;
  updated_at: string;
}

export interface LeadCreate {
  phone_number: string;
  name?: string;
  email?: string;
  company?: string;
  source?: string;
  tags?: string[];
  custom_fields?: Record<string, unknown>;
}

export interface LeadUpdate {
  name?: string;
  email?: string;
  company?: string;
  status?: LeadStatus;
  tags?: string[];
  custom_fields?: Record<string, unknown>;
}

export interface LeadImportResult {
  total: number;
  imported: number;
  skipped: number;
  errors: string[];
}

// ============ Campaigns ============

export type CampaignStatus = 'draft' | 'scheduled' | 'running' | 'paused' | 'completed' | 'cancelled';

export interface Campaign {
  id: string;
  establishment_id: string;
  name: string;
  description?: string;
  agent_id: string;
  status: CampaignStatus;
  schedule: CampaignSchedule;
  pacing: CampaignPacing;
  total_leads: number;
  leads_called: number;
  leads_connected: number;
  leads_qualified: number;
  created_at: string;
  updated_at: string;
}

export interface CampaignSchedule {
  start_date?: string;
  end_date?: string;
  days_of_week: number[];
  start_time: string;
  end_time: string;
  timezone: string;
}

export interface CampaignPacing {
  calls_per_minute: number;
  max_concurrent_calls: number;
  retry_attempts: number;
  retry_delay_minutes: number;
}

export interface CampaignCreate {
  name: string;
  description?: string;
  agent_id: string;
  schedule?: Partial<CampaignSchedule>;
  pacing?: Partial<CampaignPacing>;
}

export interface CampaignUpdate {
  name?: string;
  description?: string;
  status?: CampaignStatus;
  schedule?: Partial<CampaignSchedule>;
  pacing?: Partial<CampaignPacing>;
}

export interface CampaignStats {
  total_leads: number;
  leads_called: number;
  leads_connected: number;
  leads_qualified: number;
  average_call_duration: number;
  success_rate: number;
}

// ============ Dashboard ============

export interface DashboardKPI {
  label: string;
  value: string | number;
  trend?: string;
  trend_label?: string;
  trend_direction?: 'up' | 'down' | 'neutral';
}

export interface DashboardMetrics {
  kpis: DashboardKPI[];
  call_volume_24h: TimeSeriesPoint[];
  agent_utilization: AgentUtilization[];
  recent_activity: ActivityItem[];
}

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface AgentUtilization {
  agent_id: string;
  agent_name: string;
  status: 'online' | 'calling' | 'paused' | 'offline';
  calls_today: number;
  utilization_percent: number;
}

export interface ActivityItem {
  id: string;
  type: 'call_completed' | 'call_started' | 'agent_deployed' | 'lead_qualified' | 'alert';
  message: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  is_read: boolean;
  created_at: string;
}

// ============ Reports ============

export interface ReportFilters {
  date_from: string;
  date_to: string;
  agent_ids?: string[];
  granularity?: 'hour' | 'day' | 'week' | 'month';
}

export interface CallVolumeReport {
  data: TimeSeriesPoint[];
  total_calls: number;
  total_duration_minutes: number;
  average_duration_seconds: number;
}

export interface CostReport {
  data: TimeSeriesPoint[];
  total_cost: number;
  cost_per_call: number;
  cost_by_agent: { agent_id: string; agent_name: string; cost: number }[];
}

export interface PerformanceReport {
  success_rate: number;
  average_handle_time: number;
  first_call_resolution: number;
  sentiment_breakdown: { positive: number; neutral: number; negative: number };
}

// ============ Integrations ============

export type IntegrationType = 'calendar' | 'crm' | 'voice' | 'llm' | 'sms' | 'email' | 'webhook';

export interface Integration {
  id: string;
  name: string;
  slug: string;
  type: IntegrationType;
  description: string;
  logo_url?: string;
  is_available: boolean;
  requires_oauth: boolean;
  config_schema: Record<string, unknown>;
}

export interface IntegrationConnection {
  id: string;
  integration_id: string;
  integration_name: string;
  is_active: boolean;
  config: Record<string, unknown>;
  last_sync_at?: string;
  sync_status?: 'syncing' | 'synced' | 'error';
  created_at: string;
  updated_at: string;
}

export interface IntegrationConnectionCreate {
  integration_id: string;
  config?: Record<string, unknown>;
}

// ============ SSE Events ============

export interface SSEEvent<T = unknown> {
  type: string;
  data: T;
  timestamp: string;
}

export interface CallStartedEvent {
  call_id: string;
  from_number: string;
  to_number: string;
  agent_id?: string;
  direction: CallDirection;
}

export interface CallEndedEvent {
  call_id: string;
  duration_seconds: number;
  status: CallStatus;
  outcome?: string;
}

export interface TranscriptEvent {
  call_id: string;
  segment: CallTranscriptSegment;
}

export interface MetricUpdatedEvent {
  metric: string;
  value: unknown;
}

// ============ API Keys ============

export type APIKeyScope = 'calls' | 'agents' | 'leads' | 'campaigns' | 'reports' | 'admin';

export interface APIKey {
  id: string;
  name: string;
  prefix: string;  // First 8 characters of the key for identification
  scopes: APIKeyScope[];
  rate_limit: number;  // Requests per minute
  last_used_at?: string;
  expires_at?: string;
  created_at: string;
  is_active: boolean;
}

export interface APIKeyCreate {
  name: string;
  scopes: APIKeyScope[];
  rate_limit?: number;
  expires_in_days?: number;
}

export interface APIKeyCreated extends APIKey {
  key: string;  // Full key, only shown once at creation
}

