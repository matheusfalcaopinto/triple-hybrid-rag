/**
 * Establishments API service
 */

import { apiClient } from '../client';
import type {
  Establishment,
  EstablishmentCreate,
  EstablishmentUpdate,
  EstablishmentWithStats,
  PhoneNumber,
  PaginatedResponse,
  MessageResponse,
} from '../types';

export interface EstablishmentMember {
  id: string;
  user_id: string;
  user_email: string;
  user_name?: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  joined_at: string;
}

export const establishmentsApi = {
  /**
   * List all establishments for current user
   */
  list: async (): Promise<EstablishmentWithStats[]> => {
    return apiClient.get<EstablishmentWithStats[]>('/establishments');
  },

  /**
   * Get establishment by ID
   */
  get: async (id: string): Promise<EstablishmentWithStats> => {
    return apiClient.get<EstablishmentWithStats>(`/establishments/${id}`);
  },

  /**
   * Create new establishment
   */
  create: async (data: EstablishmentCreate): Promise<Establishment> => {
    return apiClient.post<Establishment>('/establishments', data);
  },

  /**
   * Update establishment
   */
  update: async (id: string, data: EstablishmentUpdate): Promise<Establishment> => {
    return apiClient.patch<Establishment>(`/establishments/${id}`, data);
  },

  /**
   * Delete establishment
   */
  delete: async (id: string): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(`/establishments/${id}`);
  },

  // ============ Members ============

  /**
   * List establishment members
   */
  listMembers: async (establishmentId: string): Promise<EstablishmentMember[]> => {
    return apiClient.get<EstablishmentMember[]>(
      `/establishments/${establishmentId}/members`
    );
  },

  /**
   * Invite member to establishment
   */
  inviteMember: async (
    establishmentId: string,
    email: string,
    role: 'admin' | 'member' | 'viewer'
  ): Promise<MessageResponse> => {
    return apiClient.post<MessageResponse>(
      `/establishments/${establishmentId}/members/invite`,
      { email, role }
    );
  },

  /**
   * Update member role
   */
  updateMemberRole: async (
    establishmentId: string,
    memberId: string,
    role: 'admin' | 'member' | 'viewer'
  ): Promise<EstablishmentMember> => {
    return apiClient.patch<EstablishmentMember>(
      `/establishments/${establishmentId}/members/${memberId}`,
      { role }
    );
  },

  /**
   * Remove member from establishment
   */
  removeMember: async (
    establishmentId: string,
    memberId: string
  ): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(
      `/establishments/${establishmentId}/members/${memberId}`
    );
  },

  // ============ Phone Numbers ============

  /**
   * List phone numbers for establishment
   */
  listPhoneNumbers: async (establishmentId: string): Promise<PhoneNumber[]> => {
    return apiClient.get<PhoneNumber[]>(
      `/establishments/${establishmentId}/phone-numbers`
    );
  },

  /**
   * Add phone number to establishment
   */
  addPhoneNumber: async (
    establishmentId: string,
    data: { e164: string; display_name?: string; routing_agent_id?: string }
  ): Promise<PhoneNumber> => {
    return apiClient.post<PhoneNumber>(
      `/establishments/${establishmentId}/phone-numbers`,
      data
    );
  },

  /**
   * Update phone number routing
   */
  updatePhoneNumber: async (
    establishmentId: string,
    phoneNumberId: string,
    data: { display_name?: string; routing_agent_id?: string; is_active?: boolean }
  ): Promise<PhoneNumber> => {
    return apiClient.patch<PhoneNumber>(
      `/establishments/${establishmentId}/phone-numbers/${phoneNumberId}`,
      data
    );
  },

  /**
   * Remove phone number
   */
  removePhoneNumber: async (
    establishmentId: string,
    phoneNumberId: string
  ): Promise<MessageResponse> => {
    return apiClient.delete<MessageResponse>(
      `/establishments/${establishmentId}/phone-numbers/${phoneNumberId}`
    );
  },
};
