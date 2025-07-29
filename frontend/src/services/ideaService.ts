import { apiClient } from './apiClient'
import type { 
  Idea, 
  IdeaCreateRequest, 
  IdeaListResponse, 
  WorkflowTransition, 
  WorkflowTransitionResponse 
} from '../types/idea'

export const ideaService = {
  /**
   * Create a new idea
   */
  async createIdea(idea: IdeaCreateRequest): Promise<Idea> {
    return apiClient.post<Idea>('/ideas/', idea)
  },

  /**
   * Get list of ideas with filtering and pagination
   */
  async getIdeas(params?: {
    page?: number
    page_size?: number
    status_filter?: string
    priority_filter?: string
    search?: string
  }): Promise<IdeaListResponse> {
    const searchParams = new URLSearchParams()
    if (params?.page !== undefined) searchParams.append('page', params.page.toString())
    if (params?.page_size !== undefined) searchParams.append('page_size', params.page_size.toString())
    if (params?.status_filter) searchParams.append('status_filter', params.status_filter)
    if (params?.priority_filter) searchParams.append('priority_filter', params.priority_filter)
    if (params?.search) searchParams.append('search', params.search)
    
    const url = `/v1/ideas/${searchParams.toString() ? '?' + searchParams.toString() : ''}`
    return apiClient.get<IdeaListResponse>(url)
  },

  /**
   * Get a specific idea by ID
   */
  async getIdea(ideaId: string): Promise<Idea> {
    return apiClient.get<Idea>(`/v1/ideas/${ideaId}`)
  },

  /**
   * Update an existing idea
   */
  async updateIdea(ideaId: string, updates: Partial<IdeaCreateRequest>): Promise<Idea> {
    return apiClient.put<Idea>(`/v1/ideas/${ideaId}`, updates)
  },

  /**
   * Delete an idea
   */
  async deleteIdea(ideaId: string): Promise<{ message: string }> {
    return apiClient.delete<{ message: string }>(`/v1/ideas/${ideaId}`)
  },

  /**
   * Trigger a workflow transition for an idea
   */
  async transitionWorkflow(ideaId: string, transition: WorkflowTransition): Promise<WorkflowTransitionResponse> {
    return apiClient.post<WorkflowTransitionResponse>(`/v1/ideas/${ideaId}/transition`, transition)
  },
}