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
    return apiClient.post<Idea>('/v1/ideas/', idea)
  },

  /**
   * Get list of ideas with filtering and pagination
   */
  async getIdeas(
    page: number = 1,
    pageSize: number = 20,
    statusFilter?: string,
    priorityFilter?: string,
    search?: string
  ): Promise<IdeaListResponse> {
    const params = new URLSearchParams()
    params.append('page', page.toString())
    params.append('page_size', pageSize.toString())
    if (statusFilter) params.append('status_filter', statusFilter)
    if (priorityFilter) params.append('priority_filter', priorityFilter)
    if (search) params.append('search', search)
    
    const url = `/v1/ideas/?${params.toString()}`
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