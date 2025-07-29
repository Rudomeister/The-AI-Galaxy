import { apiClient } from './apiClient'
import type { Agent, AgentListResponse, AgentCommand, AgentCommandResponse } from '../types/agent'

export const agentService = {
  /**
   * Get list of all agents
   */
  async getAgents(statusFilter?: string, agentTypeFilter?: string): Promise<AgentListResponse> {
    const params = new URLSearchParams()
    if (statusFilter) params.append('status_filter', statusFilter)
    if (agentTypeFilter) params.append('agent_type_filter', agentTypeFilter)
    
    const url = `/agents${params.toString() ? '?' + params.toString() : ''}`
    return apiClient.get<AgentListResponse>(url)
  },

  /**
   * Get detailed information about a specific agent
   */
  async getAgent(agentName: string): Promise<Agent> {
    return apiClient.get<Agent>(`/agents/${agentName}`)
  },

  /**
   * Send a command to an agent
   */
  async sendCommand(agentName: string, command: AgentCommand): Promise<AgentCommandResponse> {
    return apiClient.post<AgentCommandResponse>(`/agents/${agentName}/command`, command)
  },

  /**
   * Get recent tasks for an agent
   */
  async getAgentTasks(agentName: string, statusFilter?: string, limit: number = 50): Promise<any[]> {
    const params = new URLSearchParams()
    if (statusFilter) params.append('status_filter', statusFilter)
    params.append('limit', limit.toString())
    
    const url = `/agents/${agentName}/tasks?${params.toString()}`
    return apiClient.get<any[]>(url)
  },

  /**
   * Restart an agent
   */
  async restartAgent(agentName: string): Promise<AgentCommandResponse> {
    return apiClient.post<AgentCommandResponse>(`/agents/${agentName}/restart`)
  },
}