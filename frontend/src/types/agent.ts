export interface Agent {
  name: string
  agent_type: string
  status: 'active' | 'inactive' | 'busy' | 'error' | 'offline'
  last_heartbeat?: string
  capabilities: string[]
  current_task?: string
  error_count: number
  total_tasks_completed: number
  average_task_time: number
  metadata: Record<string, any>
}

export interface AgentListResponse {
  agents: Agent[]
  total_count: number
  active_count: number
  inactive_count: number
  timestamp: string
}

export interface AgentCommand {
  command: string
  parameters?: Record<string, any>
  priority?: 'low' | 'normal' | 'high' | 'critical'
}

export interface AgentCommandResponse {
  command_id: string
  agent_name: string
  command: string
  status: string
  timestamp: string
  message: string
}