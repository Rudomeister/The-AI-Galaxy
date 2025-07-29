export interface WebSocketMessage {
  type: string
  timestamp: string
  data?: any
}

export interface AgentEvent extends WebSocketMessage {
  type: 'agent_event' | 'agent_status' | 'agent_detail'
  channel?: string
  agent_name?: string
}

export interface SystemEvent extends WebSocketMessage {
  type: 'system_metrics_update' | 'system_metrics'
}

export interface WorkflowEvent extends WebSocketMessage {
  type: 'workflow_event' | 'workflow_subscription'
  idea_id?: string
}

export interface WebSocketStats {
  connection_stats: {
    total_connections: number
    connections_by_channel: Record<string, number>
  }
  timestamp: string
}