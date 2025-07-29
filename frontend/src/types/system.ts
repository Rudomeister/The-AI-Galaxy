export interface SystemMetrics {
  timestamp: string
  agent_metrics: {
    total_agents: number
    active_agents: number
    agent_details: Record<string, any>
  }
  system_metrics: {
    uptime_seconds: number
    active_tasks: number
    active_workflows: number
    total_tasks_processed: number
  }
  service_health: {
    orchestrator: boolean
    redis: boolean
    vector_search: boolean
  }
  redis_metrics: {
    is_healthy: boolean
    ping_time_ms: number
    connected_clients: number
    memory_used_mb: number
    total_keys: number
  }
  vector_search_metrics: {
    total_documents: number
    documents_by_type: Record<string, number>
    last_updated: string
    index_size_mb: number
  }
}

export interface HealthStatus {
  status: 'healthy' | 'warning' | 'critical'
  timestamp: string
  services: Record<string, boolean>
  uptime_seconds: number
}

export interface DetailedStatus {
  status: 'healthy' | 'warning' | 'critical'
  timestamp: string
  version: string
  uptime_seconds: number
  agent_summary: {
    total_count: number
    active_count: number
    agents: Record<string, any>
    task_performance: {
      active_tasks: number
      completed_tasks: number
      average_task_time: number
    }
  }
  service_details: {
    redis: {
      status: 'healthy' | 'unhealthy'
      ping_time_ms: number
      memory_used_mb: number
      connected_clients: number
      uptime_seconds: number
    }
    vector_search: {
      status: 'healthy' | 'unhealthy'
      total_documents: number
      collections: number
      last_updated: string
    }
  }
  performance_metrics: {
    system_load: number
    memory_usage: number
    cpu_usage: number
    request_rate: number
  }
}