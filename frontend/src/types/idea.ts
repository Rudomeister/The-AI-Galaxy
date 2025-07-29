export interface Idea {
  id: string
  title: string
  description: string
  status: 'created' | 'validated' | 'council_review' | 'approved' | 'in_development' | 'completed' | 'archived' | 'rejected'
  priority: 'low' | 'medium' | 'high' | 'critical'
  created_at: string
  updated_at: string
  tags: string[]
  metadata: Record<string, any>
  department_assignment?: string
  institution_assignment?: string
  processing_history: Array<{
    timestamp: string
    status: string
    agent: string
    message: string
  }>
}

export interface IdeaCreateRequest {
  title: string
  description: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  tags?: string[]
  metadata?: Record<string, any>
  department_hint?: string
}

export interface IdeaListResponse {
  ideas: Idea[]
  total_count: number
  page: number
  page_size: number
  has_next: boolean
}

export interface WorkflowTransition {
  target_state: string
  agent_name?: string
  parameters?: Record<string, any>
}

export interface WorkflowTransitionResponse {
  idea_id: string
  from_state: string
  to_state: string
  agent_name: string
  transition_id: string
  timestamp: string
  status: string
  message: string
}