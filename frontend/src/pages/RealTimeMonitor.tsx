import React, { useState } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Grid,
  Paper,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Alert,
  Badge,
} from '@mui/material'
import {
  Circle as DotIcon,
  SmartToy as AgentIcon,
  Psychology as IdeaIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as SuccessIcon,
  Clear as ClearIcon,
} from '@mui/icons-material'

import useWebSocket, { WebSocketMessage } from '../hooks/useWebSocket'

interface ActivityItem {
  id: string
  timestamp: string
  type: 'agent' | 'idea' | 'system' | 'error'
  severity: 'info' | 'warning' | 'error' | 'success'
  title: string
  description: string
  data?: any
}

const RealTimeMonitor: React.FC = () => {
  const [activities, setActivities] = useState<ActivityItem[]>([])
  const [isPaused, setIsPaused] = useState(false)
  const [filters, setFilters] = useState({
    agent: true,
    idea: true,
    system: true,
    error: true,
  })
  const [maxItems] = useState(100)

  // WebSocket connection
  const {
    isConnected,
    connectionError,
    reconnectCount,
  } = useWebSocket('/ws/monitor', {
    onMessage: handleWebSocketMessage,
  })

  function handleWebSocketMessage(message: WebSocketMessage) {
    if (isPaused) return

    const activity: ActivityItem = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: message.timestamp || new Date().toISOString(),
      type: getActivityType(message.type),
      severity: getActivitySeverity(message.type),
      title: getActivityTitle(message.type),
      description: getActivityDescription(message.type, message.data),
      data: message.data,
    }

    setActivities(prev => {
      const newActivities = [activity, ...prev]
      return newActivities.slice(0, maxItems)
    })
  }

  const getActivityType = (messageType: string): ActivityItem['type'] => {
    if (messageType.includes('agent')) return 'agent'
    if (messageType.includes('idea') || messageType.includes('workflow')) return 'idea'
    if (messageType.includes('error')) return 'error'
    return 'system'
  }

  const getActivitySeverity = (messageType: string): ActivityItem['severity'] => {
    if (messageType.includes('error') || messageType.includes('failed')) return 'error'
    if (messageType.includes('warning')) return 'warning'
    if (messageType.includes('completed') || messageType.includes('success')) return 'success'
    return 'info'
  }

  const getActivityTitle = (messageType: string): string => {
    switch (messageType) {
      case 'agent_status_update':
        return `Agent ${data.agent_name} status changed`
      case 'agent_heartbeat':
        return `Agent ${data.agent_name} heartbeat`
      case 'idea_status_update':
        return `Idea "${data.title}" status updated`
      case 'workflow_transition':
        return `Workflow transition: ${data.from_state} → ${data.to_state}`
      case 'system_metrics_update':
        return 'System metrics updated'
      case 'error':
        return `Error: ${data.message || 'Unknown error'}`
      default:
        return messageType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
  }

  const getActivityDescription = (messageType: string, data: any): string => {
    switch (messageType) {
      case 'agent_status_update':
        return `Status: ${data.status}, Tasks: ${data.total_tasks_completed || 0}`
      case 'agent_heartbeat':
        return `Last seen: ${new Date(data.timestamp).toLocaleTimeString()}`
      case 'idea_status_update':
        return `New status: ${data.status}, Priority: ${data.priority}`
      case 'workflow_transition':
        return `Agent: ${data.agent_name}, Transition ID: ${data.transition_id}`
      case 'system_metrics_update':
        return `CPU: ${data.cpu_usage}%, Memory: ${data.memory_usage}%`
      default:
        return JSON.stringify(data).slice(0, 100) + '...'
    }
  }

  const getActivityIcon = (type: ActivityItem['type'], severity: ActivityItem['severity']) => {
    if (severity === 'error') return <ErrorIcon color="error" />
    if (severity === 'warning') return <WarningIcon color="warning" />
    if (severity === 'success') return <SuccessIcon color="success" />
    
    switch (type) {
      case 'agent':
        return <AgentIcon color="primary" />
      case 'idea':
        return <IdeaIcon color="secondary" />
      default:
        return <InfoIcon color="info" />
    }
  }

  const filteredActivities = activities.filter(activity => filters[activity.type])

  const clearActivities = () => {
    setActivities([])
  }

  const toggleFilter = (filterType: keyof typeof filters) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: !prev[filterType]
    }))
  }

  // Connection status indicator
  const getConnectionStatus = () => {
    if (connectionError) return { color: 'error', text: 'Disconnected' }
    if (reconnectCount > 0) return { color: 'warning', text: `Reconnecting... (${reconnectCount})` }
    if (isConnected) return { color: 'success', text: 'Connected' }
    return { color: 'info', text: 'Connecting...' }
  }

  const connectionStatus = getConnectionStatus()

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Real-time Monitor
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            icon={<DotIcon />}
            label={connectionStatus.text}
            color={connectionStatus.color as any}
            size="small"
          />
          <FormControlLabel
            control={
              <Switch
                checked={!isPaused}
                onChange={(e) => setIsPaused(!e.target.checked)}
              />
            }
            label="Live Updates"
          />
        </Box>
      </Box>

      {connectionError && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {connectionError}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Filters and Controls */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Activity Filters
                </Typography>
                <Tooltip title="Clear all activities">
                  <IconButton onClick={clearActivities} size="small">
                    <ClearIcon />
                  </IconButton>
                </Tooltip>
              </Box>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={filters.agent}
                      onChange={() => toggleFilter('agent')}
                      size="small"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <AgentIcon fontSize="small" />
                      <span>Agent Events</span>
                      <Badge 
                        badgeContent={activities.filter(a => a.type === 'agent').length} 
                        color="primary" 
                        max={99}
                      />
                    </Box>
                  }
                />
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={filters.idea}
                      onChange={() => toggleFilter('idea')}
                      size="small"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <IdeaIcon fontSize="small" />
                      <span>Idea Events</span>
                      <Badge 
                        badgeContent={activities.filter(a => a.type === 'idea').length} 
                        color="secondary" 
                        max={99}
                      />
                    </Box>
                  }
                />
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={filters.system}
                      onChange={() => toggleFilter('system')}
                      size="small"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <InfoIcon fontSize="small" />
                      <span>System Events</span>
                      <Badge 
                        badgeContent={activities.filter(a => a.type === 'system').length} 
                        color="info" 
                        max={99}
                      />
                    </Box>
                  }
                />
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={filters.error}
                      onChange={() => toggleFilter('error')}
                      size="small"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <ErrorIcon fontSize="small" />
                      <span>Error Events</span>
                      <Badge 
                        badgeContent={activities.filter(a => a.type === 'error').length} 
                        color="error" 
                        max={99}
                      />
                    </Box>
                  }
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Activity Feed */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Live Activity Feed
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {filteredActivities.length} events
                </Typography>
              </Box>
              
              <Paper sx={{ maxHeight: 600, overflow: 'auto' }}>
                <List dense>
                  {filteredActivities.length === 0 ? (
                    <ListItem>
                      <ListItemText
                        primary="No activities to display"
                        secondary="Activities will appear here when the system is active"
                      />
                    </ListItem>
                  ) : (
                    filteredActivities.map((activity) => (
                      <ListItem
                        key={activity.id}
                        sx={{
                          borderLeft: `4px solid`,
                          borderLeftColor: 
                            activity.severity === 'error' ? 'error.main' :
                            activity.severity === 'warning' ? 'warning.main' :
                            activity.severity === 'success' ? 'success.main' :
                            'info.main',
                          mb: 1,
                          backgroundColor: 'background.paper',
                        }}
                      >
                        <ListItemIcon>
                          {getActivityIcon(activity.type, activity.severity)}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body2" fontWeight="medium">
                                {activity.title}
                              </Typography>
                              <Chip 
                                label={activity.type} 
                                size="small" 
                                variant="outlined"
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                {activity.description}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {new Date(activity.timestamp).toLocaleString()}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                    ))
                  )}
                </List>
              </Paper>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default RealTimeMonitor