import React, { useState, useEffect } from 'react'
import {
  Grid,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
} from '@mui/material'
import {
  SmartToy as AgentIcon,
  Speed as PerformanceIcon,
  Assignment as TaskIcon,
  Error as ErrorIcon,
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

import MetricCard from '../components/ui/MetricCard'
import DataTable, { Column } from '../components/ui/DataTable'
import StatusChip from '../components/ui/StatusChip'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import { Agent, AgentCommand } from '../types/agent'
import { agentService } from '../services/agentService'

const AgentOverview: React.FC = () => {
  const [commandDialogOpen, setCommandDialogOpen] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState<string>('')
  const [command, setCommand] = useState('')
  const [commandParams, setCommandParams] = useState('')
  const queryClient = useQueryClient()

  // Fetch agents data
  const { data: agentsData, isLoading, error, refetch } = useQuery({
    queryKey: ['agents'],
    queryFn: agentService.getAgents,
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  // Command mutation
  const commandMutation = useMutation({
    mutationFn: ({ agentName, command }: { agentName: string; command: AgentCommand }) =>
      agentService.sendCommand(agentName, command),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['agents'] })
      setCommandDialogOpen(false)
      setCommand('')
      setCommandParams('')
      setSelectedAgent('')
    },
  })

  const handleSendCommand = () => {
    if (!selectedAgent || !command) return

    let parsedParams = {}
    if (commandParams) {
      try {
        parsedParams = JSON.parse(commandParams)
      } catch (e) {
        console.error('Invalid JSON parameters:', e)
        return
      }
    }

    commandMutation.mutate({
      agentName: selectedAgent,
      command: {
        command,
        parameters: parsedParams,
        priority: 'normal',
      },
    })
  }

  // Calculate metrics
  const agents = agentsData?.agents || []
  const activeAgents = agents.filter(agent => agent.status === 'active').length
  const totalTasks = agents.reduce((sum, agent) => sum + agent.total_tasks_completed, 0)
  const avgResponseTime = agents.length > 0 
    ? agents.reduce((sum, agent) => sum + agent.average_task_time, 0) / agents.length
    : 0
  const errorCount = agents.reduce((sum, agent) => sum + agent.error_count, 0)

  // Table columns
  const columns: Column<Agent>[] = [
    {
      id: 'name',
      label: 'Agent Name',
      minWidth: 150,
      format: (value: string) => (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AgentIcon color="primary" />
          <Typography variant="body2" fontWeight="medium">
            {value}
          </Typography>
        </Box>
      ),
    },
    {
      id: 'agent_type',
      label: 'Type',
      minWidth: 120,
    },
    {
      id: 'status',
      label: 'Status',
      minWidth: 100,
      format: (value: Agent['status']) => <StatusChip status={value} />,
    },
    {
      id: 'capabilities',
      label: 'Capabilities',
      minWidth: 200,
      format: (value: string[]) => (
        <Box>
          {value.slice(0, 2).map((capability, index) => (
            <Typography key={index} variant="caption" display="block">
              {capability}
            </Typography>
          ))}
          {value.length > 2 && (
            <Typography variant="caption" color="text.secondary">
              +{value.length - 2} more
            </Typography>
          )}
        </Box>
      ),
    },
    {
      id: 'total_tasks_completed',
      label: 'Tasks Completed',
      align: 'right',
      minWidth: 120,
    },
    {
      id: 'average_task_time',
      label: 'Avg Time (s)',
      align: 'right',
      minWidth: 100,
      format: (value: number) => value.toFixed(2),
    },
    {
      id: 'error_count',
      label: 'Errors',
      align: 'right',
      minWidth: 80,
      format: (value: number) => (
        <Typography
          variant="body2"
          color={value > 0 ? 'error.main' : 'text.primary'}
        >
          {value}
        </Typography>
      ),
    },
    {
      id: 'last_heartbeat',
      label: 'Last Heartbeat',
      minWidth: 120,
      format: (value: string) =>
        value ? new Date(value).toLocaleTimeString() : 'Never',
    },
  ]

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          Failed to load agents data. Please try again.
        </Alert>
      </Box>
    )
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Agent Overview
      </Typography>

      {/* Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Agents"
            value={activeAgents}
            subtitle={`${agents.length} total agents`}
            icon={AgentIcon}
            color="success"
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Tasks"
            value={totalTasks}
            subtitle="All time completed"
            icon={TaskIcon}
            color="primary"
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg Response"
            value={`${avgResponseTime.toFixed(1)}s`}
            subtitle="Task completion time"
            icon={PerformanceIcon}
            color="info"
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Error Count"
            value={errorCount}
            subtitle="System-wide errors"
            icon={ErrorIcon}
            color="error"
            loading={isLoading}
          />
        </Grid>
      </Grid>

      {/* Agent Actions */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
        <Button
          variant="contained"
          onClick={() => setCommandDialogOpen(true)}
          disabled={agents.length === 0}
        >
          Send Command
        </Button>
        <Button variant="outlined" onClick={() => refetch()}>
          Refresh
        </Button>
      </Box>

      {/* Agents Table */}
      <DataTable
        columns={columns}
        rows={agents}
        loading={isLoading}
        title="Active Agents"
        onRefresh={refetch}
        emptyMessage="No agents are currently registered"
      />

      {/* Command Dialog */}
      <Dialog
        open={commandDialogOpen}
        onClose={() => setCommandDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Send Agent Command</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <FormControl fullWidth>
              <InputLabel>Select Agent</InputLabel>
              <Select
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value)}
                label="Select Agent"
              >
                {agents
                  .filter(agent => agent.status === 'active')
                  .map(agent => (
                    <MenuItem key={agent.name} value={agent.name}>
                      {agent.name} ({agent.agent_type})
                    </MenuItem>
                  ))}
              </Select>
            </FormControl>

            <TextField
              label="Command"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="e.g., process_idea, analyze_data"
              fullWidth
            />

            <TextField
              label="Parameters (JSON)"
              value={commandParams}
              onChange={(e) => setCommandParams(e.target.value)}
              placeholder='{"key": "value"}'
              multiline
              rows={3}
              fullWidth
            />

            {commandMutation.error && (
              <Alert severity="error">
                Failed to send command. Please try again.
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCommandDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSendCommand}
            variant="contained"
            disabled={!selectedAgent || !command || commandMutation.isPending}
          >
            {commandMutation.isPending ? 'Sending...' : 'Send Command'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default AgentOverview