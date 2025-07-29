import React from 'react'
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
} from '@mui/material'
import {
  Memory as MemoryIcon,
  Speed as CpuIcon,
  Storage as StorageIcon,
  NetworkCheck as NetworkIcon,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from 'recharts'

import MetricCard from '../components/ui/MetricCard'
import { systemService } from '../services/systemService'

// Mock data for demonstration - replace with actual data from systemService
const mockPerformanceData = [
  { time: '00:00', cpu: 45, memory: 62, tasks: 12 },
  { time: '01:00', cpu: 52, memory: 58, tasks: 15 },
  { time: '02:00', cpu: 38, memory: 65, tasks: 8 },
  { time: '03:00', cpu: 71, memory: 72, tasks: 22 },
  { time: '04:00', cpu: 56, memory: 68, tasks: 18 },
  { time: '05:00', cpu: 43, memory: 61, tasks: 11 },
  { time: '06:00', cpu: 67, memory: 75, tasks: 25 },
]

const mockAgentPerformance = [
  { name: 'Router Agent', tasks: 45, avgTime: 2.3, success: 98 },
  { name: 'Validator Agent', tasks: 38, avgTime: 5.1, success: 95 },
  { name: 'Council Agent', tasks: 12, avgTime: 15.2, success: 100 },
  { name: 'Creator Agent', tasks: 22, avgTime: 8.7, success: 92 },
  { name: 'Programmer Agent', tasks: 15, avgTime: 45.3, success: 88 },
]

const SystemMetrics: React.FC = () => {
  // Fetch system metrics
  const { isLoading, error } = useQuery({
    queryKey: ['system-metrics'],
    queryFn: systemService.getSystemMetrics,
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error">
          Failed to load system metrics. Using demo data.
        </Typography>
      </Box>
    )
  }

  // Use mock data for now, replace with actual metrics when available
  const systemHealth = {
    cpu_usage: 56,
    memory_usage: 68,
    disk_usage: 42,
    network_latency: 23,
    uptime_hours: 72.5,
    active_connections: 15,
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        System Metrics
      </Typography>

      {/* System Health Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="CPU Usage"
            value={`${systemHealth.cpu_usage}%`}
            icon={CpuIcon}
            color={systemHealth.cpu_usage > 80 ? 'error' : systemHealth.cpu_usage > 60 ? 'warning' : 'success'}
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Memory Usage"
            value={`${systemHealth.memory_usage}%`}
            icon={MemoryIcon}
            color={systemHealth.memory_usage > 80 ? 'error' : systemHealth.memory_usage > 60 ? 'warning' : 'success'}
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Disk Usage"
            value={`${systemHealth.disk_usage}%`}
            icon={StorageIcon}
            color={systemHealth.disk_usage > 80 ? 'error' : systemHealth.disk_usage > 60 ? 'warning' : 'success'}
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Network Latency"
            value={`${systemHealth.network_latency}ms`}
            icon={NetworkIcon}
            color={systemHealth.network_latency > 100 ? 'error' : systemHealth.network_latency > 50 ? 'warning' : 'success'}
            loading={isLoading}
          />
        </Grid>
      </Grid>

      {/* Performance Charts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Performance Over Time
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mockPerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="cpu" 
                      stroke="#8884d8" 
                      name="CPU %"
                      strokeWidth={2}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="memory" 
                      stroke="#82ca9d" 
                      name="Memory %"
                      strokeWidth={2}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="tasks" 
                      stroke="#ffc658" 
                      name="Active Tasks"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" gutterBottom>
                  CPU Usage ({systemHealth.cpu_usage}%)
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={systemHealth.cpu_usage} 
                  sx={{ mb: 2 }}
                  color={systemHealth.cpu_usage > 80 ? 'error' : systemHealth.cpu_usage > 60 ? 'warning' : 'success'}
                />
                
                <Typography variant="body2" gutterBottom>
                  Memory Usage ({systemHealth.memory_usage}%)
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={systemHealth.memory_usage} 
                  sx={{ mb: 2 }}
                  color={systemHealth.memory_usage > 80 ? 'error' : systemHealth.memory_usage > 60 ? 'warning' : 'success'}
                />
                
                <Typography variant="body2" gutterBottom>
                  Disk Usage ({systemHealth.disk_usage}%)
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={systemHealth.disk_usage}
                  color={systemHealth.disk_usage > 80 ? 'error' : systemHealth.disk_usage > 60 ? 'warning' : 'success'}
                />
              </Box>
              
              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" color="text.secondary">
                  System Uptime: {systemHealth.uptime_hours.toFixed(1)} hours
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Active Connections: {systemHealth.active_connections}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Agent Performance */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Agent Task Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={mockAgentPerformance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="tasks" fill="#8884d8" name="Tasks Completed" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Agent Response Times
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mockAgentPerformance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="avgTime" 
                      stroke="#82ca9d" 
                      fill="#82ca9d" 
                      name="Avg Time (s)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default SystemMetrics