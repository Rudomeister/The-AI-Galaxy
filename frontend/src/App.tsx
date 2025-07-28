import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'

import DashboardLayout from './components/layout/DashboardLayout'
import AgentOverview from './pages/AgentOverview'
import SystemMetrics from './pages/SystemMetrics'
import RealTimeMonitor from './pages/RealTimeMonitor'
import IdeaManagement from './pages/IdeaManagement'

function App() {
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <DashboardLayout>
        <Routes>
          <Route path="/" element={<AgentOverview />} />
          <Route path="/agents" element={<AgentOverview />} />
          <Route path="/metrics" element={<SystemMetrics />} />
          <Route path="/monitor" element={<RealTimeMonitor />} />
          <Route path="/ideas" element={<IdeaManagement />} />
        </Routes>
      </DashboardLayout>
    </Box>
  )
}

export default App