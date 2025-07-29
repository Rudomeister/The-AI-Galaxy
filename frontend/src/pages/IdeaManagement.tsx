import React, { useState } from 'react'
import {
  Box,
  Typography,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
} from '@mui/material'
import {
  Add as AddIcon,
  Psychology as IdeaIcon,
  TrendingUp as TrendIcon,
  Assignment as TaskIcon,
  CheckCircle as CompletedIcon,
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

import MetricCard from '../components/ui/MetricCard'
import DataTable, { Column } from '../components/ui/DataTable'
import StatusChip from '../components/ui/StatusChip'
import { Idea, IdeaCreateRequest } from '../types/idea'
import { ideaService } from '../services/ideaService'

const workflowSteps = [
  'Created',
  'Validated',
  'Council Review',
  'Approved',
  'In Development',
  'Completed'
]

const IdeaManagement: React.FC = () => {
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [selectedIdea, setSelectedIdea] = useState<Idea | null>(null)
  const [detailDialogOpen, setDetailDialogOpen] = useState(false)
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(10)
  const queryClient = useQueryClient()

  // Form state
  const [newIdea, setNewIdea] = useState<IdeaCreateRequest>({
    title: '',
    description: '',
    priority: 'medium',
    tags: [],
    metadata: {},
  })
  const [tagInput, setTagInput] = useState('')

  // Fetch ideas
  const { data: ideasData, isLoading, error, refetch } = useQuery({
    queryKey: ['ideas', page, pageSize],
    queryFn: () => ideaService.getIdeas({ page, page_size: pageSize }),
    refetchInterval: 10000, // Refresh every 10 seconds
  })

  // Create idea mutation
  const createMutation = useMutation({
    mutationFn: ideaService.createIdea,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ideas'] })
      setCreateDialogOpen(false)
      resetForm()
    },
  })

  const resetForm = () => {
    setNewIdea({
      title: '',
      description: '',
      priority: 'medium',
      tags: [],
      metadata: {},
    })
    setTagInput('')
  }

  const handleCreateIdea = () => {
    createMutation.mutate(newIdea)
  }

  const handleAddTag = () => {
    if (tagInput.trim() && !newIdea.tags?.includes(tagInput.trim())) {
      setNewIdea(prev => ({
        ...prev,
        tags: [...(prev.tags || []), tagInput.trim()]
      }))
      setTagInput('')
    }
  }

  const handleRemoveTag = (tagToRemove: string) => {
    setNewIdea(prev => ({
      ...prev,
      tags: prev.tags?.filter(tag => tag !== tagToRemove) || []
    }))
  }

  const handleViewDetails = (idea: Idea) => {
    setSelectedIdea(idea)
    setDetailDialogOpen(true)
  }

  const getWorkflowStepIndex = (status: Idea['status']) => {
    const statusMap: Record<string, number> = {
      'created': 0,
      'validated': 1,
      'council_review': 2,
      'approved': 3,
      'in_development': 4,
      'coding_in_progress': 4,
      'completed': 5,
      'archived': 5,
      'rejected': -1,
    }
    return statusMap[status] ?? 0
  }

  // Calculate metrics
  const ideas = ideasData?.ideas || []
  const totalIdeas = ideasData?.total_count || 0
  const completedIdeas = ideas.filter(idea => idea.status === 'completed').length
  const inProgressIdeas = ideas.filter(idea => 
    ['validated', 'council_review', 'approved', 'in_development', 'coding_in_progress'].includes(idea.status)
  ).length
  const rejectedIdeas = ideas.filter(idea => idea.status === 'rejected').length

  // Table columns
  const columns: Column<Idea>[] = [
    {
      id: 'title',
      label: 'Title',
      minWidth: 200,
      format: (value: string, row: Idea) => (
        <Box>
          <Typography 
            variant="body2" 
            fontWeight="medium"
            sx={{ cursor: 'pointer', color: 'primary.main' }}
            onClick={() => handleViewDetails(row)}
          >
            {value}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {row.id}
          </Typography>
        </Box>
      ),
    },
    {
      id: 'status',
      label: 'Status',
      minWidth: 120,
      format: (value: Idea['status']) => <StatusChip status={value} />,
    },
    {
      id: 'priority',
      label: 'Priority',
      minWidth: 100,
      format: (value: Idea['priority']) => (
        <Chip
          label={value}
          color={
            value === 'critical' ? 'error' :
            value === 'high' ? 'warning' :
            value === 'medium' ? 'info' : 'default'
          }
          size="small"
        />
      ),
    },
    {
      id: 'tags',
      label: 'Tags',
      minWidth: 150,
      format: (value: string[]) => (
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
          {value.slice(0, 2).map((tag, index) => (
            <Chip key={index} label={tag} size="small" variant="outlined" />
          ))}
          {value.length > 2 && (
            <Chip label={`+${value.length - 2}`} size="small" />
          )}
        </Box>
      ),
    },
    {
      id: 'created_at',
      label: 'Created',
      minWidth: 120,
      format: (value: string) => new Date(value).toLocaleDateString(),
    },
    {
      id: 'updated_at',
      label: 'Updated',
      minWidth: 120,
      format: (value: string) => new Date(value).toLocaleDateString(),
    },
  ]

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Idea Management
      </Typography>

      {/* Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Ideas"
            value={totalIdeas}
            subtitle="All submitted ideas"
            icon={IdeaIcon}
            color="primary"
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="In Progress"
            value={inProgressIdeas}
            subtitle="Being developed"
            icon={TrendIcon}
            color="info"
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Completed"
            value={completedIdeas}
            subtitle="Successfully delivered"
            icon={CompletedIcon}
            color="success"
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Rejected"
            value={rejectedIdeas}
            subtitle="Not viable"
            icon={TaskIcon}
            color="error"
            loading={isLoading}
          />
        </Grid>
      </Grid>

      {/* Actions */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Submit New Idea
        </Button>
        <Button variant="outlined" onClick={() => refetch()}>
          Refresh
        </Button>
      </Box>

      {/* Ideas Table */}
      <DataTable
        columns={columns}
        rows={ideas}
        loading={isLoading}
        title="Ideas"
        onRefresh={refetch}
        pagination={{
          page,
          pageSize,
          totalCount: totalIdeas,
          onPageChange: setPage,
          onPageSizeChange: setPageSize,
        }}
        emptyMessage="No ideas have been submitted yet"
      />

      {/* Create Idea Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Submit New Idea</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Title"
              value={newIdea.title}
              onChange={(e) => setNewIdea(prev => ({ ...prev, title: e.target.value }))}
              fullWidth
              required
            />

            <TextField
              label="Description"
              value={newIdea.description}
              onChange={(e) => setNewIdea(prev => ({ ...prev, description: e.target.value }))}
              multiline
              rows={4}
              fullWidth
              required
            />

            <FormControl fullWidth>
              <InputLabel>Priority</InputLabel>
              <Select
                value={newIdea.priority}
                onChange={(e) => setNewIdea(prev => ({ 
                  ...prev, 
                  priority: e.target.value as IdeaCreateRequest['priority']
                }))}
                label="Priority"
              >
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
              </Select>
            </FormControl>

            <Box>
              <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                <TextField
                  label="Add Tag"
                  value={tagInput}
                  onChange={(e) => setTagInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
                  size="small"
                />
                <Button onClick={handleAddTag} variant="outlined">
                  Add
                </Button>
              </Box>
              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {newIdea.tags?.map((tag, index) => (
                  <Chip
                    key={index}
                    label={tag}
                    onDelete={() => handleRemoveTag(tag)}
                    size="small"
                  />
                ))}
              </Box>
            </Box>

            {createMutation.error && (
              <Alert severity="error">
                Failed to create idea. Please try again.
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleCreateIdea}
            variant="contained"
            disabled={!newIdea.title || !newIdea.description || createMutation.isPending}
          >
            {createMutation.isPending ? 'Creating...' : 'Submit Idea'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Idea Details Dialog */}
      <Dialog
        open={detailDialogOpen}
        onClose={() => setDetailDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {selectedIdea?.title}
        </DialogTitle>
        <DialogContent>
          {selectedIdea && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="body1" paragraph>
                {selectedIdea.description}
              </Typography>

              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Workflow Progress
                </Typography>
                <Stepper activeStep={getWorkflowStepIndex(selectedIdea.status)} alternativeLabel>
                  {workflowSteps.map((label) => (
                    <Step key={label}>
                      <StepLabel>{label}</StepLabel>
                    </Step>
                  ))}
                </Stepper>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Status:</Typography>
                  <StatusChip status={selectedIdea.status} />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Priority:</Typography>
                  <Chip label={selectedIdea.priority} size="small" />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Created:</Typography>
                  <Typography variant="body2">
                    {new Date(selectedIdea.created_at).toLocaleString()}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Updated:</Typography>
                  <Typography variant="body2">
                    {new Date(selectedIdea.updated_at).toLocaleString()}
                  </Typography>
                </Grid>
              </Grid>

              {selectedIdea.tags.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Tags:</Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {selectedIdea.tags.map((tag, index) => (
                      <Chip key={index} label={tag} size="small" variant="outlined" />
                    ))}
                  </Box>
                </Box>
              )}

              {selectedIdea.processing_history.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Processing History
                  </Typography>
                  {selectedIdea.processing_history.map((entry, index) => (
                    <Card key={index} sx={{ mb: 1 }}>
                      <CardContent sx={{ py: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Box>
                            <Typography variant="body2" fontWeight="medium">
                              {entry.agent} - {entry.status}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {entry.message}
                            </Typography>
                          </Box>
                          <Typography variant="caption" color="text.secondary">
                            {new Date(entry.timestamp).toLocaleString()}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default IdeaManagement