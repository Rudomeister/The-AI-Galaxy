import React from 'react'
import { Chip, ChipProps } from '@mui/material'
import {
  CheckCircle as ActiveIcon,
  Cancel as InactiveIcon,
  HourglassEmpty as BusyIcon,
  Error as ErrorIcon,
  CloudOff as OfflineIcon,
  CheckCircle,
  HourglassEmpty,
} from '@mui/icons-material'

export type StatusType = 'active' | 'inactive' | 'busy' | 'error' | 'offline' | 'pending' | 'completed' | 'failed' | 'created' | 'validated' | 'council_review' | 'approved' | 'in_development' | 'archived' | 'rejected'

interface StatusChipProps extends Omit<ChipProps, 'color'> {
  status: StatusType
  showIcon?: boolean
}

const statusConfig = {
  active: {
    color: 'success' as const,
    icon: ActiveIcon,
    label: 'Active',
  },
  inactive: {
    color: 'default' as const,
    icon: InactiveIcon,
    label: 'Inactive',
  },
  busy: {
    color: 'warning' as const,
    icon: BusyIcon,
    label: 'Busy',
  },
  error: {
    color: 'error' as const,
    icon: ErrorIcon,
    label: 'Error',
  },
  offline: {
    color: 'error' as const,
    icon: OfflineIcon,
    label: 'Offline',
  },
  pending: {
    color: 'info' as const,
    icon: HourglassEmpty,
    label: 'Pending',
  },
  completed: {
    color: 'success' as const,
    icon: CheckCircle,
    label: 'Completed',
  },
  failed: {
    color: 'error' as const,
    icon: ErrorIcon,
    label: 'Failed',
  },
  created: {
    color: 'info' as const,
    icon: HourglassEmpty,
    label: 'Created',
  },
  validated: {
    color: 'info' as const,
    icon: CheckCircle,
    label: 'Validated',
  },
  council_review: {
    color: 'warning' as const,
    icon: BusyIcon,
    label: 'Council Review',
  },
  approved: {
    color: 'success' as const,
    icon: CheckCircle,
    label: 'Approved',
  },
  in_development: {
    color: 'warning' as const,
    icon: BusyIcon,
    label: 'In Development',
  },
  archived: {
    color: 'default' as const,
    icon: InactiveIcon,
    label: 'Archived',
  },
  rejected: {
    color: 'error' as const,
    icon: ErrorIcon,
    label: 'Rejected',
  },
}

const StatusChip: React.FC<StatusChipProps> = ({ 
  status, 
  showIcon = true, 
  label,
  ...props 
}) => {
  const config = statusConfig[status]
  const IconComponent = config.icon
  
  return (
    <Chip
      {...props}
      color={config.color}
      label={label || config.label}
      icon={showIcon ? <IconComponent /> : null}
      variant="outlined"
      size="small"
    />
  )
}

export default StatusChip