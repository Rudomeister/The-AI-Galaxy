import React from 'react'
import {
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Skeleton,
} from '@mui/material'
import { SvgIconComponent } from '@mui/icons-material'

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon?: SvgIconComponent
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info'
  loading?: boolean
  trend?: {
    value: number
    period: string
  }
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: IconComponent,
  color = 'primary',
  loading = false,
  trend,
}) => {
  if (loading) {
    return (
      <Card sx={{ height: '100%', minHeight: 120 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Skeleton variant="circular" width={40} height={40} />
            <Box sx={{ ml: 2, flexGrow: 1 }}>
              <Skeleton variant="text" width="60%" />
            </Box>
          </Box>
          <Skeleton variant="text" width="40%" height={32} />
          <Skeleton variant="text" width="80%" />
        </CardContent>
      </Card>
    )
  }

  const getTrendColor = (trendValue: number) => {
    if (trendValue > 0) return 'success.main'
    if (trendValue < 0) return 'error.main'
    return 'text.secondary'
  }

  const getTrendIcon = (trendValue: number) => {
    if (trendValue > 0) return '↗'
    if (trendValue < 0) return '↘'
    return '→'
  }

  return (
    <Card sx={{ height: '100%', minHeight: 120 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {IconComponent && (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 40,
                height: 40,
                borderRadius: 1,
                backgroundColor: `${color}.main`,
                color: `${color}.contrastText`,
                mr: 2,
              }}
            >
              <IconComponent fontSize="small" />
            </Box>
          )}
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            {title}
          </Typography>
        </Box>
        
        <Typography
          variant="h4"
          component="div"
          sx={{
            color: `${color}.main`,
            fontWeight: 'bold',
            mb: 1,
          }}
        >
          {value}
        </Typography>
        
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}
        
        {trend && (
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
            <Typography
              variant="body2"
              sx={{
                color: getTrendColor(trend.value),
                fontWeight: 'medium',
              }}
            >
              {getTrendIcon(trend.value)} {Math.abs(trend.value)}%
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
              vs {trend.period}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  )
}

export default MetricCard