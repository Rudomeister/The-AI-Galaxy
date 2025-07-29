import React from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Box,
  Typography,
  Skeleton,
  IconButton,
  Tooltip,
} from '@mui/material'
import { Refresh as RefreshIcon } from '@mui/icons-material'

export interface Column<T> {
  id: keyof T
  label: string
  minWidth?: number
  align?: 'right' | 'left' | 'center'
  format?: (value: any) => React.ReactNode
}

interface DataTableProps<T> {
  columns: Column<T>[]
  rows: T[]
  loading?: boolean
  pagination?: {
    page: number
    pageSize: number
    totalCount: number
    onPageChange: (page: number) => void
    onPageSizeChange: (pageSize: number) => void
  }
  onRefresh?: () => void
  emptyMessage?: string
  title?: string
}

function DataTable<T extends Record<string, any>>({
  columns,
  rows,
  loading = false,
  pagination,
  onRefresh,
  emptyMessage = 'No data available',
  title,
}: DataTableProps<T>) {
  const handlePageChange = (event: unknown, newPage: number) => {
    pagination?.onPageChange(newPage)
  }

  const handlePageSizeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    pagination?.onPageSizeChange(parseInt(event.target.value, 10))
    pagination?.onPageChange(0)
  }

  const renderSkeletonRows = () => {
    return Array.from({ length: pagination?.pageSize || 5 }).map((_, index) => (
      <TableRow key={index}>
        {columns.map((column) => (
          <TableCell key={String(column.id)}>
            <Skeleton variant="text" />
          </TableCell>
        ))}
      </TableRow>
    ))
  }

  const renderEmptyState = () => (
    <TableRow>
      <TableCell colSpan={columns.length} align="center" sx={{ py: 4 }}>
        <Typography variant="body2" color="text.secondary">
          {emptyMessage}
        </Typography>
      </TableCell>
    </TableRow>
  )

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      {(title || onRefresh) && (
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          p: 2,
          borderBottom: '1px solid',
          borderColor: 'divider'
        }}>
          {title && (
            <Typography variant="h6" component="div">
              {title}
            </Typography>
          )}
          {onRefresh && (
            <Tooltip title="Refresh">
              <IconButton onClick={onRefresh} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      )}
      
      <TableContainer sx={{ maxHeight: 440 }}>
        <Table stickyHeader aria-label="data table">
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell
                  key={String(column.id)}
                  align={column.align}
                  style={{ minWidth: column.minWidth }}
                  sx={{ 
                    backgroundColor: 'background.paper',
                    fontWeight: 'bold'
                  }}
                >
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          
          <TableBody>
            {loading ? (
              renderSkeletonRows()
            ) : rows.length === 0 ? (
              renderEmptyState()
            ) : (
              rows.map((row, index) => (
                <TableRow hover role="checkbox" tabIndex={-1} key={index}>
                  {columns.map((column) => {
                    const value = row[column.id]
                    return (
                      <TableCell key={String(column.id)} align={column.align}>
                        {column.format ? column.format(value) : value}
                      </TableCell>
                    )
                  })}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
      
      {pagination && (
        <TablePagination
          rowsPerPageOptions={[5, 10, 25, 50]}
          component="div"
          count={pagination.totalCount}
          rowsPerPage={pagination.pageSize}
          page={pagination.page}
          onPageChange={handlePageChange}
          onRowsPerPageChange={handlePageSizeChange}
        />
      )}
    </Paper>
  )
}

export default DataTable