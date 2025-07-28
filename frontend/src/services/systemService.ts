import { apiClient } from './apiClient'
import type { SystemMetrics, HealthStatus, DetailedStatus } from '../types/system'

export const systemService = {
  /**
   * Get basic health status
   */
  async getHealth(): Promise<HealthStatus> {
    return apiClient.get<HealthStatus>('/health')
  },

  /**
   * Get comprehensive system metrics
   */
  async getMetrics(): Promise<SystemMetrics> {
    return apiClient.get<SystemMetrics>('/health/metrics')
  },

  /**
   * Get detailed system status
   */
  async getStatus(): Promise<DetailedStatus> {
    return apiClient.get<DetailedStatus>('/health/status')
  },
}