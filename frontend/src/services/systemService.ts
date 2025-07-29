import { apiClient } from './apiClient'
import type { SystemMetrics, HealthStatus, DetailedStatus } from '../types/system'

export const systemService = {
  /**
   * Get basic health status
   */
  async getHealth(): Promise<HealthStatus> {
    return apiClient.get<HealthStatus>('http://localhost:8080/health/')
  },

  /**
   * Get comprehensive system metrics
   */
  async getMetrics(): Promise<SystemMetrics> {
    return apiClient.get<SystemMetrics>('/health/metrics')
  },

  /**
   * Get system metrics (alias for compatibility)
   */
  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.getMetrics()
  },

  /**
   * Get detailed system status
   */
  async getStatus(): Promise<DetailedStatus> {
    return apiClient.get<DetailedStatus>('/health/status')
  },
}