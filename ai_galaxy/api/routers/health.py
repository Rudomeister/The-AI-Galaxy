"""
Health and System Status API Routes.

This module provides endpoints for system health checks, metrics,
and status monitoring for the AI-Galaxy ecosystem.
"""

from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...orchestrator import SystemOrchestrator
from ...services.redis.db import RedisService
from ...services.vector_search.query_service import VectorSearchService
from ..dependencies import (
    get_orchestrator,
    get_redis_service, 
    get_vector_search_service,
    verify_service_health
)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    services: Dict[str, bool]
    uptime_seconds: float


class SystemMetricsResponse(BaseModel):
    """System metrics response model."""
    timestamp: str
    agent_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]
    service_health: Dict[str, bool]
    redis_metrics: Dict[str, Any]
    vector_search_metrics: Dict[str, Any]


class DetailedStatusResponse(BaseModel):
    """Detailed system status response model."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    agent_summary: Dict[str, Any]
    service_details: Dict[str, Any]
    performance_metrics: Dict[str, Any]


@router.get("/", response_model=HealthResponse)
async def get_health_status(
    service_health: Dict[str, bool] = Depends(verify_service_health),
    orchestrator: SystemOrchestrator = Depends(get_orchestrator)
):
    """
    Get basic health status of the AI-Galaxy system.
    
    Returns:
        Basic health information including service status and uptime
    """
    try:
        # Get system metrics for uptime
        metrics = await orchestrator.get_system_metrics()
        uptime_seconds = metrics.get('uptime_seconds', 0)
        
        # Determine overall status
        all_services_healthy = all(service_health.values())
        status = "healthy" if all_services_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            services=service_health,
            uptime_seconds=uptime_seconds
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get health status: {str(e)}"
        )


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    orchestrator: SystemOrchestrator = Depends(get_orchestrator),
    redis_service: RedisService = Depends(get_redis_service),
    vector_service: VectorSearchService = Depends(get_vector_search_service),
    service_health: Dict[str, bool] = Depends(verify_service_health)
):
    """
    Get comprehensive system metrics.
    
    Returns:
        Detailed metrics including agent performance, system resources,
        and service-specific metrics
    """
    try:
        # Get orchestrator metrics
        orchestrator_metrics = await orchestrator.get_system_metrics()
        
        # Get Redis metrics
        redis_health = await redis_service.get_health_status()
        
        # Get vector search metrics
        vector_stats = await vector_service.get_index_statistics()
        
        return SystemMetricsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_metrics={
                "total_agents": orchestrator_metrics.get('agent_count', 0),
                "active_agents": orchestrator_metrics.get('active_agents', 0),
                "agent_details": orchestrator_metrics.get('agents', {})
            },
            system_metrics={
                "uptime_seconds": orchestrator_metrics.get('uptime_seconds', 0),
                "active_tasks": orchestrator_metrics.get('active_tasks', 0),
                "active_workflows": orchestrator_metrics.get('active_workflows', 0),
                "total_tasks_processed": orchestrator_metrics.get('orchestrator_metrics', {}).get('total_tasks_processed', 0)
            },
            service_health=service_health,
            redis_metrics={
                "is_healthy": redis_health.get('is_healthy', False),
                "ping_time_ms": redis_health.get('ping_time_ms', 0),
                "connected_clients": redis_health.get('connected_clients', 0),
                "memory_used_mb": redis_health.get('memory_used_mb', 0),
                "total_keys": redis_health.get('total_keys', 0)
            },
            vector_search_metrics={
                "total_documents": vector_stats.total_documents,
                "documents_by_type": vector_stats.documents_by_type,
                "last_updated": vector_stats.last_updated.isoformat(),
                "index_size_mb": vector_stats.index_size_mb
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system metrics: {str(e)}"
        )


@router.get("/status", response_model=DetailedStatusResponse)
async def get_detailed_status(
    orchestrator: SystemOrchestrator = Depends(get_orchestrator),
    redis_service: RedisService = Depends(get_redis_service),
    vector_service: VectorSearchService = Depends(get_vector_search_service),
    service_health: Dict[str, bool] = Depends(verify_service_health)
):
    """
    Get detailed system status information.
    
    Returns:
        Comprehensive status including agent details, service information,
        and performance metrics
    """
    try:
        # Get comprehensive metrics
        orchestrator_metrics = await orchestrator.get_system_metrics()
        redis_health = await redis_service.get_health_status()
        vector_stats = await vector_service.get_index_statistics()
        
        # Determine overall status
        all_services_healthy = all(service_health.values())
        active_agents = orchestrator_metrics.get('active_agents', 0)
        total_agents = orchestrator_metrics.get('agent_count', 0)
        
        if not all_services_healthy:
            status = "critical"
        elif active_agents < total_agents:
            status = "warning"
        else:
            status = "healthy"
        
        return DetailedStatusResponse(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.0",
            uptime_seconds=orchestrator_metrics.get('uptime_seconds', 0),
            agent_summary={
                "total_count": total_agents,
                "active_count": active_agents,
                "agents": orchestrator_metrics.get('agents', {}),
                "task_performance": {
                    "active_tasks": orchestrator_metrics.get('active_tasks', 0),
                    "completed_tasks": orchestrator_metrics.get('orchestrator_metrics', {}).get('total_tasks_processed', 0),
                    "average_task_time": orchestrator_metrics.get('orchestrator_metrics', {}).get('average_task_time', 0)
                }
            },
            service_details={
                "redis": {
                    "status": "healthy" if redis_health.get('is_healthy', False) else "unhealthy",
                    "ping_time_ms": redis_health.get('ping_time_ms', 0),
                    "memory_used_mb": redis_health.get('memory_used_mb', 0),
                    "connected_clients": redis_health.get('connected_clients', 0),
                    "uptime_seconds": redis_health.get('uptime_seconds', 0)
                },
                "vector_search": {
                    "status": "healthy" if service_health.get('vector_search', False) else "unhealthy",
                    "total_documents": vector_stats.total_documents,
                    "collections": len(vector_stats.documents_by_type),
                    "last_updated": vector_stats.last_updated.isoformat()
                }
            },
            performance_metrics={
                "system_load": orchestrator_metrics.get('system_load', 0),
                "memory_usage": orchestrator_metrics.get('memory_usage', 0),
                "cpu_usage": orchestrator_metrics.get('cpu_usage', 0),
                "request_rate": orchestrator_metrics.get('request_rate', 0)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get detailed status: {str(e)}"
        )