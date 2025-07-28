"""
FastAPI Dependency Injection for AI-Galaxy Services.

This module provides dependency injection for core AI-Galaxy services,
enabling clean separation of concerns and easy testing.
"""

from typing import Optional, AsyncGenerator
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..orchestrator import SystemOrchestrator
from ..services.redis.db import RedisService
from ..services.vector_search.query_service import VectorSearchService
from ..shared.logger import get_logger

logger = get_logger("api.dependencies")
security = HTTPBearer(auto_error=False)


async def get_orchestrator(request: Request) -> SystemOrchestrator:
    """
    Get the system orchestrator instance.
    
    Args:
        request: FastAPI request object
        
    Returns:
        SystemOrchestrator instance
        
    Raises:
        HTTPException: If orchestrator is not available
    """
    orchestrator = getattr(request.app.state, 'orchestrator', None)
    if not orchestrator:
        logger.error("Orchestrator not available in app state")
        raise HTTPException(
            status_code=503,
            detail="System orchestrator not available. Please check system status."
        )
    return orchestrator


async def get_redis_service(request: Request) -> RedisService:
    """
    Get the Redis service instance.
    
    Args:
        request: FastAPI request object
        
    Returns:
        RedisService instance
        
    Raises:
        HTTPException: If Redis service is not available
    """
    redis_service = getattr(request.app.state, 'redis_service', None)
    if not redis_service:
        logger.error("Redis service not available in app state")
        raise HTTPException(
            status_code=503,
            detail="Redis service not available. Please check Redis connection."
        )
    return redis_service


async def get_vector_search_service(request: Request) -> VectorSearchService:
    """
    Get the vector search service instance.
    
    Args:
        request: FastAPI request object
        
    Returns:
        VectorSearchService instance
        
    Raises:
        HTTPException: If vector search service is not available
    """
    vector_service = getattr(request.app.state, 'vector_search_service', None)
    if not vector_service:
        logger.error("Vector search service not available in app state")
        raise HTTPException(
            status_code=503,
            detail="Vector search service not available. Please check ChromaDB connection."
        )
    return vector_service


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Get current authenticated user (optional dependency).
    
    Args:
        credentials: JWT credentials from Authorization header
        
    Returns:
        Username if authenticated, None otherwise
        
    Note:
        This is a placeholder implementation. In production, you would
        validate JWT tokens and extract user information.
    """
    if not credentials:
        return None
    
    # TODO: Implement JWT token validation
    # For now, return a placeholder user
    return "system_user"


async def require_authentication(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Require authentication for protected endpoints.
    
    Args:
        credentials: JWT credentials from Authorization header
        
    Returns:
        Username of authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # TODO: Implement JWT token validation
    # For now, accept any token as valid
    return "authenticated_user"


class ServiceHealthChecker:
    """Health checker for dependent services."""
    
    @staticmethod
    async def check_redis_health(redis_service: RedisService) -> bool:
        """Check if Redis service is healthy."""
        try:
            health_status = await redis_service.get_health_status()
            return health_status.get('is_healthy', False)
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False
    
    @staticmethod
    async def check_vector_search_health(vector_service: VectorSearchService) -> bool:
        """Check if vector search service is healthy."""
        try:
            # Try to get index statistics as a health check
            stats = await vector_service.get_index_statistics()
            return stats is not None
        except Exception as e:
            logger.error(f"Vector search health check failed: {str(e)}")
            return False
    
    @staticmethod
    async def check_orchestrator_health(orchestrator: SystemOrchestrator) -> bool:
        """Check if orchestrator is healthy."""
        try:
            metrics = await orchestrator.get_system_metrics()
            return metrics is not None
        except Exception as e:
            logger.error(f"Orchestrator health check failed: {str(e)}")
            return False


async def verify_service_health(
    orchestrator: SystemOrchestrator = Depends(get_orchestrator),
    redis_service: RedisService = Depends(get_redis_service),
    vector_service: VectorSearchService = Depends(get_vector_search_service)
) -> dict:
    """
    Verify health of all dependent services.
    
    Returns:
        Dictionary with health status of all services
    """
    health_checker = ServiceHealthChecker()
    
    return {
        "orchestrator": await health_checker.check_orchestrator_health(orchestrator),
        "redis": await health_checker.check_redis_health(redis_service),
        "vector_search": await health_checker.check_vector_search_health(vector_service)
    }