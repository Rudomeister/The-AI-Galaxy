"""
FastAPI Application Factory for AI-Galaxy.

This module creates and configures the FastAPI application with all necessary
middleware, routes, and integrations for the AI-Galaxy ecosystem.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from ..shared.logger import get_logger
from .dependencies import get_orchestrator, get_redis_service, get_vector_search_service
from .routers import health, agents, ideas, departments, institutions, websockets


logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting AI-Galaxy API server...")
    
    # Startup
    try:
        # Initialize services are handled by dependencies
        logger.info("AI-Galaxy API server started successfully")
        yield
    finally:
        # Shutdown
        logger.info("Shutting down AI-Galaxy API server...")


def create_app(orchestrator=None, redis_service=None, vector_search_service=None) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        orchestrator: Optional orchestrator instance for dependency injection
        redis_service: Optional Redis service instance 
        vector_search_service: Optional vector search service instance
        
    Returns:
        Configured FastAPI application
    """
    
    app = FastAPI(
        title="AI-Galaxy Meta-Layer API",
        description="""
        Agent-driven microservice civilization API for the AI-Galaxy ecosystem.
        
        This API provides comprehensive access to:
        - Agent management and monitoring
        - Idea submission and workflow control
        - System metrics and health monitoring  
        - Real-time communication with agents
        - Department and institution management
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,  # Custom docs URL
        redoc_url=None  # Custom redoc URL
    )
    
    # Store service instances for dependency injection
    if orchestrator:
        app.state.orchestrator = orchestrator
    if redis_service:
        app.state.redis_service = redis_service
    if vector_search_service:
        app.state.vector_search_service = vector_search_service
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request tracking middleware
    @app.middleware("http")
    async def add_request_tracking(request: Request, call_next):
        """Add request ID and timing to all requests."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Add request ID to response headers
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log request metrics
        logger.info(
            f"Request {request_id} {request.method} {request.url.path} "
            f"completed in {process_time:.3f}s with status {response.status_code}"
        )
        
        return response
    
    # Custom exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        logger.error(f"Unhandled exception in request {request_id}: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": request_id
            }
        )
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
    app.include_router(ideas.router, prefix="/api/v1/ideas", tags=["Ideas"])
    app.include_router(departments.router, prefix="/api/v1/departments", tags=["Departments"])
    app.include_router(institutions.router, prefix="/api/v1/institutions", tags=["Institutions"])
    app.include_router(websockets.router, prefix="/ws", tags=["WebSocket"])
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="AI-Galaxy API",
            version="1.0.0",
            description="Agent orchestration and microservice management API",
            routes=app.routes,
        )
        
        # Add custom security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Custom documentation endpoints
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Custom Swagger UI documentation."""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="AI-Galaxy API Documentation",
            swagger_favicon_url="/static/favicon.ico"
        )
    
    @app.get("/", include_in_schema=False)
    async def root():
        """API root endpoint."""
        return {
            "message": "AI-Galaxy Meta-Layer API",
            "version": "1.0.0",
            "documentation": "/docs",
            "health": "/health"
        }
    
    return app