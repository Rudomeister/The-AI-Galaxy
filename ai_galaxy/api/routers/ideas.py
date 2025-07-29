"""
Ideas Management API Routes.

This module provides endpoints for idea submission, management,
and workflow control in the AI-Galaxy ecosystem.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ...shared.models import Idea, IdeaStatus
from ...orchestrator import SystemOrchestrator
from ...services.vector_search.query_service import VectorSearchService
from ...services.redis.db import RedisService
from ..dependencies import get_orchestrator, get_vector_search_service, get_redis_service

router = APIRouter()


def convert_priority_to_int(priority_str: str) -> int:
    """Convert string priority to integer."""
    priority_map = {
        "low": 3,
        "medium": 5,
        "high": 8,
        "critical": 10
    }
    return priority_map.get(priority_str.lower(), 5)


def convert_priority_to_str(priority_int: int) -> str:
    """Convert integer priority to string."""
    if priority_int >= 9:
        return "critical"
    elif priority_int >= 7:
        return "high"
    elif priority_int >= 4:
        return "medium"
    else:
        return "low"


class IdeaCreateRequest(BaseModel):
    """Request model for creating a new idea."""
    title: str = Field(..., min_length=3, max_length=200, description="Idea title")
    description: str = Field(..., min_length=10, description="Detailed idea description")
    priority: str = Field(default="medium", description="Idea priority (low, medium, high, critical)")
    tags: List[str] = Field(default_factory=list, description="Optional tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    department_hint: Optional[str] = Field(None, description="Suggested department for processing")


class IdeaUpdateRequest(BaseModel):
    """Request model for updating an existing idea."""
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, min_length=10)
    priority: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class IdeaResponse(BaseModel):
    """Response model for idea information."""
    id: str
    title: str
    description: str
    status: str
    priority: str
    created_at: str
    updated_at: str
    tags: List[str]
    metadata: Dict[str, Any]
    department_assignment: Optional[str]
    institution_assignment: Optional[str]
    processing_history: List[Dict[str, Any]]


class IdeaListResponse(BaseModel):
    """Response model for idea listing."""
    ideas: List[IdeaResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class WorkflowTransitionRequest(BaseModel):
    """Request model for workflow state transitions."""
    target_state: str = Field(..., description="Target workflow state")
    agent_name: Optional[str] = Field(None, description="Specific agent to handle transition")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Transition parameters")


class WorkflowTransitionResponse(BaseModel):
    """Response model for workflow transitions."""
    idea_id: str
    from_state: str
    to_state: str
    agent_name: str
    transition_id: str
    timestamp: str
    status: str
    message: str


@router.post("/", response_model=IdeaResponse)
async def create_idea(
    idea_request: IdeaCreateRequest,
    background_tasks: BackgroundTasks,
    orchestrator: SystemOrchestrator = Depends(get_orchestrator),
    vector_service: VectorSearchService = Depends(get_vector_search_service),
    redis_service: RedisService = Depends(get_redis_service)
):
    """
    Create a new idea and start processing workflow.
    
    Args:
        idea_request: Idea creation details
        background_tasks: FastAPI background tasks
        
    Returns:
        Created idea with initial processing status
    """
    try:
        # Create idea instance
        idea_id = uuid4()
        current_time = datetime.now(timezone.utc)
        
        idea = Idea(
            id=idea_id,
            title=idea_request.title,
            description=idea_request.description,
            status=IdeaStatus.CREATED,
            priority=idea_request.priority,
            created_at=current_time,
            updated_at=current_time,
            metadata=idea_request.metadata or {}
        )
        
        # Add tags to metadata
        if idea_request.tags:
            idea.metadata['tags'] = idea_request.tags
        
        # Add department hint if provided
        if idea_request.department_hint:
            idea.metadata['department_hint'] = idea_request.department_hint
        
        # Store idea in Redis
        await redis_service.set_state(f"idea:{idea_id}", idea.dict())
        
        # Index idea for semantic search
        await vector_service.index_idea(idea)
        
        # Start processing workflow in background
        background_tasks.add_task(
            start_idea_workflow,
            str(idea_id),
            orchestrator,
            redis_service
        )
        
        return IdeaResponse(
            id=str(idea.id),
            title=idea.title,
            description=idea.description,
            status=idea.status.value,
            priority=idea.priority,
            created_at=idea.created_at.isoformat(),
            updated_at=idea.updated_at.isoformat(),
            tags=idea_request.tags,
            metadata=idea.metadata,
            department_assignment=None,
            institution_assignment=None,
            processing_history=[]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create idea: {str(e)}"
        )


@router.get("/", response_model=IdeaListResponse)
async def list_ideas(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    priority_filter: Optional[str] = Query(None, description="Filter by priority"),
    search: Optional[str] = Query(None, description="Search in title and description"),
    redis_service: RedisService = Depends(get_redis_service),
    vector_service: VectorSearchService = Depends(get_vector_search_service)
):
    """
    List ideas with filtering and pagination.
    
    Args:
        page: Page number for pagination
        page_size: Number of items per page
        status_filter: Optional status filter
        priority_filter: Optional priority filter
        search: Optional text search
        
    Returns:
        Paginated list of ideas
    """
    try:
        # Get all ideas from Redis using direct client access
        # Note: We need to access the Redis client directly for key scanning
        redis_client = redis_service._async_client
        if not redis_client:
            # Initialize connection if not ready
            await redis_service.initialize()
            redis_client = redis_service._async_client
        
        all_idea_keys = await redis_client.keys("idea:*")
        all_ideas = []
        
        # Fetch all ideas from Redis
        for key in all_idea_keys:
            # Convert bytes key to string if needed
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            idea_data = await redis_service.get_state(key_str)
            if idea_data:
                try:
                    idea = Idea(**idea_data)
                    all_ideas.append(idea)
                except Exception as e:
                    # Skip malformed ideas
                    continue
        
        # Apply filters
        filtered_ideas = all_ideas
        
        if status_filter:
            filtered_ideas = [idea for idea in filtered_ideas if idea.status.value == status_filter]
        
        if priority_filter:
            filtered_ideas = [idea for idea in filtered_ideas if idea.priority == priority_filter]
        
        if search:
            # Simple text search in title and description
            search_lower = search.lower()
            filtered_ideas = [
                idea for idea in filtered_ideas 
                if search_lower in idea.title.lower() or search_lower in idea.description.lower()
            ]
        
        # Sort by creation date (newest first)
        filtered_ideas.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        total_count = len(filtered_ideas)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_ideas = filtered_ideas[start_idx:end_idx]
        
        # Convert to response format
        ideas = [_convert_idea_to_response(idea) for idea in paginated_ideas]
        
        return IdeaListResponse(
            ideas=ideas,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=total_count > page * page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list ideas: {str(e)}"
        )


@router.get("/{idea_id}", response_model=IdeaResponse)
async def get_idea(
    idea_id: str,
    redis_service: RedisService = Depends(get_redis_service)
):
    """
    Get detailed information about a specific idea.
    
    Args:
        idea_id: Unique identifier of the idea
        
    Returns:
        Detailed idea information
    """
    try:
        # Get idea from Redis
        idea_data = await redis_service.get_state(f"idea:{idea_id}")
        if not idea_data:
            raise HTTPException(
                status_code=404,
                detail=f"Idea '{idea_id}' not found"
            )
        
        idea = Idea(**idea_data)
        return _convert_idea_to_response(idea)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get idea: {str(e)}"
        )


@router.put("/{idea_id}", response_model=IdeaResponse)
async def update_idea(
    idea_id: str,
    update_request: IdeaUpdateRequest,
    redis_service: RedisService = Depends(get_redis_service),
    vector_service: VectorSearchService = Depends(get_vector_search_service)
):
    """
    Update an existing idea.
    
    Args:
        idea_id: Unique identifier of the idea
        update_request: Update details
        
    Returns:
        Updated idea information
    """
    try:
        # Get existing idea
        idea_data = await redis_service.get_state(f"idea:{idea_id}")
        if not idea_data:
            raise HTTPException(
                status_code=404,
                detail=f"Idea '{idea_id}' not found"
            )
        
        idea = Idea(**idea_data)
        
        # Update fields
        if update_request.title is not None:
            idea.title = update_request.title
        if update_request.description is not None:
            idea.description = update_request.description
        if update_request.priority is not None:
            idea.priority = update_request.priority
        if update_request.tags is not None:
            idea.metadata['tags'] = update_request.tags
        if update_request.metadata is not None:
            idea.metadata.update(update_request.metadata)
        
        idea.updated_at = datetime.now(timezone.utc)
        
        # Save updated idea
        await redis_service.set_state(f"idea:{idea_id}", idea.dict())
        
        # Update search index
        await vector_service.index_idea(idea)
        
        return _convert_idea_to_response(idea)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update idea: {str(e)}"
        )


@router.post("/{idea_id}/transition", response_model=WorkflowTransitionResponse)
async def transition_idea_workflow(
    idea_id: str,
    transition_request: WorkflowTransitionRequest,
    background_tasks: BackgroundTasks,
    orchestrator: SystemOrchestrator = Depends(get_orchestrator),
    redis_service: RedisService = Depends(get_redis_service)
):
    """
    Trigger a workflow state transition for an idea.
    
    Args:
        idea_id: Unique identifier of the idea
        transition_request: Transition details
        background_tasks: FastAPI background tasks
        
    Returns:
        Transition status information
    """
    try:
        # Get existing idea
        idea_data = await redis_service.get_state(f"idea:{idea_id}")
        if not idea_data:
            raise HTTPException(
                status_code=404,
                detail=f"Idea '{idea_id}' not found"
            )
        
        idea = Idea(**idea_data)
        current_state = idea.status.value
        target_state = transition_request.target_state
        
        # Generate transition ID
        transition_id = str(uuid4())
        
        # Start workflow transition in background
        background_tasks.add_task(
            process_workflow_transition,
            idea_id,
            current_state,
            target_state,
            transition_request.agent_name,
            transition_request.parameters,
            transition_id,
            orchestrator,
            redis_service
        )
        
        return WorkflowTransitionResponse(
            idea_id=idea_id,
            from_state=current_state,
            to_state=target_state,
            agent_name=transition_request.agent_name or "auto",
            transition_id=transition_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="processing",
            message=f"Workflow transition from {current_state} to {target_state} initiated"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transition idea workflow: {str(e)}"
        )


@router.delete("/{idea_id}")
async def delete_idea(
    idea_id: str,
    redis_service: RedisService = Depends(get_redis_service),
    vector_service: VectorSearchService = Depends(get_vector_search_service)
):
    """
    Delete an idea from the system.
    
    Args:
        idea_id: Unique identifier of the idea
        
    Returns:
        Deletion confirmation
    """
    try:
        # Check if idea exists
        idea_data = await redis_service.get_state(f"idea:{idea_id}")
        if not idea_data:
            raise HTTPException(
                status_code=404,
                detail=f"Idea '{idea_id}' not found"
            )
        
        # Remove from Redis
        await redis_service.delete_key(f"idea:{idea_id}")
        
        # Remove from search index
        await vector_service.remove_from_index(idea_id, "idea")
        
        return {"message": f"Idea '{idea_id}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete idea: {str(e)}"
        )


# Helper functions

def _convert_idea_to_response(idea: Idea) -> IdeaResponse:
    """Convert Idea model to IdeaResponse."""
    return IdeaResponse(
        id=str(idea.id),
        title=idea.title,
        description=idea.description,
        status=idea.status.value,
        priority=idea.priority,
        created_at=idea.created_at.isoformat(),
        updated_at=idea.updated_at.isoformat(),
        tags=idea.metadata.get('tags', []),
        metadata=idea.metadata,
        department_assignment=str(idea.department_assignment) if idea.department_assignment else None,
        institution_assignment=str(idea.institution_assignment) if idea.institution_assignment else None,
        processing_history=idea.metadata.get('processing_history', [])
    )


async def start_idea_workflow(
    idea_id: str,
    orchestrator: SystemOrchestrator,
    redis_service: RedisService
):
    """Background task to start idea processing workflow."""
    try:
        # TODO: Implement actual workflow initiation
        # This would typically involve:
        # 1. Notifying the validator agent
        # 2. Setting up workflow tracking
        # 3. Updating idea status
        
        # For now, just log the workflow start
        await redis_service.publish_message(
            "workflow_events",
            {
                "event": "workflow_started",
                "idea_id": idea_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        # Log error but don't raise to avoid background task failure
        print(f"Error starting workflow for idea {idea_id}: {str(e)}")


async def process_workflow_transition(
    idea_id: str,
    from_state: str,
    to_state: str,
    agent_name: Optional[str],
    parameters: Dict[str, Any],
    transition_id: str,
    orchestrator: SystemOrchestrator,
    redis_service: RedisService
):
    """Background task to process workflow state transition."""
    try:
        # TODO: Implement actual workflow transition logic
        # This would involve:
        # 1. Validating the transition
        # 2. Assigning to appropriate agent
        # 3. Updating idea status
        # 4. Tracking transition history
        
        # For now, just publish transition event
        await redis_service.publish_message(
            "workflow_events",
            {
                "event": "transition_processed",
                "idea_id": idea_id,
                "from_state": from_state,
                "to_state": to_state,
                "agent_name": agent_name,
                "transition_id": transition_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        # Log error but don't raise to avoid background task failure
        print(f"Error processing transition {transition_id}: {str(e)}")