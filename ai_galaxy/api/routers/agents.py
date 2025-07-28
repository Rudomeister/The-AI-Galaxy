"""
Agents Management API Routes.

This module provides endpoints for managing and monitoring
AI-Galaxy agents including status, communication, and task management.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...orchestrator import SystemOrchestrator, AgentStatus
from ...services.redis.db import RedisService
from ..dependencies import get_orchestrator, get_redis_service

router = APIRouter()


class AgentStatusResponse(BaseModel):
    """Agent status response model."""
    name: str
    agent_type: str
    status: str
    last_heartbeat: Optional[str]
    capabilities: List[str]
    current_task: Optional[str]
    error_count: int
    total_tasks_completed: int
    average_task_time: float
    metadata: Dict[str, Any]


class AgentListResponse(BaseModel):
    """Agent list response model."""
    agents: List[AgentStatusResponse]
    total_count: int
    active_count: int
    inactive_count: int
    timestamp: str


class AgentCommandRequest(BaseModel):
    """Agent command request model."""
    command: str = Field(..., description="Command to send to agent")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")
    priority: str = Field(default="normal", description="Command priority")


class AgentCommandResponse(BaseModel):
    """Agent command response model."""
    command_id: str
    agent_name: str
    command: str
    status: str
    timestamp: str
    message: str


class AgentTaskResponse(BaseModel):
    """Agent task response model."""
    task_id: str
    agent_name: str
    task_type: str
    status: str
    created_at: str
    updated_at: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]]


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    status_filter: Optional[str] = Query(None, description="Filter by agent status"),
    agent_type_filter: Optional[str] = Query(None, description="Filter by agent type"),
    orchestrator: SystemOrchestrator = Depends(get_orchestrator)
):
    """
    List all registered agents with their current status.
    
    Args:
        status_filter: Optional status filter (active, inactive, error)
        agent_type_filter: Optional agent type filter
        
    Returns:
        List of agents with status information
    """
    try:
        # Get agent information from orchestrator
        system_metrics = await orchestrator.get_system_metrics()
        agents_data = system_metrics.get('agents', {})
        
        agents = []
        active_count = 0
        inactive_count = 0
        
        for agent_name, agent_info in agents_data.items():
            # Convert agent info to response model
            agent_status = AgentStatusResponse(
                name=agent_name,
                agent_type=agent_info.get('agent_type', 'unknown'),
                status=agent_info.get('status', 'unknown'),
                last_heartbeat=agent_info.get('last_heartbeat'),
                capabilities=agent_info.get('capabilities', []),
                current_task=agent_info.get('current_task'),
                error_count=agent_info.get('error_count', 0),
                total_tasks_completed=agent_info.get('total_tasks_completed', 0),
                average_task_time=agent_info.get('average_task_time', 0.0),
                metadata=agent_info.get('metadata', {})
            )
            
            # Apply filters
            if status_filter and agent_status.status.lower() != status_filter.lower():
                continue
            if agent_type_filter and agent_status.agent_type.lower() != agent_type_filter.lower():
                continue
            
            agents.append(agent_status)
            
            # Count status
            if agent_status.status.lower() == 'active':
                active_count += 1
            else:
                inactive_count += 1
        
        return AgentListResponse(
            agents=agents,
            total_count=len(agents),
            active_count=active_count,
            inactive_count=inactive_count,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.get("/{agent_name}", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_name: str,
    orchestrator: SystemOrchestrator = Depends(get_orchestrator)
):
    """
    Get detailed status information for a specific agent.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Detailed agent status information
    """
    try:
        # Check if agent exists
        if agent_name not in orchestrator.agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        agent_info = orchestrator.agents[agent_name]
        
        return AgentStatusResponse(
            name=agent_name,
            agent_type=agent_info.agent_type,
            status=agent_info.status.value,
            last_heartbeat=agent_info.last_heartbeat.isoformat() if agent_info.last_heartbeat else None,
            capabilities=agent_info.capabilities,
            current_task=agent_info.current_task,
            error_count=agent_info.error_count,
            total_tasks_completed=agent_info.total_tasks_completed,
            average_task_time=agent_info.average_task_time,
            metadata=agent_info.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent status: {str(e)}"
        )


@router.post("/{agent_name}/command", response_model=AgentCommandResponse)
async def send_agent_command(
    agent_name: str,
    command_request: AgentCommandRequest,
    orchestrator: SystemOrchestrator = Depends(get_orchestrator),
    redis_service: RedisService = Depends(get_redis_service)
):
    """
    Send a command to a specific agent.
    
    Args:
        agent_name: Name of the target agent
        command_request: Command details
        
    Returns:
        Command execution status
    """
    try:
        # Check if agent exists and is active
        if agent_name not in orchestrator.agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        agent_info = orchestrator.agents[agent_name]
        if agent_info.status != AgentStatus.ACTIVE:
            raise HTTPException(
                status_code=400,
                detail=f"Agent '{agent_name}' is not active (status: {agent_info.status.value})"
            )
        
        # Generate command ID
        command_id = str(uuid4())
        
        # Prepare command message
        command_message = {
            "command_id": command_id,
            "command": command_request.command,
            "parameters": command_request.parameters,
            "priority": command_request.priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender": "api"
        }
        
        # Send command via Redis pub/sub
        await redis_service.publish_message(
            f"agent_commands:{agent_name}",
            command_message
        )
        
        return AgentCommandResponse(
            command_id=command_id,
            agent_name=agent_name,
            command=command_request.command,
            status="sent",
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"Command sent to agent {agent_name}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send command to agent: {str(e)}"
        )


@router.get("/{agent_name}/tasks", response_model=List[AgentTaskResponse])
async def get_agent_tasks(
    agent_name: str,
    status_filter: Optional[str] = Query(None, description="Filter by task status"),
    limit: int = Query(50, description="Maximum number of tasks to return"),
    orchestrator: SystemOrchestrator = Depends(get_orchestrator)
):
    """
    Get recent tasks for a specific agent.
    
    Args:
        agent_name: Name of the agent
        status_filter: Optional task status filter
        limit: Maximum number of tasks to return
        
    Returns:
        List of recent agent tasks
    """
    try:
        # Check if agent exists
        if agent_name not in orchestrator.agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        # Get task history from orchestrator
        # Note: This would need to be implemented in the orchestrator
        # For now, return a placeholder response
        tasks = []
        
        # TODO: Implement task history retrieval from orchestrator
        # tasks = await orchestrator.get_agent_task_history(agent_name, status_filter, limit)
        
        return tasks
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent tasks: {str(e)}"
        )


@router.post("/{agent_name}/restart", response_model=AgentCommandResponse)
async def restart_agent(
    agent_name: str,
    orchestrator: SystemOrchestrator = Depends(get_orchestrator)
):
    """
    Restart a specific agent.
    
    Args:
        agent_name: Name of the agent to restart
        
    Returns:
        Restart command status
    """
    try:
        # Check if agent exists
        if agent_name not in orchestrator.agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        # Generate restart command
        command_id = str(uuid4())
        
        # TODO: Implement agent restart functionality in orchestrator
        # success = await orchestrator.restart_agent(agent_name)
        
        # For now, return a placeholder response
        return AgentCommandResponse(
            command_id=command_id,
            agent_name=agent_name,
            command="restart",
            status="pending",
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"Restart command issued for agent {agent_name}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart agent: {str(e)}"
        )