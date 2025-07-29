"""
System Orchestrator for AI-Galaxy Ecosystem.

This module implements the central coordinator for all agents in the AI-Galaxy
ecosystem. It manages workflow orchestration, task distribution, inter-agent
communication, system state management, and monitoring to ensure smooth
operation of the entire system as a cohesive living organism.
"""

import asyncio
import json
import time
import traceback
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .shared.logger import get_logger, LogContext
from .shared.models import (
    Idea, IdeaStatus, Department, Institution, Microservice,
    AgentMessage, MessageType, MessageStatus, SystemState
)
from .services.redis import RedisService, RedisConfig
from .services.vector_search import VectorSearchService, SearchConfig


class AgentStatus(str, Enum):
    """Status enumeration for agents."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class WorkflowStatus(str, Enum):
    """Status enumeration for workflows."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    name: str
    agent_type: str
    status: AgentStatus = AgentStatus.OFFLINE
    last_heartbeat: Optional[datetime] = None
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    error_count: int = 0
    total_tasks_completed: int = 0
    average_task_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    id: str
    task_type: str
    agent_name: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Workflow:
    """Represents a multi-step workflow involving multiple agents."""
    id: str
    name: str
    idea_id: str
    status: WorkflowStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks: List[Task] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestratorConfig(BaseModel):
    """Configuration for the system orchestrator."""
    heartbeat_interval: int = Field(default=30, description="Heartbeat check interval in seconds")
    task_timeout: int = Field(default=300, description="Default task timeout in seconds")
    max_concurrent_tasks: int = Field(default=10, description="Maximum concurrent tasks per agent")
    workflow_timeout: int = Field(default=3600, description="Default workflow timeout in seconds")
    health_check_interval: int = Field(default=60, description="Health check interval in seconds")
    cleanup_interval: int = Field(default=300, description="Cleanup interval for old data in seconds")
    agent_offline_threshold: int = Field(default=120, description="Seconds before agent considered offline")


class SystemOrchestrator:
    """
    Central orchestrator for the AI-Galaxy ecosystem.
    
    This class coordinates all agents, manages workflows, handles inter-agent
    communication, and maintains system state to ensure the ecosystem operates
    as a cohesive, autonomous system.
    """
    
    def __init__(self, 
                 redis_config: Optional[RedisConfig] = None,
                 search_config: Optional[SearchConfig] = None,
                 orchestrator_config: Optional[OrchestratorConfig] = None):
        """
        Initialize system orchestrator.
        
        Args:
            redis_config: Redis service configuration
            search_config: Vector search service configuration  
            orchestrator_config: Orchestrator configuration
        """
        self.config = orchestrator_config or OrchestratorConfig()
        self.logger = get_logger("orchestrator")
        
        # Core services
        self.redis_service = RedisService(redis_config)
        self.vector_search_service = VectorSearchService(search_config)
        
        # Agent registry and state
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_channels: Dict[str, asyncio.Queue] = {}
        
        # Task and workflow management
        self.active_tasks: Dict[str, Task] = {}
        self.active_workflows: Dict[str, Workflow] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # System state
        self.system_state = SystemState()
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._task_processor_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Performance metrics
        self.metrics = {
            'total_tasks_processed': 0,
            'total_workflows_completed': 0,
            'average_task_time': 0.0,
            'system_uptime': 0.0,
            'agent_count': 0,
            'active_task_count': 0
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the orchestrator and all core services.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            self.logger.info("Initializing AI-Galaxy System Orchestrator...")
            
            # Initialize core services
            redis_success = await self.redis_service.initialize()
            if not redis_success:
                self.logger.error("Failed to initialize Redis service")
                return False
            
            search_success = await self.vector_search_service.initialize()
            if not search_success:
                self.logger.error("Failed to initialize Vector Search service")
                return False
            
            # Set up Redis subscriptions for agent communication
            await self._setup_agent_communication()
            
            # Load existing system state
            await self._load_system_state()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            self.startup_time = datetime.now(timezone.utc)
            
            self.logger.info("System Orchestrator initialized successfully")
            
            # Emit startup event
            await self._emit_event('system_startup', {'timestamp': self.startup_time.isoformat()})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize System Orchestrator: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator and all services."""
        self.logger.info("Shutting down AI-Galaxy System Orchestrator...")
        
        self.is_running = False
        
        # Cancel background tasks
        tasks = [
            self._heartbeat_task,
            self._task_processor_task, 
            self._health_monitor_task,
            self._cleanup_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cancel active tasks
        for task in self.active_tasks.values():
            if task.status == WorkflowStatus.RUNNING:
                task.status = WorkflowStatus.CANCELLED
        
        # Save system state
        await self._save_system_state()
        
        # Emit shutdown event
        await self._emit_event('system_shutdown', {'timestamp': datetime.now(timezone.utc).isoformat()})
        
        # Shutdown core services
        await self.vector_search_service.shutdown()
        await self.redis_service.shutdown()
        
        self.logger.info("System Orchestrator shutdown completed")
    
    # === Agent Management ===
    
    async def register_agent(self, name: str, agent_type: str, capabilities: List[str] = None,
                           metadata: Dict[str, Any] = None) -> bool:
        """
        Register a new agent with the orchestrator.
        
        Args:
            name: Unique agent name
            agent_type: Type of agent (e.g., 'router', 'validator', etc.)
            capabilities: List of agent capabilities
            metadata: Additional agent metadata
            
        Returns:
            True if registration successful, False otherwise.
        """
        try:
            if name in self.agents:
                self.logger.warning(f"Agent {name} already registered, updating...")
            
            agent_info = AgentInfo(
                name=name,
                agent_type=agent_type,
                status=AgentStatus.ACTIVE,
                last_heartbeat=datetime.now(timezone.utc),
                capabilities=capabilities or [],
                metadata=metadata or {}
            )
            
            self.agents[name] = agent_info
            
            # Create communication channel
            if name not in self.agent_channels:
                self.agent_channels[name] = asyncio.Queue()
            
            # Subscribe to agent's Redis channel
            await self.redis_service.subscribe_to_channel(
                f"agent:{name}",
                self._handle_agent_message
            )
            
            # Store agent info in Redis
            await self.redis_service.set_state(f"agent:{name}", agent_info.__dict__)
            
            # Update metrics
            self.metrics['agent_count'] = len(self.agents)
            
            context = LogContext(
                agent_name=name,
                additional_context={
                    'agent_type': agent_type,
                    'capabilities': capabilities
                }
            )
            self.logger.info(f"Agent registered: {name}", context)
            
            # Emit agent registration event
            await self._emit_event('agent_registered', {
                'agent_name': name,
                'agent_type': agent_type,
                'capabilities': capabilities
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {name}: {str(e)}")
            return False
    
    async def unregister_agent(self, name: str) -> bool:
        """Unregister an agent from the orchestrator."""
        try:
            if name not in self.agents:
                self.logger.warning(f"Agent {name} not found for unregistration")
                return False
            
            # Cancel any active tasks for this agent
            tasks_to_cancel = [task for task in self.active_tasks.values() if task.agent_name == name]
            for task in tasks_to_cancel:
                task.status = WorkflowStatus.CANCELLED
                task.error = "Agent unregistered"
            
            # Remove from registry
            del self.agents[name]
            
            # Remove communication channel
            if name in self.agent_channels:
                del self.agent_channels[name]
            
            # Unsubscribe from Redis channel
            await self.redis_service.unsubscribe_from_channel(f"agent:{name}")
            
            # Remove from Redis
            await self.redis_service.delete_state(f"agent:{name}")
            
            # Update metrics
            self.metrics['agent_count'] = len(self.agents)
            
            self.logger.info(f"Agent unregistered: {name}")
            
            # Emit agent unregistration event
            await self._emit_event('agent_unregistered', {'agent_name': name})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {name}: {str(e)}")
            return False
    
    async def update_agent_status(self, name: str, status: AgentStatus, 
                                error_message: Optional[str] = None) -> bool:
        """Update agent status."""
        try:
            if name not in self.agents:
                self.logger.warning(f"Agent {name} not found for status update")
                return False
            
            agent = self.agents[name]
            old_status = agent.status
            agent.status = status
            agent.last_heartbeat = datetime.now(timezone.utc)
            
            if status == AgentStatus.ERROR and error_message:
                agent.error_count += 1
                agent.metadata['last_error'] = error_message
                agent.metadata['last_error_time'] = datetime.now(timezone.utc).isoformat()
            
            # Store updated info in Redis
            await self.redis_service.set_state(f"agent:{name}", agent.__dict__)
            
            context = LogContext(
                agent_name=name,
                additional_context={
                    'old_status': old_status,
                    'new_status': status,
                    'error_message': error_message
                }
            )
            self.logger.info(f"Agent status updated: {name}", context)
            
            # Emit status change event
            await self._emit_event('agent_status_changed', {
                'agent_name': name,
                'old_status': old_status,
                'new_status': status,
                'error_message': error_message
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update agent status for {name}: {str(e)}")
            return False
    
    async def get_agent_status(self, name: str) -> Optional[AgentInfo]:
        """Get current status of an agent."""
        return self.agents.get(name)
    
    async def list_agents(self, status_filter: Optional[AgentStatus] = None) -> List[AgentInfo]:
        """List all registered agents, optionally filtered by status."""
        agents = list(self.agents.values())
        
        if status_filter:
            agents = [agent for agent in agents if agent.status == status_filter]
        
        return agents
    
    # === Task Management ===
    
    async def submit_task(self, task_type: str, agent_name: str, payload: Dict[str, Any],
                        priority: TaskPriority = TaskPriority.NORMAL,
                        timeout_seconds: Optional[int] = None,
                        dependencies: List[str] = None) -> Optional[str]:
        """
        Submit a task for execution by an agent.
        
        Args:
            task_type: Type of task to execute
            agent_name: Name of agent to execute the task
            payload: Task data and parameters
            priority: Task priority level
            timeout_seconds: Task timeout (uses default if not specified)
            dependencies: List of task IDs this task depends on
            
        Returns:
            Task ID if submitted successfully, None otherwise.
        """
        try:
            # Validate agent exists and is available
            if agent_name not in self.agents:
                self.logger.error(f"Agent {agent_name} not found for task submission")
                return None
            
            agent = self.agents[agent_name]
            if agent.status not in [AgentStatus.ACTIVE, AgentStatus.BUSY]:
                self.logger.error(f"Agent {agent_name} is not available (status: {agent.status})")
                return None
            
            # Create task
            task_id = str(uuid4())
            task = Task(
                id=task_id,
                task_type=task_type,
                agent_name=agent_name,
                priority=priority,
                payload=payload,
                created_at=datetime.now(timezone.utc),
                timeout_seconds=timeout_seconds or self.config.task_timeout,
                dependencies=dependencies or []
            )
            
            # Add to active tasks
            self.active_tasks[task_id] = task
            
            # Add to task queue
            await self.task_queue.put(task)
            
            # Update metrics
            self.metrics['active_task_count'] = len(self.active_tasks)
            
            context = LogContext(
                agent_name=agent_name,
                additional_context={
                    'task_id': task_id,
                    'task_type': task_type,
                    'priority': priority
                }
            )
            self.logger.info(f"Task submitted: {task_type}", context)
            
            # Emit task submission event
            await self._emit_event('task_submitted', {
                'task_id': task_id,
                'task_type': task_type,
                'agent_name': agent_name,
                'priority': priority
            })
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit task: {str(e)}")
            return None
    
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get current status of a task."""
        return self.active_tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        try:
            if task_id not in self.active_tasks:
                self.logger.warning(f"Task {task_id} not found for cancellation")
                return False
            
            task = self.active_tasks[task_id]
            
            if task.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                self.logger.warning(f"Task {task_id} already finished, cannot cancel")
                return False
            
            task.status = WorkflowStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)
            task.error = "Task cancelled by orchestrator"
            
            # Notify agent if task is running
            if task.status == WorkflowStatus.RUNNING:
                await self._send_agent_message(
                    task.agent_name,
                    MessageType.NOTIFICATION,
                    {'action': 'cancel_task', 'task_id': task_id}
                )
            
            self.logger.info(f"Task cancelled: {task_id}")
            
            # Emit task cancellation event
            await self._emit_event('task_cancelled', {'task_id': task_id})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {str(e)}")
            return False
    
    # === Workflow Management ===
    
    async def create_workflow(self, name: str, idea_id: str, 
                            workflow_steps: List[Dict[str, Any]]) -> Optional[str]:
        """
        Create a new workflow for processing an idea.
        
        Args:
            name: Workflow name
            idea_id: ID of the idea being processed
            workflow_steps: List of steps with agent and task information
            
        Returns:
            Workflow ID if created successfully, None otherwise.
        """
        try:
            workflow_id = str(uuid4())
            
            # Create workflow
            workflow = Workflow(
                id=workflow_id,
                name=name,
                idea_id=idea_id,
                status=WorkflowStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                total_steps=len(workflow_steps)
            )
            
            # Create tasks for each step
            for i, step_config in enumerate(workflow_steps):
                task_id = str(uuid4())
                
                # Set dependencies (each task depends on previous one)
                dependencies = []
                if i > 0:
                    dependencies = [workflow.tasks[i-1].id]
                
                task = Task(
                    id=task_id,
                    task_type=step_config['task_type'],
                    agent_name=step_config['agent_name'],
                    priority=TaskPriority(step_config.get('priority', TaskPriority.NORMAL)),
                    payload=step_config.get('payload', {}),
                    created_at=datetime.now(timezone.utc),
                    timeout_seconds=step_config.get('timeout', self.config.task_timeout),
                    dependencies=dependencies
                )
                
                workflow.tasks.append(task)
            
            # Store workflow
            self.active_workflows[workflow_id] = workflow
            
            # Store in Redis
            await self.redis_service.set_state(f"workflow:{workflow_id}", workflow.__dict__)
            
            self.logger.info(f"Workflow created: {name} ({workflow_id})")
            
            # Emit workflow creation event
            await self._emit_event('workflow_created', {
                'workflow_id': workflow_id,
                'name': name,
                'idea_id': idea_id,
                'total_steps': len(workflow_steps)
            })
            
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {str(e)}")
            return None
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start execution of a workflow."""
        try:
            if workflow_id not in self.active_workflows:
                self.logger.error(f"Workflow {workflow_id} not found")
                return False
            
            workflow = self.active_workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.PENDING:
                self.logger.error(f"Workflow {workflow_id} already started or completed")
                return False
            
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now(timezone.utc)
            
            # Submit first task (tasks without dependencies)
            for task in workflow.tasks:
                if not task.dependencies:
                    self.active_tasks[task.id] = task
                    await self.task_queue.put(task)
            
            # Update in Redis
            await self.redis_service.set_state(f"workflow:{workflow_id}", workflow.__dict__)
            
            self.logger.info(f"Workflow started: {workflow_id}")
            
            # Emit workflow start event
            await self._emit_event('workflow_started', {'workflow_id': workflow_id})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start workflow {workflow_id}: {str(e)}")
            return False
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Workflow]:
        """Get current status of a workflow."""
        return self.active_workflows.get(workflow_id)
    
    # === Communication ===
    
    async def send_message_to_agent(self, agent_name: str, message_type: MessageType,
                                  payload: Dict[str, Any]) -> bool:
        """Send a message to a specific agent."""
        return await self._send_agent_message(agent_name, message_type, payload)
    
    async def broadcast_message(self, message_type: MessageType, payload: Dict[str, Any],
                              agent_filter: Optional[Callable[[AgentInfo], bool]] = None) -> int:
        """
        Broadcast a message to multiple agents.
        
        Args:
            message_type: Type of message to send
            payload: Message payload
            agent_filter: Optional filter function to select agents
            
        Returns:
            Number of agents message was sent to.
        """
        try:
            sent_count = 0
            
            for agent_name, agent_info in self.agents.items():
                # Apply filter if provided
                if agent_filter and not agent_filter(agent_info):
                    continue
                
                # Only send to active agents
                if agent_info.status == AgentStatus.ACTIVE:
                    success = await self._send_agent_message(agent_name, message_type, payload)
                    if success:
                        sent_count += 1
            
            self.logger.info(f"Broadcast message sent to {sent_count} agents")
            return sent_count
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast message: {str(e)}")
            return 0
    
    # === System State Management ===
    
    async def update_system_state(self) -> bool:
        """Update and persist current system state."""
        try:
            # Count active entities
            active_departments = 0
            active_institutions = 0
            total_microservices = 0
            ideas_in_progress = 0
            
            # Get counts from Redis/database (would need to implement queries)
            # For now, using basic counts
            
            self.system_state.active_departments = active_departments
            self.system_state.active_institutions = active_institutions
            self.system_state.total_microservices = total_microservices
            self.system_state.ideas_in_progress = ideas_in_progress
            self.system_state.last_updated = datetime.now(timezone.utc)
            
            # Store in Redis
            await self.redis_service.set_state('system_state', self.system_state.dict())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update system state: {str(e)}")
            return False
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            # Update uptime
            if self.startup_time:
                self.metrics['system_uptime'] = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
            
            # Get additional metrics
            redis_health = await self.redis_service.get_health_status()
            search_stats = await self.vector_search_service.get_index_statistics()
            
            # Prepare agents data for API
            agents_data = {}
            for agent_name, agent_info in self.agents.items():
                agents_data[agent_name] = {
                    'agent_type': agent_info.agent_type,
                    'status': agent_info.status.value,
                    'last_heartbeat': agent_info.last_heartbeat.isoformat() if agent_info.last_heartbeat else None,
                    'capabilities': agent_info.capabilities,
                    'current_task': agent_info.current_task,
                    'error_count': agent_info.error_count,
                    'total_tasks_completed': agent_info.total_tasks_completed,
                    'average_task_time': agent_info.average_task_time,
                    'metadata': agent_info.metadata
                }
            
            return {
                'orchestrator_metrics': self.metrics,
                'system_state': self.system_state.dict(),
                'agent_count': len(self.agents),
                'active_agents': len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]),
                'agents': agents_data,
                'active_tasks': len(self.active_tasks),
                'active_workflows': len(self.active_workflows),
                'redis_health': redis_health,
                'search_index_stats': search_stats.dict() if search_stats else None,
                'uptime_seconds': self.metrics['system_uptime']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
            return {}
    
    # === Event System ===
    
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        """Register an event handler for specific event types."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
        self.logger.debug(f"Event handler registered for: {event_type}")
    
    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event to all registered handlers."""
        try:
            if event_type in self._event_handlers:
                for handler in self._event_handlers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event_data)
                        else:
                            handler(event_data)
                    except Exception as e:
                        self.logger.error(f"Error in event handler for {event_type}: {str(e)}")
            
            # Also store event in Redis for debugging/monitoring
            event_record = {
                'type': event_type,
                'data': event_data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis_service.cache_set(
                f"event:{uuid4()}",
                event_record,
                ttl_seconds=3600  # Keep events for 1 hour
            )
            
        except Exception as e:
            self.logger.error(f"Failed to emit event {event_type}: {str(e)}")
    
    # === Internal Methods ===
    
    async def _setup_agent_communication(self):
        """Set up Redis pub/sub for agent communication."""
        # Subscribe to orchestrator control channel
        await self.redis_service.subscribe_to_channel(
            "orchestrator:control",
            self._handle_control_message
        )
        
        # Subscribe to task completion notifications
        await self.redis_service.subscribe_to_channel(
            "orchestrator:task_complete",
            self._handle_task_completion
        )
    
    async def _handle_agent_message(self, channel: str, message: Dict[str, Any]):
        """Handle incoming messages from agents."""
        try:
            agent_name = channel.split(":")[1]
            message_type = message.get('type', 'unknown')
            
            if message_type == 'heartbeat':
                await self._handle_agent_heartbeat(agent_name, message)
            elif message_type == 'task_update':
                await self._handle_task_update(message)
            elif message_type == 'error':
                await self._handle_agent_error(agent_name, message)
            else:
                self.logger.warning(f"Unknown message type from {agent_name}: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling agent message: {str(e)}")
    
    async def _handle_control_message(self, channel: str, message: Dict[str, Any]):
        """Handle control messages sent to orchestrator."""
        try:
            command = message.get('command')
            
            if command == 'shutdown':
                self.logger.info("Received shutdown command")
                await self.shutdown()
            elif command == 'health_check':
                # Respond with health status
                health_data = await self.get_system_metrics()
                await self.redis_service.publish_message(
                    "orchestrator:health_response",
                    health_data
                )
            else:
                self.logger.warning(f"Unknown control command: {command}")
                
        except Exception as e:
            self.logger.error(f"Error handling control message: {str(e)}")
    
    async def _handle_agent_heartbeat(self, agent_name: str, message: Dict[str, Any]):
        """Handle agent heartbeat messages."""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            agent.last_heartbeat = datetime.now(timezone.utc)
            agent.status = AgentStatus.ACTIVE
            
            # Update any additional info from heartbeat
            if 'current_task' in message:
                agent.current_task = message['current_task']
            
            # Store updated info
            await self.redis_service.set_state(f"agent:{agent_name}", agent.__dict__)
    
    async def _handle_task_update(self, message: Dict[str, Any]):
        """Handle task status updates from agents."""
        try:
            task_id = message.get('task_id')
            status = message.get('status')
            
            if task_id not in self.active_tasks:
                self.logger.warning(f"Received update for unknown task: {task_id}")
                return
            
            task = self.active_tasks[task_id]
            old_status = task.status
            
            # Update task status
            if status == 'started':
                task.status = WorkflowStatus.RUNNING
                task.started_at = datetime.now(timezone.utc)
            elif status == 'completed':
                task.status = WorkflowStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc)
                task.result = message.get('result')
                
                # Update agent metrics
                if task.agent_name in self.agents:
                    agent = self.agents[task.agent_name]
                    agent.total_tasks_completed += 1
                    if task.started_at:
                        task_time = (task.completed_at - task.started_at).total_seconds()
                        # Update running average
                        total_tasks = agent.total_tasks_completed
                        agent.average_task_time = ((agent.average_task_time * (total_tasks - 1)) + task_time) / total_tasks
                
                # Check if this completes a workflow step
                await self._check_workflow_progress(task)
                
            elif status == 'failed':
                task.status = WorkflowStatus.FAILED
                task.completed_at = datetime.now(timezone.utc)
                task.error = message.get('error', 'Task failed')
                
                # Handle retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = WorkflowStatus.PENDING
                    task.started_at = None
                    await self.task_queue.put(task)
                    self.logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
            
            # Update metrics
            if task.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                self.metrics['total_tasks_processed'] += 1
                self.metrics['active_task_count'] = len([t for t in self.active_tasks.values() 
                                                       if t.status == WorkflowStatus.RUNNING])
            
            context = LogContext(
                agent_name=task.agent_name,
                additional_context={
                    'task_id': task_id,
                    'old_status': old_status,
                    'new_status': task.status
                }
            )
            self.logger.info(f"Task status updated: {task_id}", context)
            
        except Exception as e:
            self.logger.error(f"Error handling task update: {str(e)}")
    
    async def _handle_agent_error(self, agent_name: str, message: Dict[str, Any]):
        """Handle error reports from agents."""
        error_message = message.get('error', 'Unknown error')
        await self.update_agent_status(agent_name, AgentStatus.ERROR, error_message)
    
    async def _handle_task_completion(self, channel: str, message: Dict[str, Any]):
        """Handle task completion notifications."""
        await self._handle_task_update(message)
    
    async def _send_agent_message(self, agent_name: str, message_type: MessageType, 
                                payload: Dict[str, Any]) -> bool:
        """Send a message to an agent via Redis pub/sub."""
        try:
            message = AgentMessage(
                sender_agent="orchestrator",
                receiver_agent=agent_name,
                message_type=message_type,
                payload=payload
            )
            
            return await self.redis_service.send_agent_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {agent_name}: {str(e)}")
            return False
    
    async def _check_workflow_progress(self, completed_task: Task):
        """Check if workflow can progress after task completion."""
        try:
            # Find workflows containing this task
            for workflow in self.active_workflows.values():
                if completed_task.id in [t.id for t in workflow.tasks]:
                    # Check if next tasks can now be started
                    for task in workflow.tasks:
                        if (task.status == WorkflowStatus.PENDING and 
                            all(self.active_tasks.get(dep_id, Task(id="", task_type="", agent_name="", 
                                                                  priority=TaskPriority.NORMAL, payload={}, 
                                                                  created_at=datetime.now(timezone.utc))).status == WorkflowStatus.COMPLETED 
                                for dep_id in task.dependencies)):
                            
                            # Start the next task
                            self.active_tasks[task.id] = task
                            await self.task_queue.put(task)
                            self.logger.info(f"Started next workflow task: {task.id}")
                    
                    # Check if workflow is complete
                    all_tasks_done = all(t.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] 
                                       for t in workflow.tasks)
                    
                    if all_tasks_done:
                        if all(t.status == WorkflowStatus.COMPLETED for t in workflow.tasks):
                            workflow.status = WorkflowStatus.COMPLETED
                            self.metrics['total_workflows_completed'] += 1
                        else:
                            workflow.status = WorkflowStatus.FAILED
                        
                        workflow.completed_at = datetime.now(timezone.utc)
                        
                        # Update in Redis
                        await self.redis_service.set_state(f"workflow:{workflow.id}", workflow.__dict__)
                        
                        # Emit workflow completion event
                        await self._emit_event('workflow_completed', {
                            'workflow_id': workflow.id,
                            'status': workflow.status,
                            'idea_id': workflow.idea_id
                        })
                        
                        self.logger.info(f"Workflow completed: {workflow.id} ({workflow.status})")
                    
                    break
                    
        except Exception as e:
            self.logger.error(f"Error checking workflow progress: {str(e)}")
    
    async def _start_background_tasks(self):
        """Start all background monitoring and processing tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self._task_processor_task = asyncio.create_task(self._task_processor())
        self._health_monitor_task = asyncio.create_task(self._health_monitor())
        self._cleanup_task = asyncio.create_task(self._cleanup_monitor())
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and update status."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                current_time = datetime.now(timezone.utc)
                offline_threshold = timedelta(seconds=self.config.agent_offline_threshold)
                
                for agent_name, agent in self.agents.items():
                    if agent.last_heartbeat and (current_time - agent.last_heartbeat) > offline_threshold:
                        if agent.status != AgentStatus.OFFLINE:
                            await self.update_agent_status(agent_name, AgentStatus.OFFLINE)
                            self.logger.warning(f"Agent {agent_name} marked as offline")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {str(e)}")
    
    async def _task_processor(self):
        """Process tasks from the task queue."""
        while self.is_running:
            try:
                # Get next task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Check if dependencies are met
                dependencies_met = all(
                    self.active_tasks.get(dep_id, Task(id="", task_type="", agent_name="", 
                                                      priority=TaskPriority.NORMAL, payload={}, 
                                                      created_at=datetime.now(timezone.utc))).status == WorkflowStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                
                if not dependencies_met:
                    # Put task back in queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)  # Wait a bit before retrying
                    continue
                
                # Send task to agent
                success = await self._send_agent_message(
                    task.agent_name,
                    MessageType.REQUEST,
                    {
                        'action': 'execute_task',
                        'task_id': task.id,
                        'task_type': task.task_type,
                        'payload': task.payload
                    }
                )
                
                if success:
                    task.status = WorkflowStatus.RUNNING
                    task.started_at = datetime.now(timezone.utc)
                    
                    # Update agent status
                    if task.agent_name in self.agents:
                        agent = self.agents[task.agent_name]
                        agent.status = AgentStatus.BUSY
                        agent.current_task = task.id
                
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task processor: {str(e)}")
    
    async def _health_monitor(self):
        """Monitor overall system health."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Update system state
                await self.update_system_state()
                
                # Check service health
                redis_healthy = (await self.redis_service.get_health_status()).get('is_healthy', False)
                
                if not redis_healthy:
                    self.logger.error("Redis service is unhealthy")
                
                # Update metrics
                active_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE])
                self.metrics['agent_count'] = len(self.agents)
                self.metrics['active_task_count'] = len([t for t in self.active_tasks.values() 
                                                       if t.status == WorkflowStatus.RUNNING])
                
                if active_agents == 0:
                    self.logger.warning("No active agents available")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {str(e)}")
    
    async def _cleanup_monitor(self):
        """Clean up old tasks and workflows."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                current_time = datetime.now(timezone.utc)
                cleanup_threshold = timedelta(hours=24)  # Keep data for 24 hours
                
                # Clean up completed tasks
                tasks_to_remove = []
                for task_id, task in self.active_tasks.items():
                    if (task.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                        task.completed_at and (current_time - task.completed_at) > cleanup_threshold):
                        tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.active_tasks[task_id]
                
                # Clean up completed workflows
                workflows_to_remove = []
                for workflow_id, workflow in self.active_workflows.items():
                    if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                        workflow.completed_at and (current_time - workflow.completed_at) > cleanup_threshold):
                        workflows_to_remove.append(workflow_id)
                
                for workflow_id in workflows_to_remove:
                    del self.active_workflows[workflow_id]
                    await self.redis_service.delete_state(f"workflow:{workflow_id}")
                
                if tasks_to_remove or workflows_to_remove:
                    self.logger.debug(f"Cleaned up {len(tasks_to_remove)} tasks and {len(workflows_to_remove)} workflows")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup monitor: {str(e)}")
    
    async def _load_system_state(self):
        """Load system state from persistent storage."""
        try:
            state_data = await self.redis_service.get_state('system_state')
            if state_data:
                self.system_state = SystemState(**state_data)
            
            # Load agent information
            # This could be extended to restore agent registry from persistence
            
        except Exception as e:
            self.logger.error(f"Error loading system state: {str(e)}")
    
    async def _save_system_state(self):
        """Save system state to persistent storage."""
        try:
            await self.redis_service.set_state('system_state', self.system_state.dict())
            
            # Save agent information
            for agent_name, agent_info in self.agents.items():
                await self.redis_service.set_state(f"agent:{agent_name}", agent_info.__dict__)
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {str(e)}")


# Factory function for easy instantiation
def create_orchestrator(redis_config: Optional[RedisConfig] = None,
                       search_config: Optional[SearchConfig] = None,
                       orchestrator_config: Optional[OrchestratorConfig] = None) -> SystemOrchestrator:
    """
    Factory function to create a system orchestrator instance.
    
    Args:
        redis_config: Redis service configuration
        search_config: Vector search service configuration
        orchestrator_config: Orchestrator configuration
        
    Returns:
        SystemOrchestrator instance
    """
    return SystemOrchestrator(redis_config, search_config, orchestrator_config)


# Export main classes
__all__ = [
    "SystemOrchestrator",
    "OrchestratorConfig",
    "AgentInfo",
    "Task",
    "Workflow",
    "AgentStatus",
    "WorkflowStatus",
    "TaskPriority",
    "create_orchestrator"
]