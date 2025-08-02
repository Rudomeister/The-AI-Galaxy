"""
Base Agent Message Handler for AI-Galaxy Ecosystem.

This module provides the core infrastructure for agents to receive, process,
and respond to task messages from the orchestrator via Redis pub/sub messaging.
All agents should inherit from this base class to gain message processing capabilities.
"""

import asyncio
import json
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from ...shared.logger import get_logger, LogContext
from ...shared.models import AgentMessage, MessageType, MessageStatus
from ...services.redis import RedisService, RedisConfig


class AgentState(str, Enum):
    """Agent operational states."""
    STARTING = "starting"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TaskExecutionResult(BaseModel):
    """Result of task execution by an agent."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentConfiguration(BaseModel):
    """Base configuration for agent message handlers."""
    agent_name: str
    agent_type: str
    capabilities: List[str] = Field(default_factory=list)
    heartbeat_interval: int = Field(default=30, description="Heartbeat interval in seconds")
    task_timeout: int = Field(default=300, description="Default task timeout in seconds")
    max_concurrent_tasks: int = Field(default=3, description="Maximum concurrent tasks")
    auto_register: bool = Field(default=True, description="Auto-register with orchestrator")
    redis_config: Optional[RedisConfig] = None


class BaseAgentHandler(ABC):
    """
    Base class for agent message handlers.
    
    Provides Redis pub/sub message processing infrastructure that allows agents
    to receive task assignments, execute their domain logic, and report completion
    back to the orchestrator.
    """
    
    def __init__(self, config: AgentConfiguration):
        """
        Initialize the base agent handler.
        
        Args:
            config: Agent configuration including name, type, and Redis settings
        """
        self.config = config
        self.logger = get_logger(f"agent.{config.agent_name}")
        
        # Core state
        self.agent_name = config.agent_name
        self.agent_type = config.agent_type
        self.state = AgentState.STARTING
        self.is_running = False
        
        # Task management
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, TaskExecutionResult] = {}
        
        # Redis service
        self.redis_service = RedisService(config.redis_config)
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_listener_task: Optional[asyncio.Task] = None
        
        # Event handlers
        self._task_handlers: Dict[str, Callable] = {}
        
        # Performance metrics
        self.metrics = {
            'tasks_executed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_execution_time': 0.0,
            'uptime_seconds': 0.0,
            'heartbeats_sent': 0
        }
        
        self.startup_time: Optional[datetime] = None
    
    async def initialize(self) -> bool:
        """
        Initialize the agent handler and start message processing.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            self.logger.info(f"Initializing agent handler: {self.agent_name}")
            
            # Initialize Redis service
            redis_success = await self.redis_service.initialize()
            if not redis_success:
                self.logger.error("Failed to initialize Redis service")
                return False
            
            # Subscribe to agent's message channel
            await self.redis_service.subscribe_to_channel(
                f"agent:{self.agent_name}",
                self._handle_message
            )
            
            # Register task handlers
            await self._register_task_handlers()
            
            # Register with orchestrator if auto-register is enabled
            if self.config.auto_register:
                await self._register_with_orchestrator()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Update state
            self.state = AgentState.ACTIVE
            self.is_running = True
            self.startup_time = datetime.now(timezone.utc)
            
            self.logger.info(f"Agent handler initialized successfully: {self.agent_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent handler: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = AgentState.ERROR
            return False
    
    async def shutdown(self):
        """Gracefully shutdown the agent handler."""
        self.logger.info(f"Shutting down agent handler: {self.agent_name}")
        
        self.is_running = False
        self.state = AgentState.SHUTDOWN
        
        # Cancel active tasks
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cancel background tasks
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._message_listener_task and not self._message_listener_task.done():
            self._message_listener_task.cancel()
            try:
                await self._message_listener_task
            except asyncio.CancelledError:
                pass
        
        # Unregister from orchestrator
        await self._unregister_from_orchestrator()
        
        # Shutdown Redis service
        await self.redis_service.shutdown()
        
        self.logger.info(f"Agent handler shutdown completed: {self.agent_name}")
    
    # === Abstract Methods for Subclasses ===
    
    @abstractmethod
    async def execute_task(self, task_type: str, payload: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute a specific task type with given payload.
        
        This method must be implemented by each agent to handle their specific
        task types and execute their domain logic.
        
        Args:
            task_type: Type of task to execute
            payload: Task data and parameters
            
        Returns:
            TaskExecutionResult with success status and results
        """
        pass
    
    def get_supported_task_types(self) -> List[str]:
        """
        Get list of task types this agent can handle.
        
        Returns:
            List of supported task type strings
        """
        return []
    
    async def on_startup(self):
        """
        Hook called after successful initialization.
        Override to add custom startup logic.
        """
        pass
    
    async def on_shutdown(self):
        """
        Hook called before shutdown.
        Override to add custom cleanup logic.
        """
        pass
    
    # === Message Handling ===
    
    async def _handle_message(self, channel: str, message: Dict[str, Any]):
        """Handle incoming messages from Redis pub/sub."""
        try:
            message_type = message.get('type', 'unknown')
            
            if message_type == 'request':
                await self._handle_task_request(message)
            elif message_type == 'notification':
                await self._handle_notification(message)
            elif message_type == 'heartbeat_request':
                await self._handle_heartbeat_request(message)
            else:
                self.logger.warning(f"Unknown message type received: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_task_request(self, message: Dict[str, Any]):
        """Handle task execution requests from orchestrator."""
        try:
            # Extract message data
            message_id = message.get('message_id')
            sender = message.get('sender', 'unknown')
            
            # Get full message from Redis
            if message_id:
                full_message = await self.redis_service.get_agent_message(message_id)
                if full_message:
                    payload = full_message.payload
                else:
                    self.logger.error(f"Could not retrieve full message: {message_id}")
                    return
            else:
                payload = message
            
            action = payload.get('action')
            if action == 'execute_task':
                await self._execute_task_async(payload)
            else:
                self.logger.warning(f"Unknown action in task request: {action}")
                
        except Exception as e:
            self.logger.error(f"Error handling task request: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_notification(self, message: Dict[str, Any]):
        """Handle notification messages."""
        try:
            action = message.get('action')
            
            if action == 'cancel_task':
                task_id = message.get('task_id')
                await self._cancel_task(task_id)
            elif action == 'shutdown':
                self.logger.info("Received shutdown notification")
                await self.shutdown()
            else:
                self.logger.debug(f"Received notification: {action}")
                
        except Exception as e:
            self.logger.error(f"Error handling notification: {str(e)}")
    
    async def _handle_heartbeat_request(self, message: Dict[str, Any]):
        """Handle heartbeat requests."""
        await self._send_heartbeat()
    
    # === Task Execution ===
    
    async def _execute_task_async(self, payload: Dict[str, Any]):
        """Execute task asynchronously and manage task lifecycle."""
        task_id = payload.get('task_id')
        task_type = payload.get('task_type')
        task_payload = payload.get('payload', {})
        
        if not task_id or not task_type:
            self.logger.error("Invalid task request: missing task_id or task_type")
            return
        
        # Check if we can handle this task type
        supported_types = self.get_supported_task_types()
        if supported_types and task_type not in supported_types:
            error_msg = f"Unsupported task type: {task_type}. Supported: {supported_types}"
            await self._report_task_failure(task_id, error_msg)
            return
        
        # Check concurrent task limit
        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
            error_msg = f"Maximum concurrent tasks limit reached: {self.config.max_concurrent_tasks}"
            await self._report_task_failure(task_id, error_msg)
            return
        
        # Create and start task
        task = asyncio.create_task(self._execute_task_with_timeout(task_id, task_type, task_payload))
        self.active_tasks[task_id] = task
        
        # Update state to busy if this is our first active task
        if len(self.active_tasks) == 1:
            self.state = AgentState.BUSY
        
        # Report task started
        await self._report_task_started(task_id)
        
        context = LogContext(
            agent_name=self.agent_name,
            additional_context={
                'task_id': task_id,
                'task_type': task_type,
                'active_tasks': len(self.active_tasks)
            }
        )
        self.logger.info(f"Started task execution: {task_type}", context)
    
    async def _execute_task_with_timeout(self, task_id: str, task_type: str, payload: Dict[str, Any]):
        """Execute task with timeout and error handling."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Execute the actual task
            result = await asyncio.wait_for(
                self.execute_task(task_type, payload),
                timeout=self.config.task_timeout
            )
            
            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Store result and update metrics
            self.task_results[task_id] = result
            await self._update_task_metrics(result, execution_time)
            
            # Report completion
            if result.success:
                await self._report_task_completed(task_id, result)
                self.metrics['tasks_completed'] += 1
            else:
                await self._report_task_failure(task_id, result.error or "Task execution failed")
                self.metrics['tasks_failed'] += 1
            
        except asyncio.TimeoutError:
            error_msg = f"Task timed out after {self.config.task_timeout} seconds"
            await self._report_task_failure(task_id, error_msg)
            self.metrics['tasks_failed'] += 1
            self.logger.warning(f"Task timeout: {task_id}")
            
        except asyncio.CancelledError:
            self.logger.info(f"Task cancelled: {task_id}")
            await self._report_task_failure(task_id, "Task was cancelled")
            self.metrics['tasks_failed'] += 1
            
        except Exception as e:
            error_msg = f"Task execution error: {str(e)}"
            await self._report_task_failure(task_id, error_msg)
            self.metrics['tasks_failed'] += 1
            self.logger.error(f"Task execution error: {task_id} - {str(e)}")
            self.logger.error(traceback.format_exc())
            
        finally:
            # Clean up task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Update state if no more active tasks
            if len(self.active_tasks) == 0:
                self.state = AgentState.ACTIVE
            
            self.metrics['tasks_executed'] += 1
    
    async def _cancel_task(self, task_id: str):
        """Cancel a running task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if not task.done():
                task.cancel()
                self.logger.info(f"Task cancelled: {task_id}")
    
    # === Communication with Orchestrator ===
    
    async def _report_task_started(self, task_id: str):
        """Report task start to orchestrator."""
        await self._send_task_update(task_id, 'started')
    
    async def _report_task_completed(self, task_id: str, result: TaskExecutionResult):
        """Report task completion to orchestrator."""
        await self._send_task_update(task_id, 'completed', result.result)
    
    async def _report_task_failure(self, task_id: str, error: str):
        """Report task failure to orchestrator."""
        await self._send_task_update(task_id, 'failed', error=error)
    
    async def _send_task_update(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Send task status update to orchestrator."""
        try:
            update_message = {
                'type': 'task_update',
                'task_id': task_id,
                'status': status,
                'agent_name': self.agent_name,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if result is not None:
                update_message['result'] = result
            
            if error is not None:
                update_message['error'] = error
            
            await self.redis_service.publish_message(
                "orchestrator:task_complete",
                update_message
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send task update: {str(e)}")
    
    async def _send_heartbeat(self):
        """Send heartbeat to orchestrator."""
        try:
            heartbeat_data = {
                'type': 'heartbeat',
                'agent_name': self.agent_name,
                'agent_type': self.agent_type,
                'state': self.state.value,
                'active_tasks': len(self.active_tasks),
                'current_task': list(self.active_tasks.keys())[0] if self.active_tasks else None,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': self.metrics
            }
            
            await self.redis_service.publish_message(
                f"agent:{self.agent_name}",
                heartbeat_data
            )
            
            self.metrics['heartbeats_sent'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat: {str(e)}")
    
    async def _register_with_orchestrator(self):
        """Register agent with the orchestrator."""
        try:
            registration_message = {
                'type': 'agent_registration',
                'agent_name': self.agent_name,
                'agent_type': self.agent_type,
                'capabilities': self.config.capabilities,
                'supported_task_types': self.get_supported_task_types(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis_service.publish_message(
                "orchestrator:control",
                registration_message
            )
            
            self.logger.info(f"Registered with orchestrator: {self.agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register with orchestrator: {str(e)}")
    
    async def _unregister_from_orchestrator(self):
        """Unregister agent from the orchestrator."""
        try:
            unregistration_message = {
                'type': 'agent_unregistration',
                'agent_name': self.agent_name,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis_service.publish_message(
                "orchestrator:control",
                unregistration_message
            )
            
            self.logger.info(f"Unregistered from orchestrator: {self.agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to unregister from orchestrator: {str(e)}")
    
    # === Background Tasks ===
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Call startup hook
        await self.on_startup()
    
    async def _heartbeat_loop(self):
        """Send regular heartbeats to orchestrator."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {str(e)}")
    
    # === Helper Methods ===
    
    async def _register_task_handlers(self):
        """Register task type handlers. Override in subclasses if needed."""
        pass
    
    async def _update_task_metrics(self, result: TaskExecutionResult, execution_time: float):
        """Update performance metrics."""
        # Update average execution time
        total_tasks = self.metrics['tasks_executed'] + 1
        current_avg = self.metrics['average_execution_time']
        new_avg = ((current_avg * (total_tasks - 1)) + execution_time) / total_tasks
        self.metrics['average_execution_time'] = new_avg
        
        # Update uptime
        if self.startup_time:
            self.metrics['uptime_seconds'] = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent performance metrics."""
        total_tasks = self.metrics['tasks_executed']
        success_rate = (self.metrics['tasks_completed'] / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            **self.metrics,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'active_tasks_count': len(self.active_tasks),
            'success_rate_percent': round(success_rate, 2),
            'configuration': self.config.dict(),
            'startup_time': self.startup_time.isoformat() if self.startup_time else None
        }


# Factory function for creating agent configurations
def create_agent_config(agent_name: str, agent_type: str, 
                       capabilities: List[str] = None,
                       redis_config: Optional[RedisConfig] = None,
                       **kwargs) -> AgentConfiguration:
    """
    Create agent configuration with sensible defaults.
    
    Args:
        agent_name: Unique name for the agent
        agent_type: Type/category of the agent
        capabilities: List of agent capabilities
        redis_config: Redis configuration
        **kwargs: Additional configuration options
        
    Returns:
        AgentConfiguration instance
    """
    return AgentConfiguration(
        agent_name=agent_name,
        agent_type=agent_type,
        capabilities=capabilities or [],
        redis_config=redis_config,
        **kwargs
    )


# Export main classes
__all__ = [
    "BaseAgentHandler",
    "AgentConfiguration", 
    "TaskExecutionResult",
    "AgentState",
    "create_agent_config"
]