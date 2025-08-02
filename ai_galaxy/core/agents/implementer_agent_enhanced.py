"""
Enhanced Implementer Agent with Message Processing Infrastructure.

This module implements the Implementer Agent with complete message processing
capabilities, enabling it to receive tasks from the orchestrator, execute
implementation orchestration logic, and report results back via Redis pub/sub messaging.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from .base_agent_handler import BaseAgentHandler, AgentConfiguration, TaskExecutionResult
from .implementer_agent import ImplementerAgent, ImplementerConfiguration, ImplementationPlan, ImplementationTask, ImplementationStatus, ImplementationPhase, Resource, ImplementationRisk, ProgressCheckpoint, ImplementationMetrics
from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext
from ...services.redis import RedisConfig


class EnhancedImplementerAgent(BaseAgentHandler):
    """
    Enhanced Implementer Agent with message processing infrastructure.
    
    Combines the domain logic from ImplementerAgent with the message handling
    capabilities from BaseAgentHandler to create a fully functional agent
    that can integrate with the orchestrator.
    """
    
    def __init__(self, 
                 implementer_config: Optional[ImplementerConfiguration] = None,
                 agent_config: Optional[AgentConfiguration] = None,
                 redis_config: Optional[RedisConfig] = None):
        """
        Initialize the enhanced implementer agent.
        
        Args:
            implementer_config: Configuration for implementer logic
            agent_config: Configuration for agent message handling
            redis_config: Redis connection configuration
        """
        # Set up agent configuration
        if agent_config is None:
            agent_config = AgentConfiguration(
                agent_name="implementer_agent",
                agent_type="implementer",
                capabilities=[
                    "implementation_orchestration",
                    "resource_allocation",
                    "environment_setup",
                    "development_coordination",
                    "progress_monitoring",
                    "risk_management",
                    "task_breakdown",
                    "agent_coordination"
                ],
                redis_config=redis_config
            )
        
        # Initialize base agent handler
        super().__init__(agent_config)
        
        # Initialize implementer domain logic
        self.implementer = ImplementerAgent(implementer_config)
        
        # Additional implementer-specific state
        self.orchestration_cache: Dict[str, ImplementationPlan] = {}
        self.progress_tracking: Dict[str, List[ProgressCheckpoint]] = {}
        
        self.logger.info("Enhanced Implementer Agent initialized")
    
    def get_supported_task_types(self) -> List[str]:
        """Get list of task types this agent can handle."""
        return [
            "orchestrate_implementation",
            "update_task_status",
            "monitor_progress",
            "allocate_resources",
            "assess_risks",
            "coordinate_agents",
            "get_implementation_status",
            "get_implementer_metrics",
            "handle_blocking_issues",
            "optimize_resource_allocation"
        ]
    
    async def execute_task(self, task_type: str, payload: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute implementer-specific tasks.
        
        Args:
            task_type: Type of implementer task to execute
            payload: Task data and parameters
            
        Returns:
            TaskExecutionResult with implementation orchestration results
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            context = LogContext(
                agent_name=self.agent_name,
                additional_context={
                    'task_type': task_type,
                    'payload_keys': list(payload.keys())
                }
            )
            
            self.logger.info(f"Executing implementer task: {task_type}", context)
            
            if task_type == "orchestrate_implementation":
                result = await self._handle_orchestrate_implementation(payload)
            elif task_type == "update_task_status":
                result = await self._handle_update_task_status(payload)
            elif task_type == "monitor_progress":
                result = await self._handle_monitor_progress(payload)
            elif task_type == "allocate_resources":
                result = await self._handle_allocate_resources(payload)
            elif task_type == "assess_risks":
                result = await self._handle_assess_risks(payload)
            elif task_type == "coordinate_agents":
                result = await self._handle_coordinate_agents(payload)
            elif task_type == "get_implementation_status":
                result = await self._handle_get_implementation_status(payload)
            elif task_type == "get_implementer_metrics":
                result = await self._handle_get_implementer_metrics(payload)
            elif task_type == "handle_blocking_issues":
                result = await self._handle_blocking_issues(payload)
            elif task_type == "optimize_resource_allocation":
                result = await self._handle_optimize_resource_allocation(payload)
            else:
                return TaskExecutionResult(
                    success=False,
                    error=f"Unknown task type: {task_type}"
                )
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.execution_time = execution_time
            
            self.logger.info(f"Implementer task completed: {task_type}", context)
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"Task execution failed: {str(e)}"
            
            self.logger.error(f"Implementer task failed: {task_type} - {error_msg}", context, exc_info=True)
            
            return TaskExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _handle_orchestrate_implementation(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle implementation orchestration task."""
        try:
            # Extract idea and template data from payload
            idea_data = payload.get('idea')
            template_data = payload.get('template_data', {})
            
            if not idea_data:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea' in payload"
                )
            
            # Convert to Idea object
            if isinstance(idea_data, dict):
                idea = Idea(**idea_data)
            else:
                idea = idea_data
            
            # Orchestrate implementation
            implementation_plan = await self.implementer.orchestrate_implementation(idea, template_data)
            
            # Cache the implementation plan
            self.orchestration_cache[str(idea.id)] = implementation_plan
            
            # Prepare result
            result_data = {
                'implementation_plan': implementation_plan.dict(),
                'idea_id': str(idea.id),
                'plan_id': implementation_plan.plan_id,
                'project_name': implementation_plan.project_name,
                'estimated_duration_hours': implementation_plan.estimated_duration_hours,
                'target_completion_date': implementation_plan.target_completion_date.isoformat(),
                'phases_count': len(implementation_plan.phases),
                'tasks_count': len(implementation_plan.tasks),
                'allocated_resources_count': len(implementation_plan.allocated_resources),
                'identified_risks_count': len(implementation_plan.identified_risks),
                'quality_gates_count': len(implementation_plan.quality_gates),
                'success_criteria': implementation_plan.success_criteria,
                'environment_config': implementation_plan.environment_config.dict() if implementation_plan.environment_config else None
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'orchestration_timestamp': datetime.now(timezone.utc).isoformat(),
                    'implementation_complexity': 'high' if implementation_plan.estimated_duration_hours > 200 else 'medium'
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Implementation orchestration failed: {str(e)}"
            )
    
    async def _handle_update_task_status(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle task status update."""
        try:
            # Extract required fields
            idea_id = payload.get('idea_id')
            task_id = payload.get('task_id')
            new_status = payload.get('new_status')
            completion_percentage = payload.get('completion_percentage', 0.0)
            notes = payload.get('notes')
            
            if not all([idea_id, task_id, new_status]):
                return TaskExecutionResult(
                    success=False,
                    error="Missing required fields: idea_id, task_id, new_status"
                )
            
            # Convert status string to enum
            try:
                status_enum = ImplementationStatus(new_status)
            except ValueError:
                return TaskExecutionResult(
                    success=False,
                    error=f"Invalid status: {new_status}"
                )
            
            # Update task status
            await self.implementer.update_task_status(
                idea_id, task_id, status_enum, completion_percentage, notes
            )
            
            # Get updated status
            progress_status = self.implementer.get_progress_status(idea_id)
            
            result_data = {
                'idea_id': idea_id,
                'task_id': task_id,
                'updated_status': new_status,
                'completion_percentage': completion_percentage,
                'notes': notes,
                'update_timestamp': datetime.now(timezone.utc).isoformat(),
                'current_progress': progress_status
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Task status update failed: {str(e)}"
            )
    
    async def _handle_monitor_progress(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle progress monitoring task."""
        try:
            idea_id = payload.get('idea_id')
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get progress status
            progress_status = self.implementer.get_progress_status(idea_id)
            
            if not progress_status:
                return TaskExecutionResult(
                    success=False,
                    error=f"No active implementation found for idea: {idea_id}"
                )
            
            # Get detailed metrics
            active_implementations = self.implementer.get_active_implementations()
            implementation_plan = active_implementations.get(idea_id)
            
            # Calculate additional metrics
            if implementation_plan:
                completed_tasks = len([t for t in implementation_plan.tasks if t.status == ImplementationStatus.COMPLETED])
                in_progress_tasks = len([t for t in implementation_plan.tasks if t.status == ImplementationStatus.IN_PROGRESS])
                blocked_tasks = len([t for t in implementation_plan.tasks if t.status == ImplementationStatus.BLOCKED])
                
                # Add to progress tracking cache
                if idea_id not in self.progress_tracking:
                    self.progress_tracking[idea_id] = []
                
                current_checkpoint = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'completion_percentage': progress_status['completion_percentage'],
                    'completed_tasks': completed_tasks,
                    'in_progress_tasks': in_progress_tasks,
                    'blocked_tasks': blocked_tasks,
                    'current_phase': progress_status['current_phase']
                }
                self.progress_tracking[idea_id].append(current_checkpoint)
            
            result_data = {
                'progress_status': progress_status,
                'monitoring_timestamp': datetime.now(timezone.utc).isoformat(),
                'detailed_metrics': {
                    'completed_tasks': completed_tasks if implementation_plan else 0,
                    'in_progress_tasks': in_progress_tasks if implementation_plan else 0,
                    'blocked_tasks': blocked_tasks if implementation_plan else 0,
                    'total_tasks': len(implementation_plan.tasks) if implementation_plan else 0
                }
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Progress monitoring failed: {str(e)}"
            )
    
    async def _handle_allocate_resources(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle resource allocation task."""
        try:
            idea_id = payload.get('idea_id')
            resource_requirements = payload.get('resource_requirements', {})
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get current resource utilization
            resource_utilization = self.implementer.get_resource_utilization()
            
            # Simulate resource allocation logic
            allocation_result = {
                'idea_id': idea_id,
                'requested_resources': resource_requirements,
                'current_utilization': resource_utilization,
                'allocation_status': 'allocated',
                'allocated_resources': [],
                'allocation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # For each resource type, try to allocate
            for resource_type, count in resource_requirements.items():
                if resource_type in resource_utilization:
                    util_info = resource_utilization[resource_type]
                    if util_info['average_utilization'] < 70:  # If utilization below 70%
                        allocation_result['allocated_resources'].append({
                            'type': resource_type,
                            'count': count,
                            'status': 'allocated'
                        })
                    else:
                        allocation_result['allocated_resources'].append({
                            'type': resource_type,
                            'count': 0,
                            'status': 'insufficient_capacity'
                        })
            
            return TaskExecutionResult(
                success=True,
                result=allocation_result
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Resource allocation failed: {str(e)}"
            )
    
    async def _handle_assess_risks(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle risk assessment task."""
        try:
            idea_id = payload.get('idea_id')
            risk_data = payload.get('risk_data', {})
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get active implementation
            active_implementations = self.implementer.get_active_implementations()
            implementation_plan = active_implementations.get(idea_id)
            
            if not implementation_plan:
                return TaskExecutionResult(
                    success=False,
                    error=f"No active implementation found for idea: {idea_id}"
                )
            
            # Analyze current risks
            current_risks = implementation_plan.identified_risks
            risk_summary = {
                'total_risks': len(current_risks),
                'high_risks': len([r for r in current_risks if r.risk_level.value == 'high']),
                'medium_risks': len([r for r in current_risks if r.risk_level.value == 'medium']),
                'low_risks': len([r for r in current_risks if r.risk_level.value == 'low']),
                'active_risks': len([r for r in current_risks if r.status == 'active']),
                'mitigated_risks': len([r for r in current_risks if r.status == 'mitigated'])
            }
            
            # Calculate overall risk score
            total_risk_score = sum(r.risk_score for r in current_risks)
            average_risk_score = total_risk_score / len(current_risks) if current_risks else 0
            
            result_data = {
                'idea_id': idea_id,
                'risk_assessment': {
                    'risk_summary': risk_summary,
                    'total_risk_score': total_risk_score,
                    'average_risk_score': average_risk_score,
                    'risk_level': 'high' if average_risk_score > 0.7 else 'medium' if average_risk_score > 0.4 else 'low'
                },
                'current_risks': [r.dict() for r in current_risks],
                'assessment_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Risk assessment failed: {str(e)}"
            )
    
    async def _handle_coordinate_agents(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle agent coordination task."""
        try:
            idea_id = payload.get('idea_id')
            coordination_type = payload.get('coordination_type', 'general')
            target_agents = payload.get('target_agents', [])
            coordination_data = payload.get('coordination_data', {})
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Simulate agent coordination
            coordination_result = {
                'idea_id': idea_id,
                'coordination_type': coordination_type,
                'target_agents': target_agents,
                'coordination_status': 'initiated',
                'coordination_messages': [],
                'coordination_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Create coordination messages for each target agent
            for agent in target_agents:
                message = {
                    'target_agent': agent,
                    'message_type': coordination_type,
                    'data': coordination_data,
                    'status': 'sent',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                coordination_result['coordination_messages'].append(message)
            
            return TaskExecutionResult(
                success=True,
                result=coordination_result
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Agent coordination failed: {str(e)}"
            )
    
    async def _handle_get_implementation_status(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get implementation status task."""
        try:
            idea_id = payload.get('idea_id')
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get comprehensive status
            progress_status = self.implementer.get_progress_status(idea_id)
            active_implementations = self.implementer.get_active_implementations()
            
            if not progress_status:
                # Check if it's in history
                history = self.implementer.get_implementation_history(limit=100)
                for plan in history:
                    if plan.idea_id == idea_id:
                        result_data = {
                            'idea_id': idea_id,
                            'status': 'completed',
                            'implementation_plan': plan.dict(),
                            'completion_date': plan.updated_at.isoformat(),
                            'final_status': 'historical'
                        }
                        return TaskExecutionResult(success=True, result=result_data)
                
                return TaskExecutionResult(
                    success=False,
                    error=f"No implementation found for idea: {idea_id}"
                )
            
            # Get detailed implementation plan
            implementation_plan = active_implementations.get(idea_id)
            cached_plan = self.orchestration_cache.get(idea_id)
            
            result_data = {
                'idea_id': idea_id,
                'status': 'active',
                'progress_status': progress_status,
                'implementation_plan_summary': {
                    'plan_id': implementation_plan.plan_id if implementation_plan else None,
                    'project_name': implementation_plan.project_name if implementation_plan else None,
                    'phases': [phase.value for phase in implementation_plan.phases] if implementation_plan else [],
                    'tasks_summary': {
                        'total': len(implementation_plan.tasks) if implementation_plan else 0,
                        'completed': len([t for t in implementation_plan.tasks if t.status == ImplementationStatus.COMPLETED]) if implementation_plan else 0,
                        'in_progress': len([t for t in implementation_plan.tasks if t.status == ImplementationStatus.IN_PROGRESS]) if implementation_plan else 0,
                        'blocked': len([t for t in implementation_plan.tasks if t.status == ImplementationStatus.BLOCKED]) if implementation_plan else 0
                    }
                },
                'progress_history': self.progress_tracking.get(idea_id, []),
                'status_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Failed to get implementation status: {str(e)}"
            )
    
    async def _handle_get_implementer_metrics(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get implementer metrics task."""
        try:
            # Get implementer metrics
            implementer_metrics = self.implementer.get_implementation_metrics()
            
            # Get agent metrics
            agent_metrics = self.get_agent_metrics()
            
            # Get resource utilization
            resource_utilization = self.implementer.get_resource_utilization()
            
            # Get active implementations summary
            active_implementations = self.implementer.get_active_implementations()
            active_summary = {
                'total_active': len(active_implementations),
                'by_phase': {},
                'average_completion': 0
            }
            
            if active_implementations:
                phase_counts = {}
                total_completion = 0
                
                for plan in active_implementations.values():
                    progress = self.implementer.get_progress_status(plan.idea_id)
                    if progress:
                        phase = progress['current_phase']
                        phase_counts[phase] = phase_counts.get(phase, 0) + 1
                        total_completion += progress['completion_percentage']
                
                active_summary['by_phase'] = phase_counts
                active_summary['average_completion'] = total_completion / len(active_implementations)
            
            # Enhanced metrics
            enhanced_metrics = {
                'implementer_metrics': implementer_metrics.dict(),
                'agent_metrics': agent_metrics,
                'resource_utilization': resource_utilization,
                'active_implementations_summary': active_summary,
                'cache_metrics': {
                    'orchestration_cache_size': len(self.orchestration_cache),
                    'progress_tracking_entries': len(self.progress_tracking)
                },
                'performance_metrics': {
                    'total_orchestrations': len(self.orchestration_cache),
                    'average_task_execution_time': agent_metrics.get('average_task_time', 0),
                    'success_rate': (implementer_metrics.successful_implementations / max(implementer_metrics.total_implementations, 1)) * 100
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result={'metrics': enhanced_metrics}
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Failed to get implementer metrics: {str(e)}"
            )
    
    async def _handle_blocking_issues(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle blocking issues resolution task."""
        try:
            idea_id = payload.get('idea_id')
            issues = payload.get('issues', [])
            resolution_strategy = payload.get('resolution_strategy', 'escalate')
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get active implementation
            active_implementations = self.implementer.get_active_implementations()
            implementation_plan = active_implementations.get(idea_id)
            
            if not implementation_plan:
                return TaskExecutionResult(
                    success=False,
                    error=f"No active implementation found for idea: {idea_id}"
                )
            
            # Process each blocking issue
            resolution_results = []
            for issue in issues:
                resolution_result = {
                    'issue': issue,
                    'strategy': resolution_strategy,
                    'status': 'processed',
                    'actions_taken': [],
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Simulate resolution actions based on strategy
                if resolution_strategy == 'escalate':
                    resolution_result['actions_taken'].append('Escalated to project manager')
                    resolution_result['actions_taken'].append('Notified stakeholders')
                elif resolution_strategy == 'reallocate':
                    resolution_result['actions_taken'].append('Reallocated resources')
                    resolution_result['actions_taken'].append('Updated task priorities')
                elif resolution_strategy == 'reschedule':
                    resolution_result['actions_taken'].append('Extended timeline')
                    resolution_result['actions_taken'].append('Adjusted milestones')
                
                resolution_results.append(resolution_result)
            
            result_data = {
                'idea_id': idea_id,
                'issues_processed': len(issues),
                'resolution_strategy': resolution_strategy,
                'resolution_results': resolution_results,
                'processing_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Blocking issues handling failed: {str(e)}"
            )
    
    async def _handle_optimize_resource_allocation(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle resource allocation optimization task."""
        try:
            optimization_type = payload.get('optimization_type', 'efficiency')
            target_utilization = payload.get('target_utilization', 75.0)
            
            # Get current resource utilization
            current_utilization = self.implementer.get_resource_utilization()
            
            # Analyze optimization opportunities
            optimization_recommendations = []
            for resource_type, utilization_data in current_utilization.items():
                current_avg = utilization_data['average_utilization']
                
                if current_avg < target_utilization - 10:
                    optimization_recommendations.append({
                        'resource_type': resource_type,
                        'current_utilization': current_avg,
                        'target_utilization': target_utilization,
                        'recommendation': 'increase_allocation',
                        'potential_savings': (target_utilization - current_avg) * utilization_data['resource_count']
                    })
                elif current_avg > target_utilization + 10:
                    optimization_recommendations.append({
                        'resource_type': resource_type,
                        'current_utilization': current_avg,
                        'target_utilization': target_utilization,
                        'recommendation': 'reduce_allocation',
                        'potential_savings': (current_avg - target_utilization) * utilization_data['resource_count']
                    })
            
            # Calculate overall optimization score
            total_deviation = sum(abs(data['average_utilization'] - target_utilization) 
                                for data in current_utilization.values())
            optimization_score = max(0, 100 - total_deviation)
            
            result_data = {
                'optimization_type': optimization_type,
                'target_utilization': target_utilization,
                'current_utilization': current_utilization,
                'optimization_score': optimization_score,
                'recommendations': optimization_recommendations,
                'optimization_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Resource optimization failed: {str(e)}"
            )
    
    async def on_startup(self):
        """Custom startup logic for implementer agent."""
        self.logger.info("Enhanced Implementer Agent startup completed")
        
        # Clear caches on startup
        self.orchestration_cache.clear()
        self.progress_tracking.clear()
        
        # Log implementer configuration
        implementer_metrics = self.implementer.get_implementation_metrics()
        self.logger.info(f"Implementer metrics: {implementer_metrics.dict()}")
    
    async def on_shutdown(self):
        """Custom shutdown logic for implementer agent."""
        self.logger.info("Enhanced Implementer Agent shutdown starting")
        
        # Save final metrics
        final_metrics = self.get_agent_metrics()
        implementer_metrics = self.implementer.get_implementation_metrics()
        
        self.logger.info(f"Final implementer metrics: {implementer_metrics.dict()}")
        self.logger.info(f"Final agent metrics: {final_metrics}")
        
        # Clear caches
        self.orchestration_cache.clear()
        self.progress_tracking.clear()


# Convenience function for creating and starting implementer agent
async def create_and_start_implementer_agent(
    implementer_config: Optional[ImplementerConfiguration] = None,
    redis_config: Optional[RedisConfig] = None
) -> EnhancedImplementerAgent:
    """
    Create and start an enhanced implementer agent.
    
    Args:
        implementer_config: Implementer domain logic configuration
        redis_config: Redis connection configuration
        
    Returns:
        Initialized and started EnhancedImplementerAgent
    """
    agent = EnhancedImplementerAgent(implementer_config, redis_config=redis_config)
    
    success = await agent.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Enhanced Implementer Agent")
    
    return agent


# Export main classes
__all__ = [
    "EnhancedImplementerAgent",
    "create_and_start_implementer_agent"
]