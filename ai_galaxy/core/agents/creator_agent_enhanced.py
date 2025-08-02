"""
Enhanced Creator Agent with Message Processing Infrastructure.

This module implements the Creator Agent with complete message processing
capabilities, enabling it to receive tasks from the orchestrator, execute
creator logic, and report results back via Redis pub/sub messaging.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from .base_agent_handler import BaseAgentHandler, AgentConfiguration, TaskExecutionResult
from .creator_agent import CreatorAgent, CreatorConfiguration, ProjectTemplate, DepartmentRoutingDecision, ProjectType, TechnologyStack, ProjectComplexity, DepartmentType
from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext
from ...services.redis import RedisConfig


class EnhancedCreatorAgent(BaseAgentHandler):
    """
    Enhanced Creator Agent with message processing infrastructure.
    
    Combines the domain logic from CreatorAgent with the message handling
    capabilities from BaseAgentHandler to create a fully functional agent
    that can integrate with the orchestrator.
    """
    
    def __init__(self, 
                 creator_config: Optional[CreatorConfiguration] = None,
                 agent_config: Optional[AgentConfiguration] = None,
                 redis_config: Optional[RedisConfig] = None):
        """
        Initialize the enhanced creator agent.
        
        Args:
            creator_config: Configuration for creator logic
            agent_config: Configuration for agent message handling
            redis_config: Redis connection configuration
        """
        # Set up agent configuration
        if agent_config is None:
            agent_config = AgentConfiguration(
                agent_name="creator_agent",
                agent_type="creator",
                capabilities=[
                    "template_creation",
                    "technology_selection",
                    "project_scaffolding",
                    "implementation_planning",
                    "department_routing",
                    "resource_estimation"
                ],
                redis_config=redis_config
            )
        
        # Initialize base agent handler
        super().__init__(agent_config)
        
        # Initialize creator domain logic
        self.creator = CreatorAgent(creator_config)
        
        # Additional creator-specific state
        self.template_cache: Dict[str, ProjectTemplate] = {}
        self.routing_cache: Dict[str, DepartmentRoutingDecision] = {}
        
        self.logger.info("Enhanced Creator Agent initialized")
    
    def get_supported_task_types(self) -> List[str]:
        """Get list of task types this agent can handle."""
        return [
            "create_project_template",
            "generate_project_scaffold",
            "get_template_summary",
            "get_department_routing",
            "get_creator_metrics",
            "update_idea_status",
            "analyze_idea_requirements",
            "select_technology_stack"
        ]
    
    async def execute_task(self, task_type: str, payload: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute creator-specific tasks.
        
        Args:
            task_type: Type of creator task to execute
            payload: Task data and parameters
            
        Returns:
            TaskExecutionResult with creator results
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
            
            self.logger.info(f"Executing creator task: {task_type}", context)
            
            if task_type == "create_project_template":
                result = await self._handle_create_project_template(payload)
            elif task_type == "generate_project_scaffold":
                result = await self._handle_generate_project_scaffold(payload)
            elif task_type == "get_template_summary":
                result = await self._handle_get_template_summary(payload)
            elif task_type == "get_department_routing":
                result = await self._handle_get_department_routing(payload)
            elif task_type == "get_creator_metrics":
                result = await self._handle_get_creator_metrics(payload)
            elif task_type == "update_idea_status":
                result = await self._handle_update_idea_status(payload)
            elif task_type == "analyze_idea_requirements":
                result = await self._handle_analyze_idea_requirements(payload)
            elif task_type == "select_technology_stack":
                result = await self._handle_select_technology_stack(payload)
            else:
                return TaskExecutionResult(
                    success=False,
                    error=f"Unknown task type: {task_type}"
                )
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.execution_time = execution_time
            
            self.logger.info(f"Creator task completed: {task_type}", context)
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"Task execution failed: {str(e)}"
            
            self.logger.error(f"Creator task failed: {task_type} - {error_msg}", context, exc_info=True)
            
            return TaskExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _handle_create_project_template(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle project template creation task."""
        try:
            # Extract idea data from payload
            idea_data = payload.get('idea')
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
            
            # Create project template
            template = self.creator.create_project_template(idea)
            
            # Cache the template
            self.template_cache[str(idea.id)] = template
            
            # Get department routing decision
            routing_decision = self.creator.get_department_routing_decision(str(idea.id))
            if routing_decision:
                self.routing_cache[str(idea.id)] = routing_decision
            
            # Update idea status if requested
            update_status = payload.get('update_status', True)
            status_updated = False
            if update_status:
                status_updated = self.creator.update_idea_status(idea, template)
                self.logger.info(f"Idea {idea.id} status update: {status_updated}")
            
            # Prepare result
            result_data = {
                'template': template.dict(),
                'idea_id': str(idea.id),
                'template_id': template.template_id,
                'project_name': template.project_name,
                'project_type': template.project_type.value,
                'technology_stack': template.technology_stack.value,
                'complexity': template.complexity.value,
                'recommended_department': template.recommended_department.value,
                'recommended_institution': template.recommended_institution,
                'estimated_completion_date': template.estimated_completion_date.isoformat(),
                'implementation_phases': len(template.implementation_phases),
                'priority_score': template.priority_score,
                'status_updated': status_updated,
                'success_metrics': template.success_metrics,
                'integration_points': template.integration_points,
                'resource_requirements': template.resource_requirements.dict(),
                'testing_strategy': template.testing_strategy.dict(),
                'deployment_config': template.deployment_config.dict()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'creation_timestamp': template.creation_timestamp.isoformat(),
                    'phases_count': len(template.implementation_phases),
                    'estimated_weeks': template.resource_requirements.timeline_weeks
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Template creation failed: {str(e)}"
            )
    
    async def _handle_generate_project_scaffold(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle project scaffold generation task."""
        try:
            # Extract template and output path from payload
            template_data = payload.get('template')
            output_path = payload.get('output_path')
            
            if not template_data:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'template' in payload"
                )
            
            if not output_path:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'output_path' in payload"
                )
            
            # Convert to ProjectTemplate object
            if isinstance(template_data, dict):
                template = ProjectTemplate(**template_data)
            else:
                template = template_data
            
            # Generate project scaffold
            success = self.creator.generate_project_scaffold(template, output_path)
            
            # Count generated files
            files_created = len(template.initial_files) + len(template.configuration_files)
            
            result_data = {
                'scaffold_generated': success,
                'template_id': template.template_id,
                'project_name': template.project_name,
                'output_path': output_path,
                'files_created': files_created,
                'directories_created': len(template.directory_structure.get('subdirectories', [])),
                'generation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=success,
                result=result_data,
                metadata={
                    'template_id': template.template_id,
                    'output_location': output_path
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Scaffold generation failed: {str(e)}"
            )
    
    async def _handle_get_template_summary(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get template summary task."""
        try:
            idea_id = payload.get('idea_id')
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get summary from creator
            summary = self.creator.get_template_summary(idea_id)
            
            if summary is None:
                return TaskExecutionResult(
                    success=False,
                    error=f"No template summary found for idea: {idea_id}"
                )
            
            # Add cache information
            cached_template = self.template_cache.get(idea_id)
            cached_routing = self.routing_cache.get(idea_id)
            
            if cached_template:
                summary['cached_template'] = {
                    'template_id': cached_template.template_id,
                    'creation_timestamp': cached_template.creation_timestamp.isoformat(),
                    'project_type': cached_template.project_type.value
                }
            
            if cached_routing:
                summary['cached_routing'] = {
                    'department': cached_routing.recommended_department.value,
                    'confidence': cached_routing.confidence_score,
                    'reasoning': cached_routing.reasoning
                }
            
            return TaskExecutionResult(
                success=True,
                result={'template_summary': summary}
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Failed to get template summary: {str(e)}"
            )
    
    async def _handle_get_department_routing(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get department routing decision task."""
        try:
            idea_id = payload.get('idea_id')
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get routing decision from creator
            routing_decision = self.creator.get_department_routing_decision(idea_id)
            
            if routing_decision is None:
                return TaskExecutionResult(
                    success=False,
                    error=f"No routing decision found for idea: {idea_id}"
                )
            
            result_data = {
                'routing_decision': routing_decision.dict(),
                'idea_id': idea_id,
                'recommended_department': routing_decision.recommended_department.value,
                'recommended_institution': routing_decision.recommended_institution,
                'confidence_score': routing_decision.confidence_score,
                'reasoning': routing_decision.reasoning,
                'alternative_departments': [dept.value for dept in routing_decision.alternative_departments],
                'new_institution_needed': routing_decision.new_institution_needed,
                'proposed_institution_name': routing_decision.proposed_institution_name
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Failed to get department routing: {str(e)}"
            )
    
    async def _handle_get_creator_metrics(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get creator metrics task."""
        try:
            # Get creator metrics
            creator_metrics = self.creator.get_creator_metrics()
            
            # Get agent metrics
            agent_metrics = self.get_agent_metrics()
            
            # Add enhanced metrics
            enhanced_metrics = {
                'creator_metrics': creator_metrics,
                'agent_metrics': agent_metrics,
                'cache_metrics': {
                    'cached_templates': len(self.template_cache),
                    'cached_routing_decisions': len(self.routing_cache)
                },
                'recent_activity': {
                    'last_template_creation': max(
                        [template.creation_timestamp for template in self.template_cache.values()],
                        default=datetime.min.replace(tzinfo=timezone.utc)
                    ).isoformat(),
                    'total_cached_templates': len(self.template_cache)
                },
                'technology_distribution': creator_metrics.get('technology_stack_usage', {}),
                'department_distribution': creator_metrics.get('department_distribution', {}),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result={'metrics': enhanced_metrics}
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Failed to get creator metrics: {str(e)}"
            )
    
    async def _handle_update_idea_status(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle idea status update task."""
        try:
            # Extract idea and template data
            idea_data = payload.get('idea')
            template_id = payload.get('template_id')
            
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
            
            # Get template
            template = None
            if template_id and template_id in self.template_cache:
                template = self.template_cache[template_id]
            elif str(idea.id) in self.template_cache:
                template = self.template_cache[str(idea.id)]
            else:
                return TaskExecutionResult(
                    success=False,
                    error=f"No template found for idea: {idea.id}"
                )
            
            # Update status
            success = self.creator.update_idea_status(idea, template)
            
            result_data = {
                'idea_id': str(idea.id),
                'template_id': template.template_id,
                'status_updated': success,
                'new_status': idea.status.value if success else None,
                'update_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=success,
                result=result_data,
                metadata={
                    'template_id': template.template_id
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Status update failed: {str(e)}"
            )
    
    async def _handle_analyze_idea_requirements(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle idea requirements analysis task."""
        try:
            # Extract idea data from payload
            idea_data = payload.get('idea')
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
            
            # Use creator's private method for analysis (accessing via the creator instance)
            analysis = self.creator._analyze_idea_requirements(idea)
            
            result_data = {
                'idea_id': str(idea.id),
                'analysis': {
                    'project_type': analysis['project_type'].value,
                    'complexity': analysis['complexity'].value,
                    'requirements': analysis['requirements'],
                    'scalability_needs': analysis['scalability_needs'],
                    'performance_requirements': analysis['performance_requirements'],
                    'estimated_scope': analysis['estimated_scope'],
                    'integration_complexity': analysis['integration_complexity']
                },
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Requirements analysis failed: {str(e)}"
            )
    
    async def _handle_select_technology_stack(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle technology stack selection task."""
        try:
            # Extract idea and analysis data from payload
            idea_data = payload.get('idea')
            analysis_data = payload.get('analysis')
            
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
            
            # If no analysis provided, generate it
            if not analysis_data:
                analysis = self.creator._analyze_idea_requirements(idea)
            else:
                # Convert analysis data to proper format
                analysis = {
                    'project_type': ProjectType(analysis_data.get('project_type', 'microservice')),
                    'complexity': ProjectComplexity(analysis_data.get('complexity', 'moderate')),
                    'requirements': analysis_data.get('requirements', []),
                    'scalability_needs': analysis_data.get('scalability_needs', 'medium'),
                    'performance_requirements': analysis_data.get('performance_requirements', []),
                    'estimated_scope': analysis_data.get('estimated_scope', 'medium'),
                    'integration_complexity': analysis_data.get('integration_complexity', 'medium')
                }
            
            # Select technology stack using creator's private method
            selected_stack = self.creator._select_technology_stack(idea, analysis)
            
            # Get technology profile information
            tech_profile = self.creator.technology_profiles.get(selected_stack)
            
            result_data = {
                'idea_id': str(idea.id),
                'selected_technology_stack': selected_stack.value,
                'technology_profile': {
                    'name': tech_profile.name if tech_profile else selected_stack.value,
                    'description': tech_profile.description if tech_profile else "",
                    'suitable_for': [pt.value for pt in tech_profile.suitable_for] if tech_profile else [],
                    'complexity_level': tech_profile.complexity_level.value if tech_profile else "moderate",
                    'dependencies': tech_profile.dependencies if tech_profile else [],
                    'setup_time_hours': tech_profile.setup_time_hours if tech_profile else 8,
                    'scalability_score': tech_profile.scalability_score if tech_profile else 7.0,
                    'learning_curve': tech_profile.learning_curve if tech_profile else 5.0,
                    'ecosystem_maturity': tech_profile.ecosystem_maturity if tech_profile else 7.0,
                    'performance_score': tech_profile.performance_score if tech_profile else 7.0
                },
                'selection_reasoning': {
                    'project_type': analysis['project_type'].value,
                    'complexity': analysis['complexity'].value,
                    'scalability_needs': analysis['scalability_needs']
                },
                'selection_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Technology stack selection failed: {str(e)}"
            )
    
    async def on_startup(self):
        """Custom startup logic for creator agent."""
        self.logger.info("Enhanced Creator Agent startup completed")
        
        # Clear caches on startup
        self.template_cache.clear()
        self.routing_cache.clear()
        
        # Log creator configuration
        creator_metrics = self.creator.get_creator_metrics()
        self.logger.info(f"Creator configuration: {creator_metrics.get('configuration', {})}")
    
    async def on_shutdown(self):
        """Custom shutdown logic for creator agent."""
        self.logger.info("Enhanced Creator Agent shutdown starting")
        
        # Save final metrics
        final_metrics = self.get_agent_metrics()
        creator_metrics = self.creator.get_creator_metrics()
        
        self.logger.info(f"Final creator metrics: {creator_metrics}")
        self.logger.info(f"Final agent metrics: {final_metrics}")
        
        # Clear caches
        self.template_cache.clear()
        self.routing_cache.clear()


# Convenience function for creating and starting creator agent
async def create_and_start_creator_agent(
    creator_config: Optional[CreatorConfiguration] = None,
    redis_config: Optional[RedisConfig] = None
) -> EnhancedCreatorAgent:
    """
    Create and start an enhanced creator agent.
    
    Args:
        creator_config: Creator domain logic configuration
        redis_config: Redis connection configuration
        
    Returns:
        Initialized and started EnhancedCreatorAgent
    """
    agent = EnhancedCreatorAgent(creator_config, redis_config=redis_config)
    
    success = await agent.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Enhanced Creator Agent")
    
    return agent


# Export main classes
__all__ = [
    "EnhancedCreatorAgent",
    "create_and_start_creator_agent"
]