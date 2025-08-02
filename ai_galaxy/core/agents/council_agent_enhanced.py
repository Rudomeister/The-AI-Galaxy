"""
Enhanced Council Agent with Message Processing Infrastructure.

This module implements the Council Agent with complete message processing
capabilities, enabling it to receive tasks from the orchestrator, execute
council logic, and report results back via Redis pub/sub messaging.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from .base_agent_handler import BaseAgentHandler, AgentConfiguration, TaskExecutionResult
from .council_agent import CouncilAgent, CouncilConfiguration, EvaluationReport, AppealRequest, DecisionOutcome
from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext
from ...services.redis import RedisConfig


class EnhancedCouncilAgent(BaseAgentHandler):
    """
    Enhanced Council Agent with message processing infrastructure.
    
    Combines the domain logic from CouncilAgent with the message handling
    capabilities from BaseAgentHandler to create a fully functional agent
    that can integrate with the orchestrator.
    """
    
    def __init__(self, 
                 council_config: Optional[CouncilConfiguration] = None,
                 agent_config: Optional[AgentConfiguration] = None,
                 redis_config: Optional[RedisConfig] = None):
        """
        Initialize the enhanced council agent.
        
        Args:
            council_config: Configuration for council logic
            agent_config: Configuration for agent message handling
            redis_config: Redis connection configuration
        """
        # Set up agent configuration
        if agent_config is None:
            agent_config = AgentConfiguration(
                agent_name="council_agent",
                agent_type="council",
                capabilities=[
                    "idea_evaluation",
                    "feasibility_assessment", 
                    "strategic_alignment",
                    "resource_allocation",
                    "priority_setting",
                    "decision_making"
                ],
                redis_config=redis_config
            )
        
        # Initialize base agent handler
        super().__init__(agent_config)
        
        # Initialize council domain logic
        self.council = CouncilAgent(council_config)
        
        # Additional council-specific state
        self.evaluation_cache: Dict[str, EvaluationReport] = {}
        self.active_appeals: Dict[str, AppealRequest] = {}
        
        self.logger.info("Enhanced Council Agent initialized")
    
    def get_supported_task_types(self) -> List[str]:
        """Get list of task types this agent can handle."""
        return [
            "evaluate_idea",
            "batch_evaluate_ideas",
            "process_appeal",
            "get_evaluation_summary",
            "get_council_metrics",
            "update_idea_status"
        ]
    
    async def execute_task(self, task_type: str, payload: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute council-specific tasks.
        
        Args:
            task_type: Type of council task to execute
            payload: Task data and parameters
            
        Returns:
            TaskExecutionResult with council evaluation results
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
            
            self.logger.info(f"Executing council task: {task_type}", context)
            
            if task_type == "evaluate_idea":
                result = await self._handle_evaluate_idea(payload)
            elif task_type == "batch_evaluate_ideas":
                result = await self._handle_batch_evaluate_ideas(payload)
            elif task_type == "process_appeal":
                result = await self._handle_process_appeal(payload)
            elif task_type == "get_evaluation_summary":
                result = await self._handle_get_evaluation_summary(payload)
            elif task_type == "get_council_metrics":
                result = await self._handle_get_council_metrics(payload)
            elif task_type == "update_idea_status":
                result = await self._handle_update_idea_status(payload)
            else:
                return TaskExecutionResult(
                    success=False,
                    error=f"Unknown task type: {task_type}"
                )
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.execution_time = execution_time
            
            self.logger.info(f"Council task completed: {task_type}", context)
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"Task execution failed: {str(e)}"
            
            self.logger.error(f"Council task failed: {task_type} - {error_msg}", context, exc_info=True)
            
            return TaskExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _handle_evaluate_idea(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle single idea evaluation task."""
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
            
            # Perform council evaluation
            evaluation_report = self.council.evaluate_idea(idea)
            
            # Cache the evaluation result
            self.evaluation_cache[str(idea.id)] = evaluation_report
            
            # Update idea status if requested
            update_status = payload.get('update_status', True)
            status_updated = False
            if update_status:
                status_updated = self.council.update_idea_status(idea, evaluation_report)
                self.logger.info(f"Idea {idea.id} status update: {status_updated}")
            
            # Prepare result
            result_data = {
                'evaluation_report': evaluation_report.dict(),
                'idea_id': str(idea.id),
                'final_decision': evaluation_report.final_decision.value,
                'consensus_score': evaluation_report.consensus_score,
                'strategic_alignment': evaluation_report.strategic_alignment_score,
                'voting_result': evaluation_report.voting_result.value,
                'approval_percentage': evaluation_report.approval_percentage,
                'implementation_priority': evaluation_report.implementation_priority,
                'appeal_eligibility': evaluation_report.appeal_eligibility,
                'status_updated': status_updated,
                'key_strengths': evaluation_report.key_strengths,
                'key_concerns': evaluation_report.key_concerns,
                'recommendations': evaluation_report.recommendations,
                'resource_requirements': evaluation_report.resource_requirements
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'evaluation_timestamp': evaluation_report.evaluation_timestamp.isoformat(),
                    'council_votes_count': len(evaluation_report.council_votes),
                    'decision_outcome': evaluation_report.final_decision.value
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Idea evaluation failed: {str(e)}"
            )
    
    
    async def _handle_batch_evaluate_ideas(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle batch idea evaluation task."""
        try:
            # Extract ideas data from payload
            ideas_data = payload.get('ideas')
            if not ideas_data:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'ideas' in payload"
                )
            
            # Convert to Idea objects
            ideas = []
            for idea_data in ideas_data:
                if isinstance(idea_data, dict):
                    ideas.append(Idea(**idea_data))
                else:
                    ideas.append(idea_data)
            
            # Perform batch evaluation
            evaluation_reports = []
            for idea in ideas:
                try:
                    report = self.council.evaluate_idea(idea)
                    evaluation_reports.append(report)
                    self.evaluation_cache[str(idea.id)] = report
                except Exception as e:
                    self.logger.error(f"Failed to evaluate idea {idea.id}: {e}")
                    continue
            
            # Calculate batch summary
            total_ideas = len(evaluation_reports)
            approved_count = len([r for r in evaluation_reports if r.final_decision == DecisionOutcome.APPROVED])
            rejected_count = len([r for r in evaluation_reports if r.final_decision == DecisionOutcome.REJECTED])
            revision_count = total_ideas - approved_count - rejected_count
            
            average_consensus = sum(r.consensus_score for r in evaluation_reports) / total_ideas if total_ideas > 0 else 0
            average_strategic_alignment = sum(r.strategic_alignment_score for r in evaluation_reports) / total_ideas if total_ideas > 0 else 0
            
            result_data = {
                'batch_summary': {
                    'total_ideas': total_ideas,
                    'approved_count': approved_count,
                    'rejected_count': rejected_count,
                    'revision_count': revision_count,
                    'approval_rate': (approved_count / total_ideas * 100) if total_ideas > 0 else 0,
                    'average_consensus_score': average_consensus,
                    'average_strategic_alignment': average_strategic_alignment
                },
                'evaluation_reports': [report.dict() for report in evaluation_reports],
                'idea_summaries': [
                    {
                        'idea_id': report.idea_id,
                        'decision': report.final_decision.value,
                        'consensus_score': report.consensus_score,
                        'strategic_alignment': report.strategic_alignment_score,
                        'implementation_priority': report.implementation_priority
                    }
                    for report in evaluation_reports
                ]
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'batch_size': total_ideas,
                    'processing_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Batch evaluation failed: {str(e)}"
            )
    
    async def _handle_process_appeal(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle appeal processing task."""
        try:
            # Extract appeal request data
            appeal_data = payload.get('appeal_request')
            if not appeal_data:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'appeal_request' in payload"
                )
            
            # Convert to AppealRequest object
            if isinstance(appeal_data, dict):
                appeal_request = AppealRequest(**appeal_data)
            else:
                appeal_request = appeal_data
            
            # Store appeal as active
            self.active_appeals[appeal_request.idea_id] = appeal_request
            
            # Process the appeal
            appeal_evaluation = self.council.process_appeal(appeal_request)
            
            # Cache the new evaluation
            self.evaluation_cache[appeal_request.idea_id] = appeal_evaluation
            
            # Prepare result
            result_data = {
                'appeal_evaluation': appeal_evaluation.dict(),
                'idea_id': appeal_request.idea_id,
                'original_decision': appeal_request.original_decision.value,
                'appeal_decision': appeal_evaluation.final_decision.value,
                'decision_changed': appeal_request.original_decision != appeal_evaluation.final_decision,
                'new_consensus_score': appeal_evaluation.consensus_score,
                'appeal_reason': appeal_request.appeal_reason,
                'processed_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'appeal_timestamp': appeal_request.appeal_timestamp.isoformat(),
                    'priority_escalation': appeal_request.priority_escalation
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Appeal processing failed: {str(e)}"
            )
    
    async def _handle_get_evaluation_summary(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get evaluation summary task."""
        try:
            idea_id = payload.get('idea_id')
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get summary from council
            summary = self.council.get_evaluation_summary(idea_id)
            
            if summary is None:
                return TaskExecutionResult(
                    success=False,
                    error=f"No evaluation summary found for idea: {idea_id}"
                )
            
            # Add cache information
            cached_report = self.evaluation_cache.get(idea_id)
            if cached_report:
                summary['cached_evaluation'] = {
                    'timestamp': cached_report.evaluation_timestamp.isoformat(),
                    'decision': cached_report.final_decision.value
                }
            
            return TaskExecutionResult(
                success=True,
                result={'evaluation_summary': summary}
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Failed to get evaluation summary: {str(e)}"
            )
    
    async def _handle_get_council_metrics(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get council metrics task."""
        try:
            # Get council metrics
            council_metrics = self.council.get_council_metrics()
            
            # Get agent metrics
            agent_metrics = self.get_agent_metrics()
            
            # Add enhanced metrics
            enhanced_metrics = {
                'council_metrics': council_metrics,
                'agent_metrics': agent_metrics,
                'cache_metrics': {
                    'cached_evaluations': len(self.evaluation_cache),
                    'active_appeals': len(self.active_appeals)
                },
                'recent_activity': {
                    'last_evaluation': max(
                        [report.evaluation_timestamp for report in self.evaluation_cache.values()],
                        default=datetime.min.replace(tzinfo=timezone.utc)
                    ).isoformat(),
                    'total_cached_evaluations': len(self.evaluation_cache)
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
                error=f"Failed to get council metrics: {str(e)}"
            )
    
    async def _handle_update_idea_status(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle idea status update task."""
        try:
            # Extract idea and evaluation data
            idea_data = payload.get('idea')
            evaluation_id = payload.get('evaluation_id')
            
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
            
            # Get evaluation report
            evaluation_report = None
            if evaluation_id and evaluation_id in self.evaluation_cache:
                evaluation_report = self.evaluation_cache[evaluation_id]
            elif str(idea.id) in self.evaluation_cache:
                evaluation_report = self.evaluation_cache[str(idea.id)]
            else:
                return TaskExecutionResult(
                    success=False,
                    error=f"No evaluation report found for idea: {idea.id}"
                )
            
            # Update status
            success = self.council.update_idea_status(idea, evaluation_report)
            
            result_data = {
                'idea_id': str(idea.id),
                'status_updated': success,
                'new_status': idea.status.value if success else None,
                'evaluation_decision': evaluation_report.final_decision.value,
                'update_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=success,
                result=result_data,
                metadata={
                    'evaluation_timestamp': evaluation_report.evaluation_timestamp.isoformat()
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Status update failed: {str(e)}"
            )
    
    async def on_startup(self):
        """Custom startup logic for council agent."""
        self.logger.info("Enhanced Council Agent startup completed")
        
        # Clear caches on startup
        self.evaluation_cache.clear()
        self.active_appeals.clear()
        
        # Log council configuration
        council_metrics = self.council.get_council_metrics()
        self.logger.info(f"Council configuration: {council_metrics.get('configuration', {})}")
    
    async def on_shutdown(self):
        """Custom shutdown logic for council agent."""
        self.logger.info("Enhanced Council Agent shutdown starting")
        
        # Save final metrics
        final_metrics = self.get_agent_metrics()
        council_metrics = self.council.get_council_metrics()
        
        self.logger.info(f"Final council metrics: {council_metrics}")
        self.logger.info(f"Final agent metrics: {final_metrics}")
        
        # Clear caches
        self.evaluation_cache.clear()
        self.active_appeals.clear()


# Convenience function for creating and starting council agent
async def create_and_start_council_agent(
    council_config: Optional[CouncilConfiguration] = None,
    redis_config: Optional[RedisConfig] = None
) -> EnhancedCouncilAgent:
    """
    Create and start an enhanced council agent.
    
    Args:
        council_config: Council domain logic configuration
        redis_config: Redis connection configuration
        
    Returns:
        Initialized and started EnhancedCouncilAgent
    """
    agent = EnhancedCouncilAgent(council_config, redis_config=redis_config)
    
    success = await agent.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Enhanced Council Agent")
    
    return agent


# Export main classes
__all__ = [
    "EnhancedCouncilAgent",
    "create_and_start_council_agent"
]