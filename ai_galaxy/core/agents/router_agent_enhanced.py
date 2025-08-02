"""
Enhanced Router Agent with Message Processing Infrastructure.

This module implements the Router Agent with complete message processing
capabilities, enabling it to receive tasks from the orchestrator, execute
intelligent semantic routing logic, and report results back via Redis pub/sub messaging.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from .base_agent_handler import BaseAgentHandler, AgentConfiguration, TaskExecutionResult
from .router_agent import RouterAgent, RouterConfiguration, RoutingDecision, RoutingContext, SimilarityMatch, DepartmentWorkload, RoutingMetrics, RoutingConfidence, RoutingPriority, DepartmentCapability
from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext
from ...services.redis import RedisConfig


class EnhancedRouterAgent(BaseAgentHandler):
    """
    Enhanced Router Agent with message processing infrastructure.
    
    Combines the domain logic from RouterAgent with the message handling
    capabilities from BaseAgentHandler to create a fully functional agent
    that can integrate with the orchestrator.
    """
    
    def __init__(self, 
                 router_config: Optional[RouterConfiguration] = None,
                 agent_config: Optional[AgentConfiguration] = None,
                 redis_config: Optional[RedisConfig] = None):
        """
        Initialize the enhanced router agent.
        
        Args:
            router_config: Configuration for router logic
            agent_config: Configuration for agent message handling
            redis_config: Redis connection configuration
        """
        # Set up agent configuration
        if agent_config is None:
            agent_config = AgentConfiguration(
                agent_name="router_agent",
                agent_type="router",
                capabilities=[
                    "semantic_analysis",
                    "intelligent_routing",
                    "domain_classification",
                    "similarity_matching", 
                    "workload_assessment",
                    "department_mapping",
                    "institution_selection",
                    "cross_department_coordination",
                    "adaptive_learning",
                    "routing_optimization"
                ],
                redis_config=redis_config
            )
        
        # Initialize base agent handler
        super().__init__(agent_config)
        
        # Initialize router domain logic
        self.router = RouterAgent(router_config)
        
        # Additional router-specific state
        self.routing_cache: Dict[str, RoutingDecision] = {}
        self.semantic_analysis_cache: Dict[str, RoutingContext] = {}
        self.workload_monitoring: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Enhanced Router Agent initialized")
    
    def get_supported_task_types(self) -> List[str]:
        """Get list of task types this agent can handle."""
        return [
            "route_idea",
            "analyze_semantic_content",
            "assess_department_workloads",
            "find_similar_projects", 
            "validate_routing_decision",
            "provide_routing_feedback",
            "get_routing_metrics",
            "get_routing_history",
            "export_routing_knowledge",
            "import_routing_knowledge",
            "optimize_department_allocation",
            "suggest_new_institutions"
        ]
    
    async def execute_task(self, task_type: str, payload: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute router-specific tasks.
        
        Args:
            task_type: Type of router task to execute
            payload: Task data and parameters
            
        Returns:
            TaskExecutionResult with routing analysis results
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
            
            self.logger.info(f"Executing router task: {task_type}", context)
            
            if task_type == "route_idea":
                result = await self._handle_route_idea(payload)
            elif task_type == "analyze_semantic_content":
                result = await self._handle_analyze_semantic_content(payload)
            elif task_type == "assess_department_workloads":
                result = await self._handle_assess_department_workloads(payload)
            elif task_type == "find_similar_projects":
                result = await self._handle_find_similar_projects(payload)
            elif task_type == "validate_routing_decision":
                result = await self._handle_validate_routing_decision(payload)
            elif task_type == "provide_routing_feedback":
                result = await self._handle_provide_routing_feedback(payload)
            elif task_type == "get_routing_metrics":
                result = await self._handle_get_routing_metrics(payload)
            elif task_type == "get_routing_history":
                result = await self._handle_get_routing_history(payload)
            elif task_type == "export_routing_knowledge":
                result = await self._handle_export_routing_knowledge(payload)
            elif task_type == "import_routing_knowledge":
                result = await self._handle_import_routing_knowledge(payload)
            elif task_type == "optimize_department_allocation":
                result = await self._handle_optimize_department_allocation(payload)
            elif task_type == "suggest_new_institutions":
                result = await self._handle_suggest_new_institutions(payload)
            else:
                return TaskExecutionResult(
                    success=False,
                    error=f"Unknown task type: {task_type}"
                )
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.execution_time = execution_time
            
            self.logger.info(f"Router task completed: {task_type}", context)
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"Task execution failed: {str(e)}"
            
            self.logger.error(f"Router task failed: {task_type} - {error_msg}", context, exc_info=True)
            
            return TaskExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _handle_route_idea(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle idea routing task."""
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
            
            # Perform intelligent routing
            routing_decision = await self.router.route_idea(idea)
            
            # Cache the routing decision
            self.routing_cache[str(idea.id)] = routing_decision
            
            # Prepare comprehensive result
            result_data = {
                'idea_id': str(idea.id),
                'routing_decision': routing_decision.dict(),
                'primary_department': routing_decision.primary_department,
                'primary_institution': routing_decision.primary_institution,
                'confidence_level': routing_decision.confidence_level.value,
                'confidence_score': routing_decision.confidence_score,
                'priority': routing_decision.priority.value,
                'reasoning': routing_decision.reasoning,
                'alternative_departments': routing_decision.alternative_departments,
                'alternative_institutions': routing_decision.alternative_institutions,
                'semantic_keywords': routing_decision.semantic_keywords,
                'domain_classification': routing_decision.domain_classification,
                'new_institution_needed': routing_decision.new_institution_needed,
                'proposed_institution_name': routing_decision.proposed_institution_name,
                'proposed_institution_capabilities': routing_decision.proposed_institution_capabilities,
                'cross_department_collaboration': routing_decision.cross_department_collaboration,
                'escalation_needed': routing_decision.escalation_needed,
                'escalation_reason': routing_decision.escalation_reason,
                'validation_notes': routing_decision.validation_notes,
                'routing_timestamp': routing_decision.analysis_timestamp.isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'routing_confidence': routing_decision.confidence_level.value,
                    'requires_escalation': routing_decision.escalation_needed,
                    'cross_department_collab': len(routing_decision.cross_department_collaboration) > 0
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Idea routing failed: {str(e)}"
            )
    
    async def _handle_analyze_semantic_content(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle semantic content analysis task."""
        try:
            # Extract content data from payload
            idea_data = payload.get('idea')
            text_content = payload.get('text_content')
            
            if not idea_data and not text_content:
                return TaskExecutionResult(
                    success=False,
                    error="Missing either 'idea' or 'text_content' in payload"
                )
            
            idea = None
            if idea_data:
                if isinstance(idea_data, dict):
                    idea = Idea(**idea_data)
                else:
                    idea = idea_data
            
            # Perform semantic analysis
            if idea:
                semantic_context = await self.router._analyze_idea_semantics(idea)
                idea_id = str(idea.id)
            else:
                # Create temporary idea from text content
                temp_idea = Idea(title="Semantic Analysis", description=text_content)
                semantic_context = await self.router._analyze_idea_semantics(temp_idea)
                idea_id = "temp_analysis"
            
            # Cache semantic analysis
            self.semantic_analysis_cache[idea_id] = semantic_context
            
            # Prepare detailed semantic analysis result
            result_data = {
                'idea_id': idea_id,
                'semantic_analysis': {
                    'semantic_keywords': [
                        {
                            'term': kw.term,
                            'frequency': kw.frequency,
                            'importance_score': kw.importance_score,
                            'domain_relevance': kw.domain_relevance,
                            'context': kw.context
                        }
                        for kw in semantic_context.semantic_keywords
                    ],
                    'domain_scores': semantic_context.domain_scores,
                    'requirement_complexity': semantic_context.requirement_complexity,
                    'integration_needs': semantic_context.integration_needs,
                    'performance_requirements': semantic_context.performance_requirements,
                    'timeline_constraints': semantic_context.timeline_constraints,
                    'resource_constraints': semantic_context.resource_constraints
                },
                'top_domains': dict(sorted(semantic_context.domain_scores.items(), 
                                         key=lambda x: x[1], reverse=True)[:3]),
                'recommended_departments': [
                    self.router._map_domain_to_department((domain, score))
                    for domain, score in sorted(semantic_context.domain_scores.items(), 
                                              key=lambda x: x[1], reverse=True)[:3]
                ],
                'complexity_analysis': {
                    'level': semantic_context.requirement_complexity,
                    'factors': {
                        'integration_complexity': len(semantic_context.integration_needs),
                        'performance_complexity': len(semantic_context.performance_requirements),
                        'timeline_pressure': semantic_context.timeline_constraints == 'urgent'
                    }
                },
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'keywords_extracted': len(semantic_context.semantic_keywords),
                    'domains_identified': len(semantic_context.domain_scores),
                    'complexity_level': semantic_context.requirement_complexity
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Semantic analysis failed: {str(e)}"
            )
    
    async def _handle_assess_department_workloads(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle department workload assessment task."""
        try:
            # Get current workload assessment
            workload_assessment = self.router._assess_department_workloads()
            
            # Store in monitoring cache
            self.workload_monitoring[datetime.now(timezone.utc).isoformat()] = {
                'workload_data': {dept_id: workload.dict() for dept_id, workload in workload_assessment.items()},
                'assessment_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Calculate system-wide metrics
            total_capacity = sum(w.total_capacity for w in workload_assessment.values())
            total_load = sum(w.current_load for w in workload_assessment.values())
            system_utilization = (total_load / total_capacity * 100) if total_capacity > 0 else 0
            
            # Identify bottlenecks and recommendations
            overloaded_departments = [
                dept_id for dept_id, workload in workload_assessment.items()
                if workload.utilization_percentage > 85
            ]
            
            underutilized_departments = [
                dept_id for dept_id, workload in workload_assessment.items()
                if workload.utilization_percentage < 40
            ]
            
            # Generate recommendations
            recommendations = []
            if overloaded_departments:
                recommendations.append(f"Consider load balancing for overloaded departments: {', '.join(overloaded_departments)}")
            
            if underutilized_departments:
                recommendations.append(f"Opportunity to route more work to underutilized departments: {', '.join(underutilized_departments)}")
            
            if system_utilization > 90:
                recommendations.append("System approaching capacity - consider scaling or prioritization")
            
            result_data = {
                'system_overview': {
                    'total_capacity': total_capacity,
                    'total_load': total_load,
                    'system_utilization_percentage': system_utilization,
                    'department_count': len(workload_assessment),
                    'overloaded_departments': overloaded_departments,
                    'underutilized_departments': underutilized_departments
                },
                'department_workloads': {
                    dept_id: {
                        'department_id': workload.department_id,
                        'utilization_percentage': workload.utilization_percentage,
                        'pending_ideas': workload.pending_ideas,
                        'active_projects': workload.active_projects,
                        'average_completion_time': workload.average_completion_time,
                        'recent_success_rate': workload.recent_success_rate,
                        'available_institutions': workload.available_institutions,
                        'overloaded_institutions': workload.overloaded_institutions,
                        'status': (
                            'overloaded' if workload.utilization_percentage > 85 else
                            'high_utilization' if workload.utilization_percentage > 70 else
                            'normal' if workload.utilization_percentage > 40 else
                            'underutilized'
                        )
                    }
                    for dept_id, workload in workload_assessment.items()
                },
                'recommendations': recommendations,
                'assessment_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'system_utilization': system_utilization,
                    'bottlenecks_detected': len(overloaded_departments)
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Workload assessment failed: {str(e)}"
            )
    
    async def _handle_find_similar_projects(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle similar projects finding task."""
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
            
            # Get semantic analysis for the idea
            semantic_context = await self.router._analyze_idea_semantics(idea)
            
            # Find similar projects
            similar_projects = await self.router._find_similar_projects(idea, semantic_context)
            
            # Prepare result with similarity analysis
            result_data = {
                'idea_id': str(idea.id),
                'similar_projects_found': len(similar_projects),
                'similar_projects': [
                    {
                        'idea_id': match.idea_id,
                        'similarity_score': match.similarity_score,
                        'matched_keywords': match.matched_keywords,
                        'previous_routing': {
                            'primary_department': match.previous_routing.primary_department,
                            'primary_institution': match.previous_routing.primary_institution,
                            'confidence_level': match.previous_routing.confidence_level.value,
                            'confidence_score': match.previous_routing.confidence_score,
                            'analysis_timestamp': match.previous_routing.analysis_timestamp.isoformat()
                        },
                        'outcome_success': match.outcome_success,
                        'lessons_learned': match.lessons_learned
                    }
                    for match in similar_projects
                ],
                'similarity_analysis': {
                    'highest_similarity': max([m.similarity_score for m in similar_projects], default=0.0),
                    'average_similarity': sum([m.similarity_score for m in similar_projects]) / len(similar_projects) if similar_projects else 0.0,
                    'common_keywords': list(set().union(*[m.matched_keywords for m in similar_projects])) if similar_projects else [],
                    'suggested_departments': list(set([m.previous_routing.primary_department for m in similar_projects])),
                    'success_rate': sum(1 for m in similar_projects if m.outcome_success is not False) / len(similar_projects) if similar_projects else 0.0
                },
                'routing_recommendations': self._generate_similarity_recommendations(similar_projects),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'similar_projects_count': len(similar_projects),
                    'highest_similarity': max([m.similarity_score for m in similar_projects], default=0.0)
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Similar projects search failed: {str(e)}"
            )
    
    async def _handle_validate_routing_decision(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle routing decision validation task."""
        try:
            # Extract routing decision data
            routing_data = payload.get('routing_decision')
            idea_data = payload.get('idea')
            
            if not routing_data:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'routing_decision' in payload"
                )
            
            # Convert to RoutingDecision object
            if isinstance(routing_data, dict):
                routing_decision = RoutingDecision(**routing_data)
            else:
                routing_decision = routing_data
            
            # Get idea if provided
            idea = None
            if idea_data:
                if isinstance(idea_data, dict):
                    idea = Idea(**idea_data)
                else:
                    idea = idea_data
            
            # Validate the routing decision
            validated_decision = self.router._validate_routing_decision(routing_decision, idea)
            
            # Perform additional validation checks
            validation_results = {
                'original_decision': routing_decision.dict(),
                'validated_decision': validated_decision.dict(),
                'validation_changes': self._compare_routing_decisions(routing_decision, validated_decision),
                'validation_summary': {
                    'escalation_needed': validated_decision.escalation_needed,
                    'escalation_reason': validated_decision.escalation_reason,
                    'validation_notes_count': len(validated_decision.validation_notes),
                    'confidence_acceptable': validated_decision.confidence_score >= self.router.config.confidence_threshold,
                    'department_available': True,  # Would check actual department availability
                    'institution_available': validated_decision.primary_institution is not None or validated_decision.new_institution_needed
                },
                'recommendation_score': self._calculate_recommendation_score(validated_decision),
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=validation_results,
                metadata={
                    'validation_passed': not validated_decision.escalation_needed,
                    'confidence_level': validated_decision.confidence_level.value
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Routing validation failed: {str(e)}"
            )
    
    async def _handle_provide_routing_feedback(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle routing feedback provision task."""
        try:
            # Extract feedback data
            idea_id = payload.get('idea_id')
            was_successful = payload.get('was_successful')
            actual_department = payload.get('actual_department')
            notes = payload.get('notes')
            
            if not idea_id or was_successful is None:
                return TaskExecutionResult(
                    success=False,
                    error="Missing required fields: 'idea_id' and 'was_successful'"
                )
            
            # Provide feedback to the router
            await self.router.provide_routing_feedback(
                idea_id=idea_id,
                was_successful=was_successful,
                actual_department=actual_department,
                notes=notes
            )
            
            # Get updated metrics after feedback
            updated_metrics = self.router.get_routing_metrics()
            
            result_data = {
                'idea_id': idea_id,
                'feedback_processed': True,
                'feedback_summary': {
                    'was_successful': was_successful,
                    'actual_department': actual_department,
                    'notes': notes,
                    'feedback_timestamp': datetime.now(timezone.utc).isoformat()
                },
                'updated_metrics': {
                    'total_feedback_received': len(self.router.feedback_data),
                    'feedback_accuracy': updated_metrics.feedback_accuracy,
                    'learning_iterations': updated_metrics.learning_iterations,
                    'routing_accuracy': updated_metrics.routing_accuracy
                },
                'learning_impact': {
                    'adaptive_learning_enabled': self.router.config.enable_adaptive_learning,
                    'pattern_reinforcement': was_successful,
                    'weight_adjustment': not was_successful and actual_department is not None
                }
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'feedback_successful': was_successful,
                    'learning_enabled': self.router.config.enable_adaptive_learning
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Routing feedback processing failed: {str(e)}"
            )
    
    async def _handle_get_routing_metrics(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get routing metrics task."""
        try:
            # Get comprehensive routing metrics
            routing_metrics = self.router.get_routing_metrics()
            
            # Get agent metrics
            agent_metrics = self.get_agent_metrics()
            
            # Calculate additional insights
            performance_insights = {
                'efficiency_rating': 'excellent' if routing_metrics.routing_accuracy > 0.9 else 
                                   'good' if routing_metrics.routing_accuracy > 0.8 else
                                   'average' if routing_metrics.routing_accuracy > 0.7 else
                                   'needs_improvement',
                'confidence_reliability': 'high' if routing_metrics.confidence_accuracy > 0.85 else
                                        'medium' if routing_metrics.confidence_accuracy > 0.7 else
                                        'low',
                'learning_progress': 'active' if routing_metrics.learning_iterations > 10 else
                                   'developing' if routing_metrics.learning_iterations > 0 else
                                   'initial',
                'system_maturity': 'mature' if routing_metrics.total_routings > 100 else
                                 'developing' if routing_metrics.total_routings > 20 else
                                 'new'
            }
            
            # Enhanced metrics
            enhanced_metrics = {
                'routing_metrics': routing_metrics.dict(),
                'agent_metrics': agent_metrics,
                'performance_insights': performance_insights,
                'cache_metrics': {
                    'routing_cache_size': len(self.routing_cache),
                    'semantic_analysis_cache_size': len(self.semantic_analysis_cache),
                    'workload_monitoring_entries': len(self.workload_monitoring)
                },
                'historical_performance': {
                    'total_ideas_routed': routing_metrics.total_routings,
                    'success_rate': (routing_metrics.successful_routings / max(routing_metrics.total_routings, 1)) * 100,
                    'escalation_rate': (routing_metrics.escalated_routings / max(routing_metrics.total_routings, 1)) * 100,
                    'average_analysis_time': routing_metrics.average_analysis_time,
                    'average_confidence': routing_metrics.average_confidence_score
                },
                'department_insights': {
                    'most_routed_department': max(routing_metrics.department_distribution.items(), key=lambda x: x[1])[0] if routing_metrics.department_distribution else None,
                    'department_diversity': len(routing_metrics.department_distribution),
                    'distribution_balance': self._calculate_distribution_balance(routing_metrics.department_distribution)
                },
                'learning_analytics': {
                    'feedback_received': len(self.router.feedback_data),
                    'feedback_accuracy': routing_metrics.feedback_accuracy,
                    'improvement_rate': routing_metrics.improvement_rate,
                    'adaptation_cycles': routing_metrics.adaptation_cycles
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
                error=f"Failed to get routing metrics: {str(e)}"
            )
    
    async def _handle_get_routing_history(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get routing history task."""
        try:
            # Extract parameters
            limit = payload.get('limit', 50)
            department_filter = payload.get('department_filter')
            confidence_filter = payload.get('confidence_filter')
            date_from = payload.get('date_from')
            date_to = payload.get('date_to')
            
            # Get routing history
            routing_history = self.router.get_routing_history(limit)
            
            # Apply filters
            filtered_history = routing_history
            
            if department_filter:
                filtered_history = [
                    decision for decision in filtered_history
                    if decision.primary_department == department_filter
                ]
            
            if confidence_filter:
                filtered_history = [
                    decision for decision in filtered_history
                    if decision.confidence_level.value == confidence_filter
                ]
            
            if date_from:
                from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
                filtered_history = [
                    decision for decision in filtered_history
                    if decision.analysis_timestamp >= from_date
                ]
            
            if date_to:
                to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                filtered_history = [
                    decision for decision in filtered_history
                    if decision.analysis_timestamp <= to_date
                ]
            
            # Prepare history analysis
            history_analysis = self._analyze_routing_history(filtered_history)
            
            result_data = {
                'routing_history': [decision.dict() for decision in filtered_history],
                'history_summary': {
                    'total_entries': len(filtered_history),
                    'date_range': {
                        'earliest': min([d.analysis_timestamp for d in filtered_history]).isoformat() if filtered_history else None,
                        'latest': max([d.analysis_timestamp for d in filtered_history]).isoformat() if filtered_history else None
                    },
                    'filters_applied': {
                        'limit': limit,
                        'department_filter': department_filter,
                        'confidence_filter': confidence_filter,
                        'date_from': date_from,
                        'date_to': date_to
                    }
                },
                'history_analysis': history_analysis,
                'retrieval_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'entries_returned': len(filtered_history),
                    'filters_applied': bool(department_filter or confidence_filter or date_from or date_to)
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Failed to get routing history: {str(e)}"
            )
    
    async def _handle_export_routing_knowledge(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle routing knowledge export task."""
        try:
            # Export routing knowledge
            knowledge_data = self.router.export_routing_knowledge()
            
            # Add enhanced cache data
            enhanced_knowledge = knowledge_data.copy()
            enhanced_knowledge.update({
                'enhanced_agent_cache': {
                    'routing_cache': {
                        idea_id: decision.dict() 
                        for idea_id, decision in self.routing_cache.items()
                    },
                    'semantic_analysis_cache': {
                        idea_id: {
                            'idea_id': context.idea_id,
                            'semantic_keywords': [
                                {
                                    'term': kw.term,
                                    'frequency': kw.frequency,
                                    'importance_score': kw.importance_score,
                                    'domain_relevance': kw.domain_relevance,
                                    'context': kw.context
                                }
                                for kw in context.semantic_keywords
                            ],
                            'domain_scores': context.domain_scores,
                            'requirement_complexity': context.requirement_complexity,
                            'integration_needs': context.integration_needs,
                            'performance_requirements': context.performance_requirements,
                            'timeline_constraints': context.timeline_constraints,
                            'resource_constraints': context.resource_constraints
                        }
                        for idea_id, context in self.semantic_analysis_cache.items()
                    },
                    'workload_monitoring': self.workload_monitoring
                },
                'enhanced_agent_metrics': self.get_agent_metrics(),
                'export_metadata': {
                    'export_type': 'enhanced_router_agent',
                    'cache_sizes': {
                        'routing_cache': len(self.routing_cache),
                        'semantic_cache': len(self.semantic_analysis_cache),
                        'workload_monitoring': len(self.workload_monitoring)
                    },
                    'enhanced_export_timestamp': datetime.now(timezone.utc).isoformat()
                }
            })
            
            # Calculate export statistics
            export_stats = {
                'total_routing_decisions': len(knowledge_data.get('routing_history', {})),
                'total_feedback_entries': len(knowledge_data.get('feedback_data', {})),
                'domain_signatures_count': len(knowledge_data.get('domain_signatures', {})),
                'enhanced_cache_entries': len(self.routing_cache) + len(self.semantic_analysis_cache),
                'knowledge_completeness': self._calculate_knowledge_completeness(enhanced_knowledge)
            }
            
            result_data = {
                'export_successful': True,
                'knowledge_data': enhanced_knowledge,
                'export_statistics': export_stats,
                'export_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'knowledge_size': len(str(enhanced_knowledge)),
                    'completeness_score': export_stats['knowledge_completeness']
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Routing knowledge export failed: {str(e)}"
            )
    
    async def _handle_import_routing_knowledge(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle routing knowledge import task."""
        try:
            # Extract knowledge data
            knowledge_data = payload.get('knowledge_data')
            if not knowledge_data:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'knowledge_data' in payload"
                )
            
            # Import to base router
            self.router.import_routing_knowledge(knowledge_data)
            
            # Import enhanced cache data if available
            enhanced_cache = knowledge_data.get('enhanced_agent_cache', {})
            
            if 'routing_cache' in enhanced_cache:
                for idea_id, decision_data in enhanced_cache['routing_cache'].items():
                    self.routing_cache[idea_id] = RoutingDecision(**decision_data)
            
            if 'semantic_analysis_cache' in enhanced_cache:
                # Import semantic analysis cache with proper object reconstruction
                # This would require more complex deserialization
                self.semantic_analysis_cache.update(enhanced_cache['semantic_analysis_cache'])
            
            if 'workload_monitoring' in enhanced_cache:
                self.workload_monitoring.update(enhanced_cache['workload_monitoring'])
            
            # Calculate import statistics
            import_stats = {
                'routing_decisions_imported': len(knowledge_data.get('routing_history', {})),
                'feedback_entries_imported': len(knowledge_data.get('feedback_data', {})),
                'enhanced_cache_imported': len(enhanced_cache),
                'total_cache_size': len(self.routing_cache) + len(self.semantic_analysis_cache),
                'import_successful': True
            }
            
            result_data = {
                'import_successful': True,
                'import_statistics': import_stats,
                'knowledge_integration': {
                    'base_router_updated': True,
                    'enhanced_cache_updated': bool(enhanced_cache),
                    'total_routing_history': len(self.router.routing_history),
                    'total_feedback_data': len(self.router.feedback_data)
                },
                'import_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'entries_imported': import_stats['routing_decisions_imported'],
                    'cache_enhanced': bool(enhanced_cache)
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Routing knowledge import failed: {str(e)}"
            )
    
    async def _handle_optimize_department_allocation(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle department allocation optimization task."""
        try:
            # Get current workload assessment
            workload_assessment = self.router._assess_department_workloads()
            
            # Analyze allocation patterns
            allocation_analysis = self._analyze_allocation_patterns(workload_assessment)
            
            # Generate optimization recommendations
            optimization_recommendations = []
            
            # Check for overloaded departments
            for dept_id, workload in workload_assessment.items():
                if workload.utilization_percentage > 85:
                    optimization_recommendations.append({
                        'type': 'load_balancing',
                        'department': dept_id,
                        'current_utilization': workload.utilization_percentage,
                        'recommendation': f'Redistribute {workload.pending_ideas} pending ideas to alternative departments',
                        'priority': 'high',
                        'estimated_impact': 'reduce_utilization_by_15_percent'
                    })
                
                if len(workload.overloaded_institutions) > 0:
                    optimization_recommendations.append({
                        'type': 'institution_scaling',
                        'department': dept_id,
                        'overloaded_institutions': workload.overloaded_institutions,
                        'recommendation': 'Create additional institutions or redistribute workload',
                        'priority': 'medium',
                        'estimated_impact': 'improve_institution_capacity'
                    })
            
            # Check for underutilized departments
            underutilized = [
                dept_id for dept_id, workload in workload_assessment.items()
                if workload.utilization_percentage < 40
            ]
            
            if underutilized:
                optimization_recommendations.append({
                    'type': 'capacity_utilization',
                    'departments': underutilized,
                    'recommendation': 'Route more compatible ideas to underutilized departments',
                    'priority': 'low',
                    'estimated_impact': 'increase_system_efficiency'
                })
            
            # Performance optimization suggestions
            performance_optimizations = []
            
            # Semantic routing improvements
            if self.router.routing_metrics.routing_accuracy < 0.8:
                performance_optimizations.append({
                    'area': 'semantic_analysis',
                    'recommendation': 'Enhance keyword patterns and domain signatures',
                    'current_accuracy': self.router.routing_metrics.routing_accuracy,
                    'target_improvement': '10_percent_accuracy_increase'
                })
            
            # Cross-department collaboration optimization
            cross_collab_opportunities = self._identify_collaboration_opportunities(workload_assessment)
            
            result_data = {
                'optimization_analysis': {
                    'current_allocation': allocation_analysis,
                    'optimization_opportunities': len(optimization_recommendations),
                    'performance_improvements': len(performance_optimizations),
                    'collaboration_opportunities': len(cross_collab_opportunities)
                },
                'optimization_recommendations': optimization_recommendations,
                'performance_optimizations': performance_optimizations,
                'collaboration_opportunities': cross_collab_opportunities,
                'implementation_priority': {
                    'high_priority': [r for r in optimization_recommendations if r.get('priority') == 'high'],
                    'medium_priority': [r for r in optimization_recommendations if r.get('priority') == 'medium'],
                    'low_priority': [r for r in optimization_recommendations if r.get('priority') == 'low']
                },
                'optimization_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'recommendations_count': len(optimization_recommendations),
                    'high_priority_items': len([r for r in optimization_recommendations if r.get('priority') == 'high'])
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Department allocation optimization failed: {str(e)}"
            )
    
    async def _handle_suggest_new_institutions(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle new institution suggestion task."""
        try:
            # Extract parameters
            department_id = payload.get('department_id')
            capability_requirements = payload.get('capability_requirements', [])
            workload_pressure = payload.get('workload_pressure', 'medium')
            
            # Get current workload assessment
            workload_assessment = self.router._assess_department_workloads()
            
            # Analyze institution needs
            institution_suggestions = []
            
            if department_id:
                # Specific department analysis
                if department_id in workload_assessment:
                    workload = workload_assessment[department_id]
                    suggestions = self._analyze_department_institution_needs(department_id, workload, capability_requirements)
                    institution_suggestions.extend(suggestions)
            else:
                # System-wide analysis
                for dept_id, workload in workload_assessment.items():
                    if workload.utilization_percentage > 80 or len(workload.available_institutions) == 0:
                        suggestions = self._analyze_department_institution_needs(dept_id, workload, capability_requirements)
                        institution_suggestions.extend(suggestions)
            
            # Prioritize suggestions
            prioritized_suggestions = self._prioritize_institution_suggestions(institution_suggestions, workload_assessment)
            
            # Generate implementation roadmap
            implementation_roadmap = self._generate_institution_roadmap(prioritized_suggestions)
            
            result_data = {
                'institution_analysis': {
                    'total_suggestions': len(institution_suggestions),
                    'high_priority_suggestions': len([s for s in prioritized_suggestions if s.get('priority') == 'high']),
                    'departments_analyzed': len(workload_assessment) if not department_id else 1,
                    'capability_gaps_identified': len(set().union(*[s.get('capabilities', []) for s in institution_suggestions]))
                },
                'institution_suggestions': prioritized_suggestions,
                'implementation_roadmap': implementation_roadmap,
                'resource_requirements': {
                    'estimated_total_institutions': len(prioritized_suggestions),
                    'development_phases': len(implementation_roadmap.get('phases', [])),
                    'capability_coverage': self._calculate_capability_coverage(prioritized_suggestions)
                },
                'suggestion_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'suggestions_generated': len(institution_suggestions),
                    'high_priority_count': len([s for s in prioritized_suggestions if s.get('priority') == 'high'])
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Institution suggestion failed: {str(e)}"
            )
    
    def _generate_similarity_recommendations(self, similar_projects: List[SimilarityMatch]) -> List[str]:
        """Generate routing recommendations based on similar projects."""
        recommendations = []
        
        if not similar_projects:
            recommendations.append("No similar projects found - routing based on semantic analysis only")
            return recommendations
        
        # Analyze success patterns
        successful_projects = [p for p in similar_projects if p.outcome_success is not False]
        if successful_projects:
            common_departments = {}
            for project in successful_projects:
                dept = project.previous_routing.primary_department
                common_departments[dept] = common_departments.get(dept, 0) + 1
            
            most_successful_dept = max(common_departments.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Similar successful projects often routed to: {most_successful_dept}")
        
        # High similarity recommendations
        high_similarity = [p for p in similar_projects if p.similarity_score > 0.8]
        if high_similarity:
            recommendations.append(f"Found {len(high_similarity)} highly similar projects - high confidence routing recommended")
        
        return recommendations
    
    def _compare_routing_decisions(self, original: RoutingDecision, validated: RoutingDecision) -> Dict[str, Any]:
        """Compare two routing decisions to identify changes."""
        changes = {}
        
        if original.primary_department != validated.primary_department:
            changes['primary_department'] = {
                'original': original.primary_department,
                'validated': validated.primary_department
            }
        
        if original.primary_institution != validated.primary_institution:
            changes['primary_institution'] = {
                'original': original.primary_institution,
                'validated': validated.primary_institution
            }
        
        if original.confidence_score != validated.confidence_score:
            changes['confidence_score'] = {
                'original': original.confidence_score,
                'validated': validated.confidence_score
            }
        
        if original.escalation_needed != validated.escalation_needed:
            changes['escalation_needed'] = {
                'original': original.escalation_needed,
                'validated': validated.escalation_needed
            }
        
        return changes
    
    def _calculate_recommendation_score(self, decision: RoutingDecision) -> float:
        """Calculate a recommendation score for the routing decision."""
        score = decision.confidence_score
        
        # Boost score for clear department mapping
        if decision.primary_institution:
            score += 0.1
        
        # Reduce score for escalation needs
        if decision.escalation_needed:
            score -= 0.2
        
        # Boost score for cross-department collaboration
        if decision.cross_department_collaboration:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _calculate_distribution_balance(self, distribution: Dict[str, int]) -> float:
        """Calculate how balanced the department distribution is."""
        if not distribution:
            return 0.0
        
        values = list(distribution.values())
        if len(values) <= 1:
            return 1.0
        
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        
        # Normalized balance score (higher = more balanced)
        balance_score = 1.0 / (1.0 + variance / (mean_val + 0.1))
        return balance_score
    
    def _analyze_routing_history(self, history: List[RoutingDecision]) -> Dict[str, Any]:
        """Analyze routing history for patterns and insights."""
        if not history:
            return {}
        
        # Department distribution
        dept_distribution = {}
        confidence_distribution = {}
        escalation_rate = 0
        
        for decision in history:
            dept = decision.primary_department
            dept_distribution[dept] = dept_distribution.get(dept, 0) + 1
            
            conf_level = decision.confidence_level.value
            confidence_distribution[conf_level] = confidence_distribution.get(conf_level, 0) + 1
            
            if decision.escalation_needed:
                escalation_rate += 1
        
        escalation_rate = (escalation_rate / len(history)) * 100
        
        return {
            'department_distribution': dept_distribution,
            'confidence_distribution': confidence_distribution,
            'escalation_rate': escalation_rate,
            'average_confidence': sum(d.confidence_score for d in history) / len(history),
            'new_institutions_suggested': sum(1 for d in history if d.new_institution_needed),
            'cross_department_collaborations': sum(len(d.cross_department_collaboration) for d in history)
        }
    
    def _calculate_knowledge_completeness(self, knowledge_data: Dict[str, Any]) -> float:
        """Calculate how complete the exported knowledge is."""
        completeness_factors = []
        
        # Check for routing history
        if knowledge_data.get('routing_history'):
            completeness_factors.append(0.3)
        
        # Check for feedback data
        if knowledge_data.get('feedback_data'):
            completeness_factors.append(0.2)
        
        # Check for domain signatures
        if knowledge_data.get('domain_signatures'):
            completeness_factors.append(0.2)
        
        # Check for enhanced cache
        if knowledge_data.get('enhanced_agent_cache'):
            completeness_factors.append(0.2)
        
        # Check for metrics
        if knowledge_data.get('routing_metrics'):
            completeness_factors.append(0.1)
        
        return sum(completeness_factors)
    
    def _analyze_allocation_patterns(self, workload_assessment: Dict[str, DepartmentWorkload]) -> Dict[str, Any]:
        """Analyze current department allocation patterns."""
        total_capacity = sum(w.total_capacity for w in workload_assessment.values())
        total_load = sum(w.current_load for w in workload_assessment.values())
        
        allocation_efficiency = {}
        for dept_id, workload in workload_assessment.items():
            efficiency = workload.recent_success_rate * (1 - abs(workload.utilization_percentage - 70) / 100)
            allocation_efficiency[dept_id] = efficiency
        
        return {
            'system_utilization': (total_load / total_capacity) * 100 if total_capacity > 0 else 0,
            'department_efficiency': allocation_efficiency,
            'load_distribution': {dept_id: w.current_load for dept_id, w in workload_assessment.items()},
            'capacity_distribution': {dept_id: w.total_capacity for dept_id, w in workload_assessment.items()}
        }
    
    def _identify_collaboration_opportunities(self, workload_assessment: Dict[str, DepartmentWorkload]) -> List[Dict[str, Any]]:
        """Identify cross-department collaboration opportunities."""
        opportunities = []
        
        # Look for complementary capabilities
        underutilized = [dept_id for dept_id, w in workload_assessment.items() if w.utilization_percentage < 50]
        overloaded = [dept_id for dept_id, w in workload_assessment.items() if w.utilization_percentage > 85]
        
        for overloaded_dept in overloaded:
            for underutilized_dept in underutilized:
                opportunities.append({
                    'type': 'load_sharing',
                    'from_department': overloaded_dept,
                    'to_department': underutilized_dept,
                    'potential_benefit': 'redistribute_workload',
                    'estimated_impact': 'medium'
                })
        
        return opportunities
    
    def _analyze_department_institution_needs(self, dept_id: str, workload: DepartmentWorkload, requirements: List[str]) -> List[Dict[str, Any]]:
        """Analyze institution needs for a specific department."""
        suggestions = []
        
        # High utilization suggests need for more institutions
        if workload.utilization_percentage > 80:
            suggestions.append({
                'department_id': dept_id,
                'institution_name': f"{dept_id}_expansion_institution",
                'reason': 'high_utilization',
                'priority': 'high',
                'capabilities': requirements or ['general_purpose'],
                'estimated_capacity': workload.total_capacity // 2
            })
        
        # No institutions suggests need for foundational institution
        if len(workload.available_institutions) == 0:
            suggestions.append({
                'department_id': dept_id,
                'institution_name': f"{dept_id}_foundation_institution",
                'reason': 'no_existing_institutions',
                'priority': 'critical',
                'capabilities': requirements or ['foundational'],
                'estimated_capacity': 50
            })
        
        return suggestions
    
    def _prioritize_institution_suggestions(self, suggestions: List[Dict[str, Any]], workload_assessment: Dict[str, DepartmentWorkload]) -> List[Dict[str, Any]]:
        """Prioritize institution suggestions based on multiple factors."""
        prioritized = suggestions.copy()
        
        # Sort by priority level and workload pressure
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        prioritized.sort(key=lambda s: (
            priority_order.get(s.get('priority', 'low'), 3),
            -workload_assessment.get(s.get('department_id', ''), DepartmentWorkload(
                department_id='', total_capacity=1, current_load=0, 
                utilization_percentage=0, pending_ideas=0, active_projects=0,
                average_completion_time=0, recent_success_rate=0
            )).utilization_percentage
        ))
        
        return prioritized
    
    def _generate_institution_roadmap(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate implementation roadmap for new institutions."""
        if not suggestions:
            return {}
        
        phases = []
        current_phase = []
        phase_capacity = 3  # Max institutions per phase
        
        for suggestion in suggestions:
            if len(current_phase) >= phase_capacity:
                phases.append(current_phase)
                current_phase = []
            current_phase.append(suggestion)
        
        if current_phase:
            phases.append(current_phase)
        
        return {
            'total_phases': len(phases),
            'phases': [
                {
                    'phase_number': i + 1,
                    'institutions': phase,
                    'estimated_duration': '2-4 weeks',
                    'dependencies': 'previous_phase_completion' if i > 0 else 'none'
                }
                for i, phase in enumerate(phases)
            ],
            'estimated_total_duration': f"{len(phases) * 3} weeks"
        }
    
    def _calculate_capability_coverage(self, suggestions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate capability coverage from institution suggestions."""
        coverage = {}
        for suggestion in suggestions:
            for capability in suggestion.get('capabilities', []):
                coverage[capability] = coverage.get(capability, 0) + 1
        return coverage
    
    async def on_startup(self):
        """Custom startup logic for router agent."""
        self.logger.info("Enhanced Router Agent startup completed")
        
        # Clear caches on startup
        self.routing_cache.clear()
        self.semantic_analysis_cache.clear()
        self.workload_monitoring.clear()
        
        # Log router configuration
        router_metrics = self.router.get_routing_metrics()
        self.logger.info(f"Router metrics: {router_metrics.dict()}")
        
        # Log semantic capabilities
        domain_signatures = len(self.router.domain_signatures)
        self.logger.info(f"Domain signatures loaded: {domain_signatures}")
    
    async def on_shutdown(self):
        """Custom shutdown logic for router agent."""
        self.logger.info("Enhanced Router Agent shutdown starting")
        
        # Save final metrics
        final_metrics = self.get_agent_metrics()
        router_metrics = self.router.get_routing_metrics()
        
        self.logger.info(f"Final router metrics: {router_metrics.dict()}")
        self.logger.info(f"Final agent metrics: {final_metrics}")
        
        # Export knowledge if beneficial
        if len(self.router.routing_history) > 10:
            try:
                knowledge_export = self.router.export_routing_knowledge()
                self.logger.info(f"Exported routing knowledge: {len(knowledge_export)} items")
            except Exception as e:
                self.logger.error(f"Failed to export routing knowledge: {e}")
        
        # Clear caches
        self.routing_cache.clear()
        self.semantic_analysis_cache.clear()
        self.workload_monitoring.clear()


# Convenience function for creating and starting router agent
async def create_and_start_router_agent(
    router_config: Optional[RouterConfiguration] = None,
    redis_config: Optional[RedisConfig] = None
) -> EnhancedRouterAgent:
    """
    Create and start an enhanced router agent.
    
    Args:
        router_config: Router domain logic configuration
        redis_config: Redis connection configuration
        
    Returns:
        Initialized and started EnhancedRouterAgent
    """
    agent = EnhancedRouterAgent(router_config, redis_config=redis_config)
    
    success = await agent.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Enhanced Router Agent")
    
    return agent


# Export main classes
__all__ = [
    "EnhancedRouterAgent",
    "create_and_start_router_agent"
]