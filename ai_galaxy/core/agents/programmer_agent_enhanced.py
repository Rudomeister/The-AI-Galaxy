"""
Enhanced Programmer Agent with Message Processing Infrastructure.

This module implements the Programmer Agent with complete message processing
capabilities, enabling it to receive tasks from the orchestrator, execute
advanced code generation logic, and report results back via Redis pub/sub messaging.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from .base_agent_handler import BaseAgentHandler, AgentConfiguration, TaskExecutionResult
from .programmer_agent import ProgrammerAgent, ProgrammerConfiguration, CodeSpec, CodeArtifact, OptimizationResult, TestSuite, DevelopmentMetrics, ProgrammingLanguage, Framework, DesignPattern, CodeQuality, TestingFramework, OptimizationType
from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext
from ...services.redis import RedisConfig


class EnhancedProgrammerAgent(BaseAgentHandler):
    """
    Enhanced Programmer Agent with message processing infrastructure.
    
    Combines the domain logic from ProgrammerAgent with the message handling
    capabilities from BaseAgentHandler to create a fully functional agent
    that can integrate with the orchestrator.
    """
    
    def __init__(self, 
                 programmer_config: Optional[ProgrammerConfiguration] = None,
                 agent_config: Optional[AgentConfiguration] = None,
                 redis_config: Optional[RedisConfig] = None):
        """
        Initialize the enhanced programmer agent.
        
        Args:
            programmer_config: Configuration for programmer logic
            agent_config: Configuration for agent message handling
            redis_config: Redis connection configuration
        """
        # Set up agent configuration
        if agent_config is None:
            agent_config = AgentConfiguration(
                agent_name="programmer_agent",
                agent_type="programmer",
                capabilities=[
                    "code_generation",
                    "architecture_design",
                    "test_suite_creation",
                    "code_optimization",
                    "documentation_generation",
                    "quality_assurance",
                    "multi_language_support",
                    "framework_integration",
                    "pattern_implementation",
                    "performance_optimization"
                ],
                redis_config=redis_config
            )
        
        # Initialize base agent handler
        super().__init__(agent_config)
        
        # Initialize programmer domain logic
        self.programmer = ProgrammerAgent(programmer_config)
        
        # Additional programmer-specific state
        self.generation_cache: Dict[str, List[CodeArtifact]] = {}
        self.optimization_cache: Dict[str, OptimizationResult] = {}
        self.active_sessions: Dict[str, str] = {}  # session_id -> idea_id mapping
        
        self.logger.info("Enhanced Programmer Agent initialized")
    
    def get_supported_task_types(self) -> List[str]:
        """Get list of task types this agent can handle."""
        return [
            "generate_code_from_specification",
            "optimize_existing_code",
            "generate_test_suite",
            "validate_code_specification",
            "generate_documentation",
            "perform_quality_assurance",
            "get_generation_status",
            "get_programmer_metrics",
            "export_code_artifacts",
            "analyze_code_complexity",
            "generate_api_endpoints",
            "create_integration_layer"
        ]
    
    async def execute_task(self, task_type: str, payload: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute programmer-specific tasks.
        
        Args:
            task_type: Type of programmer task to execute
            payload: Task data and parameters
            
        Returns:
            TaskExecutionResult with code generation results
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
            
            self.logger.info(f"Executing programmer task: {task_type}", context)
            
            if task_type == "generate_code_from_specification":
                result = await self._handle_generate_code_from_specification(payload)
            elif task_type == "optimize_existing_code":
                result = await self._handle_optimize_existing_code(payload)
            elif task_type == "generate_test_suite":
                result = await self._handle_generate_test_suite(payload)
            elif task_type == "validate_code_specification":
                result = await self._handle_validate_code_specification(payload)
            elif task_type == "generate_documentation":
                result = await self._handle_generate_documentation(payload)
            elif task_type == "perform_quality_assurance":
                result = await self._handle_perform_quality_assurance(payload)
            elif task_type == "get_generation_status":
                result = await self._handle_get_generation_status(payload)
            elif task_type == "get_programmer_metrics":
                result = await self._handle_get_programmer_metrics(payload)
            elif task_type == "export_code_artifacts":
                result = await self._handle_export_code_artifacts(payload)
            elif task_type == "analyze_code_complexity":
                result = await self._handle_analyze_code_complexity(payload)
            elif task_type == "generate_api_endpoints":
                result = await self._handle_generate_api_endpoints(payload)
            elif task_type == "create_integration_layer":
                result = await self._handle_create_integration_layer(payload)
            else:
                return TaskExecutionResult(
                    success=False,
                    error=f"Unknown task type: {task_type}"
                )
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.execution_time = execution_time
            
            self.logger.info(f"Programmer task completed: {task_type}", context)
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"Task execution failed: {str(e)}"
            
            self.logger.error(f"Programmer task failed: {task_type} - {error_msg}", context, exc_info=True)
            
            return TaskExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _handle_generate_code_from_specification(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle code generation from specification task."""
        try:
            # Extract idea and specification data from payload
            idea_data = payload.get('idea')
            specification = payload.get('specification', {})
            
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
            
            # Generate code from specification
            artifacts = await self.programmer.generate_code_from_specification(idea, specification)
            
            # Cache the generated artifacts
            self.generation_cache[str(idea.id)] = artifacts
            
            # Prepare comprehensive result
            result_data = {
                'idea_id': str(idea.id),
                'generation_successful': True,
                'artifacts_generated': len(artifacts),
                'total_lines_of_code': sum(artifact.lines_of_code for artifact in artifacts),
                'languages_used': list(set(artifact.language.value for artifact in artifacts)),
                'artifact_types': list(set(artifact.artifact_type for artifact in artifacts)),
                'artifacts_summary': [
                    {
                        'id': artifact.id,
                        'name': artifact.name,
                        'file_path': artifact.file_path,
                        'language': artifact.language.value,
                        'artifact_type': artifact.artifact_type,
                        'lines_of_code': artifact.lines_of_code,
                        'complexity_score': artifact.complexity_score,
                        'linting_passed': artifact.linting_passed,
                        'created_at': artifact.created_at.isoformat()
                    }
                    for artifact in artifacts
                ],
                'code_quality_metrics': {
                    'average_complexity': sum(a.complexity_score for a in artifacts) / len(artifacts) if artifacts else 0,
                    'linting_pass_rate': sum(1 for a in artifacts if a.linting_passed) / len(artifacts) * 100 if artifacts else 0,
                    'total_dependencies': len(set().union(*(a.dependencies for a in artifacts))),
                    'test_coverage_estimate': sum(a.test_coverage for a in artifacts) / len(artifacts) if artifacts else 0
                },
                'output_directory': self.programmer.config.output_directory,
                'generation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'generation_type': 'full_implementation',
                    'specification_complexity': len(specification),
                    'frameworks_used': specification.get('framework', 'none')
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Code generation failed: {str(e)}"
            )
    
    async def _handle_optimize_existing_code(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle code optimization task."""
        try:
            # Extract required fields
            code_content = payload.get('code_content')
            optimization_type = payload.get('optimization_type', 'performance')
            idea_id = payload.get('idea_id')
            
            if not code_content:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'code_content' in payload"
                )
            
            # Convert optimization type string to enum
            try:
                opt_type = OptimizationType(optimization_type)
            except ValueError:
                return TaskExecutionResult(
                    success=False,
                    error=f"Invalid optimization type: {optimization_type}"
                )
            
            # Perform code optimization
            optimization_result = await self.programmer.optimize_existing_code(code_content, opt_type)
            
            # Cache optimization result if idea_id provided
            if idea_id:
                self.optimization_cache[idea_id] = optimization_result
            
            # Prepare result
            result_data = {
                'optimization_successful': True,
                'optimization_type': optimization_type,
                'improvements_made': optimization_result.improvements,
                'performance_gain': optimization_result.performance_gain,
                'metrics_comparison': {
                    'before': optimization_result.metrics_before,
                    'after': optimization_result.metrics_after
                },
                'original_artifact': {
                    'lines_of_code': optimization_result.original_artifact.lines_of_code,
                    'complexity_score': optimization_result.original_artifact.complexity_score
                },
                'optimized_artifact': {
                    'lines_of_code': optimization_result.optimized_artifact.lines_of_code,
                    'complexity_score': optimization_result.optimized_artifact.complexity_score,
                    'content': optimization_result.optimized_artifact.content
                },
                'optimization_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'optimization_type': optimization_type,
                    'improvements_count': len(optimization_result.improvements)
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Code optimization failed: {str(e)}"
            )
    
    async def _handle_generate_test_suite(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle test suite generation task."""
        try:
            # Extract required fields
            idea_id = payload.get('idea_id')
            code_spec_data = payload.get('code_spec')
            artifacts_data = payload.get('artifacts', [])
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get code spec - either from payload or from programmer's active generations
            if code_spec_data:
                code_spec = CodeSpec(**code_spec_data)
            else:
                active_generations = self.programmer.get_active_generations()
                if idea_id not in active_generations:
                    return TaskExecutionResult(
                        success=False,
                        error=f"No active generation found for idea: {idea_id}"
                    )
                code_spec = active_generations[idea_id]
            
            # Get artifacts - either from payload or from generation cache
            if artifacts_data:
                artifacts = [CodeArtifact(**artifact_data) for artifact_data in artifacts_data]
            else:
                artifacts = self.generation_cache.get(idea_id, [])
                if not artifacts:
                    return TaskExecutionResult(
                        success=False,
                        error=f"No generated artifacts found for idea: {idea_id}"
                    )
            
            # Generate test suite
            test_artifacts = await self.programmer._generate_test_suite(code_spec, artifacts)
            
            # Update generation cache with test artifacts
            if idea_id in self.generation_cache:
                self.generation_cache[idea_id].extend(test_artifacts)
            
            # Prepare result
            result_data = {
                'idea_id': idea_id,
                'test_generation_successful': True,
                'test_artifacts_generated': len(test_artifacts),
                'test_framework': code_spec.testing_framework.value,
                'target_coverage': code_spec.test_coverage_target,
                'test_types': list(set(artifact.artifact_type for artifact in test_artifacts)),
                'test_artifacts_summary': [
                    {
                        'id': artifact.id,
                        'name': artifact.name,
                        'file_path': artifact.file_path,
                        'lines_of_code': artifact.lines_of_code,
                        'test_type': artifact.artifact_type
                    }
                    for artifact in test_artifacts
                ],
                'generation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'test_framework': code_spec.testing_framework.value,
                    'target_coverage': code_spec.test_coverage_target
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Test suite generation failed: {str(e)}"
            )
    
    async def _handle_validate_code_specification(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle code specification validation task."""
        try:
            # Extract code specification data
            code_spec_data = payload.get('code_spec')
            
            if not code_spec_data:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'code_spec' in payload"
                )
            
            # Convert to CodeSpec object
            if isinstance(code_spec_data, dict):
                code_spec = CodeSpec(**code_spec_data)
            else:
                code_spec = code_spec_data
            
            # Validate the code specification
            validation_result = await self.programmer.validate_code_spec(code_spec)
            
            # Add enhanced validation information
            enhanced_validation = {
                'validation_result': validation_result,
                'specification_analysis': {
                    'complexity_estimate': 'high' if len(code_spec.functionality) > 10 else 'medium' if len(code_spec.functionality) > 5 else 'low',
                    'estimated_development_time': len(code_spec.functionality) * 2,  # hours estimate
                    'risk_assessment': {
                        'language_compatibility': validation_result['valid'],
                        'framework_support': code_spec.framework is not None,
                        'testing_coverage': code_spec.test_coverage_target >= 80,
                        'documentation_requirements': code_spec.include_docstrings and code_spec.include_type_hints
                    },
                    'recommended_improvements': validation_result.get('recommendations', [])
                },
                'supported_features': {
                    'language': code_spec.language.value,
                    'framework': code_spec.framework.value if code_spec.framework else None,
                    'design_patterns': [pattern.value for pattern in code_spec.design_patterns],
                    'integrations': code_spec.integrations,
                    'quality_level': code_spec.quality_level.value
                },
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=validation_result['valid'],
                result=enhanced_validation,
                metadata={
                    'issues_count': len(validation_result.get('issues', [])),
                    'recommendations_count': len(validation_result.get('recommendations', []))
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Code specification validation failed: {str(e)}"
            )
    
    async def _handle_generate_documentation(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle documentation generation task."""
        try:
            # Extract required fields
            idea_id = payload.get('idea_id')
            code_spec_data = payload.get('code_spec')
            artifacts_data = payload.get('artifacts', [])
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get code spec and artifacts
            if code_spec_data:
                code_spec = CodeSpec(**code_spec_data)
            else:
                active_generations = self.programmer.get_active_generations()
                if idea_id not in active_generations:
                    return TaskExecutionResult(
                        success=False,
                        error=f"No active generation found for idea: {idea_id}"
                    )
                code_spec = active_generations[idea_id]
            
            if artifacts_data:
                artifacts = [CodeArtifact(**artifact_data) for artifact_data in artifacts_data]
            else:
                artifacts = self.generation_cache.get(idea_id, [])
            
            # Generate documentation
            doc_artifacts = await self.programmer._generate_documentation(code_spec, artifacts)
            
            # Update generation cache with documentation artifacts
            if idea_id in self.generation_cache:
                self.generation_cache[idea_id].extend(doc_artifacts)
            
            # Prepare result
            result_data = {
                'idea_id': idea_id,
                'documentation_generated': True,
                'documentation_artifacts': len(doc_artifacts),
                'documentation_types': list(set(artifact.name.split('.')[-1] for artifact in doc_artifacts)),
                'documentation_summary': [
                    {
                        'id': artifact.id,
                        'name': artifact.name,
                        'file_path': artifact.file_path,
                        'lines_of_code': artifact.lines_of_code,
                        'documentation_type': artifact.artifact_type
                    }
                    for artifact in doc_artifacts
                ],
                'includes_readme': any('README' in artifact.name for artifact in doc_artifacts),
                'includes_api_docs': any('api' in artifact.name.lower() for artifact in doc_artifacts),
                'generation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'documentation_completeness': 'comprehensive' if len(doc_artifacts) > 2 else 'basic'
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Documentation generation failed: {str(e)}"
            )
    
    async def _handle_perform_quality_assurance(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle quality assurance task."""
        try:
            # Extract required fields
            idea_id = payload.get('idea_id')
            artifacts_data = payload.get('artifacts', [])
            
            if not idea_id and not artifacts_data:
                return TaskExecutionResult(
                    success=False,
                    error="Missing either 'idea_id' or 'artifacts' in payload"
                )
            
            # Get artifacts
            if artifacts_data:
                artifacts = [CodeArtifact(**artifact_data) for artifact_data in artifacts_data]
            else:
                artifacts = self.generation_cache.get(idea_id, [])
                if not artifacts:
                    return TaskExecutionResult(
                        success=False,
                        error=f"No artifacts found for QA for idea: {idea_id}"
                    )
            
            # Perform quality assurance
            await self.programmer._perform_quality_assurance(artifacts)
            
            # Collect QA results
            qa_results = {
                'idea_id': idea_id,
                'qa_completed': True,
                'total_artifacts_checked': len(artifacts),
                'artifacts_passed': sum(1 for artifact in artifacts if artifact.linting_passed),
                'artifacts_failed': sum(1 for artifact in artifacts if not artifact.linting_passed),
                'quality_metrics': {
                    'average_complexity': sum(a.complexity_score for a in artifacts) / len(artifacts) if artifacts else 0,
                    'pass_rate': sum(1 for a in artifacts if a.linting_passed) / len(artifacts) * 100 if artifacts else 0,
                    'type_checking_pass_rate': sum(1 for a in artifacts if a.type_checking_passed) / len(artifacts) * 100 if artifacts else 0,
                    'security_scan_pass_rate': sum(1 for a in artifacts if a.security_scan_passed) / len(artifacts) * 100 if artifacts else 0
                },
                'artifacts_qa_summary': [
                    {
                        'id': artifact.id,
                        'name': artifact.name,
                        'linting_passed': artifact.linting_passed,
                        'type_checking_passed': artifact.type_checking_passed,
                        'security_scan_passed': artifact.security_scan_passed,
                        'complexity_score': artifact.complexity_score
                    }
                    for artifact in artifacts
                ],
                'qa_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=qa_results,
                metadata={
                    'overall_quality_score': qa_results['quality_metrics']['pass_rate']
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Quality assurance failed: {str(e)}"
            )
    
    async def _handle_get_generation_status(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get generation status task."""
        try:
            idea_id = payload.get('idea_id')
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get active generations
            active_generations = self.programmer.get_active_generations()
            generation_history = self.programmer.get_generation_history(idea_id)
            cached_artifacts = self.generation_cache.get(idea_id, [])
            
            # Determine status
            if idea_id in active_generations:
                status = 'active'
                current_spec = active_generations[idea_id]
                status_details = {
                    'status': status,
                    'current_specification': {
                        'name': current_spec.name,
                        'language': current_spec.language.value,
                        'framework': current_spec.framework.value if current_spec.framework else None,
                        'functionality_count': len(current_spec.functionality),
                        'quality_level': current_spec.quality_level.value
                    }
                }
            elif generation_history:
                status = 'completed'
                status_details = {
                    'status': status,
                    'artifacts_generated': len(generation_history),
                    'completion_summary': {
                        'total_lines': sum(a.lines_of_code for a in generation_history),
                        'languages_used': list(set(a.language.value for a in generation_history)),
                        'artifact_types': list(set(a.artifact_type for a in generation_history))
                    }
                }
            elif cached_artifacts:
                status = 'cached'
                status_details = {
                    'status': status,
                    'cached_artifacts': len(cached_artifacts),
                    'cache_summary': {
                        'total_lines': sum(a.lines_of_code for a in cached_artifacts),
                        'languages_used': list(set(a.language.value for a in cached_artifacts)),
                        'artifact_types': list(set(a.artifact_type for a in cached_artifacts))
                    }
                }
            else:
                status = 'not_found'
                status_details = {
                    'status': status,
                    'message': f'No generation found for idea: {idea_id}'
                }
            
            result_data = {
                'idea_id': idea_id,
                'generation_status': status_details,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={'status': status}
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Failed to get generation status: {str(e)}"
            )
    
    async def _handle_get_programmer_metrics(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle get programmer metrics task."""
        try:
            # Get programmer metrics
            programmer_metrics = self.programmer.get_development_metrics()
            
            # Get agent metrics
            agent_metrics = self.get_agent_metrics()
            
            # Get supported capabilities
            supported_languages = [lang.value for lang in self.programmer.get_supported_languages()]
            supported_frameworks = [fw.value for fw in self.programmer.get_supported_frameworks()]
            
            # Enhanced metrics
            enhanced_metrics = {
                'development_metrics': programmer_metrics.dict(),
                'agent_metrics': agent_metrics,
                'capabilities': {
                    'supported_languages': supported_languages,
                    'supported_frameworks': supported_frameworks,
                    'supported_patterns': [pattern.value for pattern in DesignPattern],
                    'supported_optimizations': [opt.value for opt in OptimizationType],
                    'quality_levels': [quality.value for quality in CodeQuality],
                    'testing_frameworks': [test.value for test in TestingFramework]
                },
                'cache_metrics': {
                    'generation_cache_size': len(self.generation_cache),
                    'optimization_cache_size': len(self.optimization_cache),
                    'active_sessions': len(self.active_sessions)
                },
                'performance_metrics': {
                    'total_generations': programmer_metrics.total_code_generated,
                    'average_generation_time': programmer_metrics.average_generation_time,
                    'optimization_success_rate': programmer_metrics.optimization_success_rate,
                    'code_quality_score': programmer_metrics.code_quality_score
                },
                'knowledge_base_metrics': {
                    'architecture_templates': len(self.programmer.architecture_templates),
                    'code_patterns': sum(len(patterns) for patterns in self.programmer.code_patterns.values()),
                    'best_practices': sum(len(practices) for practices in self.programmer.best_practices.values()),
                    'learned_patterns': sum(len(patterns) for patterns in self.programmer.learned_patterns.values())
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
                error=f"Failed to get programmer metrics: {str(e)}"
            )
    
    async def _handle_export_code_artifacts(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle code artifacts export task."""
        try:
            idea_id = payload.get('idea_id')
            export_format = payload.get('export_format', 'json')
            include_content = payload.get('include_content', False)
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get artifacts
            artifacts = self.generation_cache.get(idea_id, [])
            if not artifacts:
                # Try generation history
                artifacts = self.programmer.get_generation_history(idea_id)
            
            if not artifacts:
                return TaskExecutionResult(
                    success=False,
                    error=f"No artifacts found for export for idea: {idea_id}"
                )
            
            # Prepare export data
            export_data = {
                'idea_id': idea_id,
                'export_format': export_format,
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'artifacts_count': len(artifacts),
                'artifacts': []
            }
            
            for artifact in artifacts:
                artifact_data = {
                    'id': artifact.id,
                    'name': artifact.name,
                    'file_path': artifact.file_path,
                    'language': artifact.language.value,
                    'artifact_type': artifact.artifact_type,
                    'lines_of_code': artifact.lines_of_code,
                    'complexity_score': artifact.complexity_score,
                    'dependencies': artifact.dependencies,
                    'created_at': artifact.created_at.isoformat(),
                    'last_modified': artifact.last_modified.isoformat(),
                    'version': artifact.version,
                    'author': artifact.author,
                    'linting_passed': artifact.linting_passed,
                    'type_checking_passed': artifact.type_checking_passed,
                    'security_scan_passed': artifact.security_scan_passed
                }
                
                if include_content:
                    artifact_data['content'] = artifact.content
                
                export_data['artifacts'].append(artifact_data)
            
            # Add summary statistics
            export_data['summary'] = {
                'total_lines_of_code': sum(a.lines_of_code for a in artifacts),
                'languages_distribution': {},
                'artifact_types_distribution': {},
                'average_complexity': sum(a.complexity_score for a in artifacts) / len(artifacts) if artifacts else 0,
                'quality_metrics': {
                    'linting_pass_rate': sum(1 for a in artifacts if a.linting_passed) / len(artifacts) * 100,
                    'type_checking_pass_rate': sum(1 for a in artifacts if a.type_checking_passed) / len(artifacts) * 100,
                    'security_scan_pass_rate': sum(1 for a in artifacts if a.security_scan_passed) / len(artifacts) * 100
                }
            }
            
            # Calculate distributions
            for artifact in artifacts:
                lang = artifact.language.value
                artifact_type = artifact.artifact_type
                
                export_data['summary']['languages_distribution'][lang] = \
                    export_data['summary']['languages_distribution'].get(lang, 0) + 1
                export_data['summary']['artifact_types_distribution'][artifact_type] = \
                    export_data['summary']['artifact_types_distribution'].get(artifact_type, 0) + 1
            
            return TaskExecutionResult(
                success=True,
                result=export_data,
                metadata={
                    'export_format': export_format,
                    'artifacts_count': len(artifacts),
                    'includes_content': include_content
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Code artifacts export failed: {str(e)}"
            )
    
    async def _handle_analyze_code_complexity(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle code complexity analysis task."""
        try:
            idea_id = payload.get('idea_id')
            code_content = payload.get('code_content')
            
            if not idea_id and not code_content:
                return TaskExecutionResult(
                    success=False,
                    error="Missing either 'idea_id' or 'code_content' in payload"
                )
            
            artifacts_to_analyze = []
            
            if code_content:
                # Analyze provided code content
                temp_artifact = CodeArtifact(
                    name="temp_analysis.py",
                    content=code_content,
                    language=ProgrammingLanguage.PYTHON
                )
                artifacts_to_analyze = [temp_artifact]
            else:
                # Analyze cached artifacts
                artifacts_to_analyze = self.generation_cache.get(idea_id, [])
                if not artifacts_to_analyze:
                    artifacts_to_analyze = self.programmer.get_generation_history(idea_id)
                
                if not artifacts_to_analyze:
                    return TaskExecutionResult(
                        success=False,
                        error=f"No artifacts found for complexity analysis for idea: {idea_id}"
                    )
            
            # Analyze complexity for each artifact
            complexity_results = []
            total_complexity = 0
            
            for artifact in artifacts_to_analyze:
                complexity_score = self.programmer._calculate_complexity(artifact.content)
                artifact.complexity_score = complexity_score
                total_complexity += complexity_score
                
                complexity_results.append({
                    'artifact_id': artifact.id,
                    'artifact_name': artifact.name,
                    'complexity_score': complexity_score,
                    'lines_of_code': artifact.lines_of_code,
                    'complexity_per_line': complexity_score / max(artifact.lines_of_code, 1),
                    'complexity_rating': (
                        'low' if complexity_score < 5 else
                        'medium' if complexity_score < 10 else
                        'high' if complexity_score < 20 else
                        'very_high'
                    )
                })
            
            # Calculate overall metrics
            average_complexity = total_complexity / len(artifacts_to_analyze) if artifacts_to_analyze else 0
            
            result_data = {
                'analysis_type': 'complexity_analysis',
                'artifacts_analyzed': len(artifacts_to_analyze),
                'overall_metrics': {
                    'total_complexity': total_complexity,
                    'average_complexity': average_complexity,
                    'complexity_distribution': {
                        'low': len([r for r in complexity_results if r['complexity_rating'] == 'low']),
                        'medium': len([r for r in complexity_results if r['complexity_rating'] == 'medium']),
                        'high': len([r for r in complexity_results if r['complexity_rating'] == 'high']),
                        'very_high': len([r for r in complexity_results if r['complexity_rating'] == 'very_high'])
                    },
                    'overall_rating': (
                        'low' if average_complexity < 5 else
                        'medium' if average_complexity < 10 else
                        'high' if average_complexity < 20 else
                        'very_high'
                    )
                },
                'artifact_complexity_details': complexity_results,
                'recommendations': self._generate_complexity_recommendations(complexity_results),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'average_complexity': average_complexity,
                    'highest_complexity': max((r['complexity_score'] for r in complexity_results), default=0)
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Code complexity analysis failed: {str(e)}"
            )
    
    async def _handle_generate_api_endpoints(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle API endpoints generation task."""
        try:
            # Extract required fields
            idea_id = payload.get('idea_id')
            code_spec_data = payload.get('code_spec')
            api_specification = payload.get('api_specification', {})
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get code spec
            if code_spec_data:
                code_spec = CodeSpec(**code_spec_data)
            else:
                active_generations = self.programmer.get_active_generations()
                if idea_id not in active_generations:
                    return TaskExecutionResult(
                        success=False,
                        error=f"No active generation found for idea: {idea_id}"
                    )
                code_spec = active_generations[idea_id]
            
            # Update code spec with API specification if provided
            if api_specification:
                code_spec.api_endpoints = api_specification.get('endpoints', code_spec.api_endpoints)
                code_spec.framework = Framework(api_specification.get('framework', code_spec.framework.value)) if api_specification.get('framework') else code_spec.framework
            
            # Generate API endpoints
            api_artifacts = await self.programmer._generate_api_layer(code_spec)
            
            # Update generation cache
            if idea_id in self.generation_cache:
                self.generation_cache[idea_id].extend(api_artifacts)
            else:
                self.generation_cache[idea_id] = api_artifacts
            
            result_data = {
                'idea_id': idea_id,
                'api_generation_successful': True,
                'api_artifacts_generated': len(api_artifacts),
                'framework_used': code_spec.framework.value if code_spec.framework else 'none',
                'endpoints_count': len(code_spec.api_endpoints),
                'api_artifacts_summary': [
                    {
                        'id': artifact.id,
                        'name': artifact.name,
                        'file_path': artifact.file_path,
                        'lines_of_code': artifact.lines_of_code
                    }
                    for artifact in api_artifacts
                ],
                'generation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'framework': code_spec.framework.value if code_spec.framework else 'none',
                    'endpoints_count': len(code_spec.api_endpoints)
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"API endpoints generation failed: {str(e)}"
            )
    
    async def _handle_create_integration_layer(self, payload: Dict[str, Any]) -> TaskExecutionResult:
        """Handle integration layer creation task."""
        try:
            # Extract required fields
            idea_id = payload.get('idea_id')
            code_spec_data = payload.get('code_spec')
            integration_specs = payload.get('integration_specs', [])
            
            if not idea_id:
                return TaskExecutionResult(
                    success=False,
                    error="Missing 'idea_id' in payload"
                )
            
            # Get code spec
            if code_spec_data:
                code_spec = CodeSpec(**code_spec_data)
            else:
                active_generations = self.programmer.get_active_generations()
                if idea_id not in active_generations:
                    return TaskExecutionResult(
                        success=False,
                        error=f"No active generation found for idea: {idea_id}"
                    )
                code_spec = active_generations[idea_id]
            
            # Update integrations if provided
            if integration_specs:
                code_spec.integrations.extend(integration_specs)
                code_spec.integrations = list(set(code_spec.integrations))  # Remove duplicates
            
            # Generate integration layer
            integration_artifacts = await self.programmer._generate_integrations(code_spec)
            
            # Update generation cache
            if idea_id in self.generation_cache:
                self.generation_cache[idea_id].extend(integration_artifacts)
            else:
                self.generation_cache[idea_id] = integration_artifacts
            
            result_data = {
                'idea_id': idea_id,
                'integration_generation_successful': True,
                'integration_artifacts_generated': len(integration_artifacts),
                'integrations_implemented': code_spec.integrations,
                'integration_artifacts_summary': [
                    {
                        'id': artifact.id,
                        'name': artifact.name,
                        'file_path': artifact.file_path,
                        'lines_of_code': artifact.lines_of_code,
                        'integration_type': 'integration'
                    }
                    for artifact in integration_artifacts
                ],
                'generation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result=result_data,
                metadata={
                    'integrations_count': len(code_spec.integrations),
                    'artifacts_generated': len(integration_artifacts)
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error=f"Integration layer creation failed: {str(e)}"
            )
    
    def _generate_complexity_recommendations(self, complexity_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on complexity analysis."""
        recommendations = []
        
        high_complexity_artifacts = [r for r in complexity_results if r['complexity_rating'] in ['high', 'very_high']]
        
        if high_complexity_artifacts:
            recommendations.append("Consider refactoring high-complexity functions into smaller, more focused functions")
            recommendations.append("Implement the Single Responsibility Principle for complex classes")
            recommendations.append("Add comprehensive unit tests for high-complexity code sections")
        
        very_high_complexity = [r for r in complexity_results if r['complexity_rating'] == 'very_high']
        if very_high_complexity:
            recommendations.append("Urgent: Review very high complexity code for potential bugs and maintenance issues")
            recommendations.append("Consider implementing design patterns like Strategy or Command to reduce complexity")
        
        avg_complexity_per_line = sum(r['complexity_per_line'] for r in complexity_results) / len(complexity_results) if complexity_results else 0
        if avg_complexity_per_line > 0.5:
            recommendations.append("Consider breaking down large functions and improving code structure")
        
        return recommendations
    
    async def on_startup(self):
        """Custom startup logic for programmer agent."""
        self.logger.info("Enhanced Programmer Agent startup completed")
        
        # Clear caches on startup
        self.generation_cache.clear()
        self.optimization_cache.clear()
        self.active_sessions.clear()
        
        # Log programmer configuration
        programmer_metrics = self.programmer.get_development_metrics()
        self.logger.info(f"Programmer metrics: {programmer_metrics.dict()}")
        
        # Log supported capabilities
        supported_languages = [lang.value for lang in self.programmer.get_supported_languages()]
        supported_frameworks = [fw.value for fw in self.programmer.get_supported_frameworks()]
        self.logger.info(f"Supported languages: {supported_languages}")
        self.logger.info(f"Supported frameworks: {supported_frameworks}")
    
    async def on_shutdown(self):
        """Custom shutdown logic for programmer agent."""
        self.logger.info("Enhanced Programmer Agent shutdown starting")
        
        # Save final metrics
        final_metrics = self.get_agent_metrics()
        programmer_metrics = self.programmer.get_development_metrics()
        
        self.logger.info(f"Final programmer metrics: {programmer_metrics.dict()}")
        self.logger.info(f"Final agent metrics: {final_metrics}")
        
        # Export knowledge base if configured
        if self.programmer.config.backup_generated_code:
            try:
                knowledge_export = self.programmer.export_code_knowledge()
                self.logger.info(f"Exported code knowledge: {len(knowledge_export)} items")
            except Exception as e:
                self.logger.error(f"Failed to export code knowledge: {e}")
        
        # Clear caches
        self.generation_cache.clear()
        self.optimization_cache.clear()
        self.active_sessions.clear()


# Convenience function for creating and starting programmer agent
async def create_and_start_programmer_agent(
    programmer_config: Optional[ProgrammerConfiguration] = None,
    redis_config: Optional[RedisConfig] = None
) -> EnhancedProgrammerAgent:
    """
    Create and start an enhanced programmer agent.
    
    Args:
        programmer_config: Programmer domain logic configuration
        redis_config: Redis connection configuration
        
    Returns:
        Initialized and started EnhancedProgrammerAgent
    """
    agent = EnhancedProgrammerAgent(programmer_config, redis_config=redis_config)
    
    success = await agent.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Enhanced Programmer Agent")
    
    return agent


# Export main classes
__all__ = [
    "EnhancedProgrammerAgent",
    "create_and_start_programmer_agent"
]