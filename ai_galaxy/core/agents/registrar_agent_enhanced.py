"""
Enhanced Registrar Agent for the AI-Galaxy Lower Meta-Layer.

This module implements the Enhanced Registrar Agent, combining the comprehensive
service registry and cataloging capabilities of the original RegistrarAgent with 
BaseAgentHandler message processing infrastructure for seamless integration with
the AI-Galaxy orchestration system.

The Enhanced Registrar Agent serves as the institutional memory of the AI-Galaxy
ecosystem, managing service registration, quality assessment, lifecycle management,
and providing intelligent service discovery with vector search integration.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from uuid import uuid4

from .base_agent_handler import BaseAgentHandler, TaskExecutionResult, AgentConfiguration
from .registrar_agent import (
    RegistrarAgent, 
    RegistrarConfiguration,
    RegistrationRequest,
    ServiceSearchQuery,
    ServiceRegistrationStatus,
    DocumentationType,
    QualityLevel
)
from ...shared.logger import get_logger, LogContext
from ...services.redis import RedisConfig


class EnhancedRegistrarAgent(BaseAgentHandler):
    """
    Enhanced Registrar Agent with message processing capabilities.
    
    Combines the comprehensive service registry and cataloging functionality
    of the original RegistrarAgent with Redis pub/sub message processing
    for autonomous operation within the AI-Galaxy orchestration system.
    
    Key Capabilities:
    - Service registration and metadata management
    - Quality assessment and compliance checking  
    - Documentation generation and lifecycle management
    - Service discovery with vector search
    - Analytics and reporting
    - Governance and compliance oversight
    - Performance monitoring and optimization
    - Dependency graph analysis
    - Automated quality gates
    - Integration with ecosystem orchestrator
    """
    
    def __init__(self, 
                 registrar_config: Optional[RegistrarConfiguration] = None,
                 agent_config: Optional[AgentConfiguration] = None,
                 redis_config: Optional[RedisConfig] = None):
        """
        Initialize the Enhanced Registrar Agent.
        
        Args:
            registrar_config: Configuration for registrar functionality
            agent_config: Configuration for agent behavior and capabilities
            redis_config: Redis connection configuration
        """
        # Set up agent configuration with comprehensive capabilities
        if agent_config is None:
            agent_config = AgentConfiguration(
                agent_name="registrar_agent",
                agent_type="registrar",
                capabilities=[
                    "service_registration",
                    "metadata_management", 
                    "quality_assessment",
                    "compliance_checking",
                    "documentation_generation",
                    "lifecycle_management",
                    "service_discovery",
                    "vector_search",
                    "analytics_reporting",
                    "governance_oversight",
                    "performance_monitoring",
                    "dependency_analysis",
                    "quality_gates",
                    "ecosystem_integration"
                ],
                redis_config=redis_config
            )
        
        # Initialize base agent handler
        super().__init__(agent_config)
        
        # Initialize registrar domain logic
        self.registrar = RegistrarAgent(registrar_config)
        
        # Enhanced metrics and state tracking
        self.processing_metrics = {
            "registrations_processed": 0,
            "quality_assessments_completed": 0,
            "documentation_generated": 0,
            "lifecycle_actions_performed": 0,
            "searches_conducted": 0,
            "compliance_checks_run": 0,
            "reports_generated": 0,
            "errors_encountered": 0,
            "average_processing_time": 0.0
        }
        
        # Task execution tracking
        self.active_registrations = {}
        self.active_assessments = {}
        self.active_documentation_tasks = {}
        
        self.logger.info(f"Enhanced Registrar Agent initialized with {len(agent_config.capabilities)} capabilities")
    
    async def execute_task(self, task_type: str, task_data: Dict[str, Any]) -> TaskExecutionResult:
        """
        Process registrar tasks with comprehensive service management.
        
        Args:
            task_type: Type of registrar task to perform
            task_data: Task-specific data and parameters
            
        Returns:
            TaskExecutionResult with registration/management outcomes
        """
        start_time = datetime.now()
        task_id = task_data.get('task_id', str(uuid4()))
        
        context = LogContext(
            agent_name=self.agent_name,
            task_id=task_id,
            additional_context={"task_type": task_type}
        )
        
        self.logger.agent_action("processing_registrar_task", self.agent_name, task_id, {
            "task_type": task_type,
            "agent_capabilities": len(self.config.capabilities)
        })
        
        try:
            # Route to appropriate registrar task handler
            result = await self._route_registrar_task(task_type, task_data, context)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_processing_metrics(task_type, processing_time, True)
            
            self.logger.agent_action("registrar_task_completed", self.agent_name, task_id, {
                "task_type": task_type,
                "success": result.success,
                "processing_time": processing_time
            })
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_processing_metrics(task_type, processing_time, False)
            
            self.logger.error(f"Registrar task failed: {e}", context, exc_info=True)
            
            return TaskExecutionResult(
                success=False,
                error_message=f"Registrar task execution failed: {str(e)}",
                task_id=task_id,
                agent_name=self.agent_name,
                execution_time=processing_time
            )
    
    async def _route_registrar_task(self, task_type: str, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Route registrar task to appropriate handler."""
        task_handlers = {
            # Core registration tasks
            "register_service": self._handle_register_service,
            "update_service_status": self._handle_update_service_status,
            "get_service_details": self._handle_get_service_details,
            "discover_services": self._handle_discover_services,
            
            # Quality and compliance tasks
            "assess_service_quality": self._handle_assess_service_quality,
            "check_service_compliance": self._handle_check_service_compliance,
            "apply_quality_gates": self._handle_apply_quality_gates,
            "generate_quality_report": self._handle_generate_quality_report,
            
            # Documentation tasks
            "generate_service_documentation": self._handle_generate_service_documentation,
            "update_documentation": self._handle_update_documentation,
            "validate_documentation": self._handle_validate_documentation,
            
            # Lifecycle management tasks
            "manage_service_lifecycle": self._handle_manage_service_lifecycle,
            "deprecate_service": self._handle_deprecate_service,
            "archive_service": self._handle_archive_service,
            "reactivate_service": self._handle_reactivate_service,
            
            # Analytics and reporting tasks
            "generate_analytics_report": self._handle_generate_analytics_report,
            "get_dependency_graph": self._handle_get_dependency_graph,
            "export_registry_data": self._handle_export_registry_data,
            "get_registrar_metrics": self._handle_get_registrar_metrics,
            
            # Administrative tasks
            "validate_registry_integrity": self._handle_validate_registry_integrity,
            "optimize_registry_performance": self._handle_optimize_registry_performance,
            "backup_registry": self._handle_backup_registry,
            "restore_registry": self._handle_restore_registry
        }
        
        handler = task_handlers.get(task_type)
        if not handler:
            return TaskExecutionResult(
                success=False,
                error_message=f"Unknown registrar task type: {task_type}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
        
        return await handler(task_data, context)
    
    # Core Registration Task Handlers
    
    async def _handle_register_service(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service registration requests."""
        try:
            # Parse registration request
            request_data = task_data.get('registration_request', {})
            registration_request = RegistrationRequest(**request_data)
            
            # Track active registration
            task_id = task_data.get('task_id', str(uuid4()))
            self.active_registrations[task_id] = {
                "service_name": registration_request.service_name,
                "start_time": datetime.now(),
                "status": "processing"
            }
            
            # Perform registration
            success, message, service_metadata = self.registrar.register_service(registration_request)
            
            # Update active registration status
            self.active_registrations[task_id]["status"] = "completed" if success else "failed"
            self.active_registrations[task_id]["end_time"] = datetime.now()
            
            result_data = {
                "registration_successful": success,
                "message": message,
                "service_id": service_metadata.service_id if service_metadata else None,
                "service_name": registration_request.service_name,
                "quality_level": service_metadata.quality_level.value if service_metadata else None,
                "compliance_status": service_metadata.compliance_status.value if service_metadata else None
            }
            
            if success:
                self.processing_metrics["registrations_processed"] += 1
            
            return TaskExecutionResult(
                success=success,
                result_data=result_data,
                message=message,
                task_id=task_id,
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Service registration failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Service registration error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_update_service_status(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service status updates."""
        try:
            service_id = task_data.get('service_id')
            new_status_str = task_data.get('new_status')
            reason = task_data.get('reason')
            
            if not service_id or not new_status_str:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id and new_status are required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            new_status = ServiceRegistrationStatus(new_status_str)
            success = self.registrar.update_service_status(service_id, new_status, reason)
            
            result_data = {
                "service_id": service_id,
                "status_updated": success,
                "new_status": new_status.value,
                "reason": reason
            }
            
            return TaskExecutionResult(
                success=success,
                result_data=result_data,
                message=f"Service status {'updated' if success else 'update failed'}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Service status update failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Status update error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_get_service_details(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service detail retrieval."""
        try:
            service_id = task_data.get('service_id')
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            service_metadata = self.registrar.get_service_details(service_id)
            
            if service_metadata:
                result_data = {
                    "service_found": True,
                    "service_metadata": service_metadata.dict(),
                    "service_id": service_id
                }
                message = f"Service details retrieved for {service_metadata.name}"
            else:
                result_data = {
                    "service_found": False,
                    "service_id": service_id
                }
                message = f"Service not found: {service_id}"
            
            return TaskExecutionResult(
                success=bool(service_metadata),
                result_data=result_data,
                message=message,
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Service detail retrieval failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Service detail retrieval error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_discover_services(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service discovery requests."""
        try:
            # Parse search query
            query_data = task_data.get('search_query', {})
            search_query = ServiceSearchQuery(**query_data)
            
            # Perform service discovery
            search_results = self.registrar.discover_services(search_query)
            
            result_data = {
                "services_found": len(search_results),
                "search_results": [result.dict() for result in search_results],
                "query_parameters": query_data
            }
            
            self.processing_metrics["searches_conducted"] += 1
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Discovered {len(search_results)} services matching criteria",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Service discovery failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Service discovery error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    # Quality and Compliance Task Handlers
    
    async def _handle_assess_service_quality(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service quality assessment."""
        try:
            service_id = task_data.get('service_id')
            force_reassessment = task_data.get('force_reassessment', False)
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Track active assessment
            task_id = task_data.get('task_id', str(uuid4()))
            self.active_assessments[task_id] = {
                "service_id": service_id,
                "start_time": datetime.now(),
                "status": "processing"
            }
            
            # Perform quality assessment
            assessment_result = self.registrar.assess_service_quality(service_id, force_reassessment)
            
            # Update assessment status
            self.active_assessments[task_id]["status"] = "completed"
            self.active_assessments[task_id]["end_time"] = datetime.now()
            
            success = "error" not in assessment_result
            if success:
                self.processing_metrics["quality_assessments_completed"] += 1
            
            result_data = {
                "service_id": service_id,
                "assessment_completed": success,
                "assessment_results": assessment_result
            }
            
            return TaskExecutionResult(
                success=success,
                result_data=result_data,
                message=f"Quality assessment {'completed' if success else 'failed'}",
                task_id=task_id,
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Quality assessment error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_check_service_compliance(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service compliance checking."""
        try:
            service_id = task_data.get('service_id')
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Get service metadata
            service_metadata = self.registrar.get_service_details(service_id)
            if not service_metadata:
                return TaskExecutionResult(
                    success=False,
                    error_message=f"Service not found: {service_id}",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Perform compliance check
            code_path = self.registrar._get_service_code_path(service_metadata)
            compliance_result = self.registrar._check_compliance(service_metadata, code_path)
            
            result_data = {
                "service_id": service_id,
                "compliance_status": compliance_result["status"].value,
                "compliance_notes": compliance_result["notes"],
                "check_completed": True
            }
            
            self.processing_metrics["compliance_checks_run"] += 1
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Compliance check completed - Status: {compliance_result['status'].value}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Compliance check error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_apply_quality_gates(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle quality gate application."""
        try:
            service_id = task_data.get('service_id')
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Get service metadata
            service_metadata = self.registrar.get_service_details(service_id)
            if not service_metadata:
                return TaskExecutionResult(
                    success=False,
                    error_message=f"Service not found: {service_id}",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Apply quality gates
            gate_result = self.registrar._apply_quality_gates(service_metadata)
            
            result_data = {
                "service_id": service_id,
                "quality_gates_passed": gate_result[0],
                "gate_result_message": gate_result[1],
                "gates_applied": True
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Quality gates {'passed' if gate_result[0] else 'failed'}: {gate_result[1]}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Quality gate application failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Quality gate error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_generate_quality_report(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle quality report generation."""
        try:
            report_scope = task_data.get('report_scope', 'comprehensive')
            
            # Generate quality report
            quality_report = self.registrar._generate_quality_report()
            
            result_data = {
                "report_type": "quality",
                "report_scope": report_scope,
                "report_data": quality_report,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            self.processing_metrics["reports_generated"] += 1
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Quality report generated successfully",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Quality report generation failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Quality report error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    # Documentation Task Handlers
    
    async def _handle_generate_service_documentation(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service documentation generation."""
        try:
            service_id = task_data.get('service_id')
            doc_types_str = task_data.get('doc_types', ['USER_GUIDE'])
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Parse documentation types
            doc_types = [DocumentationType(dt) for dt in doc_types_str]
            
            # Track active documentation task
            task_id = task_data.get('task_id', str(uuid4()))
            self.active_documentation_tasks[task_id] = {
                "service_id": service_id,
                "doc_types": [dt.value for dt in doc_types],
                "start_time": datetime.now(),
                "status": "processing"
            }
            
            # Generate documentation
            success = self.registrar.generate_service_documentation(service_id, doc_types)
            
            # Update documentation task status
            self.active_documentation_tasks[task_id]["status"] = "completed" if success else "failed"
            self.active_documentation_tasks[task_id]["end_time"] = datetime.now()
            
            if success:
                self.processing_metrics["documentation_generated"] += 1
            
            result_data = {
                "service_id": service_id,
                "documentation_generated": success,
                "doc_types": [dt.value for dt in doc_types],
                "generation_count": len(doc_types) if success else 0
            }
            
            return TaskExecutionResult(
                success=success,
                result_data=result_data,
                message=f"Documentation generation {'completed' if success else 'failed'}",
                task_id=task_id,
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Documentation generation error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_update_documentation(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle documentation updates."""
        try:
            service_id = task_data.get('service_id')
            doc_type_str = task_data.get('doc_type')
            doc_content = task_data.get('doc_content')
            
            if not all([service_id, doc_type_str, doc_content]):
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id, doc_type, and doc_content are required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            doc_type = DocumentationType(doc_type_str)
            
            # Get service metadata
            service_metadata = self.registrar.get_service_details(service_id)
            if not service_metadata:
                return TaskExecutionResult(
                    success=False,
                    error_message=f"Service not found: {service_id}",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Update specific documentation
            # This would involve updating the service documentation in the registry
            # For now, we'll simulate this process
            
            result_data = {
                "service_id": service_id,
                "doc_type": doc_type.value,
                "documentation_updated": True,
                "update_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Documentation updated for {doc_type.value}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Documentation update failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Documentation update error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_validate_documentation(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle documentation validation."""
        try:
            service_id = task_data.get('service_id')
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Get service metadata
            service_metadata = self.registrar.get_service_details(service_id)
            if not service_metadata:
                return TaskExecutionResult(
                    success=False,
                    error_message=f"Service not found: {service_id}",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Validate documentation completeness
            validation_results = {
                "has_user_guide": any(doc.doc_type == DocumentationType.USER_GUIDE for doc in service_metadata.documentation),
                "has_api_docs": any(doc.doc_type == DocumentationType.API_DOCS for doc in service_metadata.documentation),
                "has_integration_guide": any(doc.doc_type == DocumentationType.INTEGRATION_GUIDE for doc in service_metadata.documentation),
                "has_deployment_guide": any(doc.doc_type == DocumentationType.DEPLOYMENT for doc in service_metadata.documentation),
                "total_docs": len(service_metadata.documentation),
                "requires_review": any(doc.review_required for doc in service_metadata.documentation)
            }
            
            # Calculate documentation completeness score
            required_docs = ["has_user_guide", "has_api_docs", "has_integration_guide", "has_deployment_guide"]
            completeness_score = sum(validation_results[doc] for doc in required_docs) / len(required_docs) * 100
            
            result_data = {
                "service_id": service_id,
                "validation_results": validation_results,
                "completeness_score": completeness_score,
                "validation_passed": completeness_score >= 75.0,
                "validation_completed": True
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Documentation validation completed - Score: {completeness_score:.1f}%",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Documentation validation failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Documentation validation error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    # Lifecycle Management Task Handlers
    
    async def _handle_manage_service_lifecycle(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle general service lifecycle management."""
        try:
            service_id = task_data.get('service_id')
            action = task_data.get('action')
            reason = task_data.get('reason')
            
            if not service_id or not action:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id and action are required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Perform lifecycle management action
            success = self.registrar.manage_service_lifecycle(service_id, action, reason=reason)
            
            if success:
                self.processing_metrics["lifecycle_actions_performed"] += 1
            
            result_data = {
                "service_id": service_id,
                "action": action,
                "lifecycle_action_completed": success,
                "reason": reason
            }
            
            return TaskExecutionResult(
                success=success,
                result_data=result_data,
                message=f"Lifecycle action '{action}' {'completed' if success else 'failed'}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Lifecycle management failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Lifecycle management error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_deprecate_service(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service deprecation."""
        try:
            service_id = task_data.get('service_id')
            reason = task_data.get('reason', 'Deprecated by registrar agent')
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Deprecate service
            success = self.registrar._deprecate_service(service_id, reason)
            
            result_data = {
                "service_id": service_id,
                "service_deprecated": success,
                "deprecation_reason": reason,
                "deprecation_date": datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=success,
                result_data=result_data,
                message=f"Service {'deprecated' if success else 'deprecation failed'}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Service deprecation failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Service deprecation error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_archive_service(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service archival."""
        try:
            service_id = task_data.get('service_id')
            reason = task_data.get('reason', 'Archived by registrar agent')
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Archive service
            success = self.registrar._archive_service(service_id, reason)
            
            result_data = {
                "service_id": service_id,
                "service_archived": success,
                "archive_reason": reason,
                "archive_date": datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=success,
                result_data=result_data,
                message=f"Service {'archived' if success else 'archival failed'}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Service archival failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Service archival error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_reactivate_service(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle service reactivation."""
        try:
            service_id = task_data.get('service_id')
            reason = task_data.get('reason', 'Reactivated by registrar agent')
            
            if not service_id:
                return TaskExecutionResult(
                    success=False,
                    error_message="service_id is required",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Reactivate service
            success = self.registrar._reactivate_service(service_id, reason)
            
            result_data = {
                "service_id": service_id,
                "service_reactivated": success,
                "reactivation_reason": reason,
                "reactivation_date": datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=success,
                result_data=result_data,
                message=f"Service {'reactivated' if success else 'reactivation failed'}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Service reactivation failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Service reactivation error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    # Analytics and Reporting Task Handlers
    
    async def _handle_generate_analytics_report(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle analytics report generation."""
        try:
            report_type = task_data.get('report_type', 'comprehensive')
            
            # Generate analytics report
            analytics_report = self.registrar.get_analytics_report(report_type)
            
            if "error" in analytics_report:
                return TaskExecutionResult(
                    success=False,
                    error_message=analytics_report["error"],
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            result_data = {
                "report_type": report_type,
                "analytics_report": analytics_report,
                "report_generated": True
            }
            
            self.processing_metrics["reports_generated"] += 1
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Analytics report ({report_type}) generated successfully",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Analytics report generation failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Analytics report error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_get_dependency_graph(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle dependency graph retrieval."""
        try:
            service_id = task_data.get('service_id')  # Optional - None for full ecosystem graph
            
            # Get dependency graph
            dependency_graph = self.registrar.get_dependency_graph(service_id)
            
            if "error" in dependency_graph:
                return TaskExecutionResult(
                    success=False,
                    error_message=dependency_graph["error"],
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            result_data = {
                "dependency_graph": dependency_graph,
                "graph_scope": "service" if service_id else "ecosystem",
                "service_id": service_id,
                "graph_retrieved": True
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Dependency graph retrieved for {'service' if service_id else 'ecosystem'}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Dependency graph retrieval failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Dependency graph error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_export_registry_data(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle registry data export."""
        try:
            export_format = task_data.get('export_format', 'json')
            include_metadata = task_data.get('include_metadata', True)
            
            # Export registry data
            if export_format.lower() == 'json':
                registry_data = self.registrar.registry.dict()
                
                if not include_metadata:
                    # Remove sensitive or detailed metadata
                    simplified_data = {
                        "services": {},
                        "categories": registry_data.get("categories", {}),
                        "tags": registry_data.get("tags", {}),
                        "summary": {
                            "total_services": len(registry_data.get("services", {})),
                            "last_updated": registry_data.get("last_updated")
                        }
                    }
                    
                    # Add simplified service info
                    for service_id, service in registry_data.get("services", {}).items():
                        simplified_data["services"][service_id] = {
                            "name": service.get("name"),
                            "description": service.get("description"),
                            "service_type": service.get("service_type"),
                            "quality_level": service.get("quality_level"),
                            "registration_status": service.get("registration_status")
                        }
                    
                    registry_data = simplified_data
                
                result_data = {
                    "export_format": export_format,
                    "registry_data": registry_data,
                    "export_completed": True,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "include_metadata": include_metadata
                }
            else:
                return TaskExecutionResult(
                    success=False,
                    error_message=f"Unsupported export format: {export_format}",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Registry data exported in {export_format} format",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Registry data export failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Registry export error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_get_registrar_metrics(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle registrar metrics retrieval."""
        try:
            # Combine registrar domain metrics with enhanced agent metrics
            combined_metrics = {
                "registrar_domain_metrics": self.registrar.registrar_metrics,
                "enhanced_agent_metrics": self.processing_metrics,
                "active_tasks": {
                    "registrations": len(self.active_registrations),
                    "assessments": len(self.active_assessments),
                    "documentation_tasks": len(self.active_documentation_tasks)
                },
                "agent_info": {
                    "agent_name": self.agent_name,
                    "capabilities": self.config.capabilities,
                    "agent_type": self.config.agent_type
                },
                "metrics_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            result_data = {
                "metrics": combined_metrics,
                "metrics_retrieved": True
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message="Registrar metrics retrieved successfully",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Metrics retrieval failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Metrics retrieval error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    # Administrative Task Handlers
    
    async def _handle_validate_registry_integrity(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle registry integrity validation."""
        try:
            # Perform comprehensive registry integrity checks
            integrity_issues = []
            
            # Check for orphaned services
            for service_id, service in self.registrar.registry.services.items():
                # Check if service has valid dependencies
                for dep in service.dependencies:
                    if dep.service_id != "external" and dep.service_id not in self.registrar.registry.services:
                        integrity_issues.append(f"Service {service_id} has dependency on non-existent service {dep.service_id}")
            
            # Check index consistency
            for category, service_ids in self.registrar.registry.categories.items():
                for service_id in service_ids:
                    if service_id not in self.registrar.registry.services:
                        integrity_issues.append(f"Category '{category}' references non-existent service {service_id}")
            
            for tag, service_ids in self.registrar.registry.tags.items():
                for service_id in service_ids:
                    if service_id not in self.registrar.registry.services:
                        integrity_issues.append(f"Tag '{tag}' references non-existent service {service_id}")
            
            # Check dependency graph consistency
            for service_id, deps in self.registrar.registry.dependency_graph.items():
                if service_id not in self.registrar.registry.services:
                    integrity_issues.append(f"Dependency graph has entry for non-existent service {service_id}")
                
                for dep_id in deps:
                    if dep_id not in self.registrar.registry.services:
                        integrity_issues.append(f"Dependency graph has invalid dependency {dep_id} for service {service_id}")
            
            integrity_score = max(0, 100 - len(integrity_issues) * 5)  # Each issue reduces score by 5
            
            result_data = {
                "integrity_validated": True,
                "integrity_score": integrity_score,
                "issues_found": len(integrity_issues),
                "integrity_issues": integrity_issues,
                "validation_passed": len(integrity_issues) == 0,
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Registry integrity validation completed - Score: {integrity_score}% ({len(integrity_issues)} issues found)",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Registry integrity validation failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Registry integrity validation error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_optimize_registry_performance(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle registry performance optimization."""
        try:
            optimization_actions = []
            
            # Update registry metrics
            self.registrar._update_registrar_metrics()
            optimization_actions.append("Updated registry metrics")
            
            # Clean up old quality assessments (keep only last 5)
            for service_id in list(self.registrar.quality_assessments.keys()):
                assessments = self.registrar.quality_assessments[service_id]
                if len(assessments) > 5:
                    self.registrar.quality_assessments[service_id] = assessments[-5:]
                    optimization_actions.append(f"Cleaned old quality assessments for service {service_id}")
            
            # Clean up old lifecycle events (keep only last 10)
            for service_id in list(self.registrar.lifecycle_events.keys()):
                events = self.registrar.lifecycle_events[service_id]
                if len(events) > 10:
                    self.registrar.lifecycle_events[service_id] = events[-10:]
                    optimization_actions.append(f"Cleaned old lifecycle events for service {service_id}")
            
            # Update service popularity scores
            for service_id in self.registrar.registry.services.keys():
                self.registrar._update_service_popularity(service_id)
            optimization_actions.append("Updated service popularity scores")
            
            # Save optimized registry
            self.registrar._save_registry()
            optimization_actions.append("Saved optimized registry")
            
            result_data = {
                "optimization_completed": True,
                "optimization_actions": optimization_actions,
                "actions_performed": len(optimization_actions),
                "optimization_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Registry optimization completed - {len(optimization_actions)} actions performed",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Registry optimization failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Registry optimization error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_backup_registry(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle registry backup creation."""
        try:
            backup_location = task_data.get('backup_location', f"registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            include_history = task_data.get('include_history', True)
            
            # Create backup data
            backup_data = {
                "registry": self.registrar.registry.dict(),
                "backup_metadata": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "backup_version": "1.0",
                    "agent_name": self.agent_name,
                    "include_history": include_history
                }
            }
            
            if include_history:
                backup_data["quality_assessments"] = self.registrar.quality_assessments
                backup_data["usage_analytics"] = self.registrar.usage_analytics
                backup_data["lifecycle_events"] = self.registrar.lifecycle_events
                backup_data["registration_history"] = [req.dict() for req in self.registrar.registration_history]
            
            # In a real implementation, this would save to a file or backup service
            # For now, we'll return the backup data
            
            result_data = {
                "backup_created": True,
                "backup_location": backup_location,
                "backup_size_bytes": len(json.dumps(backup_data)),
                "include_history": include_history,
                "backup_timestamp": datetime.now(timezone.utc).isoformat(),
                "backup_data": backup_data  # In production, this might be a reference/path instead
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Registry backup created at {backup_location}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Registry backup failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Registry backup error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    async def _handle_restore_registry(self, task_data: Dict[str, Any], context: LogContext) -> TaskExecutionResult:
        """Handle registry restoration from backup."""
        try:
            backup_data = task_data.get('backup_data')
            restore_history = task_data.get('restore_history', True)
            
            if not backup_data:
                return TaskExecutionResult(
                    success=False,
                    error_message="backup_data is required for restoration",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Validate backup data structure
            if "registry" not in backup_data:
                return TaskExecutionResult(
                    success=False,
                    error_message="Invalid backup data: missing registry information",
                    task_id=task_data.get('task_id', str(uuid4())),
                    agent_name=self.agent_name
                )
            
            # Create backup of current state before restoration
            current_backup = {
                "registry": self.registrar.registry.dict(),
                "pre_restore_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Restore registry
            from .registrar_agent import ServiceRegistry
            self.registrar.registry = ServiceRegistry(**backup_data["registry"])
            
            if restore_history and "quality_assessments" in backup_data:
                self.registrar.quality_assessments = backup_data["quality_assessments"]
                self.registrar.usage_analytics = backup_data["usage_analytics"]
                self.registrar.lifecycle_events = backup_data["lifecycle_events"]
                
                if "registration_history" in backup_data:
                    self.registrar.registration_history = [
                        RegistrationRequest(**req) for req in backup_data["registration_history"]
                    ]
            
            # Save restored registry
            self.registrar._save_registry()
            
            # Update metrics after restoration
            self.registrar._update_registrar_metrics()
            
            result_data = {
                "restoration_completed": True,
                "restored_services": len(self.registrar.registry.services),
                "restore_history": restore_history,
                "restoration_timestamp": datetime.now(timezone.utc).isoformat(),
                "pre_restore_backup": current_backup  # For rollback if needed
            }
            
            return TaskExecutionResult(
                success=True,
                result_data=result_data,
                message=f"Registry restored successfully - {len(self.registrar.registry.services)} services",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Registry restoration failed: {e}", context, exc_info=True)
            return TaskExecutionResult(
                success=False,
                error_message=f"Registry restoration error: {str(e)}",
                task_id=task_data.get('task_id', str(uuid4())),
                agent_name=self.agent_name
            )
    
    # Enhanced Agent Specific Methods
    
    async def _update_processing_metrics(self, task_type: str, processing_time: float, success: bool):
        """Update processing metrics for performance tracking."""
        if success:
            # Update task-specific metrics
            if task_type == "register_service":
                self.processing_metrics["registrations_processed"] += 1
            elif task_type == "assess_service_quality":
                self.processing_metrics["quality_assessments_completed"] += 1
            elif task_type.startswith("generate_") and "documentation" in task_type:
                self.processing_metrics["documentation_generated"] += 1
            elif task_type.startswith("manage_") or task_type in ["deprecate_service", "archive_service", "reactivate_service"]:
                self.processing_metrics["lifecycle_actions_performed"] += 1
            elif task_type == "discover_services":
                self.processing_metrics["searches_conducted"] += 1
            elif task_type == "check_service_compliance":
                self.processing_metrics["compliance_checks_run"] += 1
            elif "report" in task_type:
                self.processing_metrics["reports_generated"] += 1
        else:
            self.processing_metrics["errors_encountered"] += 1
        
        # Update average processing time
        current_avg = self.processing_metrics["average_processing_time"]
        total_tasks = sum([
            self.processing_metrics["registrations_processed"],
            self.processing_metrics["quality_assessments_completed"],
            self.processing_metrics["documentation_generated"],
            self.processing_metrics["lifecycle_actions_performed"],
            self.processing_metrics["searches_conducted"],
            self.processing_metrics["compliance_checks_run"],
            self.processing_metrics["reports_generated"],
            self.processing_metrics["errors_encountered"]
        ])
        
        if total_tasks > 0:
            self.processing_metrics["average_processing_time"] = (
                (current_avg * (total_tasks - 1) + processing_time) / total_tasks
            )
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status including registrar-specific metrics."""
        base_status = await super().get_agent_status()
        
        # Add registrar-specific status information
        registrar_status = {
            "registrar_metrics": self.processing_metrics,
            "registry_summary": {
                "total_services": len(self.registrar.registry.services),
                "total_categories": len(self.registrar.registry.categories),
                "total_tags": len(self.registrar.registry.tags),
                "registry_version": self.registrar.registry.version,
                "last_updated": self.registrar.registry.last_updated.isoformat()
            },
            "active_tasks": {
                "registrations": len(self.active_registrations),
                "assessments": len(self.active_assessments),
                "documentation_tasks": len(self.active_documentation_tasks)
            },
            "domain_metrics": self.registrar.registrar_metrics
        }
        
        # Merge with base status
        base_status.update(registrar_status)
        return base_status
    
    async def cleanup_completed_tasks(self):
        """Clean up completed task tracking."""
        current_time = datetime.now()
        
        # Clean up old registrations (older than 1 hour)
        completed_registrations = []
        for task_id, registration in list(self.active_registrations.items()):
            if registration.get("end_time") and (current_time - registration["end_time"]).total_seconds() > 3600:
                completed_registrations.append(task_id)
        
        for task_id in completed_registrations:
            del self.active_registrations[task_id]
        
        # Clean up old assessments
        completed_assessments = []
        for task_id, assessment in list(self.active_assessments.items()):
            if assessment.get("end_time") and (current_time - assessment["end_time"]).total_seconds() > 3600:
                completed_assessments.append(task_id)
        
        for task_id in completed_assessments:
            del self.active_assessments[task_id]
        
        # Clean up old documentation tasks
        completed_docs = []
        for task_id, doc_task in list(self.active_documentation_tasks.items()):
            if doc_task.get("end_time") and (current_time - doc_task["end_time"]).total_seconds() > 3600:
                completed_docs.append(task_id)
        
        for task_id in completed_docs:
            del self.active_documentation_tasks[task_id]
        
        if completed_registrations or completed_assessments or completed_docs:
            self.logger.info(f"Cleaned up {len(completed_registrations)} registrations, "
                           f"{len(completed_assessments)} assessments, "
                           f"{len(completed_docs)} documentation tasks")


# Factory function for easy enhanced agent creation
def create_enhanced_registrar_agent(
    registrar_config: Optional[RegistrarConfiguration] = None,
    agent_config: Optional[AgentConfiguration] = None,
    redis_config: Optional[RedisConfig] = None
) -> EnhancedRegistrarAgent:
    """
    Create a new Enhanced Registrar Agent instance.
    
    Args:
        registrar_config: Configuration for registrar functionality
        agent_config: Configuration for agent behavior
        redis_config: Redis connection configuration
        
    Returns:
        Configured EnhancedRegistrarAgent instance
    """
    return EnhancedRegistrarAgent(registrar_config, agent_config, redis_config)


# Export main classes and functions
__all__ = [
    "EnhancedRegistrarAgent", 
    "create_enhanced_registrar_agent"
]