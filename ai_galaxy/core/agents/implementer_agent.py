"""
Implementer Agent for the AI-Galaxy Lower Meta-Layer.

This module implements the Implementer Agent, the orchestrator of idea-to-code
progression in the AI-Galaxy ecosystem. The Implementer manages the entire
implementation lifecycle, coordinates between different agents and resources,
and ensures smooth progression from approved templates to working microservices.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ...shared.models import Idea, IdeaStatus, Department, Institution, Microservice
from ...shared.logger import get_logger, LogContext
from ..state_machine.router import StateMachineRouter, TransitionResult


class ImplementationPhase(str, Enum):
    """Implementation phases in the idea-to-code progression."""
    TEMPLATE_ANALYSIS = "template_analysis"
    RESOURCE_ALLOCATION = "resource_allocation"
    ENVIRONMENT_SETUP = "environment_setup"
    DEVELOPMENT_KICKOFF = "development_kickoff"
    ACTIVE_DEVELOPMENT = "active_development"
    TESTING_VALIDATION = "testing_validation"
    INTEGRATION_VERIFICATION = "integration_verification"
    DEPLOYMENT_PREPARATION = "deployment_preparation"
    PRODUCTION_DEPLOYMENT = "production_deployment"
    POST_DEPLOYMENT_MONITORING = "post_deployment_monitoring"
    COMPLETION_REGISTRATION = "completion_registration"


class ImplementationStatus(str, Enum):
    """Status of implementation tasks and phases."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class ResourceType(str, Enum):
    """Types of resources managed by the Implementer."""
    DEVELOPER = "developer"
    INFRASTRUCTURE = "infrastructure"
    COMPUTING_RESOURCES = "computing_resources"
    EXTERNAL_SERVICES = "external_services"
    BUDGET = "budget"
    TIME_ALLOCATION = "time_allocation"


class RiskLevel(str, Enum):
    """Risk levels for implementation risks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentRole(str, Enum):
    """Roles of different agents in the implementation."""
    PROGRAMMER = "programmer"
    ROUTER = "router"
    REGISTRAR = "registrar"
    COUNCIL = "council"
    QA_VALIDATOR = "qa_validator"
    DEPLOYMENT_MANAGER = "deployment_manager"


@dataclass
class ImplementationTask:
    """Represents a specific task in the implementation pipeline."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    phase: ImplementationPhase = ImplementationPhase.TEMPLATE_ANALYSIS
    status: ImplementationStatus = ImplementationStatus.PENDING
    assigned_agent: Optional[str] = None
    assigned_resources: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    completion_percentage: float = 0.0
    blocking_issues: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)


@dataclass
class Resource:
    """Represents a resource in the implementation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    type: ResourceType = ResourceType.DEVELOPER
    capacity: float = 100.0  # Percentage capacity
    current_utilization: float = 0.0
    skills: List[str] = field(default_factory=list)
    availability_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    availability_end: Optional[datetime] = None
    cost_per_hour: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImplementationRisk:
    """Represents a risk in the implementation process."""
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    probability: float = 0.5  # 0.0 to 1.0
    impact: float = 0.5  # 0.0 to 1.0
    risk_score: float = field(init=False)
    category: str = ""
    affected_phases: List[ImplementationPhase] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    contingency_plans: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    status: str = "active"  # active, mitigated, realized, closed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        self.risk_score = self.probability * self.impact


class EnvironmentConfig(BaseModel):
    """Configuration for development environment setup."""
    environment_name: str
    environment_type: str = "development"  # development, staging, production
    python_version: str = "3.9"
    dependencies: List[str] = Field(default_factory=list)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    infrastructure_requirements: Dict[str, Any] = Field(default_factory=dict)
    service_integrations: List[str] = Field(default_factory=list)
    ci_cd_pipeline: Dict[str, Any] = Field(default_factory=dict)
    testing_framework: str = "pytest"
    deployment_target: str = "local"


class QualityGate(BaseModel):
    """Represents a quality gate in the implementation process."""
    name: str
    phase: ImplementationPhase
    criteria: List[str] = Field(default_factory=list)
    validation_script: Optional[str] = None
    required_approvals: List[str] = Field(default_factory=list)
    automated_checks: List[str] = Field(default_factory=list)
    manual_checks: List[str] = Field(default_factory=list)
    blocking: bool = True
    timeout_hours: int = 24


class ProgressCheckpoint(BaseModel):
    """Represents a progress checkpoint with metrics."""
    checkpoint_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phase: ImplementationPhase
    completion_percentage: float = Field(ge=0.0, le=100.0)
    tasks_completed: int = 0
    tasks_remaining: int = 0
    issues_identified: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    resource_utilization: Dict[str, float] = Field(default_factory=dict)
    next_milestones: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AgentCoordination(BaseModel):
    """Coordination information for agent interactions."""
    agent_role: AgentRole
    agent_endpoint: Optional[str] = None
    communication_protocol: str = "async_message"
    required_capabilities: List[str] = Field(default_factory=list)
    current_assignments: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    availability_status: str = "available"  # available, busy, offline
    last_communication: Optional[datetime] = None


class ImplementationPlan(BaseModel):
    """Comprehensive implementation plan for an idea."""
    idea_id: str
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Plan overview
    project_name: str
    description: str
    estimated_duration_hours: float
    target_completion_date: datetime
    
    # Implementation breakdown
    phases: List[ImplementationPhase] = Field(default_factory=list)
    tasks: List[ImplementationTask] = Field(default_factory=list)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Resource allocation
    resource_requirements: Dict[ResourceType, int] = Field(default_factory=dict)
    allocated_resources: List[Resource] = Field(default_factory=list)
    budget_allocation: Dict[str, float] = Field(default_factory=dict)
    
    # Risk management
    identified_risks: List[ImplementationRisk] = Field(default_factory=list)
    risk_mitigation_plan: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Quality and monitoring
    quality_gates: List[QualityGate] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    kpis: Dict[str, float] = Field(default_factory=dict)
    
    # Environment and deployment
    environment_config: Optional[EnvironmentConfig] = None
    deployment_strategy: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent coordination
    agent_assignments: Dict[AgentRole, AgentCoordination] = Field(default_factory=dict)


class ImplementationMetrics(BaseModel):
    """Comprehensive metrics for implementation tracking."""
    total_implementations: int = 0
    successful_implementations: int = 0
    failed_implementations: int = 0
    average_completion_time: float = 0.0
    average_resource_utilization: float = 0.0
    
    # Phase-specific metrics
    phase_completion_rates: Dict[ImplementationPhase, float] = Field(default_factory=dict)
    phase_average_durations: Dict[ImplementationPhase, float] = Field(default_factory=dict)
    
    # Quality metrics
    quality_gate_pass_rates: Dict[str, float] = Field(default_factory=dict)
    defect_rates: Dict[ImplementationPhase, float] = Field(default_factory=dict)
    
    # Resource metrics
    resource_efficiency: Dict[ResourceType, float] = Field(default_factory=dict)
    agent_performance: Dict[AgentRole, float] = Field(default_factory=dict)
    
    # Risk metrics
    risk_realization_rate: float = 0.0
    average_risk_impact: float = 0.0
    mitigation_success_rate: float = 0.0


class ImplementerConfiguration(BaseModel):
    """Configuration for the Implementer Agent."""
    # Orchestration settings
    max_concurrent_implementations: int = 5
    default_implementation_timeout_hours: int = 168  # 1 week
    enable_adaptive_scheduling: bool = True
    enable_cross_agent_coordination: bool = True
    
    # Resource management
    resource_allocation_strategy: str = "balanced"  # balanced, performance, cost_optimized
    resource_overcommit_percentage: float = 10.0
    enable_dynamic_scaling: bool = True
    
    # Quality assurance
    require_quality_gates: bool = True
    quality_gate_timeout_hours: int = 24
    enable_automated_testing: bool = True
    
    # Risk management
    risk_assessment_frequency_hours: int = 24
    auto_escalate_high_risks: bool = True
    enable_predictive_risk_analysis: bool = True
    
    # Environment management
    auto_provision_environments: bool = True
    environment_cleanup_delay_hours: int = 72
    enable_environment_monitoring: bool = True
    
    # Communication settings
    agent_communication_timeout_seconds: int = 30
    progress_update_frequency_hours: int = 8
    enable_real_time_notifications: bool = True


class ImplementerAgent:
    """
    The Implementer Agent - orchestrator of idea-to-code progression.
    
    Manages the entire implementation lifecycle, coordinates between agents,
    handles resource allocation, and ensures quality through the complete
    development pipeline from approved templates to working microservices.
    """
    
    def __init__(self, config: Optional[ImplementerConfiguration] = None,
                 state_router: Optional[StateMachineRouter] = None):
        """
        Initialize the Implementer Agent.
        
        Args:
            config: Implementer configuration parameters
            state_router: State machine router for workflow transitions
        """
        self.logger = get_logger("implementer_agent")
        self.config = config or ImplementerConfiguration()
        self.state_router = state_router
        
        # Implementation tracking
        self.active_implementations: Dict[str, ImplementationPlan] = {}
        self.implementation_history: Dict[str, ImplementationPlan] = {}
        self.progress_checkpoints: Dict[str, List[ProgressCheckpoint]] = defaultdict(list)
        
        # Resource management
        self.available_resources: Dict[ResourceType, List[Resource]] = defaultdict(list)
        self.resource_allocations: Dict[str, List[Resource]] = defaultdict(list)
        self.resource_utilization_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Agent coordination
        self.registered_agents: Dict[AgentRole, AgentCoordination] = {}
        self.agent_communication_queue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Risk management
        self.global_risks: List[ImplementationRisk] = []
        self.risk_patterns: Dict[str, List[ImplementationRisk]] = defaultdict(list)
        
        # Performance metrics
        self.implementation_metrics = ImplementationMetrics()
        
        # Environment management
        self.environment_templates: Dict[str, EnvironmentConfig] = {}
        self.active_environments: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default resources and templates
        self._initialize_default_resources()
        self._initialize_environment_templates()
        self._initialize_agent_registrations()
        
        self.logger.agent_action("implementer_agent_initialized", "implementer_agent",
                                additional_context={
                                    "max_concurrent": self.config.max_concurrent_implementations,
                                    "adaptive_scheduling": self.config.enable_adaptive_scheduling,
                                    "auto_provision": self.config.auto_provision_environments,
                                    "quality_gates": self.config.require_quality_gates
                                })
    
    async def orchestrate_implementation(self, idea: Idea, template_data: Dict[str, Any]) -> ImplementationPlan:
        """
        Orchestrate the complete implementation of an approved idea.
        
        Args:
            idea: The approved idea to implement
            template_data: Template data from the Council Agent
            
        Returns:
            Comprehensive implementation plan with orchestration details
        """
        start_time = datetime.now(timezone.utc)
        idea_id = str(idea.id)
        
        context = LogContext(
            agent_name="implementer_agent",
            idea_id=idea_id,
            additional_context={"orchestration_start": start_time.isoformat()}
        )
        
        self.logger.agent_action("starting_implementation_orchestration", "implementer_agent", idea_id)
        
        try:
            # Phase 1: Analyze template and create implementation plan
            implementation_plan = await self._analyze_template_and_create_plan(idea, template_data)
            
            # Phase 2: Allocate resources
            await self._allocate_implementation_resources(implementation_plan)
            
            # Phase 3: Set up development environment
            await self._setup_development_environment(implementation_plan)
            
            # Phase 4: Coordinate with agents and initiate development
            await self._coordinate_development_kickoff(implementation_plan)
            
            # Phase 5: Monitor and orchestrate active development
            await self._orchestrate_active_development(implementation_plan)
            
            # Store implementation plan
            self.active_implementations[idea_id] = implementation_plan
            
            # Create initial progress checkpoint
            initial_checkpoint = ProgressCheckpoint(
                checkpoint_id=str(uuid4()),
                phase=ImplementationPhase.DEVELOPMENT_KICKOFF,
                completion_percentage=0.0,
                next_milestones=["Environment Setup", "Agent Coordination", "Initial Development Tasks"],
                recommendations=["Monitor resource allocation", "Track progress against milestones"]
            )
            self.progress_checkpoints[idea_id].append(initial_checkpoint)
            
            # Update metrics
            self.implementation_metrics.total_implementations += 1
            
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.agent_action("implementation_orchestration_completed", "implementer_agent", idea_id, {
                "plan_id": implementation_plan.plan_id,
                "estimated_duration": implementation_plan.estimated_duration_hours,
                "phases_count": len(implementation_plan.phases),
                "tasks_count": len(implementation_plan.tasks),
                "orchestration_time": orchestration_time
            })
            
            return implementation_plan
            
        except Exception as e:
            self.logger.error(f"Implementation orchestration failed: {e}", context, exc_info=True)
            
            # Create fallback plan
            fallback_plan = self._create_fallback_implementation_plan(idea, template_data, str(e))
            self.active_implementations[idea_id] = fallback_plan
            
            return fallback_plan
    
    async def _analyze_template_and_create_plan(self, idea: Idea, template_data: Dict[str, Any]) -> ImplementationPlan:
        """Analyze template data and create detailed implementation plan."""
        idea_id = str(idea.id)
        
        # Extract key information from template
        project_name = template_data.get("project_name", idea.title)
        description = template_data.get("description", idea.description)
        complexity = template_data.get("complexity", "medium")
        technology_stack = template_data.get("technology_stack", [])
        
        # Estimate duration based on complexity and scope
        base_hours = {
            "simple": 40,
            "medium": 120,
            "complex": 300,
            "enterprise": 600
        }
        estimated_duration = base_hours.get(complexity, 120)
        
        # Create implementation phases
        phases = list(ImplementationPhase)
        
        # Create detailed tasks for each phase
        tasks = self._generate_implementation_tasks(template_data, complexity)
        
        # Identify dependencies
        dependencies = self._identify_task_dependencies(tasks)
        
        # Assess resource requirements
        resource_requirements = self._assess_resource_requirements(template_data, complexity)
        
        # Identify risks
        risks = await self._identify_implementation_risks(template_data, complexity)
        
        # Create quality gates
        quality_gates = self._create_quality_gates(phases, complexity)
        
        # Define success criteria
        success_criteria = self._define_success_criteria(template_data)
        
        # Create environment configuration
        environment_config = self._create_environment_config(template_data, technology_stack)
        
        # Calculate target completion date
        target_completion = datetime.now(timezone.utc) + timedelta(hours=estimated_duration)
        
        # Create implementation plan
        implementation_plan = ImplementationPlan(
            idea_id=idea_id,
            project_name=project_name,
            description=description,
            estimated_duration_hours=estimated_duration,
            target_completion_date=target_completion,
            phases=phases,
            tasks=tasks,
            dependencies=dependencies,
            resource_requirements=resource_requirements,
            identified_risks=risks,
            quality_gates=quality_gates,
            success_criteria=success_criteria,
            environment_config=environment_config
        )
        
        self.logger.info(f"Implementation plan created with {len(tasks)} tasks and {len(risks)} risks", 
                        LogContext(agent_name="implementer_agent", idea_id=idea_id))
        
        return implementation_plan
    
    def _generate_implementation_tasks(self, template_data: Dict[str, Any], complexity: str) -> List[ImplementationTask]:
        """Generate detailed implementation tasks based on template analysis."""
        tasks = []
        
        # Template analysis tasks
        tasks.append(ImplementationTask(
            name="Template Requirements Analysis",
            description="Analyze template requirements and create technical specifications",
            phase=ImplementationPhase.TEMPLATE_ANALYSIS,
            estimated_hours=8,
            validation_criteria=["Requirements documented", "Technical specs approved"],
            deliverables=["requirements_doc.md", "technical_specifications.md"]
        ))
        
        # Environment setup tasks
        tasks.append(ImplementationTask(
            name="Development Environment Setup",
            description="Set up development environment with required dependencies",
            phase=ImplementationPhase.ENVIRONMENT_SETUP,
            estimated_hours=4,
            validation_criteria=["Environment provisioned", "Dependencies installed", "Tests passing"],
            deliverables=["environment_config.yaml", "setup_verification.log"]
        ))
        
        # Development tasks based on complexity
        if complexity in ["medium", "complex", "enterprise"]:
            tasks.append(ImplementationTask(
                name="Core Architecture Implementation",
                description="Implement core system architecture and foundations",
                phase=ImplementationPhase.ACTIVE_DEVELOPMENT,
                estimated_hours=40 if complexity == "medium" else 80,
                validation_criteria=["Architecture implemented", "Core tests passing"],
                deliverables=["core_modules/", "architecture_tests/"]
            ))
            
            tasks.append(ImplementationTask(
                name="Business Logic Implementation",
                description="Implement main business logic and functionality",
                phase=ImplementationPhase.ACTIVE_DEVELOPMENT,
                estimated_hours=30 if complexity == "medium" else 60,
                validation_criteria=["Business logic complete", "Unit tests coverage > 80%"],
                deliverables=["business_logic/", "unit_tests/"]
            ))
        
        # Integration tasks
        tasks.append(ImplementationTask(
            name="Integration Implementation",
            description="Implement required integrations with external services",
            phase=ImplementationPhase.INTEGRATION_VERIFICATION,
            estimated_hours=16,
            validation_criteria=["Integrations working", "Integration tests passing"],
            deliverables=["integrations/", "integration_tests/"]
        ))
        
        # Testing tasks
        tasks.append(ImplementationTask(
            name="Comprehensive Testing",
            description="Execute comprehensive testing including unit, integration, and e2e tests",
            phase=ImplementationPhase.TESTING_VALIDATION,
            estimated_hours=20,
            validation_criteria=["All tests passing", "Coverage targets met"],
            deliverables=["test_results.xml", "coverage_report.html"]
        ))
        
        # Deployment preparation
        tasks.append(ImplementationTask(
            name="Deployment Preparation",
            description="Prepare application for deployment with CI/CD pipeline",
            phase=ImplementationPhase.DEPLOYMENT_PREPARATION,
            estimated_hours=8,
            validation_criteria=["CI/CD pipeline working", "Deployment scripts tested"],
            deliverables=["Dockerfile", "deployment.yaml", "ci_cd_pipeline.yaml"]
        ))
        
        # Production deployment
        tasks.append(ImplementationTask(
            name="Production Deployment",
            description="Deploy application to production environment",
            phase=ImplementationPhase.PRODUCTION_DEPLOYMENT,
            estimated_hours=4,
            validation_criteria=["Application deployed", "Health checks passing"],
            deliverables=["deployment_logs/", "health_check_results.json"]
        ))
        
        return tasks
    
    def _identify_task_dependencies(self, tasks: List[ImplementationTask]) -> Dict[str, List[str]]:
        """Identify dependencies between implementation tasks."""
        dependencies = {}
        
        # Create a mapping of phase order
        phase_order = {phase: i for i, phase in enumerate(ImplementationPhase)}
        
        for task in tasks:
            task_deps = []
            
            # Tasks depend on previous phase tasks
            current_phase_order = phase_order[task.phase]
            
            for other_task in tasks:
                other_phase_order = phase_order[other_task.phase]
                
                # Depend on tasks from previous phases
                if other_phase_order < current_phase_order:
                    task_deps.append(other_task.id)
                
                # Special dependencies within same phase
                elif (other_phase_order == current_phase_order and 
                      other_task.id != task.id):
                    # Core architecture before business logic
                    if (task.name == "Business Logic Implementation" and 
                        other_task.name == "Core Architecture Implementation"):
                        task_deps.append(other_task.id)
            
            dependencies[task.id] = task_deps
        
        return dependencies
    
    def _assess_resource_requirements(self, template_data: Dict[str, Any], complexity: str) -> Dict[ResourceType, int]:
        """Assess resource requirements based on template complexity."""
        base_requirements = {
            "simple": {
                ResourceType.DEVELOPER: 1,
                ResourceType.COMPUTING_RESOURCES: 1,
                ResourceType.INFRASTRUCTURE: 1
            },
            "medium": {
                ResourceType.DEVELOPER: 2,
                ResourceType.COMPUTING_RESOURCES: 2,
                ResourceType.INFRASTRUCTURE: 2
            },
            "complex": {
                ResourceType.DEVELOPER: 4,
                ResourceType.COMPUTING_RESOURCES: 4,
                ResourceType.INFRASTRUCTURE: 3,
                ResourceType.EXTERNAL_SERVICES: 2
            },
            "enterprise": {
                ResourceType.DEVELOPER: 6,
                ResourceType.COMPUTING_RESOURCES: 8,
                ResourceType.INFRASTRUCTURE: 5,
                ResourceType.EXTERNAL_SERVICES: 4
            }
        }
        
        requirements = base_requirements.get(complexity, base_requirements["medium"])
        
        # Adjust based on specific template requirements
        technology_stack = template_data.get("technology_stack", [])
        if "kubernetes" in technology_stack:
            requirements[ResourceType.INFRASTRUCTURE] += 1
        if "machine_learning" in template_data.get("capabilities", []):
            requirements[ResourceType.COMPUTING_RESOURCES] += 2
        
        return requirements
    
    async def _identify_implementation_risks(self, template_data: Dict[str, Any], complexity: str) -> List[ImplementationRisk]:
        """Identify potential risks in the implementation process."""
        risks = []
        
        # Complexity-based risks
        if complexity in ["complex", "enterprise"]:
            risks.append(ImplementationRisk(
                title="High Complexity Risk",
                description="Complex implementation may exceed time estimates",
                risk_level=RiskLevel.MEDIUM,
                probability=0.4,
                impact=0.7,
                category="schedule",
                affected_phases=[ImplementationPhase.ACTIVE_DEVELOPMENT],
                mitigation_strategies=[
                    "Break down into smaller milestones",
                    "Regular progress reviews",
                    "Parallel development streams"
                ],
                contingency_plans=[
                    "Scope reduction if behind schedule",
                    "Additional resource allocation"
                ]
            ))
        
        # Technology stack risks
        technology_stack = template_data.get("technology_stack", [])
        if len(technology_stack) > 5:
            risks.append(ImplementationRisk(
                title="Technology Integration Risk",
                description="Multiple technologies may cause integration challenges",
                risk_level=RiskLevel.MEDIUM,
                probability=0.5,
                impact=0.6,
                category="technical",
                affected_phases=[ImplementationPhase.INTEGRATION_VERIFICATION],
                mitigation_strategies=[
                    "Early integration testing",
                    "Technology compatibility verification",
                    "Fallback technology options"
                ]
            ))
        
        # Resource availability risk
        risks.append(ImplementationRisk(
            title="Resource Availability Risk",
            description="Required resources may not be available when needed",
            risk_level=RiskLevel.LOW,
            probability=0.3,
            impact=0.5,
            category="resource",
            affected_phases=[ImplementationPhase.RESOURCE_ALLOCATION],
            mitigation_strategies=[
                "Early resource reservation",
                "Cross-training team members",
                "External resource options"
            ]
        ))
        
        # Quality risk
        if template_data.get("performance_requirements"):
            risks.append(ImplementationRisk(
                title="Performance Requirements Risk",
                description="Application may not meet performance requirements",
                risk_level=RiskLevel.MEDIUM,
                probability=0.4,
                impact=0.8,
                category="quality",
                affected_phases=[ImplementationPhase.TESTING_VALIDATION],
                mitigation_strategies=[
                    "Early performance testing",
                    "Performance monitoring setup",
                    "Load testing automation"
                ]
            ))
        
        return risks
    
    def _create_quality_gates(self, phases: List[ImplementationPhase], complexity: str) -> List[QualityGate]:
        """Create quality gates for the implementation phases."""
        quality_gates = []
        
        # Template analysis gate
        quality_gates.append(QualityGate(
            name="Template Analysis Complete",
            phase=ImplementationPhase.TEMPLATE_ANALYSIS,
            criteria=[
                "Requirements documented and approved",
                "Technical specifications complete",
                "Architecture design reviewed"
            ],
            required_approvals=["lead_developer", "project_manager"],
            automated_checks=["documentation_completeness"],
            manual_checks=["technical_review", "stakeholder_approval"]
        ))
        
        # Development milestone gates
        quality_gates.append(QualityGate(
            name="Development Milestone",
            phase=ImplementationPhase.ACTIVE_DEVELOPMENT,
            criteria=[
                "Core functionality implemented",
                "Unit test coverage >= 80%",
                "Code review completed",
                "No critical security vulnerabilities"
            ],
            automated_checks=["unit_tests", "code_coverage", "security_scan"],
            manual_checks=["code_review", "functionality_verification"]
        ))
        
        # Integration gate
        quality_gates.append(QualityGate(
            name="Integration Complete",
            phase=ImplementationPhase.INTEGRATION_VERIFICATION,
            criteria=[
                "All integrations working",
                "Integration tests passing",
                "Data flow verified",
                "Error handling tested"
            ],
            automated_checks=["integration_tests", "api_tests"],
            manual_checks=["end_to_end_verification"]
        ))
        
        # Deployment readiness gate
        quality_gates.append(QualityGate(
            name="Deployment Ready",
            phase=ImplementationPhase.DEPLOYMENT_PREPARATION,
            criteria=[
                "All tests passing",
                "Performance benchmarks met",
                "Security scan clean",
                "Documentation complete",
                "CI/CD pipeline working"
            ],
            automated_checks=["full_test_suite", "performance_tests", "security_scan", "pipeline_test"],
            manual_checks=["final_review", "deployment_checklist"]
        ))
        
        return quality_gates
    
    def _define_success_criteria(self, template_data: Dict[str, Any]) -> List[str]:
        """Define success criteria for the implementation."""
        criteria = [
            "All functional requirements implemented",
            "All tests passing with >= 80% coverage",
            "Performance requirements met",
            "Security requirements satisfied",
            "Documentation complete",
            "Successfully deployed to production",
            "Post-deployment monitoring active"
        ]
        
        # Add template-specific criteria
        if template_data.get("api_requirements"):
            criteria.append("API endpoints operational and documented")
        
        if template_data.get("ui_requirements"):
            criteria.append("User interface meets design specifications")
        
        if template_data.get("integration_requirements"):
            criteria.append("All external integrations working correctly")
        
        return criteria
    
    def _create_environment_config(self, template_data: Dict[str, Any], technology_stack: List[str]) -> EnvironmentConfig:
        """Create environment configuration based on template requirements."""
        return EnvironmentConfig(
            environment_name=f"dev_{template_data.get('project_name', 'project').lower()}",
            python_version=template_data.get("python_version", "3.9"),
            dependencies=technology_stack,
            environment_variables={
                "ENV": "development",
                "DEBUG": "true",
                "LOG_LEVEL": "INFO"
            },
            infrastructure_requirements={
                "cpu": template_data.get("cpu_requirements", "2 cores"),
                "memory": template_data.get("memory_requirements", "4GB"),
                "storage": template_data.get("storage_requirements", "20GB")
            },
            service_integrations=template_data.get("integrations", []),
            ci_cd_pipeline={
                "trigger": "push",
                "stages": ["test", "build", "deploy"],
                "notifications": True
            },
            testing_framework=template_data.get("testing_framework", "pytest"),
            deployment_target=template_data.get("deployment_target", "local")
        )
    
    async def _allocate_implementation_resources(self, plan: ImplementationPlan):
        """Allocate resources for the implementation plan."""
        self.logger.info(f"Allocating resources for implementation {plan.plan_id}",
                        LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
        
        allocated_resources = []
        
        for resource_type, required_count in plan.resource_requirements.items():
            available = self.available_resources.get(resource_type, [])
            
            # Sort by availability and utilization
            available.sort(key=lambda r: r.current_utilization)
            
            allocated = 0
            for resource in available:
                if allocated >= required_count:
                    break
                
                if resource.current_utilization < 80.0:  # 80% utilization threshold
                    allocated_resources.append(resource)
                    resource.current_utilization += 50.0  # Reserve 50% capacity
                    allocated += 1
        
        plan.allocated_resources = allocated_resources
        self.resource_allocations[plan.idea_id] = allocated_resources
        
        self.logger.info(f"Allocated {len(allocated_resources)} resources",
                        LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
    
    async def _setup_development_environment(self, plan: ImplementationPlan):
        """Set up development environment for the implementation."""
        if not self.config.auto_provision_environments or not plan.environment_config:
            return
        
        self.logger.info(f"Setting up development environment for {plan.plan_id}",
                        LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
        
        try:
            env_config = plan.environment_config
            
            # Create environment directory
            env_path = Path(tempfile.mkdtemp(prefix=f"aidev_{plan.idea_id}_"))
            
            # Create basic project structure
            (env_path / "src").mkdir()
            (env_path / "tests").mkdir()
            (env_path / "docs").mkdir()
            (env_path / "config").mkdir()
            
            # Create requirements file
            if env_config.dependencies:
                requirements_file = env_path / "requirements.txt"
                requirements_file.write_text("\n".join(env_config.dependencies))
            
            # Create environment configuration file
            config_file = env_path / "config" / "environment.yaml"
            config_file.write_text(yaml.dump(env_config.dict()))
            
            # Create basic CI/CD pipeline file
            if env_config.ci_cd_pipeline:
                pipeline_file = env_path / ".github" / "workflows" / "ci.yaml"
                pipeline_file.parent.mkdir(parents=True, exist_ok=True)
                pipeline_file.write_text(yaml.dump(env_config.ci_cd_pipeline))
            
            # Store environment information
            self.active_environments[plan.idea_id] = {
                "path": str(env_path),
                "config": env_config.dict(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            
            self.logger.info(f"Development environment created at {env_path}",
                            LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
            
        except Exception as e:
            self.logger.error(f"Failed to setup development environment: {e}",
                            LogContext(agent_name="implementer_agent", idea_id=plan.idea_id),
                            exc_info=True)
    
    async def _coordinate_development_kickoff(self, plan: ImplementationPlan):
        """Coordinate with other agents to kick off development."""
        self.logger.info(f"Coordinating development kickoff for {plan.plan_id}",
                        LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
        
        # Coordinate with Programmer Agent
        if AgentRole.PROGRAMMER in self.registered_agents:
            await self._coordinate_with_agent(
                AgentRole.PROGRAMMER,
                "development_assignment",
                {
                    "plan_id": plan.plan_id,
                    "idea_id": plan.idea_id,
                    "tasks": [task.__dict__ for task in plan.tasks if task.phase == ImplementationPhase.ACTIVE_DEVELOPMENT],
                    "environment_config": plan.environment_config.dict() if plan.environment_config else None,
                    "success_criteria": plan.success_criteria
                }
            )
        
        # Coordinate with Router Agent for resource routing
        if AgentRole.ROUTER in self.registered_agents:
            await self._coordinate_with_agent(
                AgentRole.ROUTER,
                "resource_routing_update",
                {
                    "plan_id": plan.plan_id,
                    "allocated_resources": [r.__dict__ for r in plan.allocated_resources],
                    "implementation_priority": "high"
                }
            )
        
        # Update agent assignments in plan
        for agent_role in [AgentRole.PROGRAMMER, AgentRole.QA_VALIDATOR]:
            if agent_role in self.registered_agents:
                plan.agent_assignments[agent_role] = self.registered_agents[agent_role]
    
    async def _orchestrate_active_development(self, plan: ImplementationPlan):
        """Orchestrate the active development phase with monitoring."""
        self.logger.info(f"Starting active development orchestration for {plan.plan_id}",
                        LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
        
        # Start monitoring tasks in background
        asyncio.create_task(self._monitor_implementation_progress(plan))
        
        # Start risk monitoring
        asyncio.create_task(self._monitor_implementation_risks(plan))
        
        # Start resource utilization monitoring
        asyncio.create_task(self._monitor_resource_utilization(plan))
    
    async def _monitor_implementation_progress(self, plan: ImplementationPlan):
        """Monitor implementation progress and create checkpoints."""
        while plan.idea_id in self.active_implementations:
            try:
                # Create progress checkpoint
                checkpoint = await self._create_progress_checkpoint(plan)
                self.progress_checkpoints[plan.idea_id].append(checkpoint)
                
                # Check if implementation is complete
                if checkpoint.completion_percentage >= 100.0:
                    await self._handle_implementation_completion(plan)
                    break
                
                # Check for blocking issues
                if checkpoint.issues_identified:
                    await self._handle_blocking_issues(plan, checkpoint.issues_identified)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config.progress_update_frequency_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Error monitoring implementation progress: {e}",
                                LogContext(agent_name="implementer_agent", idea_id=plan.idea_id),
                                exc_info=True)
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def _create_progress_checkpoint(self, plan: ImplementationPlan) -> ProgressCheckpoint:
        """Create a progress checkpoint for the implementation."""
        # Calculate completion percentage based on completed tasks
        total_tasks = len(plan.tasks)
        completed_tasks = sum(1 for task in plan.tasks if task.status == ImplementationStatus.COMPLETED)
        completion_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Identify current phase
        current_phase = ImplementationPhase.TEMPLATE_ANALYSIS
        for phase in ImplementationPhase:
            phase_tasks = [t for t in plan.tasks if t.phase == phase]
            if phase_tasks and any(t.status in [ImplementationStatus.IN_PROGRESS, ImplementationStatus.PENDING] for t in phase_tasks):
                current_phase = phase
                break
        
        # Identify issues
        issues = []
        for task in plan.tasks:
            if task.status == ImplementationStatus.BLOCKED:
                issues.extend(task.blocking_issues)
        
        # Calculate resource utilization
        resource_utilization = {}
        for resource in plan.allocated_resources:
            resource_utilization[f"{resource.type}_{resource.id}"] = resource.current_utilization
        
        # Generate recommendations
        recommendations = []
        if completion_percentage < 50 and datetime.now(timezone.utc) > plan.target_completion_date - timedelta(days=2):
            recommendations.append("Consider scope reduction or timeline extension")
        
        if len(issues) > 3:
            recommendations.append("Focus on resolving blocking issues before proceeding")
        
        return ProgressCheckpoint(
            checkpoint_id=str(uuid4()),
            phase=current_phase,
            completion_percentage=completion_percentage,
            tasks_completed=completed_tasks,
            tasks_remaining=total_tasks - completed_tasks,
            issues_identified=issues,
            resource_utilization=resource_utilization,
            next_milestones=[t.name for t in plan.tasks if t.status == ImplementationStatus.PENDING][:3],
            recommendations=recommendations
        )
    
    async def _monitor_implementation_risks(self, plan: ImplementationPlan):
        """Monitor implementation risks and trigger mitigation strategies."""
        while plan.idea_id in self.active_implementations:
            try:
                for risk in plan.identified_risks:
                    if risk.status == "active":
                        # Check if risk conditions are met
                        risk_triggered = await self._assess_risk_trigger(risk, plan)
                        
                        if risk_triggered:
                            await self._trigger_risk_mitigation(risk, plan)
                
                await asyncio.sleep(self.config.risk_assessment_frequency_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Error monitoring implementation risks: {e}",
                                LogContext(agent_name="implementer_agent", idea_id=plan.idea_id),
                                exc_info=True)
                await asyncio.sleep(3600)
    
    async def _monitor_resource_utilization(self, plan: ImplementationPlan):
        """Monitor resource utilization and optimize allocation."""
        while plan.idea_id in self.active_implementations:
            try:
                for resource in plan.allocated_resources:
                    # Track utilization history
                    utilization_entry = (datetime.now(timezone.utc), resource.current_utilization)
                    self.resource_utilization_history[resource.id].append(utilization_entry)
                    
                    # Optimize if under/over utilized
                    if resource.current_utilization < 30:
                        await self._optimize_resource_allocation(resource, plan, "underutilized")
                    elif resource.current_utilization > 90:
                        await self._optimize_resource_allocation(resource, plan, "overutilized")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error monitoring resource utilization: {e}",
                                LogContext(agent_name="implementer_agent", idea_id=plan.idea_id),
                                exc_info=True)
                await asyncio.sleep(3600)
    
    async def _coordinate_with_agent(self, agent_role: AgentRole, action: str, data: Dict[str, Any]):
        """Coordinate with a specific agent role."""
        if agent_role not in self.registered_agents:
            self.logger.warning(f"Agent {agent_role.value} not registered for coordination")
            return
        
        coordination = self.registered_agents[agent_role]
        
        # Create coordination message
        message = {
            "action": action,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_agent": "implementer_agent",
            "correlation_id": str(uuid4())
        }
        
        # Add to communication queue
        self.agent_communication_queue[agent_role.value].append(message)
        
        # Update coordination metrics
        coordination.last_communication = datetime.now(timezone.utc)
        
        self.logger.info(f"Coordinated with {agent_role.value}: {action}",
                        LogContext(agent_name="implementer_agent"))
    
    def _initialize_default_resources(self):
        """Initialize default resources for implementations."""
        # Developer resources
        for i in range(5):
            dev_resource = Resource(
                name=f"Developer_{i+1}",
                type=ResourceType.DEVELOPER,
                capacity=100.0,
                current_utilization=0.0,
                skills=["python", "javascript", "sql", "docker"],
                cost_per_hour=50.0
            )
            self.available_resources[ResourceType.DEVELOPER].append(dev_resource)
        
        # Computing resources
        for i in range(10):
            compute_resource = Resource(
                name=f"ComputeNode_{i+1}",
                type=ResourceType.COMPUTING_RESOURCES,
                capacity=100.0,
                current_utilization=0.0,
                skills=["cpu_intensive", "memory_intensive"],
                cost_per_hour=5.0
            )
            self.available_resources[ResourceType.COMPUTING_RESOURCES].append(compute_resource)
        
        # Infrastructure resources
        for i in range(3):
            infra_resource = Resource(
                name=f"InfraCluster_{i+1}",
                type=ResourceType.INFRASTRUCTURE,
                capacity=100.0,
                current_utilization=0.0,
                skills=["kubernetes", "docker", "monitoring"],
                cost_per_hour=20.0
            )
            self.available_resources[ResourceType.INFRASTRUCTURE].append(infra_resource)
    
    def _initialize_environment_templates(self):
        """Initialize default environment templates."""
        self.environment_templates["python_web"] = EnvironmentConfig(
            environment_name="python_web_template",
            python_version="3.9",
            dependencies=["flask", "requests", "pytest", "black", "flake8"],
            environment_variables={"FLASK_ENV": "development"},
            testing_framework="pytest"
        )
        
        self.environment_templates["ml_project"] = EnvironmentConfig(
            environment_name="ml_project_template",
            python_version="3.9",
            dependencies=["scikit-learn", "pandas", "numpy", "matplotlib", "jupyter", "pytest"],
            environment_variables={"PYTHONPATH": "/app"},
            testing_framework="pytest"
        )
    
    def _initialize_agent_registrations(self):
        """Initialize default agent registrations."""
        self.registered_agents[AgentRole.PROGRAMMER] = AgentCoordination(
            agent_role=AgentRole.PROGRAMMER,
            required_capabilities=["code_generation", "testing", "debugging"],
            availability_status="available"
        )
        
        self.registered_agents[AgentRole.ROUTER] = AgentCoordination(
            agent_role=AgentRole.ROUTER,
            required_capabilities=["resource_routing", "load_balancing"],
            availability_status="available"
        )
        
        self.registered_agents[AgentRole.QA_VALIDATOR] = AgentCoordination(
            agent_role=AgentRole.QA_VALIDATOR,
            required_capabilities=["quality_validation", "testing", "compliance_check"],
            availability_status="available"
        )
    
    async def _assess_risk_trigger(self, risk: ImplementationRisk, plan: ImplementationPlan) -> bool:
        """Assess if a risk has been triggered."""
        # Simplified risk trigger logic
        if risk.category == "schedule":
            progress_checkpoints = self.progress_checkpoints.get(plan.idea_id, [])
            if progress_checkpoints:
                latest = progress_checkpoints[-1]
                time_elapsed = datetime.now(timezone.utc) - plan.created_at
                expected_progress = (time_elapsed.total_seconds() / 3600) / plan.estimated_duration_hours * 100
                return latest.completion_percentage < (expected_progress * 0.8)
        
        return False
    
    async def _trigger_risk_mitigation(self, risk: ImplementationRisk, plan: ImplementationPlan):
        """Trigger risk mitigation strategies."""
        self.logger.warning(f"Risk triggered: {risk.title}",
                          LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
        
        for strategy in risk.mitigation_strategies:
            self.logger.info(f"Applying mitigation strategy: {strategy}",
                           LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
        
        risk.status = "mitigated"
    
    async def _optimize_resource_allocation(self, resource: Resource, plan: ImplementationPlan, issue_type: str):
        """Optimize resource allocation based on utilization patterns."""
        if issue_type == "underutilized":
            # Reduce allocation slightly
            resource.current_utilization = max(0, resource.current_utilization - 10)
            self.logger.info(f"Reduced utilization for underutilized resource {resource.name}",
                           LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
        
        elif issue_type == "overutilized":
            # Try to find additional resources
            additional_resources = [r for r in self.available_resources[resource.type] 
                                 if r.current_utilization < 50 and r.id != resource.id]
            if additional_resources:
                new_resource = additional_resources[0]
                new_resource.current_utilization += 30
                plan.allocated_resources.append(new_resource)
                self.logger.info(f"Allocated additional resource {new_resource.name}",
                               LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
    
    async def _handle_implementation_completion(self, plan: ImplementationPlan):
        """Handle completion of an implementation."""
        self.logger.agent_action("implementation_completed", "implementer_agent", plan.idea_id)
        
        # Move from active to history
        del self.active_implementations[plan.idea_id]
        plan.updated_at = datetime.now(timezone.utc)
        self.implementation_history[plan.idea_id] = plan
        
        # Release resources
        for resource in plan.allocated_resources:
            resource.current_utilization = max(0, resource.current_utilization - 50)
        
        # Update metrics
        self.implementation_metrics.successful_implementations += 1
        
        # Coordinate with Registrar Agent for completion registration
        if AgentRole.REGISTRAR in self.registered_agents:
            await self._coordinate_with_agent(
                AgentRole.REGISTRAR,
                "implementation_completion",
                {
                    "plan_id": plan.plan_id,
                    "idea_id": plan.idea_id,
                    "completion_time": datetime.now(timezone.utc).isoformat(),
                    "deliverables": [task.deliverables for task in plan.tasks if task.deliverables]
                }
            )
    
    async def _handle_blocking_issues(self, plan: ImplementationPlan, issues: List[str]):
        """Handle blocking issues in implementation."""
        for issue in issues:
            self.logger.warning(f"Blocking issue identified: {issue}",
                              LogContext(agent_name="implementer_agent", idea_id=plan.idea_id))
        
        # Coordinate with relevant agents to resolve issues
        if any("resource" in issue.lower() for issue in issues):
            await self._coordinate_with_agent(
                AgentRole.ROUTER,
                "resource_issue_resolution",
                {"plan_id": plan.plan_id, "issues": issues}
            )
    
    def _create_fallback_implementation_plan(self, idea: Idea, template_data: Dict[str, Any], error: str) -> ImplementationPlan:
        """Create a minimal fallback implementation plan."""
        return ImplementationPlan(
            idea_id=str(idea.id),
            project_name=idea.title,
            description=f"Fallback plan due to error: {error}",
            estimated_duration_hours=40,
            target_completion_date=datetime.now(timezone.utc) + timedelta(hours=40),
            phases=[ImplementationPhase.TEMPLATE_ANALYSIS, ImplementationPhase.ACTIVE_DEVELOPMENT],
            tasks=[
                ImplementationTask(
                    name="Manual Implementation Review",
                    description="Manual review and implementation required due to orchestration failure",
                    phase=ImplementationPhase.TEMPLATE_ANALYSIS,
                    estimated_hours=40,
                    status=ImplementationStatus.PENDING
                )
            ],
            success_criteria=["Manual implementation completed", "Basic functionality working"]
        )
    
    # Public API methods
    
    def get_active_implementations(self) -> Dict[str, ImplementationPlan]:
        """Get all currently active implementations."""
        return self.active_implementations.copy()
    
    def get_implementation_history(self, limit: Optional[int] = None) -> List[ImplementationPlan]:
        """Get implementation history with optional limit."""
        history = list(self.implementation_history.values())
        history.sort(key=lambda p: p.updated_at, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    def get_implementation_metrics(self) -> ImplementationMetrics:
        """Get comprehensive implementation metrics."""
        return self.implementation_metrics
    
    def get_resource_utilization(self) -> Dict[str, Dict[str, float]]:
        """Get current resource utilization across all types."""
        utilization = {}
        
        for resource_type, resources in self.available_resources.items():
            utilization[resource_type.value] = {
                "total_capacity": sum(r.capacity for r in resources),
                "total_utilization": sum(r.current_utilization for r in resources),
                "average_utilization": sum(r.current_utilization for r in resources) / len(resources) if resources else 0,
                "resource_count": len(resources)
            }
        
        return utilization
    
    def get_progress_status(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed progress status for a specific implementation."""
        if idea_id not in self.active_implementations:
            return None
        
        plan = self.active_implementations[idea_id]
        checkpoints = self.progress_checkpoints.get(idea_id, [])
        
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        
        return {
            "plan_id": plan.plan_id,
            "project_name": plan.project_name,
            "current_phase": latest_checkpoint.phase.value if latest_checkpoint else "unknown",
            "completion_percentage": latest_checkpoint.completion_percentage if latest_checkpoint else 0,
            "estimated_completion": plan.target_completion_date.isoformat(),
            "tasks_completed": latest_checkpoint.tasks_completed if latest_checkpoint else 0,
            "tasks_remaining": latest_checkpoint.tasks_remaining if latest_checkpoint else len(plan.tasks),
            "blocking_issues": latest_checkpoint.issues_identified if latest_checkpoint else [],
            "resource_utilization": latest_checkpoint.resource_utilization if latest_checkpoint else {},
            "recommendations": latest_checkpoint.recommendations if latest_checkpoint else []
        }
    
    async def update_task_status(self, idea_id: str, task_id: str, 
                               new_status: ImplementationStatus, 
                               completion_percentage: float = 0.0,
                               notes: Optional[str] = None):
        """
        Update the status of a specific implementation task.
        
        Args:
            idea_id: ID of the idea being implemented
            task_id: ID of the task to update
            new_status: New status for the task
            completion_percentage: Completion percentage (0-100)
            notes: Optional notes about the status update
        """
        if idea_id not in self.active_implementations:
            self.logger.warning(f"No active implementation found for idea {idea_id}")
            return
        
        plan = self.active_implementations[idea_id]
        
        for task in plan.tasks:
            if task.id == task_id:
                old_status = task.status
                task.status = new_status
                task.completion_percentage = completion_percentage
                
                if new_status == ImplementationStatus.IN_PROGRESS and not task.start_time:
                    task.start_time = datetime.now(timezone.utc)
                elif new_status == ImplementationStatus.COMPLETED:
                    task.end_time = datetime.now(timezone.utc)
                    task.completion_percentage = 100.0
                
                self.logger.agent_action("task_status_updated", "implementer_agent", idea_id, {
                    "task_id": task_id,
                    "task_name": task.name,
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "completion": completion_percentage,
                    "notes": notes
                })
                
                # Update plan timestamp
                plan.updated_at = datetime.now(timezone.utc)
                break
    
    async def add_implementation_risk(self, idea_id: str, risk: ImplementationRisk):
        """Add a new risk to an active implementation."""
        if idea_id not in self.active_implementations:
            return
        
        plan = self.active_implementations[idea_id]
        plan.identified_risks.append(risk)
        
        self.logger.warning(f"New implementation risk added: {risk.title}",
                          LogContext(agent_name="implementer_agent", idea_id=idea_id))
        
        # Auto-escalate high risks if configured
        if (self.config.auto_escalate_high_risks and 
            risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]):
            await self._trigger_risk_mitigation(risk, plan)
    
    def export_implementation_knowledge(self) -> Dict[str, Any]:
        """Export implementation knowledge for backup or transfer."""
        return {
            "active_implementations": {
                idea_id: plan.dict() for idea_id, plan in self.active_implementations.items()
            },
            "implementation_history": {
                idea_id: plan.dict() for idea_id, plan in self.implementation_history.items()
            },
            "progress_checkpoints": {
                idea_id: [cp.dict() for cp in checkpoints]
                for idea_id, checkpoints in self.progress_checkpoints.items()
            },
            "implementation_metrics": self.implementation_metrics.dict(),
            "resource_utilization_history": dict(self.resource_utilization_history),
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }


# Factory function for easy agent creation
def create_implementer_agent(config: Optional[ImplementerConfiguration] = None,
                           state_router: Optional[StateMachineRouter] = None) -> ImplementerAgent:
    """
    Create a new Implementer Agent instance.
    
    Args:
        config: Optional implementer configuration
        state_router: Optional state machine router
        
    Returns:
        Configured ImplementerAgent instance
    """
    return ImplementerAgent(config, state_router)


# Export main classes and functions
__all__ = [
    "ImplementerAgent",
    "ImplementerConfiguration",
    "ImplementationPlan",
    "ImplementationTask",
    "Resource",
    "ImplementationRisk",
    "EnvironmentConfig",
    "QualityGate",
    "ProgressCheckpoint",
    "AgentCoordination",
    "ImplementationMetrics",
    "ImplementationPhase",
    "ImplementationStatus",
    "ResourceType",
    "RiskLevel",
    "AgentRole",
    "create_implementer_agent"
]