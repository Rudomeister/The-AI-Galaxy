"""
Creator Agent for the AI-Galaxy Lower Meta-Layer.

This module implements the Creator Agent, responsible for transforming approved
ideas into structured implementation templates. The Creator analyzes idea
requirements, selects appropriate technology stacks, generates project scaffolds,
and creates comprehensive implementation plans for handoff to the Implementer Agent.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext
from ..state_machine.router import StateMachineRouter, TransitionResult


class ProjectType(str, Enum):
    """Types of projects that can be created."""
    MICROSERVICE = "microservice"
    API_SERVICE = "api_service"
    ML_MODEL = "ml_model"
    DATA_PIPELINE = "data_pipeline"
    WEB_APPLICATION = "web_application"
    LIBRARY = "library"
    CLI_TOOL = "cli_tool"
    INTEGRATION = "integration"
    ANALYTICS_DASHBOARD = "analytics_dashboard"
    WORKFLOW_AUTOMATION = "workflow_automation"
    RESEARCH_PROJECT = "research_project"
    DOCUMENTATION = "documentation"


class TechnologyStack(str, Enum):
    """Supported technology stacks."""
    PYTHON_FASTAPI = "python_fastapi"
    PYTHON_FLASK = "python_flask"
    PYTHON_DJANGO = "python_django"
    NODE_EXPRESS = "node_express"
    NODE_NESTJS = "node_nestjs"
    REACT_TYPESCRIPT = "react_typescript"
    VUE_TYPESCRIPT = "vue_typescript"
    PYTHON_ML = "python_ml"
    PYTHON_DATA = "python_data"
    GO_GIN = "go_gin"
    RUST_ACTIX = "rust_actix"
    JAVA_SPRING = "java_spring"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    DOCKER_COMPOSE = "docker_compose"


class ProjectComplexity(str, Enum):
    """Project complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class TemplateGenerationResult(str, Enum):
    """Results of template generation attempts."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    NEEDS_REVIEW = "needs_review"


class DepartmentType(str, Enum):
    """Types of departments in the ecosystem."""
    MACHINE_LEARNING = "department_of_ml"
    DATA_ENGINEERING = "department_of_data"
    WEB_DEVELOPMENT = "department_of_web"
    INFRASTRUCTURE = "department_of_infrastructure"
    SECURITY = "department_of_security"
    RESEARCH = "department_of_research"
    INTEGRATION = "department_of_integration"
    ANALYTICS = "department_of_analytics"


@dataclass
class TechnologyProfile:
    """Profile for a technology stack with capabilities and requirements."""
    stack: TechnologyStack
    name: str
    description: str
    suitable_for: List[ProjectType]
    complexity_level: ProjectComplexity
    dependencies: List[str]
    setup_time_hours: int
    scalability_score: float  # 0.0 to 10.0
    learning_curve: float  # 0.0 (easy) to 10.0 (difficult)
    ecosystem_maturity: float  # 0.0 to 10.0
    performance_score: float  # 0.0 to 10.0


@dataclass
class FileTemplate:
    """Template for generating individual files."""
    path: str
    content: str
    is_executable: bool = False
    description: str = ""
    dependencies: List[str] = field(default_factory=list)


@dataclass
class DirectoryStructure:
    """Template for project directory structure."""
    name: str
    subdirectories: List['DirectoryStructure'] = field(default_factory=list)
    files: List[FileTemplate] = field(default_factory=list)
    description: str = ""


class ImplementationPhase(BaseModel):
    """Individual phase in the implementation plan."""
    phase_number: int
    name: str
    description: str
    estimated_hours: int
    prerequisites: List[str] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)


class ResourceRequirement(BaseModel):
    """Resource requirements for implementation."""
    developers_needed: int
    skill_requirements: List[str] = Field(default_factory=list)
    external_services: List[str] = Field(default_factory=list)
    infrastructure_needs: List[str] = Field(default_factory=list)
    estimated_cost: str
    timeline_weeks: int


class TestingStrategy(BaseModel):
    """Testing strategy and framework setup."""
    test_frameworks: List[str] = Field(default_factory=list)
    test_types: List[str] = Field(default_factory=list)
    coverage_targets: Dict[str, float] = Field(default_factory=dict)
    automation_level: str
    ci_cd_integration: bool = True


class DeploymentConfiguration(BaseModel):
    """Deployment and infrastructure configuration."""
    deployment_strategy: str
    environments: List[str] = Field(default_factory=list)
    infrastructure_type: str
    scaling_strategy: str
    monitoring_setup: List[str] = Field(default_factory=list)
    backup_strategy: str


class ProjectTemplate(BaseModel):
    """Comprehensive project template."""
    template_id: str = Field(default_factory=lambda: str(uuid4()))
    idea_id: str
    project_name: str
    project_type: ProjectType
    technology_stack: TechnologyStack
    complexity: ProjectComplexity
    description: str
    
    # Structure and code
    directory_structure: Dict[str, Any] = Field(default_factory=dict)
    initial_files: List[Dict[str, Any]] = Field(default_factory=list)
    configuration_files: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Implementation planning
    implementation_phases: List[ImplementationPhase] = Field(default_factory=list)
    resource_requirements: ResourceRequirement
    testing_strategy: TestingStrategy
    deployment_config: DeploymentConfiguration
    
    # Department assignment
    recommended_department: DepartmentType
    recommended_institution: Optional[str] = None
    integration_points: List[str] = Field(default_factory=list)
    
    # Metadata
    creation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_completion_date: datetime
    priority_score: int = Field(ge=1, le=10)
    success_metrics: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class DepartmentRoutingDecision(BaseModel):
    """Decision for routing ideas to departments and institutions."""
    idea_id: str
    recommended_department: DepartmentType
    recommended_institution: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternative_departments: List[DepartmentType] = Field(default_factory=list)
    new_institution_needed: bool = False
    proposed_institution_name: Optional[str] = None


class CreatorConfiguration(BaseModel):
    """Configuration for the Creator Agent."""
    default_technology_preferences: Dict[ProjectType, List[TechnologyStack]] = Field(default_factory=dict)
    complexity_thresholds: Dict[str, int] = Field(default_factory=dict)
    max_implementation_phases: int = 10
    default_timeline_buffer: float = 1.2  # 20% buffer
    enable_advanced_features: bool = True
    auto_department_assignment: bool = True
    require_testing_strategy: bool = True
    require_deployment_config: bool = True


class CreatorAgent:
    """
    The Creator Agent - template and scaffold generator of AI-Galaxy.
    
    Transforms approved ideas into structured implementation templates with
    technology stack selection, project scaffolding, implementation planning,
    and department routing decisions.
    """
    
    def __init__(self, config: Optional[CreatorConfiguration] = None,
                 state_router: Optional[StateMachineRouter] = None):
        """
        Initialize the Creator Agent.
        
        Args:
            config: Creator configuration parameters
            state_router: State machine router for workflow transitions
        """
        self.logger = get_logger("creator_agent")
        self.config = config or CreatorConfiguration()
        self.state_router = state_router
        
        # Initialize technology profiles
        self.technology_profiles = self._initialize_technology_profiles()
        
        # Initialize template database
        self.template_history: Dict[str, List[ProjectTemplate]] = {}
        self.department_assignments: Dict[str, DepartmentRoutingDecision] = {}
        
        # Performance metrics
        self.creation_metrics = {
            "total_templates_created": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_creation_time": 0.0,
            "templates_by_type": {},
            "templates_by_complexity": {}
        }
        
        self.logger.agent_action("creator_agent_initialized", "creator_agent",
                                additional_context={
                                    "technology_stacks": len(self.technology_profiles),
                                    "auto_department_assignment": self.config.auto_department_assignment,
                                    "advanced_features": self.config.enable_advanced_features
                                })
    
    def create_project_template(self, idea: Idea) -> ProjectTemplate:
        """
        Create a comprehensive project template from an approved idea.
        
        Args:
            idea: The approved idea to convert into a template
            
        Returns:
            Detailed project template ready for implementation
        """
        start_time = datetime.now(timezone.utc)
        idea_id = str(idea.id)
        
        context = LogContext(
            agent_name="creator_agent",
            idea_id=idea_id,
            additional_context={"template_creation_start": start_time.isoformat()}
        )
        
        self.logger.agent_action("starting_template_creation", "creator_agent", idea_id)
        
        try:
            # Analyze idea to determine project characteristics
            project_analysis = self._analyze_idea_requirements(idea)
            
            # Select optimal technology stack
            tech_selection = self._select_technology_stack(idea, project_analysis)
            
            # Generate project structure
            directory_structure = self._generate_directory_structure(
                project_analysis["project_type"], tech_selection
            )
            
            # Create initial files and configurations
            initial_files = self._generate_initial_files(
                idea, project_analysis, tech_selection
            )
            configuration_files = self._generate_configuration_files(
                project_analysis, tech_selection
            )
            
            # Create implementation plan
            implementation_phases = self._create_implementation_plan(
                idea, project_analysis, tech_selection
            )
            
            # Estimate resource requirements
            resource_requirements = self._estimate_resource_requirements(
                project_analysis, implementation_phases
            )
            
            # Create testing strategy
            testing_strategy = self._create_testing_strategy(
                project_analysis["project_type"], tech_selection
            )
            
            # Create deployment configuration
            deployment_config = self._create_deployment_configuration(
                project_analysis, tech_selection
            )
            
            # Determine department assignment
            routing_decision = self._determine_department_assignment(idea, project_analysis)
            
            # Calculate project timeline
            estimated_completion = self._calculate_completion_date(
                implementation_phases, resource_requirements
            )
            
            # Generate success metrics
            success_metrics = self._generate_success_metrics(idea, project_analysis)
            
            # Create the template
            template = ProjectTemplate(
                idea_id=idea_id,
                project_name=self._generate_project_name(idea),
                project_type=project_analysis["project_type"],
                technology_stack=tech_selection,
                complexity=project_analysis["complexity"],
                description=self._generate_project_description(idea, project_analysis),
                directory_structure=self._serialize_directory_structure(directory_structure),
                initial_files=[self._serialize_file_template(f) for f in initial_files],
                configuration_files=[self._serialize_file_template(f) for f in configuration_files],
                implementation_phases=implementation_phases,
                resource_requirements=resource_requirements,
                testing_strategy=testing_strategy,
                deployment_config=deployment_config,
                recommended_department=routing_decision.recommended_department,
                recommended_institution=routing_decision.recommended_institution,
                integration_points=self._identify_integration_points(project_analysis),
                estimated_completion_date=estimated_completion,
                priority_score=self._calculate_priority_score(idea, project_analysis),
                success_metrics=success_metrics
            )
            
            # Store template and routing decision
            if idea_id not in self.template_history:
                self.template_history[idea_id] = []
            self.template_history[idea_id].append(template)
            self.department_assignments[idea_id] = routing_decision
            
            # Update metrics
            self._update_creation_metrics(template, start_time)
            
            self.logger.agent_action("template_creation_completed", "creator_agent", idea_id, {
                "project_type": template.project_type.value,
                "technology_stack": template.technology_stack.value,
                "complexity": template.complexity.value,
                "department": template.recommended_department.value,
                "phases": len(template.implementation_phases),
                "creation_duration": (datetime.now(timezone.utc) - start_time).total_seconds()
            })
            
            return template
            
        except Exception as e:
            self.logger.error(f"Template creation failed: {e}", context, exc_info=True)
            
            # Create minimal fallback template
            return self._create_fallback_template(idea, start_time)
    
    def update_idea_status(self, idea: Idea, template: ProjectTemplate) -> bool:
        """
        Update idea status to template_created using state machine.
        
        Args:
            idea: The idea to update
            template: The created template
            
        Returns:
            True if status update successful, False otherwise
        """
        idea_id = str(idea.id)
        current_state = idea.status.value
        target_state = "template_created"
        
        context = LogContext(
            agent_name="creator_agent",
            idea_id=idea_id,
            additional_context={
                "current_state": current_state,
                "template_id": template.template_id
            }
        )
        
        try:
            # Update idea metadata with template information
            idea.metadata.update({
                "template_creation": {
                    "template_id": template.template_id,
                    "project_type": template.project_type.value,
                    "technology_stack": template.technology_stack.value,
                    "complexity": template.complexity.value,
                    "recommended_department": template.recommended_department.value,
                    "recommended_institution": template.recommended_institution,
                    "estimated_completion": template.estimated_completion_date.isoformat(),
                    "implementation_phases": len(template.implementation_phases),
                    "priority_score": template.priority_score,
                    "creation_timestamp": template.creation_timestamp.isoformat()
                }
            })
            
            # Execute state transition if router available
            if self.state_router:
                result = self.state_router.execute_transition(idea, target_state, "creator_agent")
                
                if result == TransitionResult.SUCCESS:
                    self.logger.state_transition(current_state, target_state, idea_id,
                                               "creator_agent", f"Template created: {template.template_id}")
                    return True
                else:
                    self.logger.error(f"State transition failed: {result}", context)
                    return False
            else:
                # Manual status update if no router
                idea.status = IdeaStatus(target_state)
                idea.updated_at = datetime.now(timezone.utc)
                self.logger.info(f"Idea status updated to {target_state}", context)
                return True
                
        except Exception as e:
            self.logger.error(f"Status update failed: {e}", context, exc_info=True)
            return False
    
    def get_department_routing_decision(self, idea_id: str) -> Optional[DepartmentRoutingDecision]:
        """
        Get the department routing decision for a specific idea.
        
        Args:
            idea_id: ID of the idea
            
        Returns:
            Routing decision or None if not found
        """
        return self.department_assignments.get(idea_id)
    
    def generate_project_scaffold(self, template: ProjectTemplate, output_path: str) -> bool:
        """
        Generate actual project files and directories from template.
        
        Args:
            template: The project template to realize
            output_path: Base path where to create the project
            
        Returns:
            True if scaffold generation successful, False otherwise
        """
        context = LogContext(
            agent_name="creator_agent",
            idea_id=template.idea_id,
            additional_context={
                "template_id": template.template_id,
                "output_path": output_path
            }
        )
        
        try:
            base_path = Path(output_path) / template.project_name
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            self._create_directories(base_path, template.directory_structure)
            
            # Generate initial files
            for file_data in template.initial_files:
                file_path = base_path / file_data["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_data["content"])
                
                if file_data.get("is_executable", False):
                    os.chmod(file_path, 0o755)
            
            # Generate configuration files
            for config_file in template.configuration_files:
                config_path = base_path / config_file["path"]
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(config_file["content"])
            
            # Create implementation guide
            self._create_implementation_guide(base_path, template)
            
            self.logger.agent_action("project_scaffold_generated", "creator_agent", 
                                   template.idea_id, {
                                       "project_name": template.project_name,
                                       "output_path": str(base_path),
                                       "files_created": len(template.initial_files) + len(template.configuration_files)
                                   })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Scaffold generation failed: {e}", context, exc_info=True)
            return False
    
    def get_template_summary(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of templates created for a specific idea.
        
        Args:
            idea_id: ID of the idea
            
        Returns:
            Template summary or None if not found
        """
        templates = self.template_history.get(idea_id, [])
        routing_decision = self.department_assignments.get(idea_id)
        
        if not templates:
            return None
        
        latest_template = templates[-1]
        
        return {
            "idea_id": idea_id,
            "template_count": len(templates),
            "latest_template": {
                "template_id": latest_template.template_id,
                "project_name": latest_template.project_name,
                "project_type": latest_template.project_type.value,
                "technology_stack": latest_template.technology_stack.value,
                "complexity": latest_template.complexity.value,
                "estimated_completion": latest_template.estimated_completion_date.isoformat(),
                "implementation_phases": len(latest_template.implementation_phases),
                "priority_score": latest_template.priority_score
            },
            "department_assignment": {
                "department": routing_decision.recommended_department.value if routing_decision else None,
                "institution": routing_decision.recommended_institution if routing_decision else None,
                "confidence": routing_decision.confidence_score if routing_decision else 0.0
            },
            "creation_timeline": [
                {
                    "timestamp": template.creation_timestamp.isoformat(),
                    "template_id": template.template_id,
                    "project_type": template.project_type.value
                }
                for template in templates
            ]
        }
    
    def get_creator_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about creator performance.
        
        Returns:
            Dictionary with creator performance metrics
        """
        total_ideas = len(self.template_history)
        
        if total_ideas == 0:
            return {**self.creation_metrics, "total_ideas_processed": 0}
        
        # Calculate additional metrics
        success_rate = (self.creation_metrics["successful_generations"] / 
                       self.creation_metrics["total_templates_created"] * 100) if self.creation_metrics["total_templates_created"] > 0 else 0
        
        # Technology stack popularity
        tech_usage = {}
        for templates in self.template_history.values():
            for template in templates:
                stack = template.technology_stack.value
                tech_usage[stack] = tech_usage.get(stack, 0) + 1
        
        # Department distribution
        dept_distribution = {}
        for decision in self.department_assignments.values():
            dept = decision.recommended_department.value
            dept_distribution[dept] = dept_distribution.get(dept, 0) + 1
        
        return {
            **self.creation_metrics,
            "total_ideas_processed": total_ideas,
            "success_rate_percent": success_rate,
            "average_phases_per_project": self._calculate_average_phases(),
            "average_priority_score": self._calculate_average_priority(),
            "technology_stack_usage": tech_usage,
            "department_distribution": dept_distribution,
            "configuration": {
                "max_implementation_phases": self.config.max_implementation_phases,
                "timeline_buffer": self.config.default_timeline_buffer,
                "auto_department_assignment": self.config.auto_department_assignment,
                "advanced_features": self.config.enable_advanced_features
            }
        }
    
    # Private helper methods
    
    def _initialize_technology_profiles(self) -> Dict[TechnologyStack, TechnologyProfile]:
        """Initialize technology stack profiles with capabilities."""
        return {
            TechnologyStack.PYTHON_FASTAPI: TechnologyProfile(
                stack=TechnologyStack.PYTHON_FASTAPI,
                name="Python FastAPI",
                description="Modern, fast API framework for Python",
                suitable_for=[ProjectType.API_SERVICE, ProjectType.MICROSERVICE, ProjectType.ML_MODEL],
                complexity_level=ProjectComplexity.MODERATE,
                dependencies=["python", "fastapi", "uvicorn", "pydantic"],
                setup_time_hours=4,
                scalability_score=8.5,
                learning_curve=4.0,
                ecosystem_maturity=8.0,
                performance_score=9.0
            ),
            TechnologyStack.PYTHON_ML: TechnologyProfile(
                stack=TechnologyStack.PYTHON_ML,
                name="Python ML Stack",
                description="Machine learning stack with scikit-learn, pandas, numpy",
                suitable_for=[ProjectType.ML_MODEL, ProjectType.DATA_PIPELINE, ProjectType.RESEARCH_PROJECT],
                complexity_level=ProjectComplexity.COMPLEX,
                dependencies=["python", "scikit-learn", "pandas", "numpy", "jupyter"],
                setup_time_hours=6,
                scalability_score=7.0,
                learning_curve=6.0,
                ecosystem_maturity=9.0,
                performance_score=7.5
            ),
            TechnologyStack.REACT_TYPESCRIPT: TechnologyProfile(
                stack=TechnologyStack.REACT_TYPESCRIPT,
                name="React TypeScript",
                description="React with TypeScript for type-safe frontend development",
                suitable_for=[ProjectType.WEB_APPLICATION, ProjectType.ANALYTICS_DASHBOARD],
                complexity_level=ProjectComplexity.MODERATE,
                dependencies=["node", "react", "typescript", "webpack"],
                setup_time_hours=5,
                scalability_score=8.0,
                learning_curve=5.0,
                ecosystem_maturity=9.0,
                performance_score=8.0
            ),
            TechnologyStack.NODE_EXPRESS: TechnologyProfile(
                stack=TechnologyStack.NODE_EXPRESS,
                name="Node.js Express",
                description="Lightweight Node.js web framework",
                suitable_for=[ProjectType.API_SERVICE, ProjectType.MICROSERVICE, ProjectType.WEB_APPLICATION],
                complexity_level=ProjectComplexity.SIMPLE,
                dependencies=["node", "express", "cors", "helmet"],
                setup_time_hours=3,
                scalability_score=7.0,
                learning_curve=3.0,
                ecosystem_maturity=9.0,
                performance_score=7.5
            ),
            TechnologyStack.DOCKER_COMPOSE: TechnologyProfile(
                stack=TechnologyStack.DOCKER_COMPOSE,
                name="Docker Compose",
                description="Container orchestration for development environments",
                suitable_for=[ProjectType.MICROSERVICE, ProjectType.INTEGRATION, ProjectType.DATA_PIPELINE],
                complexity_level=ProjectComplexity.MODERATE,
                dependencies=["docker", "docker-compose"],
                setup_time_hours=2,
                scalability_score=8.0,
                learning_curve=4.0,
                ecosystem_maturity=9.0,
                performance_score=8.5
            ),
            TechnologyStack.KUBERNETES: TechnologyProfile(
                stack=TechnologyStack.KUBERNETES,
                name="Kubernetes",
                description="Production-grade container orchestration",
                suitable_for=[ProjectType.MICROSERVICE, ProjectType.API_SERVICE, ProjectType.ML_MODEL],
                complexity_level=ProjectComplexity.ENTERPRISE,
                dependencies=["kubernetes", "helm", "kubectl"],
                setup_time_hours=12,
                scalability_score=10.0,
                learning_curve=8.0,
                ecosystem_maturity=8.5,
                performance_score=9.5
            )
        }
    
    def _analyze_idea_requirements(self, idea: Idea) -> Dict[str, Any]:
        """Analyze idea to determine project requirements and characteristics."""
        description = idea.description.lower()
        title = idea.title.lower()
        
        # Determine project type based on keywords
        project_type = self._classify_project_type(description, title)
        
        # Assess complexity based on various factors
        complexity = self._assess_project_complexity(idea, description, title)
        
        # Identify key requirements
        requirements = self._extract_requirements(description, title)
        
        # Assess scalability needs
        scalability_needs = self._assess_scalability_needs(description, title)
        
        # Identify performance requirements
        performance_requirements = self._identify_performance_requirements(description, title)
        
        return {
            "project_type": project_type,
            "complexity": complexity,
            "requirements": requirements,
            "scalability_needs": scalability_needs,
            "performance_requirements": performance_requirements,
            "estimated_scope": self._estimate_project_scope(description, title),
            "integration_complexity": self._assess_integration_complexity(description, title)
        }
    
    def _classify_project_type(self, description: str, title: str) -> ProjectType:
        """Classify the project type based on description and title."""
        text = f"{description} {title}"
        
        # Define keyword patterns for each project type
        type_patterns = {
            ProjectType.ML_MODEL: ["machine learning", "ml", "ai", "neural", "model", "prediction", "classification"],
            ProjectType.API_SERVICE: ["api", "rest", "endpoint", "service", "microservice"],
            ProjectType.WEB_APPLICATION: ["web app", "frontend", "dashboard", "ui", "interface"],
            ProjectType.DATA_PIPELINE: ["data", "etl", "pipeline", "processing", "analytics"],
            ProjectType.MICROSERVICE: ["microservice", "service", "api", "backend"],
            ProjectType.LIBRARY: ["library", "package", "framework", "tool", "utility"],
            ProjectType.CLI_TOOL: ["cli", "command line", "terminal", "script"],
            ProjectType.INTEGRATION: ["integration", "connector", "sync", "bridge"],
            ProjectType.ANALYTICS_DASHBOARD: ["dashboard", "analytics", "metrics", "visualization"],
            ProjectType.WORKFLOW_AUTOMATION: ["automation", "workflow", "process", "orchestration"],
            ProjectType.RESEARCH_PROJECT: ["research", "experiment", "study", "analysis"],
            ProjectType.DOCUMENTATION: ["documentation", "docs", "guide", "manual"]
        }
        
        # Score each type based on keyword matches
        type_scores = {}
        for project_type, keywords in type_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                type_scores[project_type] = score
        
        # Return the highest scoring type, or default to microservice
        return max(type_scores.items(), key=lambda x: x[1])[0] if type_scores else ProjectType.MICROSERVICE
    
    def _assess_project_complexity(self, idea: Idea, description: str, title: str) -> ProjectComplexity:
        """Assess project complexity based on multiple factors."""
        complexity_score = 0
        
        # Length and detail of description
        if len(description) > 500:
            complexity_score += 2
        elif len(description) > 200:
            complexity_score += 1
        
        # Priority suggests complexity
        if idea.priority >= 8:
            complexity_score += 2
        elif idea.priority >= 6:
            complexity_score += 1
        
        # Complexity indicators in text
        complexity_indicators = {
            "distributed": 3,
            "enterprise": 3,
            "scalable": 2,
            "microservice": 2,
            "machine learning": 2,
            "real-time": 2,
            "high availability": 3,
            "multi-tenant": 3,
            "integration": 1,
            "simple": -2,
            "basic": -1,
            "prototype": -1
        }
        
        text = f"{description} {title}"
        for indicator, score in complexity_indicators.items():
            if indicator in text:
                complexity_score += score
        
        # Map score to complexity level
        if complexity_score >= 8:
            return ProjectComplexity.ENTERPRISE
        elif complexity_score >= 5:
            return ProjectComplexity.COMPLEX
        elif complexity_score >= 2:
            return ProjectComplexity.MODERATE
        else:
            return ProjectComplexity.SIMPLE
    
    def _extract_requirements(self, description: str, title: str) -> List[str]:
        """Extract key requirements from the description."""
        text = f"{description} {title}"
        requirements = []
        
        # Functional requirements patterns
        requirement_patterns = {
            r"must\s+(\w+(?:\s+\w+){0,3})": "Must {}",
            r"should\s+(\w+(?:\s+\w+){0,3})": "Should {}",
            r"need(?:s)?\s+(?:to\s+)?(\w+(?:\s+\w+){0,3})": "Needs to {}",
            r"require(?:s)?\s+(\w+(?:\s+\w+){0,3})": "Requires {}",
            r"support(?:s)?\s+(\w+(?:\s+\w+){0,3})": "Supports {}"
        }
        
        for pattern, template in requirement_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:3]:  # Limit to avoid noise
                requirements.append(template.format(match))
        
        # Add default requirements based on project characteristics
        if "api" in text:
            requirements.append("RESTful API interface")
        if "database" in text:
            requirements.append("Data persistence layer")
        if "authentication" in text or "auth" in text:
            requirements.append("User authentication system")
        
        return requirements[:5]  # Limit to top 5 requirements
    
    def _assess_scalability_needs(self, description: str, title: str) -> str:
        """Assess scalability requirements."""
        text = f"{description} {title}"
        
        high_scale_indicators = ["high volume", "scalable", "distributed", "load balancing", "millions"]
        medium_scale_indicators = ["moderate load", "growing", "expansion", "thousands"]
        low_scale_indicators = ["small", "prototype", "proof of concept", "basic"]
        
        if any(indicator in text for indicator in high_scale_indicators):
            return "high"
        elif any(indicator in text for indicator in medium_scale_indicators):
            return "medium"
        elif any(indicator in text for indicator in low_scale_indicators):
            return "low"
        else:
            return "medium"  # Default
    
    def _identify_performance_requirements(self, description: str, title: str) -> List[str]:
        """Identify performance requirements."""
        text = f"{description} {title}"
        requirements = []
        
        performance_indicators = {
            "real-time": "Real-time response requirements",
            "fast": "Fast response time requirements", 
            "low latency": "Low latency requirements",
            "high throughput": "High throughput requirements",
            "concurrent": "Concurrent user support",
            "performance": "Performance optimization needed"
        }
        
        for indicator, requirement in performance_indicators.items():
            if indicator in text:
                requirements.append(requirement)
        
        return requirements
    
    def _estimate_project_scope(self, description: str, title: str) -> str:
        """Estimate the overall project scope."""
        text = f"{description} {title}"
        
        # Count scope indicators
        large_scope_indicators = ["enterprise", "platform", "ecosystem", "comprehensive", "full-featured"]
        medium_scope_indicators = ["application", "service", "system", "solution"]
        small_scope_indicators = ["tool", "utility", "script", "component", "module"]
        
        large_count = sum(1 for indicator in large_scope_indicators if indicator in text)
        medium_count = sum(1 for indicator in medium_scope_indicators if indicator in text)
        small_count = sum(1 for indicator in small_scope_indicators if indicator in text)
        
        if large_count > 0 or len(description) > 1000:
            return "large"
        elif medium_count > 0 or len(description) > 300:
            return "medium"
        else:
            return "small"
    
    def _assess_integration_complexity(self, description: str, title: str) -> str:
        """Assess integration complexity requirements."""
        text = f"{description} {title}"
        
        integration_indicators = {
            "api integration": "high",
            "third-party": "high",
            "external service": "medium",
            "database": "medium",
            "microservice": "high",
            "standalone": "low",
            "isolated": "low"
        }
        
        for indicator, complexity in integration_indicators.items():
            if indicator in text:
                return complexity
        
        return "medium"  # Default
    
    def _select_technology_stack(self, idea: Idea, analysis: Dict[str, Any]) -> TechnologyStack:
        """Select the optimal technology stack for the project."""
        project_type = analysis["project_type"]
        complexity = analysis["complexity"]
        
        # Get suitable technology stacks for the project type
        suitable_stacks = []
        for stack, profile in self.technology_profiles.items():
            if project_type in profile.suitable_for:
                suitable_stacks.append((stack, profile))
        
        if not suitable_stacks:
            # Default fallback
            return TechnologyStack.PYTHON_FASTAPI
        
        # Score each suitable stack based on project requirements
        stack_scores = {}
        for stack, profile in suitable_stacks:
            score = 0
            
            # Complexity alignment
            if profile.complexity_level.value == complexity.value:
                score += 3
            elif abs(self._complexity_to_int(profile.complexity_level) - 
                    self._complexity_to_int(complexity)) == 1:
                score += 1
            
            # Performance requirements
            if "performance" in analysis["performance_requirements"]:
                score += profile.performance_score * 0.3
            
            # Scalability needs
            scalability_map = {"low": 5, "medium": 7, "high": 9}
            needed_scalability = scalability_map.get(analysis["scalability_needs"], 7)
            if profile.scalability_score >= needed_scalability:
                score += 2
            
            # Learning curve (prefer easier stacks for simple projects)
            if complexity == ProjectComplexity.SIMPLE:
                score += (10 - profile.learning_curve) * 0.2
            
            # Ecosystem maturity
            score += profile.ecosystem_maturity * 0.1
            
            stack_scores[stack] = score
        
        # Return the highest scoring stack
        return max(stack_scores.items(), key=lambda x: x[1])[0]
    
    def _complexity_to_int(self, complexity: ProjectComplexity) -> int:
        """Convert complexity enum to integer for comparison."""
        mapping = {
            ProjectComplexity.SIMPLE: 1,
            ProjectComplexity.MODERATE: 2,
            ProjectComplexity.COMPLEX: 3,
            ProjectComplexity.ENTERPRISE: 4
        }
        return mapping.get(complexity, 2)
    
    def _generate_directory_structure(self, project_type: ProjectType, 
                                    tech_stack: TechnologyStack) -> DirectoryStructure:
        """Generate directory structure based on project type and technology."""
        base_dirs = []
        
        # Common directories
        base_dirs.append(DirectoryStructure("src", description="Source code"))
        base_dirs.append(DirectoryStructure("tests", description="Test files"))
        base_dirs.append(DirectoryStructure("docs", description="Documentation"))
        base_dirs.append(DirectoryStructure("config", description="Configuration files"))
        
        # Technology-specific additions
        if tech_stack in [TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_ML]:
            base_dirs.extend([
                DirectoryStructure("src/api", description="API endpoints"),
                DirectoryStructure("src/models", description="Data models"),
                DirectoryStructure("src/services", description="Business logic"),
                DirectoryStructure("src/utils", description="Utility functions"),
                DirectoryStructure("scripts", description="Deployment and utility scripts")
            ])
        
        elif tech_stack == TechnologyStack.REACT_TYPESCRIPT:
            base_dirs.extend([
                DirectoryStructure("src/components", description="React components"),
                DirectoryStructure("src/pages", description="Page components"),
                DirectoryStructure("src/hooks", description="Custom React hooks"),
                DirectoryStructure("src/utils", description="Utility functions"),
                DirectoryStructure("public", description="Static assets")
            ])
        
        elif tech_stack == TechnologyStack.NODE_EXPRESS:
            base_dirs.extend([
                DirectoryStructure("src/routes", description="Express routes"),
                DirectoryStructure("src/middleware", description="Express middleware"),
                DirectoryStructure("src/controllers", description="Route controllers"),
                DirectoryStructure("src/models", description="Data models")
            ])
        
        # Project type specific additions
        if project_type == ProjectType.ML_MODEL:
            base_dirs.extend([
                DirectoryStructure("data", description="Training and test data"),
                DirectoryStructure("models", description="Trained models"),
                DirectoryStructure("notebooks", description="Jupyter notebooks")
            ])
        
        elif project_type == ProjectType.MICROSERVICE:
            base_dirs.extend([
                DirectoryStructure("docker", description="Docker configurations"),
                DirectoryStructure("k8s", description="Kubernetes manifests")
            ])
        
        return DirectoryStructure("project_root", subdirectories=base_dirs)
    
    def _generate_initial_files(self, idea: Idea, analysis: Dict[str, Any], 
                              tech_stack: TechnologyStack) -> List[FileTemplate]:
        """Generate initial project files based on technology stack."""
        files = []
        
        # Common files
        files.append(FileTemplate(
            path="README.md",
            content=self._generate_readme_content(idea, analysis, tech_stack),
            description="Project documentation"
        ))
        
        files.append(FileTemplate(
            path=".gitignore",
            content=self._generate_gitignore_content(tech_stack),
            description="Git ignore file"
        ))
        
        # Technology-specific files
        if tech_stack in [TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_ML]:
            files.extend(self._generate_python_files(idea, analysis, tech_stack))
        elif tech_stack == TechnologyStack.REACT_TYPESCRIPT:
            files.extend(self._generate_react_files(idea, analysis))
        elif tech_stack == TechnologyStack.NODE_EXPRESS:
            files.extend(self._generate_node_files(idea, analysis))
        
        return files
    
    def _generate_configuration_files(self, analysis: Dict[str, Any], 
                                    tech_stack: TechnologyStack) -> List[FileTemplate]:
        """Generate configuration files for the project."""
        configs = []
        
        # Technology-specific configurations
        if tech_stack in [TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_ML]:
            configs.append(FileTemplate(
                path="requirements.txt",
                content=self._generate_python_requirements(tech_stack),
                description="Python dependencies"
            ))
            
            configs.append(FileTemplate(
                path="pyproject.toml",
                content=self._generate_pyproject_toml(analysis),
                description="Python project configuration"
            ))
        
        elif tech_stack in [TechnologyStack.REACT_TYPESCRIPT, TechnologyStack.NODE_EXPRESS]:
            configs.append(FileTemplate(
                path="package.json",
                content=self._generate_package_json(analysis, tech_stack),
                description="Node.js dependencies and scripts"
            ))
            
            if tech_stack == TechnologyStack.REACT_TYPESCRIPT:
                configs.append(FileTemplate(
                    path="tsconfig.json",
                    content=self._generate_tsconfig(),
                    description="TypeScript configuration"
                ))
        
        # Docker configuration if applicable
        if analysis["complexity"] in [ProjectComplexity.COMPLEX, ProjectComplexity.ENTERPRISE]:
            configs.append(FileTemplate(
                path="Dockerfile",
                content=self._generate_dockerfile(tech_stack),
                description="Docker container configuration"
            ))
            
            configs.append(FileTemplate(
                path="docker-compose.yml",
                content=self._generate_docker_compose(analysis),
                description="Docker Compose configuration"
            ))
        
        return configs
    
    def _create_implementation_plan(self, idea: Idea, analysis: Dict[str, Any], 
                                  tech_stack: TechnologyStack) -> List[ImplementationPhase]:
        """Create detailed implementation plan with phases."""
        phases = []
        
        # Phase 1: Project Setup
        phases.append(ImplementationPhase(
            phase_number=1,
            name="Project Setup and Infrastructure",
            description="Set up development environment and project structure",
            estimated_hours=8,
            prerequisites=[],
            deliverables=[
                "Development environment configured",
                "Project structure created",
                "Version control initialized",
                "CI/CD pipeline basic setup"
            ],
            tasks=[
                "Set up development environment",
                "Create project directory structure",
                "Initialize git repository",
                "Configure basic CI/CD pipeline",
                "Set up dependency management"
            ],
            success_criteria=[
                "All team members can run project locally",
                "Code can be committed and pushed",
                "Basic tests can be executed"
            ],
            risk_factors=[
                "Environment compatibility issues",
                "Tool installation problems"
            ]
        ))
        
        # Phase 2: Core Implementation
        phases.append(ImplementationPhase(
            phase_number=2,
            name="Core Feature Implementation",
            description="Implement main functionality and business logic",
            estimated_hours=self._estimate_core_implementation_hours(analysis),
            prerequisites=["Phase 1 completed"],
            deliverables=[
                "Core functionality implemented",
                "Basic API/interface created",
                "Data models defined"
            ],
            tasks=self._generate_core_implementation_tasks(idea, analysis, tech_stack),
            success_criteria=[
                "Main features work as specified",
                "Core business logic is functional",
                "Basic integration tests pass"
            ],
            risk_factors=[
                "Technical complexity underestimated",
                "Integration challenges",
                "Performance bottlenecks"
            ]
        ))
        
        # Phase 3: Testing and Quality Assurance
        phases.append(ImplementationPhase(
            phase_number=3,
            name="Testing and Quality Assurance",
            description="Comprehensive testing and code quality improvements",
            estimated_hours=16,
            prerequisites=["Phase 2 completed"],
            deliverables=[
                "Unit tests with good coverage",
                "Integration tests",
                "Code quality metrics",
                "Documentation updated"
            ],
            tasks=[
                "Write comprehensive unit tests",
                "Implement integration tests",
                "Set up code quality tools",
                "Perform security review",
                "Update documentation"
            ],
            success_criteria=[
                "Test coverage above 80%",
                "All tests passing",
                "Code quality metrics meet standards",
                "Security vulnerabilities addressed"
            ],
            risk_factors=[
                "Hard-to-test edge cases",
                "Performance test failures",
                "Security vulnerabilities discovered"
            ]
        ))
        
        # Add deployment phase for complex projects
        if analysis["complexity"] in [ProjectComplexity.COMPLEX, ProjectComplexity.ENTERPRISE]:
            phases.append(ImplementationPhase(
                phase_number=4,
                name="Deployment and Operations",
                description="Production deployment and operational setup",
                estimated_hours=12,
                prerequisites=["Phase 3 completed"],
                deliverables=[
                    "Production deployment configured",
                    "Monitoring and logging set up",
                    "Backup and recovery procedures",
                    "Operations documentation"
                ],
                tasks=[
                    "Configure production environment",
                    "Set up monitoring and alerting",
                    "Implement logging and analytics",
                    "Create backup procedures",
                    "Write operations guide"
                ],
                success_criteria=[
                    "Application runs successfully in production",
                    "Monitoring captures key metrics",
                    "Backup and recovery tested",
                    "Team can operate and maintain system"
                ],
                risk_factors=[
                    "Production environment issues",
                    "Monitoring gaps",
                    "Operational complexity"
                ]
            ))
        
        return phases
    
    def _estimate_resource_requirements(self, analysis: Dict[str, Any], 
                                      phases: List[ImplementationPhase]) -> ResourceRequirement:
        """Estimate resource requirements for the project."""
        total_hours = sum(phase.estimated_hours for phase in phases)
        
        # Determine team size based on complexity and timeline
        if analysis["complexity"] == ProjectComplexity.ENTERPRISE:
            developers_needed = 3 + (total_hours // 100)
        elif analysis["complexity"] == ProjectComplexity.COMPLEX:
            developers_needed = 2 + (total_hours // 120)
        else:
            developers_needed = max(1, total_hours // 160)
        
        developers_needed = min(developers_needed, 6)  # Cap at 6 developers
        
        # Skill requirements based on project type and technology
        skill_requirements = ["Software development", "Version control (Git)"]
        
        if analysis["project_type"] == ProjectType.ML_MODEL:
            skill_requirements.extend(["Machine learning", "Python", "Data science"])
        elif analysis["project_type"] == ProjectType.WEB_APPLICATION:
            skill_requirements.extend(["Frontend development", "UI/UX design", "JavaScript/TypeScript"])
        elif analysis["project_type"] == ProjectType.API_SERVICE:
            skill_requirements.extend(["API design", "Backend development", "Database design"])
        
        # External services based on requirements
        external_services = []
        if "database" in str(analysis["requirements"]).lower():
            external_services.append("Database service")
        if "authentication" in str(analysis["requirements"]).lower():
            external_services.append("Authentication service")
        if analysis["scalability_needs"] == "high":
            external_services.append("Load balancer")
        
        # Infrastructure needs
        infrastructure_needs = ["Development environment"]
        if analysis["complexity"] in [ProjectComplexity.COMPLEX, ProjectComplexity.ENTERPRISE]:
            infrastructure_needs.extend(["CI/CD pipeline", "Staging environment", "Production environment"])
        if analysis["scalability_needs"] == "high":
            infrastructure_needs.append("Container orchestration")
        
        # Estimate cost (simplified)
        cost_per_dev_week = 4000  # Rough estimate
        timeline_weeks = max(2, total_hours // (40 * developers_needed))
        estimated_cost = f"${cost_per_dev_week * developers_needed * timeline_weeks:,}"
        
        return ResourceRequirement(
            developers_needed=developers_needed,
            skill_requirements=skill_requirements,
            external_services=external_services,
            infrastructure_needs=infrastructure_needs,
            estimated_cost=estimated_cost,
            timeline_weeks=timeline_weeks
        )
    
    def _create_testing_strategy(self, project_type: ProjectType, 
                               tech_stack: TechnologyStack) -> TestingStrategy:
        """Create testing strategy based on project characteristics."""
        test_frameworks = []
        test_types = ["Unit tests", "Integration tests"]
        coverage_targets = {"unit": 80.0, "integration": 60.0}
        
        # Technology-specific frameworks
        if tech_stack in [TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_ML]:
            test_frameworks.extend(["pytest", "coverage", "mock"])
        elif tech_stack in [TechnologyStack.REACT_TYPESCRIPT, TechnologyStack.NODE_EXPRESS]:
            test_frameworks.extend(["jest", "testing-library", "cypress"])
        
        # Project type specific testing
        if project_type == ProjectType.API_SERVICE:
            test_types.extend(["API tests", "Performance tests"])
            test_frameworks.append("postman/newman")
        elif project_type == ProjectType.ML_MODEL:
            test_types.extend(["Model validation", "Data quality tests"])
            test_frameworks.append("pytest-ml")
        elif project_type == ProjectType.WEB_APPLICATION:
            test_types.extend(["E2E tests", "Accessibility tests"])
            coverage_targets["e2e"] = 70.0
        
        automation_level = "high" if len(test_types) >= 4 else "medium"
        
        return TestingStrategy(
            test_frameworks=test_frameworks,
            test_types=test_types,
            coverage_targets=coverage_targets,
            automation_level=automation_level,
            ci_cd_integration=True
        )
    
    def _create_deployment_configuration(self, analysis: Dict[str, Any], 
                                       tech_stack: TechnologyStack) -> DeploymentConfiguration:
        """Create deployment configuration based on project requirements."""
        complexity = analysis["complexity"]
        scalability = analysis["scalability_needs"]
        
        # Deployment strategy based on complexity
        if complexity == ProjectComplexity.ENTERPRISE:
            deployment_strategy = "Blue-Green with canary releases"
            environments = ["development", "staging", "production", "disaster-recovery"]
            infrastructure_type = "Kubernetes cluster"
            scaling_strategy = "Auto-scaling with load balancing"
        elif complexity == ProjectComplexity.COMPLEX:
            deployment_strategy = "Rolling deployment"
            environments = ["development", "staging", "production"]
            infrastructure_type = "Container-based (Docker)"
            scaling_strategy = "Horizontal scaling"
        else:
            deployment_strategy = "Direct deployment"
            environments = ["development", "production"]
            infrastructure_type = "Virtual machines or cloud instances"
            scaling_strategy = "Vertical scaling"
        
        # Monitoring setup
        monitoring_setup = ["Application logs", "Performance metrics"]
        if complexity in [ProjectComplexity.COMPLEX, ProjectComplexity.ENTERPRISE]:
            monitoring_setup.extend(["Health checks", "Error tracking", "Business metrics"])
        if scalability == "high":
            monitoring_setup.append("Real-time alerting")
        
        # Backup strategy
        backup_strategy = "Daily automated backups with 30-day retention"
        if complexity == ProjectComplexity.ENTERPRISE:
            backup_strategy = "Continuous backup with point-in-time recovery"
        
        return DeploymentConfiguration(
            deployment_strategy=deployment_strategy,
            environments=environments,
            infrastructure_type=infrastructure_type,
            scaling_strategy=scaling_strategy,
            monitoring_setup=monitoring_setup,
            backup_strategy=backup_strategy
        )
    
    def _determine_department_assignment(self, idea: Idea, 
                                       analysis: Dict[str, Any]) -> DepartmentRoutingDecision:
        """Determine which department and institution should handle the project."""
        project_type = analysis["project_type"]
        description = idea.description.lower()
        
        # Department assignment logic
        if project_type in [ProjectType.ML_MODEL, ProjectType.RESEARCH_PROJECT]:
            department = DepartmentType.MACHINE_LEARNING
            institution = "keras_institution"
            confidence = 0.9
            reasoning = "Project involves machine learning or research components"
        elif project_type in [ProjectType.DATA_PIPELINE, ProjectType.ANALYTICS_DASHBOARD]:
            department = DepartmentType.DATA_ENGINEERING
            institution = None  # Would need to create specific institution
            confidence = 0.8
            reasoning = "Project focuses on data processing and analytics"
        elif project_type in [ProjectType.WEB_APPLICATION]:
            department = DepartmentType.WEB_DEVELOPMENT
            institution = None
            confidence = 0.85
            reasoning = "Project is primarily a web application"
        elif project_type in [ProjectType.MICROSERVICE, ProjectType.API_SERVICE]:
            if "ml" in description or "ai" in description:
                department = DepartmentType.MACHINE_LEARNING
                institution = "keras_institution"
                confidence = 0.7
                reasoning = "Microservice with ML/AI components"
            else:
                department = DepartmentType.WEB_DEVELOPMENT
                institution = None
                confidence = 0.75
                reasoning = "General microservice or API development"
        elif project_type == ProjectType.INTEGRATION:
            department = DepartmentType.INTEGRATION
            institution = None
            confidence = 0.8
            reasoning = "Project focuses on system integration"
        else:
            # Default assignment
            department = DepartmentType.RESEARCH
            institution = None
            confidence = 0.6
            reasoning = "General project type - assigned to research department"
        
        # Determine if new institution is needed
        new_institution_needed = institution is None
        proposed_institution_name = None
        
        if new_institution_needed:
            # Suggest institution name based on project characteristics
            if project_type == ProjectType.DATA_PIPELINE:
                proposed_institution_name = "apache_spark_institution"
            elif project_type == ProjectType.WEB_APPLICATION:
                proposed_institution_name = "react_institution"
            elif project_type == ProjectType.INTEGRATION:
                proposed_institution_name = "integration_platform_institution"
        
        # Alternative departments
        alternatives = []
        if department != DepartmentType.RESEARCH:
            alternatives.append(DepartmentType.RESEARCH)
        if department != DepartmentType.INTEGRATION and "integration" in description:
            alternatives.append(DepartmentType.INTEGRATION)
        
        return DepartmentRoutingDecision(
            idea_id=str(idea.id),
            recommended_department=department,
            recommended_institution=institution,
            confidence_score=confidence,
            reasoning=reasoning,
            alternative_departments=alternatives,
            new_institution_needed=new_institution_needed,
            proposed_institution_name=proposed_institution_name
        )
    
    def _calculate_completion_date(self, phases: List[ImplementationPhase], 
                                 resources: ResourceRequirement) -> datetime:
        """Calculate estimated completion date."""
        total_hours = sum(phase.estimated_hours for phase in phases)
        
        # Apply timeline buffer
        buffered_hours = total_hours * self.config.default_timeline_buffer
        
        # Calculate working days (assuming 8 hours per day, 5 days per week)
        working_days = buffered_hours / (8 * resources.developers_needed)
        calendar_days = working_days * (7/5)  # Convert to calendar days
        
        return datetime.now(timezone.utc) + timedelta(days=calendar_days)
    
    def _generate_success_metrics(self, idea: Idea, analysis: Dict[str, Any]) -> List[str]:
        """Generate success metrics for the project."""
        metrics = [
            "Project completed within estimated timeline",
            "All functional requirements implemented",
            "Code quality standards met (>80% test coverage)",
            "Performance requirements satisfied"
        ]
        
        # Add project-type specific metrics
        if analysis["project_type"] == ProjectType.API_SERVICE:
            metrics.extend([
                "API response time <200ms for 95% of requests",
                "API availability >99.5%"
            ])
        elif analysis["project_type"] == ProjectType.ML_MODEL:
            metrics.extend([
                "Model accuracy meets target performance",
                "Model inference time within acceptable limits"
            ])
        elif analysis["project_type"] == ProjectType.WEB_APPLICATION:
            metrics.extend([
                "Page load time <3 seconds",
                "Mobile responsiveness achieved"
            ])
        
        # Add scalability metrics if needed
        if analysis["scalability_needs"] == "high":
            metrics.append("System handles expected peak load")
        
        return metrics
    
    def _generate_project_name(self, idea: Idea) -> str:
        """Generate a project name from the idea title."""
        # Clean and format the title
        name = re.sub(r'[^a-zA-Z0-9\s-]', '', idea.title)
        name = re.sub(r'\s+', '-', name.strip())
        name = name.lower()
        
        # Ensure reasonable length
        if len(name) > 50:
            name = name[:50].rstrip('-')
        
        return name or f"project-{str(idea.id)[:8]}"
    
    def _generate_project_description(self, idea: Idea, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive project description."""
        return f"""
{idea.description}

Project Type: {analysis['project_type'].value.replace('_', ' ').title()}
Complexity: {analysis['complexity'].value.title()}
Estimated Scope: {analysis['estimated_scope'].title()}

Key Requirements:
{chr(10).join(f'- {req}' for req in analysis['requirements'])}

Performance Requirements:
{chr(10).join(f'- {req}' for req in analysis['performance_requirements'])}

This project was automatically analyzed and templated by the AI-Galaxy Creator Agent.
        """.strip()
    
    def _identify_integration_points(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify key integration points for the project."""
        integrations = []
        
        # Standard integrations
        integrations.append("AI-Galaxy logging system")
        integrations.append("AI-Galaxy state machine")
        
        # Project-specific integrations
        if analysis["project_type"] == ProjectType.ML_MODEL:
            integrations.extend([
                "Model registry service",
                "Training data pipeline",
                "Model monitoring system"
            ])
        elif analysis["project_type"] == ProjectType.API_SERVICE:
            integrations.extend([
                "API gateway",
                "Authentication service",
                "Rate limiting service"
            ])
        elif analysis["project_type"] == ProjectType.WEB_APPLICATION:
            integrations.extend([
                "Content delivery network",
                "Analytics service",
                "User management system"
            ])
        
        # Redis integration for caching
        if analysis["scalability_needs"] in ["medium", "high"]:
            integrations.append("Redis caching service")
        
        # Vector search integration for AI features
        if "search" in str(analysis["requirements"]).lower():
            integrations.append("Vector search service")
        
        return integrations
    
    def _calculate_priority_score(self, idea: Idea, analysis: Dict[str, Any]) -> int:
        """Calculate implementation priority score."""
        base_score = idea.priority
        
        # Adjust based on complexity (simpler projects get higher priority)
        complexity_adjustment = {
            ProjectComplexity.SIMPLE: 2,
            ProjectComplexity.MODERATE: 1,
            ProjectComplexity.COMPLEX: 0,
            ProjectComplexity.ENTERPRISE: -1
        }
        
        adjusted_score = base_score + complexity_adjustment.get(analysis["complexity"], 0)
        
        # Boost ML and research projects
        if analysis["project_type"] in [ProjectType.ML_MODEL, ProjectType.RESEARCH_PROJECT]:
            adjusted_score += 1
        
        return max(1, min(10, adjusted_score))
    
    # File generation helpers
    
    def _generate_readme_content(self, idea: Idea, analysis: Dict[str, Any], 
                               tech_stack: TechnologyStack) -> str:
        """Generate README.md content."""
        return f"""# {idea.title}

{idea.description}

## Project Overview

- **Type**: {analysis['project_type'].value.replace('_', ' ').title()}
- **Technology Stack**: {tech_stack.value.replace('_', ' ').title()}
- **Complexity**: {analysis['complexity'].value.title()}

## Requirements

{chr(10).join(f'- {req}' for req in analysis['requirements'])}

## Getting Started

### Prerequisites

{self._get_prerequisites_text(tech_stack)}

### Installation

1. Clone the repository
2. Install dependencies
3. Configure environment variables
4. Run the application

## Development

### Project Structure

```
src/          # Source code
tests/        # Test files
docs/         # Documentation
config/       # Configuration files
```

### Running Tests

```bash
# Run all tests
{self._get_test_command(tech_stack)}
```

## Deployment

See `docs/deployment.md` for deployment instructions.

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is part of the AI-Galaxy ecosystem.

---
*This project was generated by the AI-Galaxy Creator Agent*
"""
    
    def _get_prerequisites_text(self, tech_stack: TechnologyStack) -> str:
        """Get prerequisites text for technology stack."""
        if tech_stack in [TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_ML]:
            return "- Python 3.8+\n- pip or poetry"
        elif tech_stack in [TechnologyStack.REACT_TYPESCRIPT, TechnologyStack.NODE_EXPRESS]:
            return "- Node.js 16+\n- npm or yarn"
        else:
            return "- See technology stack documentation"
    
    def _get_test_command(self, tech_stack: TechnologyStack) -> str:
        """Get test command for technology stack."""
        if tech_stack in [TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_ML]:
            return "pytest"
        elif tech_stack in [TechnologyStack.REACT_TYPESCRIPT, TechnologyStack.NODE_EXPRESS]:
            return "npm test"
        else:
            return "# See documentation for test commands"
    
    def _generate_gitignore_content(self, tech_stack: TechnologyStack) -> str:
        """Generate .gitignore content based on technology stack."""
        base_ignore = """
# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment variables
.env
.env.local
.env.*.local

# Dependencies
node_modules/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.coverage
.pytest_cache/
.tox/
htmlcov/

# Production builds
/build
/dist
"""
        
        if tech_stack in [TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_ML]:
            base_ignore += """
# Python specific
venv/
env/
ENV/
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/
"""
        
        elif tech_stack in [TechnologyStack.REACT_TYPESCRIPT, TechnologyStack.NODE_EXPRESS]:
            base_ignore += """
# Node.js specific
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.eslintcache
"""
        
        return base_ignore.strip()
    
    def _generate_python_files(self, idea: Idea, analysis: Dict[str, Any], 
                             tech_stack: TechnologyStack) -> List[FileTemplate]:
        """Generate Python-specific files."""
        files = []
        
        if tech_stack == TechnologyStack.PYTHON_FASTAPI:
            files.append(FileTemplate(
                path="src/main.py",
                content=self._generate_fastapi_main(),
                description="FastAPI application entry point"
            ))
            
            files.append(FileTemplate(
                path="src/api/__init__.py",
                content="",
                description="API package"
            ))
            
            files.append(FileTemplate(
                path="src/api/routes.py",
                content=self._generate_fastapi_routes(),
                description="API routes"
            ))
        
        elif tech_stack == TechnologyStack.PYTHON_ML:
            files.append(FileTemplate(
                path="src/model.py",
                content=self._generate_ml_model_stub(),
                description="ML model implementation"
            ))
            
            files.append(FileTemplate(
                path="src/train.py",
                content=self._generate_ml_training_script(),
                description="Model training script"
            ))
        
        return files
    
    def _generate_fastapi_main(self) -> str:
        """Generate FastAPI main application file."""
        return '''"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

app = FastAPI(
    title="AI-Galaxy Project",
    description="Generated by AI-Galaxy Creator Agent",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI-Galaxy Project API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
'''
    
    def _generate_fastapi_routes(self) -> str:
        """Generate FastAPI routes file."""
        return '''"""
API routes for the application.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()


@router.get("/status")
async def get_status() -> Dict[str, str]:
    """Get application status."""
    return {"status": "operational", "version": "0.1.0"}


@router.get("/example")
async def get_example() -> Dict[str, Any]:
    """Example endpoint - replace with actual functionality."""
    return {
        "message": "This is an example endpoint",
        "data": {"example_field": "example_value"}
    }


# Add more routes here based on your requirements
'''
    
    def _generate_ml_model_stub(self) -> str:
        """Generate ML model stub."""
        return '''"""
Machine learning model implementation.
"""

import numpy as np
from typing import Any, Dict, List
import joblib


class MLModel:
    """Base ML model class."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        # Implement your training logic here
        # Example with sklearn:
        # from sklearn.ensemble import RandomForestClassifier
        # self.model = RandomForestClassifier()
        # self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Implement prediction logic
        # return self.model.predict(X)
        return np.array([])  # Placeholder
    
    def save(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.model, filepath)
    
    def load(self, filepath: str) -> None:
        """Load a trained model."""
        self.model = joblib.load(filepath)
        self.is_trained = True
'''
    
    def _generate_ml_training_script(self) -> str:
        """Generate ML training script."""
        return '''"""
Model training script.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from model import MLModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> tuple:
    """Load training data."""
    # Implement data loading logic
    # Example:
    # df = pd.read_csv(data_path)
    # X = df.drop('target', axis=1).values
    # y = df['target'].values
    # return X, y
    
    logger.info(f"Loading data from {data_path}")
    # Placeholder - replace with actual data loading
    return None, None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--output", required=True, help="Path to save trained model")
    
    args = parser.parse_args()
    
    # Load data
    X, y = load_data(args.data)
    
    if X is None or y is None:
        logger.error("Failed to load data")
        return
    
    # Train model
    model = MLModel()
    logger.info("Starting model training...")
    model.train(X, y)
    
    # Save model
    model.save(args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
'''
    
    def _generate_react_files(self, idea: Idea, analysis: Dict[str, Any]) -> List[FileTemplate]:
        """Generate React TypeScript files."""
        files = []
        
        files.append(FileTemplate(
            path="src/App.tsx",
            content=self._generate_react_app(),
            description="Main React application component"
        ))
        
        files.append(FileTemplate(
            path="src/index.tsx",
            content=self._generate_react_index(),
            description="React application entry point"
        ))
        
        return files
    
    def _generate_react_app(self) -> str:
        """Generate React App.tsx."""
        return '''import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>AI-Galaxy Project</h1>
        <p>Generated by AI-Galaxy Creator Agent</p>
      </header>
      <main>
        {/* Add your main application content here */}
        <p>Welcome to your new AI-Galaxy project!</p>
      </main>
    </div>
  );
}

export default App;
'''
    
    def _generate_react_index(self) -> str:
        """Generate React index.tsx."""
        return '''import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
'''
    
    def _generate_node_files(self, idea: Idea, analysis: Dict[str, Any]) -> List[FileTemplate]:
        """Generate Node.js Express files."""
        files = []
        
        files.append(FileTemplate(
            path="src/app.js",
            content=self._generate_express_app(),
            description="Express application setup"
        ))
        
        files.append(FileTemplate(
            path="src/routes/index.js",
            content=self._generate_express_routes(),
            description="Express routes"
        ))
        
        return files
    
    def _generate_express_app(self) -> str:
        """Generate Express app.js."""
        return '''const express = require('express');
const cors = require('cors');
const helmet = require('helmet');

const indexRoutes = require('./routes/index');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api', indexRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = app;
'''
    
    def _generate_express_routes(self) -> str:
        """Generate Express routes."""
        return '''const express = require('express');
const router = express.Router();

// GET /api/status
router.get('/status', (req, res) => {
  res.json({
    status: 'operational',
    version: '0.1.0',
    timestamp: new Date().toISOString()
  });
});

// GET /api/example
router.get('/example', (req, res) => {
  res.json({
    message: 'This is an example endpoint',
    data: { example_field: 'example_value' }
  });
});

// Add more routes here based on your requirements

module.exports = router;
'''
    
    def _generate_python_requirements(self, tech_stack: TechnologyStack) -> str:
        """Generate Python requirements.txt."""
        if tech_stack == TechnologyStack.PYTHON_FASTAPI:
            return '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
'''
        elif tech_stack == TechnologyStack.PYTHON_ML:
            return '''numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
joblib==1.3.2
pytest==7.4.3
'''
        else:
            return ""
    
    def _generate_pyproject_toml(self, analysis: Dict[str, Any]) -> str:
        """Generate pyproject.toml for Python projects."""
        return f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-galaxy-project"
version = "0.1.0"
description = "AI-Galaxy generated project"
readme = "README.md"
requires-python = ">=3.8"
license = {{text = "MIT"}}
authors = [
    {{name = "AI-Galaxy Creator Agent"}},
]
keywords = ["ai-galaxy", "{analysis['project_type'].value}"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

[project.urls]
Homepage = "https://github.com/ai-galaxy/project"
Repository = "https://github.com/ai-galaxy/project.git"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
'''
    
    def _generate_package_json(self, analysis: Dict[str, Any], 
                             tech_stack: TechnologyStack) -> str:
        """Generate package.json for Node.js projects."""
        if tech_stack == TechnologyStack.REACT_TYPESCRIPT:
            return '''{
  "name": "ai-galaxy-project",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^4.9.5",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^5.16.4",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "@types/jest": "^27.5.2",
    "@types/node": "^16.18.0",
    "@types/react": "^18.0.26",
    "@types/react-dom": "^18.0.9",
    "react-scripts": "5.0.1"
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}'''
        elif tech_stack == TechnologyStack.NODE_EXPRESS:
            return '''{
  "name": "ai-galaxy-project",
  "version": "0.1.0",
  "description": "AI-Galaxy generated Express.js project",
  "main": "src/app.js",
  "scripts": {
    "start": "node src/app.js",
    "dev": "nodemon src/app.js",
    "test": "jest",
    "test:watch": "jest --watch"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "helmet": "^7.1.0",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.7.0",
    "supertest": "^6.3.3"
  },
  "engines": {
    "node": ">=16.0.0"
  }
}'''
        else:
            return "{}"
    
    def _generate_tsconfig(self) -> str:
        """Generate TypeScript configuration."""
        return '''{
  "compilerOptions": {
    "target": "es5",
    "lib": [
      "dom",
      "dom.iterable",
      "es6"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": [
    "src"
  ]
}'''
    
    def _generate_dockerfile(self, tech_stack: TechnologyStack) -> str:
        """Generate Dockerfile based on technology stack."""
        if tech_stack in [TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_ML]:
            return '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        elif tech_stack in [TechnologyStack.NODE_EXPRESS, TechnologyStack.REACT_TYPESCRIPT]:
            return '''FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY src/ ./src/

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S app -u 1001
USER app

# Expose port
EXPOSE 3000

# Run application
CMD ["npm", "start"]
'''
        else:
            return ""
    
    def _generate_docker_compose(self, analysis: Dict[str, Any]) -> str:
        """Generate docker-compose.yml."""
        return '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=development
    volumes:
      - ./src:/app/src
    depends_on:
      - redis
      - db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ai_galaxy_project
      POSTGRES_USER: app
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
'''
    
    def _estimate_core_implementation_hours(self, analysis: Dict[str, Any]) -> int:
        """Estimate hours for core implementation phase."""
        base_hours = {
            ProjectComplexity.SIMPLE: 16,
            ProjectComplexity.MODERATE: 32,
            ProjectComplexity.COMPLEX: 64,
            ProjectComplexity.ENTERPRISE: 120
        }
        
        hours = base_hours.get(analysis["complexity"], 32)
        
        # Adjust based on project type
        type_multipliers = {
            ProjectType.ML_MODEL: 1.5,
            ProjectType.WEB_APPLICATION: 1.2,
            ProjectType.DATA_PIPELINE: 1.3,
            ProjectType.MICROSERVICE: 1.0,
            ProjectType.LIBRARY: 0.8
        }
        
        multiplier = type_multipliers.get(analysis["project_type"], 1.0)
        return int(hours * multiplier)
    
    def _generate_core_implementation_tasks(self, idea: Idea, analysis: Dict[str, Any], 
                                          tech_stack: TechnologyStack) -> List[str]:
        """Generate core implementation tasks."""
        tasks = [
            "Implement main business logic",
            "Create data models and schemas",
            "Set up error handling and validation"
        ]
        
        # Technology-specific tasks
        if tech_stack == TechnologyStack.PYTHON_FASTAPI:
            tasks.extend([
                "Create FastAPI endpoints",
                "Implement request/response models",
                "Add API documentation"
            ])
        elif tech_stack == TechnologyStack.PYTHON_ML:
            tasks.extend([
                "Implement model training pipeline",
                "Create prediction interface",
                "Add model evaluation metrics"
            ])
        elif tech_stack == TechnologyStack.REACT_TYPESCRIPT:
            tasks.extend([
                "Create main UI components",
                "Implement state management",
                "Add routing and navigation"
            ])
        
        # Project type specific tasks
        if analysis["project_type"] == ProjectType.API_SERVICE:
            tasks.extend([
                "Implement authentication",
                "Add rate limiting",
                "Create API versioning"
            ])
        elif analysis["project_type"] == ProjectType.WEB_APPLICATION:
            tasks.extend([
                "Implement user interface",
                "Add responsive design",
                "Integrate with backend APIs"
            ])
        
        return tasks
    
    def _serialize_directory_structure(self, structure: DirectoryStructure) -> Dict[str, Any]:
        """Serialize directory structure to dict."""
        return {
            "name": structure.name,
            "description": structure.description,
            "subdirectories": [self._serialize_directory_structure(sub) for sub in structure.subdirectories],
            "files": [self._serialize_file_template(f) for f in structure.files]
        }
    
    def _serialize_file_template(self, template: FileTemplate) -> Dict[str, Any]:
        """Serialize file template to dict."""
        return {
            "path": template.path,
            "content": template.content,
            "is_executable": template.is_executable,
            "description": template.description,
            "dependencies": template.dependencies
        }
    
    def _create_directories(self, base_path: Path, structure: Dict[str, Any]):
        """Create directory structure from template."""
        for subdir in structure.get("subdirectories", []):
            dir_path = base_path / subdir["name"]
            dir_path.mkdir(exist_ok=True)
            self._create_directories(dir_path, subdir)
    
    def _create_implementation_guide(self, base_path: Path, template: ProjectTemplate):
        """Create implementation guide document."""
        guide_content = f"""# Implementation Guide

## Project: {template.project_name}

### Overview
{template.description}

### Implementation Phases

"""
        
        for phase in template.implementation_phases:
            guide_content += f"""
#### Phase {phase.phase_number}: {phase.name}

**Description**: {phase.description}

**Estimated Hours**: {phase.estimated_hours}

**Prerequisites**: {', '.join(phase.prerequisites) if phase.prerequisites else 'None'}

**Tasks**:
{chr(10).join(f'- {task}' for task in phase.tasks)}

**Deliverables**:
{chr(10).join(f'- {deliverable}' for deliverable in phase.deliverables)}

**Success Criteria**:
{chr(10).join(f'- {criteria}' for criteria in phase.success_criteria)}

**Risk Factors**:
{chr(10).join(f'- {risk}' for risk in phase.risk_factors)}

---
"""
        
        guide_content += f"""
### Resource Requirements

- **Developers Needed**: {template.resource_requirements.developers_needed}
- **Timeline**: {template.resource_requirements.timeline_weeks} weeks
- **Estimated Cost**: {template.resource_requirements.estimated_cost}

**Required Skills**:
{chr(10).join(f'- {skill}' for skill in template.resource_requirements.skill_requirements)}

### Testing Strategy

**Test Types**: {', '.join(template.testing_strategy.test_types)}

**Frameworks**: {', '.join(template.testing_strategy.test_frameworks)}

**Automation Level**: {template.testing_strategy.automation_level}

### Success Metrics

{chr(10).join(f'- {metric}' for metric in template.success_metrics)}

### Integration Points

{chr(10).join(f'- {integration}' for integration in template.integration_points)}

---
*Generated by AI-Galaxy Creator Agent on {template.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        guide_path = base_path / "IMPLEMENTATION_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
    
    def _create_fallback_template(self, idea: Idea, start_time: datetime) -> ProjectTemplate:
        """Create a minimal fallback template when creation fails."""
        return ProjectTemplate(
            idea_id=str(idea.id),
            project_name=self._generate_project_name(idea),
            project_type=ProjectType.MICROSERVICE,
            technology_stack=TechnologyStack.PYTHON_FASTAPI,
            complexity=ProjectComplexity.SIMPLE,
            description=f"Fallback template for: {idea.description}",
            resource_requirements=ResourceRequirement(
                developers_needed=1,
                skill_requirements=["Software development"],
                external_services=[],
                infrastructure_needs=["Development environment"],
                estimated_cost="$8,000",
                timeline_weeks=2
            ),
            testing_strategy=TestingStrategy(
                test_frameworks=["pytest"],
                test_types=["Unit tests"],
                coverage_targets={"unit": 70.0},
                automation_level="medium"
            ),
            deployment_config=DeploymentConfiguration(
                deployment_strategy="Direct deployment",
                environments=["development", "production"],
                infrastructure_type="Virtual machines",
                scaling_strategy="Vertical scaling",
                monitoring_setup=["Application logs"],
                backup_strategy="Daily automated backups"
            ),
            recommended_department=DepartmentType.RESEARCH,
            estimated_completion_date=datetime.now(timezone.utc) + timedelta(weeks=2),
            priority_score=5
        )
    
    def _update_creation_metrics(self, template: ProjectTemplate, start_time: datetime):
        """Update internal creation metrics."""
        self.creation_metrics["total_templates_created"] += 1
        self.creation_metrics["successful_generations"] += 1
        
        # Update type-specific metrics
        project_type = template.project_type.value
        self.creation_metrics["templates_by_type"][project_type] = \
            self.creation_metrics["templates_by_type"].get(project_type, 0) + 1
        
        # Update complexity metrics
        complexity = template.complexity.value
        self.creation_metrics["templates_by_complexity"][complexity] = \
            self.creation_metrics["templates_by_complexity"].get(complexity, 0) + 1
        
        # Update average creation time
        creation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        current_avg = self.creation_metrics["average_creation_time"]
        total_created = self.creation_metrics["total_templates_created"]
        
        new_avg = ((current_avg * (total_created - 1)) + creation_time) / total_created
        self.creation_metrics["average_creation_time"] = new_avg
    
    def _calculate_average_phases(self) -> float:
        """Calculate average number of phases per project."""
        if not self.template_history:
            return 0.0
        
        total_phases = 0
        total_templates = 0
        
        for templates in self.template_history.values():
            for template in templates:
                total_phases += len(template.implementation_phases)
                total_templates += 1
        
        return total_phases / total_templates if total_templates > 0 else 0.0
    
    def _calculate_average_priority(self) -> float:
        """Calculate average priority score."""
        if not self.template_history:
            return 0.0
        
        total_priority = 0
        total_templates = 0
        
        for templates in self.template_history.values():
            for template in templates:
                total_priority += template.priority_score
                total_templates += 1
        
        return total_priority / total_templates if total_templates > 0 else 0.0


# Factory function for easy agent creation
def create_creator_agent(config: Optional[CreatorConfiguration] = None,
                        state_router: Optional[StateMachineRouter] = None) -> CreatorAgent:
    """
    Create a new Creator Agent instance.
    
    Args:
        config: Optional creator configuration
        state_router: Optional state machine router
        
    Returns:
        Configured CreatorAgent instance
    """
    return CreatorAgent(config, state_router)


# Export main classes and functions
__all__ = [
    "CreatorAgent",
    "CreatorConfiguration",
    "ProjectTemplate",
    "DepartmentRoutingDecision",
    "ProjectType",
    "TechnologyStack",
    "ProjectComplexity",
    "DepartmentType",
    "create_creator_agent"
]