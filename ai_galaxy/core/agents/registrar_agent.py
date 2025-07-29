"""
Registrar Agent for the AI-Galaxy Lower Meta-Layer.

This module implements the Registrar Agent, the institutional memory and cataloging
system of the AI-Galaxy ecosystem. The Registrar manages registration, documentation,
and lifecycle of completed microservices, serving as the central repository for
service discovery, quality assessment, and governance.
"""

import json
import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel, Field

from ...shared.models import Microservice, MicroserviceStatus, Department, Institution
from ...shared.logger import get_logger, LogContext


class ServiceRegistrationStatus(str, Enum):
    """Status enumeration for service registration process."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    REGISTERED = "registered"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ServiceType(str, Enum):
    """Types of services that can be registered."""
    API = "api"
    LIBRARY = "library"
    WORKER = "worker"
    INTERFACE = "interface"
    TOOL = "tool"
    INTEGRATION = "integration"
    UTILITY = "utility"


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXPERIMENTAL = "experimental"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"
    DEPRECATED = "deprecated"


class ComplianceStatus(str, Enum):
    """Compliance verification status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REVIEW_REQUIRED = "review_required"
    EXEMPT = "exempt"


class DocumentationType(str, Enum):
    """Types of documentation."""
    API_DOCS = "api_docs"
    USER_GUIDE = "user_guide"
    INTEGRATION_GUIDE = "integration_guide"
    ARCHITECTURE = "architecture"
    DEPLOYMENT = "deployment"
    CHANGELOG = "changelog"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class ServiceEndpoint:
    """Represents a service API endpoint."""
    path: str
    method: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    responses: Dict[str, Any] = field(default_factory=dict)
    authentication_required: bool = False
    rate_limit: Optional[str] = None


@dataclass
class ServiceDependency:
    """Represents a service dependency."""
    service_id: str
    service_name: str
    version_requirement: str
    dependency_type: str  # "required", "optional", "development"
    relationship: str  # "uses", "extends", "integrates_with"


@dataclass
class QualityMetric:
    """Quality assessment metric."""
    metric_name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    status: str = "unknown"  # "pass", "fail", "warning", "unknown"
    measurement_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PerformanceBenchmark:
    """Performance benchmark data."""
    benchmark_name: str
    metric_type: str  # "latency", "throughput", "memory", "cpu"
    value: float
    unit: str
    test_conditions: Dict[str, Any] = field(default_factory=dict)
    baseline_value: Optional[float] = None
    measurement_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ServiceDocumentation(BaseModel):
    """Service documentation metadata."""
    doc_type: DocumentationType
    title: str
    file_path: str
    version: str = "1.0.0"
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    auto_generated: bool = False
    review_required: bool = False
    reviewer: Optional[str] = None
    review_date: Optional[datetime] = None


class ServiceMetadata(BaseModel):
    """Comprehensive service metadata."""
    service_id: str
    name: str
    description: str
    service_type: ServiceType
    version: str
    registration_status: ServiceRegistrationStatus
    quality_level: QualityLevel
    
    # Technical details
    programming_language: str
    framework: Optional[str] = None
    database_type: Optional[str] = None
    runtime_requirements: Dict[str, str] = Field(default_factory=dict)
    
    # API information
    api_endpoints: List[ServiceEndpoint] = Field(default_factory=list)
    api_version: Optional[str] = None
    openapi_spec_path: Optional[str] = None
    
    # Dependencies
    dependencies: List[ServiceDependency] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)  # Services that depend on this one
    
    # Quality metrics
    quality_metrics: List[QualityMetric] = Field(default_factory=list)
    performance_benchmarks: List[PerformanceBenchmark] = Field(default_factory=list)
    test_coverage: Optional[float] = None
    code_quality_score: Optional[float] = None
    security_score: Optional[float] = None
    
    # Documentation
    documentation: List[ServiceDocumentation] = Field(default_factory=list)
    readme_path: Optional[str] = None
    examples_path: Optional[str] = None
    
    # Lifecycle
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deprecation_date: Optional[datetime] = None
    retirement_date: Optional[datetime] = None
    
    # Usage and analytics
    usage_count: int = 0
    popularity_score: float = 0.0
    last_accessed: Optional[datetime] = None
    
    # Governance
    owner: str
    maintainers: List[str] = Field(default_factory=list)
    compliance_status: ComplianceStatus = ComplianceStatus.REVIEW_REQUIRED
    compliance_notes: List[str] = Field(default_factory=list)
    
    # Tags and categorization
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    
    # Deployment information
    deployment_environments: List[str] = Field(default_factory=list)
    health_check_url: Optional[str] = None
    monitoring_dashboard: Optional[str] = None
    
    # Vector search embeddings
    description_embedding: Optional[List[float]] = None
    capability_embedding: Optional[List[float]] = None


class ServiceRegistry(BaseModel):
    """Central service registry."""
    registry_id: str = Field(default_factory=lambda: str(uuid4()))
    services: Dict[str, ServiceMetadata] = Field(default_factory=dict)
    categories: Dict[str, List[str]] = Field(default_factory=dict)
    tags: Dict[str, List[str]] = Field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"


class RegistrationRequest(BaseModel):
    """Request for registering a new service."""
    service_name: str
    service_type: ServiceType
    code_path: str
    owner: str
    description: str
    version: str = "1.0.0"
    programming_language: str
    framework: Optional[str] = None
    api_endpoint: Optional[str] = None
    documentation_path: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    auto_analyze: bool = True
    submission_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegistrarConfiguration(BaseModel):
    """Configuration for the Registrar Agent."""
    registry_storage_path: str = "data/service_registry.json"
    documentation_base_path: str = "docs/services"
    quality_thresholds: Dict[str, float] = Field(default_factory=dict)
    auto_documentation_enabled: bool = True
    compliance_rules: Dict[str, Any] = Field(default_factory=dict)
    vector_search_enabled: bool = True
    performance_monitoring_enabled: bool = True
    deprecation_warning_days: int = 90
    retirement_grace_period_days: int = 180
    enable_quality_gates: bool = True
    minimum_test_coverage: float = 70.0
    minimum_code_quality_score: float = 7.0


class ServiceSearchQuery(BaseModel):
    """Search query for service discovery."""
    query_text: Optional[str] = None
    service_type: Optional[ServiceType] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    quality_level: Optional[QualityLevel] = None
    programming_language: Optional[str] = None
    has_api: Optional[bool] = None
    min_quality_score: Optional[float] = None
    sort_by: str = "relevance"  # "relevance", "popularity", "created_date", "updated_date"
    limit: int = 20


class ServiceSearchResult(BaseModel):
    """Result from service search."""
    service_id: str
    name: str
    description: str
    service_type: ServiceType
    quality_level: QualityLevel
    relevance_score: float
    popularity_score: float
    last_updated: datetime
    tags: List[str] = Field(default_factory=list)
    api_available: bool = False


class RegistrarAgent:
    """
    The Registrar Agent - institutional memory and cataloging system of AI-Galaxy.
    
    Manages service registration, documentation, quality assessment, lifecycle
    management, and provides comprehensive service discovery capabilities with
    vector search integration and governance oversight.
    """
    
    def __init__(self, config: Optional[RegistrarConfiguration] = None):
        """
        Initialize the Registrar Agent.
        
        Args:
            config: Registrar configuration parameters
        """
        self.logger = get_logger("registrar_agent")
        self.config = config or RegistrarConfiguration()
        
        # Initialize service registry
        self.registry = self._load_registry()
        
        # Initialize quality thresholds if not provided
        if not self.config.quality_thresholds:
            self.config.quality_thresholds = self._default_quality_thresholds()
        
        # Initialize compliance rules if not provided
        if not self.config.compliance_rules:
            self.config.compliance_rules = self._default_compliance_rules()
        
        # Tracking and metrics
        self.registration_history: List[RegistrationRequest] = []
        self.quality_assessments: Dict[str, List[Dict[str, Any]]] = {}
        self.usage_analytics: Dict[str, Dict[str, Any]] = {}
        self.lifecycle_events: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance metrics
        self.registrar_metrics = {
            "total_services": 0,
            "services_by_type": {},
            "services_by_quality": {},
            "average_quality_score": 0.0,
            "compliance_rate": 0.0,
            "documentation_coverage": 0.0
        }
        
        self.logger.agent_action("registrar_agent_initialized", "registrar_agent",
                                additional_context={
                                    "registry_path": self.config.registry_storage_path,
                                    "quality_gates_enabled": self.config.enable_quality_gates,
                                    "vector_search_enabled": self.config.vector_search_enabled
                                })
    
    def register_service(self, request: RegistrationRequest) -> Tuple[bool, str, Optional[ServiceMetadata]]:
        """
        Register a new microservice in the registry.
        
        Args:
            request: Service registration request
            
        Returns:
            Tuple of (success, message, service_metadata)
        """
        service_id = str(uuid4())
        
        context = LogContext(
            agent_name="registrar_agent",
            additional_context={
                "service_name": request.service_name,
                "service_type": request.service_type.value,
                "owner": request.owner
            }
        )
        
        self.logger.agent_action("registering_service", "registrar_agent", service_id)
        
        try:
            # Validate registration request
            validation_result = self._validate_registration_request(request)
            if not validation_result[0]:
                return False, validation_result[1], None
            
            # Check for duplicate services
            if self._service_exists(request.service_name):
                return False, f"Service with name '{request.service_name}' already exists", None
            
            # Analyze service if auto-analysis enabled
            analysis_results = {}
            if request.auto_analyze:
                analysis_results = self._analyze_service_code(request.code_path)
            
            # Extract API endpoints if available
            api_endpoints = []
            if request.api_endpoint:
                api_endpoints = self._discover_api_endpoints(request.code_path, request.api_endpoint)
            
            # Analyze dependencies
            dependencies = self._analyze_dependencies(request.code_path)
            
            # Generate initial documentation
            documentation = []
            if self.config.auto_documentation_enabled:
                documentation = self._generate_documentation(request, analysis_results)
            
            # Create service metadata
            service_metadata = ServiceMetadata(
                service_id=service_id,
                name=request.service_name,
                description=request.description,
                service_type=request.service_type,
                version=request.version,
                registration_status=ServiceRegistrationStatus.PENDING,
                quality_level=QualityLevel.DEVELOPMENT,
                programming_language=request.programming_language,
                framework=request.framework,
                api_endpoints=api_endpoints,
                dependencies=dependencies,
                documentation=documentation,
                owner=request.owner,
                tags=request.tags,
                categories=request.categories,
                **analysis_results
            )
            
            # Perform quality assessment
            quality_assessment = self._assess_service_quality(service_metadata, request.code_path)
            service_metadata.quality_metrics = quality_assessment["metrics"]
            service_metadata.quality_level = quality_assessment["level"]
            service_metadata.test_coverage = quality_assessment.get("test_coverage")
            service_metadata.code_quality_score = quality_assessment.get("code_quality_score")
            service_metadata.security_score = quality_assessment.get("security_score")
            
            # Check compliance
            compliance_result = self._check_compliance(service_metadata, request.code_path)
            service_metadata.compliance_status = compliance_result["status"]
            service_metadata.compliance_notes = compliance_result["notes"]
            
            # Generate embeddings for vector search
            if self.config.vector_search_enabled:
                embeddings = self._generate_service_embeddings(service_metadata)
                service_metadata.description_embedding = embeddings.get("description")
                service_metadata.capability_embedding = embeddings.get("capability")
            
            # Apply quality gates
            if self.config.enable_quality_gates:
                gate_result = self._apply_quality_gates(service_metadata)
                if not gate_result[0]:
                    service_metadata.registration_status = ServiceRegistrationStatus.REJECTED
                    self._save_registry()
                    return False, f"Quality gate failure: {gate_result[1]}", service_metadata
            
            # Register the service
            service_metadata.registration_status = ServiceRegistrationStatus.REGISTERED
            self.registry.services[service_id] = service_metadata
            
            # Update registry indexes
            self._update_registry_indexes(service_metadata)
            
            # Store registration history
            self.registration_history.append(request)
            
            # Save registry
            self._save_registry()
            
            # Update metrics
            self._update_registrar_metrics()
            
            self.logger.agent_action("service_registered", "registrar_agent", service_id, {
                "service_name": request.service_name,
                "quality_level": service_metadata.quality_level.value,
                "compliance_status": service_metadata.compliance_status.value
            })
            
            return True, f"Service '{request.service_name}' registered successfully", service_metadata
            
        except Exception as e:
            self.logger.error(f"Service registration failed: {e}", context, exc_info=True)
            return False, f"Registration failed: {str(e)}", None
    
    def discover_services(self, query: ServiceSearchQuery) -> List[ServiceSearchResult]:
        """
        Discover services based on search criteria.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of matching services ranked by relevance
        """
        self.logger.agent_action("discovering_services", "registrar_agent", 
                                additional_context={"query": query.dict()})
        
        try:
            # Filter services based on criteria
            candidate_services = self._filter_services(query)
            
            # Calculate relevance scores
            scored_services = []
            for service_id, service in candidate_services.items():
                relevance_score = self._calculate_relevance_score(service, query)
                
                result = ServiceSearchResult(
                    service_id=service_id,
                    name=service.name,
                    description=service.description,
                    service_type=service.service_type,
                    quality_level=service.quality_level,
                    relevance_score=relevance_score,
                    popularity_score=service.popularity_score,
                    last_updated=service.last_updated,
                    tags=service.tags,
                    api_available=bool(service.api_endpoints)
                )
                
                scored_services.append(result)
            
            # Sort by specified criteria
            sorted_services = self._sort_search_results(scored_services, query.sort_by)
            
            # Apply limit
            limited_services = sorted_services[:query.limit]
            
            # Update usage analytics
            self._update_search_analytics(query, len(limited_services))
            
            self.logger.agent_action("services_discovered", "registrar_agent",
                                   additional_context={
                                       "query_terms": query.query_text,
                                       "results_count": len(limited_services)
                                   })
            
            return limited_services
            
        except Exception as e:
            self.logger.error(f"Service discovery failed: {e}", exc_info=True)
            return []
    
    def get_service_details(self, service_id: str) -> Optional[ServiceMetadata]:
        """
        Get detailed information about a specific service.
        
        Args:
            service_id: ID of the service
            
        Returns:
            Service metadata or None if not found
        """
        service = self.registry.services.get(service_id)
        
        if service:
            # Update access tracking
            service.last_accessed = datetime.now(timezone.utc)
            service.usage_count += 1
            self._update_service_popularity(service_id)
            self._save_registry()
        
        return service
    
    def update_service_status(self, service_id: str, new_status: ServiceRegistrationStatus,
                             reason: Optional[str] = None) -> bool:
        """
        Update the status of a registered service.
        
        Args:
            service_id: ID of the service
            new_status: New status to set
            reason: Optional reason for status change
            
        Returns:
            True if update successful, False otherwise
        """
        context = LogContext(
            agent_name="registrar_agent",
            additional_context={"service_id": service_id, "new_status": new_status.value}
        )
        
        try:
            service = self.registry.services.get(service_id)
            if not service:
                self.logger.warning(f"Service {service_id} not found", context)
                return False
            
            old_status = service.registration_status
            service.registration_status = new_status
            service.last_updated = datetime.now(timezone.utc)
            
            # Handle status-specific logic
            if new_status == ServiceRegistrationStatus.DEPRECATED:
                service.deprecation_date = datetime.now(timezone.utc)
                service.quality_level = QualityLevel.DEPRECATED
            elif new_status == ServiceRegistrationStatus.ARCHIVED:
                service.retirement_date = datetime.now(timezone.utc)
            
            # Record lifecycle event
            self._record_lifecycle_event(service_id, old_status.value, new_status.value, reason)
            
            self._save_registry()
            
            self.logger.agent_action("service_status_updated", "registrar_agent", service_id, {
                "old_status": old_status.value,
                "new_status": new_status.value,
                "reason": reason
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Status update failed: {e}", context, exc_info=True)
            return False
    
    def assess_service_quality(self, service_id: str, force_reassessment: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment of a service.
        
        Args:
            service_id: ID of the service
            force_reassessment: Force reassessment even if recent assessment exists
            
        Returns:
            Quality assessment results
        """
        context = LogContext(
            agent_name="registrar_agent",
            additional_context={"service_id": service_id}
        )
        
        try:
            service = self.registry.services.get(service_id)
            if not service:
                return {"error": "Service not found"}
            
            # Check if recent assessment exists
            if not force_reassessment and self._has_recent_quality_assessment(service_id):
                return self.quality_assessments[service_id][-1]
            
            # Perform quality assessment
            assessment = self._assess_service_quality(service, self._get_service_code_path(service))
            
            # Update service metadata
            service.quality_metrics = assessment["metrics"]
            service.quality_level = assessment["level"]
            service.test_coverage = assessment.get("test_coverage")
            service.code_quality_score = assessment.get("code_quality_score")
            service.security_score = assessment.get("security_score")
            service.last_updated = datetime.now(timezone.utc)
            
            # Store assessment history
            if service_id not in self.quality_assessments:
                self.quality_assessments[service_id] = []
            
            assessment["assessment_date"] = datetime.now(timezone.utc).isoformat()
            self.quality_assessments[service_id].append(assessment)
            
            self._save_registry()
            
            self.logger.agent_action("quality_assessed", "registrar_agent", service_id, {
                "quality_level": assessment["level"].value,
                "code_quality_score": assessment.get("code_quality_score"),
                "test_coverage": assessment.get("test_coverage")
            })
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}", context, exc_info=True)
            return {"error": str(e)}
    
    def generate_service_documentation(self, service_id: str, doc_types: List[DocumentationType]) -> bool:
        """
        Generate documentation for a service.
        
        Args:
            service_id: ID of the service
            doc_types: Types of documentation to generate
            
        Returns:
            True if generation successful, False otherwise
        """
        context = LogContext(
            agent_name="registrar_agent",
            additional_context={"service_id": service_id, "doc_types": [dt.value for dt in doc_types]}
        )
        
        try:
            service = self.registry.services.get(service_id)
            if not service:
                self.logger.warning(f"Service {service_id} not found", context)
                return False
            
            code_path = self._get_service_code_path(service)
            generated_docs = []
            
            for doc_type in doc_types:
                doc_result = self._generate_specific_documentation(service, code_path, doc_type)
                if doc_result:
                    generated_docs.append(doc_result)
            
            # Update service documentation
            service.documentation.extend(generated_docs)
            service.last_updated = datetime.now(timezone.utc)
            
            self._save_registry()
            
            self.logger.agent_action("documentation_generated", "registrar_agent", service_id, {
                "doc_types": [dt.value for dt in doc_types],
                "generated_count": len(generated_docs)
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}", context, exc_info=True)
            return False
    
    def get_dependency_graph(self, service_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dependency graph for a specific service or entire ecosystem.
        
        Args:
            service_id: Optional service ID to focus on, None for full graph
            
        Returns:
            Dependency graph data
        """
        try:
            if service_id:
                service = self.registry.services.get(service_id)
                if not service:
                    return {"error": "Service not found"}
                
                return self._build_service_dependency_graph(service_id)
            else:
                return self._build_ecosystem_dependency_graph()
                
        except Exception as e:
            self.logger.error(f"Dependency graph generation failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def get_analytics_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Generate analytics report about the service ecosystem.
        
        Args:
            report_type: Type of report to generate
            
        Returns:
            Analytics report data
        """
        try:
            report = {
                "report_type": report_type,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "registry_version": self.registry.version,
                "summary": self._generate_ecosystem_summary()
            }
            
            if report_type in ["comprehensive", "quality"]:
                report["quality_metrics"] = self._generate_quality_report()
            
            if report_type in ["comprehensive", "usage"]:
                report["usage_analytics"] = self._generate_usage_report()
            
            if report_type in ["comprehensive", "compliance"]:
                report["compliance_report"] = self._generate_compliance_report()
            
            if report_type in ["comprehensive", "lifecycle"]:
                report["lifecycle_analysis"] = self._generate_lifecycle_report()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Analytics report generation failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def manage_service_lifecycle(self, service_id: str, action: str, **kwargs) -> bool:
        """
        Manage service lifecycle actions (deprecate, retire, archive).
        
        Args:
            service_id: ID of the service
            action: Lifecycle action to perform
            **kwargs: Additional parameters for the action
            
        Returns:
            True if action successful, False otherwise
        """
        context = LogContext(
            agent_name="registrar_agent",
            additional_context={"service_id": service_id, "action": action}
        )
        
        try:
            service = self.registry.services.get(service_id)
            if not service:
                self.logger.warning(f"Service {service_id} not found", context)
                return False
            
            if action == "deprecate":
                return self._deprecate_service(service_id, kwargs.get("reason"))
            elif action == "retire":
                return self._retire_service(service_id, kwargs.get("reason"))
            elif action == "archive":
                return self._archive_service(service_id, kwargs.get("reason"))
            elif action == "reactivate":
                return self._reactivate_service(service_id, kwargs.get("reason"))
            else:
                self.logger.warning(f"Unknown lifecycle action: {action}", context)
                return False
                
        except Exception as e:
            self.logger.error(f"Lifecycle management failed: {e}", context, exc_info=True)
            return False
    
    # Private helper methods
    
    def _load_registry(self) -> ServiceRegistry:
        """Load service registry from storage."""
        registry_path = Path(self.config.registry_storage_path)
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ServiceRegistry(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load registry from {registry_path}: {e}")
        
        # Create new registry
        return ServiceRegistry()
    
    def _save_registry(self):
        """Save service registry to storage."""
        registry_path = Path(self.config.registry_storage_path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry.dict(), f, indent=2, default=str)
            
            self.registry.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def _default_quality_thresholds(self) -> Dict[str, float]:
        """Default quality assessment thresholds."""
        return {
            "test_coverage": 70.0,
            "code_quality": 7.0,
            "security_score": 8.0,
            "documentation_coverage": 80.0,
            "performance_score": 6.0
        }
    
    def _default_compliance_rules(self) -> Dict[str, Any]:
        """Default compliance rules."""
        return {
            "naming_convention": {
                "pattern": r"^[a-z][a-z0-9_]*[a-z0-9]$",
                "description": "Service names must be lowercase with underscores"
            },
            "required_files": [
                "README.md",
                "requirements.txt",
                ".gitignore"
            ],
            "security_requirements": {
                "no_hardcoded_secrets": True,
                "dependency_scan": True,
                "vulnerability_check": True
            },
            "documentation_requirements": {
                "api_documentation": True,
                "usage_examples": True,
                "deployment_guide": True
            }
        }
    
    def _validate_registration_request(self, request: RegistrationRequest) -> Tuple[bool, str]:
        """Validate service registration request."""
        # Check required fields
        if not request.service_name.strip():
            return False, "Service name is required"
        
        if not request.description.strip():
            return False, "Service description is required"
        
        if not request.owner.strip():
            return False, "Service owner is required"
        
        # Check naming convention
        naming_pattern = self.config.compliance_rules.get("naming_convention", {}).get("pattern")
        if naming_pattern and not re.match(naming_pattern, request.service_name):
            return False, f"Service name doesn't match naming convention: {naming_pattern}"
        
        # Check if code path exists
        if not Path(request.code_path).exists():
            return False, f"Code path does not exist: {request.code_path}"
        
        return True, "Validation passed"
    
    def _service_exists(self, service_name: str) -> bool:
        """Check if a service with the given name already exists."""
        for service in self.registry.services.values():
            if service.name == service_name:
                return True
        return False
    
    def _analyze_service_code(self, code_path: str) -> Dict[str, Any]:
        """Analyze service code to extract metadata."""
        analysis = {}
        
        try:
            path = Path(code_path)
            
            # Detect runtime requirements
            if (path / "requirements.txt").exists():
                with open(path / "requirements.txt", 'r') as f:
                    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    analysis["runtime_requirements"] = {"python": requirements}
            
            if (path / "package.json").exists():
                with open(path / "package.json", 'r') as f:
                    package_data = json.load(f)
                    analysis["runtime_requirements"] = {"node": package_data.get("dependencies", {})}
            
            # Detect database usage
            analysis["database_type"] = self._detect_database_type(code_path)
            
            # Count lines of code
            analysis["lines_of_code"] = self._count_lines_of_code(code_path)
            
        except Exception as e:
            self.logger.warning(f"Code analysis failed: {e}")
        
        return analysis
    
    def _discover_api_endpoints(self, code_path: str, api_base: str) -> List[ServiceEndpoint]:
        """Discover API endpoints from service code."""
        endpoints = []
        
        try:
            # This is a simplified implementation
            # In practice, you'd parse the actual code files to extract endpoints
            path = Path(code_path)
            
            # Look for common API framework patterns
            python_files = list(path.rglob("*.py"))
            
            for py_file in python_files:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Simple regex patterns for Flask/FastAPI routes
                    flask_routes = re.findall(r'@app\.route\(["\']([^"\']+)["\'](?:,\s*methods=\[([^\]]+)\])?', content)
                    fastapi_routes = re.findall(r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']', content)
                    
                    for route, methods in flask_routes:
                        methods_list = ["GET"] if not methods else [m.strip(' "\'') for m in methods.split(',')]
                        for method in methods_list:
                            endpoints.append(ServiceEndpoint(
                                path=route,
                                method=method.upper(),
                                description=f"Auto-discovered {method.upper()} endpoint"
                            ))
                    
                    for method, route in fastapi_routes:
                        endpoints.append(ServiceEndpoint(
                            path=route,
                            method=method.upper(),
                            description=f"Auto-discovered {method.upper()} endpoint"
                        ))
        
        except Exception as e:
            self.logger.warning(f"Endpoint discovery failed: {e}")
        
        return endpoints[:20]  # Limit to first 20 endpoints
    
    def _analyze_dependencies(self, code_path: str) -> List[ServiceDependency]:
        """Analyze service dependencies."""
        dependencies = []
        
        try:
            path = Path(code_path)
            
            # Analyze Python dependencies
            if (path / "requirements.txt").exists():
                with open(path / "requirements.txt", 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse dependency line
                            if '==' in line:
                                name, version = line.split('==', 1)
                            elif '>=' in line:
                                name, version = line.split('>=', 1)
                                version = f">={version}"
                            else:
                                name, version = line, "*"
                            
                            dependencies.append(ServiceDependency(
                                service_id="external",
                                service_name=name.strip(),
                                version_requirement=version.strip(),
                                dependency_type="required",
                                relationship="uses"
                            ))
        
        except Exception as e:
            self.logger.warning(f"Dependency analysis failed: {e}")
        
        return dependencies
    
    def _generate_documentation(self, request: RegistrationRequest, analysis: Dict[str, Any]) -> List[ServiceDocumentation]:
        """Generate initial documentation for the service."""
        documentation = []
        
        try:
            # Generate README documentation
            readme_doc = ServiceDocumentation(
                doc_type=DocumentationType.USER_GUIDE,
                title=f"{request.service_name} - User Guide",
                file_path=f"{request.service_name}_user_guide.md",
                auto_generated=True,
                review_required=True
            )
            documentation.append(readme_doc)
            
            # Generate API documentation if endpoints exist
            if request.api_endpoint:
                api_doc = ServiceDocumentation(
                    doc_type=DocumentationType.API_DOCS,
                    title=f"{request.service_name} - API Documentation",
                    file_path=f"{request.service_name}_api_docs.md",
                    auto_generated=True,
                    review_required=True
                )
                documentation.append(api_doc)
        
        except Exception as e:
            self.logger.warning(f"Documentation generation failed: {e}")
        
        return documentation
    
    def _assess_service_quality(self, service: ServiceMetadata, code_path: str) -> Dict[str, Any]:
        """Perform comprehensive quality assessment."""
        assessment = {
            "metrics": [],
            "level": QualityLevel.DEVELOPMENT,
            "test_coverage": None,
            "code_quality_score": None,
            "security_score": None
        }
        
        try:
            path = Path(code_path)
            
            # Mock quality metrics (in practice, integrate with actual tools)
            # Test coverage assessment
            test_coverage = self._assess_test_coverage(path)
            assessment["test_coverage"] = test_coverage
            assessment["metrics"].append(QualityMetric(
                metric_name="test_coverage",
                value=test_coverage,
                unit="percentage",
                threshold=self.config.quality_thresholds.get("test_coverage", 70.0),
                status="pass" if test_coverage >= self.config.quality_thresholds.get("test_coverage", 70.0) else "fail"
            ))
            
            # Code quality assessment
            code_quality = self._assess_code_quality(path)
            assessment["code_quality_score"] = code_quality
            assessment["metrics"].append(QualityMetric(
                metric_name="code_quality",
                value=code_quality,
                unit="score",
                threshold=self.config.quality_thresholds.get("code_quality", 7.0),
                status="pass" if code_quality >= self.config.quality_thresholds.get("code_quality", 7.0) else "fail"
            ))
            
            # Security assessment
            security_score = self._assess_security(path)
            assessment["security_score"] = security_score
            assessment["metrics"].append(QualityMetric(
                metric_name="security",
                value=security_score,
                unit="score",
                threshold=self.config.quality_thresholds.get("security_score", 8.0),
                status="pass" if security_score >= self.config.quality_thresholds.get("security_score", 8.0) else "fail"
            ))
            
            # Determine overall quality level
            overall_score = (test_coverage/10 + code_quality + security_score) / 3
            
            if overall_score >= 8.5:
                assessment["level"] = QualityLevel.ENTERPRISE
            elif overall_score >= 7.0:
                assessment["level"] = QualityLevel.PRODUCTION
            elif overall_score >= 5.0:
                assessment["level"] = QualityLevel.DEVELOPMENT
            else:
                assessment["level"] = QualityLevel.EXPERIMENTAL
        
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
        
        return assessment
    
    def _check_compliance(self, service: ServiceMetadata, code_path: str) -> Dict[str, Any]:
        """Check service compliance with governance rules."""
        result = {
            "status": ComplianceStatus.COMPLIANT,
            "notes": []
        }
        
        try:
            path = Path(code_path)
            
            # Check required files
            required_files = self.config.compliance_rules.get("required_files", [])
            for required_file in required_files:
                if not (path / required_file).exists():
                    result["notes"].append(f"Missing required file: {required_file}")
                    result["status"] = ComplianceStatus.NON_COMPLIANT
            
            # Check security requirements
            security_reqs = self.config.compliance_rules.get("security_requirements", {})
            if security_reqs.get("no_hardcoded_secrets") and self._has_hardcoded_secrets(path):
                result["notes"].append("Hardcoded secrets detected")
                result["status"] = ComplianceStatus.NON_COMPLIANT
            
            # Check documentation requirements
            doc_reqs = self.config.compliance_rules.get("documentation_requirements", {})
            if doc_reqs.get("api_documentation") and not service.api_endpoints:
                result["notes"].append("API documentation required but no endpoints found")
                if result["status"] == ComplianceStatus.COMPLIANT:
                    result["status"] = ComplianceStatus.REVIEW_REQUIRED
        
        except Exception as e:
            self.logger.warning(f"Compliance check failed: {e}")
            result["status"] = ComplianceStatus.REVIEW_REQUIRED
            result["notes"].append(f"Compliance check error: {str(e)}")
        
        return result
    
    def _generate_service_embeddings(self, service: ServiceMetadata) -> Dict[str, List[float]]:
        """Generate vector embeddings for service content."""
        # Mock implementation - in practice, integrate with actual embedding service
        import hashlib
        
        # Generate mock embeddings based on service content
        description_hash = hashlib.md5(service.description.encode()).hexdigest()
        capability_text = " ".join(service.tags + service.categories)
        capability_hash = hashlib.md5(capability_text.encode()).hexdigest()
        
        # Convert hash to mock embedding vector
        description_embedding = [float(int(c, 16)) / 15.0 for c in description_hash[:128]]
        capability_embedding = [float(int(c, 16)) / 15.0 for c in capability_hash[:128]]
        
        return {
            "description": description_embedding,
            "capability": capability_embedding
        }
    
    def _apply_quality_gates(self, service: ServiceMetadata) -> Tuple[bool, str]:
        """Apply quality gates to determine if service passes registration."""
        if not self.config.enable_quality_gates:
            return True, "Quality gates disabled"
        
        failures = []
        
        # Check minimum test coverage
        if service.test_coverage is not None and service.test_coverage < self.config.minimum_test_coverage:
            failures.append(f"Test coverage {service.test_coverage}% below minimum {self.config.minimum_test_coverage}%")
        
        # Check minimum code quality score
        if service.code_quality_score is not None and service.code_quality_score < self.config.minimum_code_quality_score:
            failures.append(f"Code quality score {service.code_quality_score} below minimum {self.config.minimum_code_quality_score}")
        
        # Check compliance status
        if service.compliance_status == ComplianceStatus.NON_COMPLIANT:
            failures.append("Service is not compliant with governance rules")
        
        if failures:
            return False, "; ".join(failures)
        
        return True, "All quality gates passed"
    
    def _update_registry_indexes(self, service: ServiceMetadata):
        """Update registry indexes for faster searching."""
        # Update categories index
        for category in service.categories:
            if category not in self.registry.categories:
                self.registry.categories[category] = []
            if service.service_id not in self.registry.categories[category]:
                self.registry.categories[category].append(service.service_id)
        
        # Update tags index
        for tag in service.tags:
            if tag not in self.registry.tags:
                self.registry.tags[tag] = []
            if service.service_id not in self.registry.tags[tag]:
                self.registry.tags[tag].append(service.service_id)
        
        # Update dependency graph
        service_deps = [dep.service_id for dep in service.dependencies if dep.service_id != "external"]
        self.registry.dependency_graph[service.service_id] = service_deps
    
    def _update_registrar_metrics(self):
        """Update registrar performance metrics."""
        total_services = len(self.registry.services)
        self.registrar_metrics["total_services"] = total_services
        
        # Services by type
        type_counts = {}
        quality_counts = {}
        compliant_count = 0
        documented_count = 0
        quality_scores = []
        
        for service in self.registry.services.values():
            # Type distribution
            service_type = service.service_type.value
            type_counts[service_type] = type_counts.get(service_type, 0) + 1
            
            # Quality distribution
            quality_level = service.quality_level.value
            quality_counts[quality_level] = quality_counts.get(quality_level, 0) + 1
            
            # Compliance rate
            if service.compliance_status == ComplianceStatus.COMPLIANT:
                compliant_count += 1
            
            # Documentation coverage
            if service.documentation:
                documented_count += 1
            
            # Quality scores
            if service.code_quality_score is not None:
                quality_scores.append(service.code_quality_score)
        
        self.registrar_metrics.update({
            "services_by_type": type_counts,
            "services_by_quality": quality_counts,
            "compliance_rate": (compliant_count / total_services * 100) if total_services > 0 else 0,
            "documentation_coverage": (documented_count / total_services * 100) if total_services > 0 else 0,
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0
        })
    
    # Additional helper methods for quality assessment, search, analytics, etc.
    
    def _assess_test_coverage(self, code_path: Path) -> float:
        """Assess test coverage (mock implementation)."""
        # Mock test coverage calculation
        test_files = list(code_path.rglob("test_*.py")) + list(code_path.rglob("*_test.py"))
        source_files = list(code_path.rglob("*.py"))
        
        if not source_files:
            return 0.0
        
        # Simple heuristic: coverage based on test file ratio
        coverage = min(100.0, (len(test_files) / len(source_files)) * 100 * 1.5)
        return round(coverage, 1)
    
    def _assess_code_quality(self, code_path: Path) -> float:
        """Assess code quality (mock implementation)."""
        # Mock code quality assessment
        python_files = list(code_path.rglob("*.py"))
        
        if not python_files:
            return 5.0
        
        # Simple heuristic based on file structure and naming
        quality_score = 6.0
        
        # Bonus for good structure
        if (code_path / "src").exists() or (code_path / "lib").exists():
            quality_score += 0.5
        
        # Bonus for documentation
        if (code_path / "docs").exists():
            quality_score += 0.5
        
        # Bonus for configuration files
        if (code_path / "setup.py").exists() or (code_path / "pyproject.toml").exists():
            quality_score += 0.5
        
        return min(10.0, quality_score)
    
    def _assess_security(self, code_path: Path) -> float:
        """Assess security (mock implementation)."""
        # Mock security assessment
        security_score = 7.0
        
        # Check for common security files
        if (code_path / ".gitignore").exists():
            security_score += 0.5
        
        if not self._has_hardcoded_secrets(code_path):
            security_score += 1.0
        else:
            security_score -= 2.0
        
        return max(0.0, min(10.0, security_score))
    
    def _has_hardcoded_secrets(self, code_path: Path) -> bool:
        """Check for hardcoded secrets (simplified)."""
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        python_files = list(code_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            return True
            except:
                continue
        
        return False
    
    def _detect_database_type(self, code_path: str) -> Optional[str]:
        """Detect database type used by service."""
        path = Path(code_path)
        
        # Check requirements for database libraries
        if (path / "requirements.txt").exists():
            with open(path / "requirements.txt", 'r') as f:
                requirements = f.read().lower()
                
                if "postgresql" in requirements or "psycopg" in requirements:
                    return "postgresql"
                elif "mysql" in requirements or "pymysql" in requirements:
                    return "mysql"
                elif "redis" in requirements:
                    return "redis"
                elif "mongodb" in requirements or "pymongo" in requirements:
                    return "mongodb"
                elif "sqlite" in requirements:
                    return "sqlite"
        
        return None
    
    def _count_lines_of_code(self, code_path: str) -> int:
        """Count lines of code."""
        path = Path(code_path)
        total_lines = 0
        
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += sum(1 for line in f if line.strip())
            except:
                continue
        
        return total_lines
    
    def _filter_services(self, query: ServiceSearchQuery) -> Dict[str, ServiceMetadata]:
        """Filter services based on search criteria."""
        filtered = {}
        
        for service_id, service in self.registry.services.items():
            # Skip non-registered services
            if service.registration_status != ServiceRegistrationStatus.REGISTERED:
                continue
            
            # Filter by service type
            if query.service_type and service.service_type != query.service_type:
                continue
            
            # Filter by quality level
            if query.quality_level and service.quality_level != query.quality_level:
                continue
            
            # Filter by programming language
            if query.programming_language and service.programming_language.lower() != query.programming_language.lower():
                continue
            
            # Filter by API availability
            if query.has_api is not None:
                has_api = bool(service.api_endpoints)
                if has_api != query.has_api:
                    continue
            
            # Filter by minimum quality score
            if query.min_quality_score and service.code_quality_score:
                if service.code_quality_score < query.min_quality_score:
                    continue
            
            # Filter by categories
            if query.categories:
                if not any(cat in service.categories for cat in query.categories):
                    continue
            
            # Filter by tags
            if query.tags:
                if not any(tag in service.tags for tag in query.tags):
                    continue
            
            filtered[service_id] = service
        
        return filtered
    
    def _calculate_relevance_score(self, service: ServiceMetadata, query: ServiceSearchQuery) -> float:
        """Calculate relevance score for search results."""
        score = 0.0
        
        if query.query_text:
            query_terms = query.query_text.lower().split()
            
            # Check name relevance
            name_matches = sum(1 for term in query_terms if term in service.name.lower())
            score += name_matches * 3.0
            
            # Check description relevance
            desc_matches = sum(1 for term in query_terms if term in service.description.lower())
            score += desc_matches * 2.0
            
            # Check tag relevance
            tag_matches = sum(1 for term in query_terms if any(term in tag.lower() for tag in service.tags))
            score += tag_matches * 1.5
        
        # Boost by quality score
        if service.code_quality_score:
            score += service.code_quality_score * 0.5
        
        # Boost by popularity
        score += service.popularity_score * 0.3
        
        # Boost recent updates
        days_since_update = (datetime.now(timezone.utc) - service.last_updated).days
        if days_since_update < 30:
            score += 1.0
        elif days_since_update < 90:
            score += 0.5
        
        return score
    
    def _sort_search_results(self, results: List[ServiceSearchResult], sort_by: str) -> List[ServiceSearchResult]:
        """Sort search results by specified criteria."""
        if sort_by == "relevance":
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == "popularity":
            return sorted(results, key=lambda x: x.popularity_score, reverse=True)
        elif sort_by == "created_date":
            return sorted(results, key=lambda x: x.last_updated, reverse=True)
        elif sort_by == "updated_date":
            return sorted(results, key=lambda x: x.last_updated, reverse=True)
        else:
            return results
    
    def _update_search_analytics(self, query: ServiceSearchQuery, result_count: int):
        """Update search analytics."""
        analytics_key = f"search_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        
        if analytics_key not in self.usage_analytics:
            self.usage_analytics[analytics_key] = {
                "total_searches": 0,
                "query_types": {},
                "result_counts": []
            }
        
        analytics = self.usage_analytics[analytics_key]
        analytics["total_searches"] += 1
        analytics["result_counts"].append(result_count)
        
        if query.service_type:
            type_key = query.service_type.value
            analytics["query_types"][type_key] = analytics["query_types"].get(type_key, 0) + 1
    
    def _update_service_popularity(self, service_id: str):
        """Update service popularity score based on usage."""
        service = self.registry.services.get(service_id)
        if service:
            # Simple popularity calculation based on usage count and recency
            base_score = min(10.0, service.usage_count * 0.1)
            
            # Boost for recent access
            if service.last_accessed:
                days_since_access = (datetime.now(timezone.utc) - service.last_accessed).days
                recency_boost = max(0, 1.0 - (days_since_access / 30.0))
                base_score += recency_boost
            
            service.popularity_score = base_score
    
    def _get_service_code_path(self, service: ServiceMetadata) -> str:
        """Get the code path for a service (simplified)."""
        # In practice, this would be stored in service metadata
        return f"./services/{service.name}"
    
    def _has_recent_quality_assessment(self, service_id: str) -> bool:
        """Check if service has recent quality assessment."""
        if service_id not in self.quality_assessments:
            return False
        
        latest_assessment = self.quality_assessments[service_id][-1]
        assessment_date = datetime.fromisoformat(latest_assessment["assessment_date"])
        
        # Consider assessment recent if within 30 days
        return (datetime.now(timezone.utc) - assessment_date).days < 30
    
    def _record_lifecycle_event(self, service_id: str, old_status: str, new_status: str, reason: Optional[str]):
        """Record a lifecycle event for tracking."""
        if service_id not in self.lifecycle_events:
            self.lifecycle_events[service_id] = []
        
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_status": old_status,
            "new_status": new_status,
            "reason": reason,
            "event_type": "status_change"
        }
        
        self.lifecycle_events[service_id].append(event)
    
    def _deprecate_service(self, service_id: str, reason: Optional[str] = None) -> bool:
        """Deprecate a service."""
        service = self.registry.services.get(service_id)
        if not service:
            return False
        
        service.registration_status = ServiceRegistrationStatus.DEPRECATED
        service.deprecation_date = datetime.now(timezone.utc)
        service.quality_level = QualityLevel.DEPRECATED
        service.last_updated = datetime.now(timezone.utc)
        
        # Add deprecation notice to compliance notes
        deprecation_note = f"Service deprecated on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        if reason:
            deprecation_note += f": {reason}"
        service.compliance_notes.append(deprecation_note)
        
        self._record_lifecycle_event(service_id, "active", "deprecated", reason)
        self._save_registry()
        
        return True
    
    def _retire_service(self, service_id: str, reason: Optional[str] = None) -> bool:
        """Retire a service."""
        service = self.registry.services.get(service_id)
        if not service:
            return False
        
        service.registration_status = ServiceRegistrationStatus.ARCHIVED
        service.retirement_date = datetime.now(timezone.utc)
        service.last_updated = datetime.now(timezone.utc)
        
        retirement_note = f"Service retired on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        if reason:
            retirement_note += f": {reason}"
        service.compliance_notes.append(retirement_note)
        
        self._record_lifecycle_event(service_id, service.registration_status.value, "retired", reason)
        self._save_registry()
        
        return True
    
    def _archive_service(self, service_id: str, reason: Optional[str] = None) -> bool:
        """Archive a service."""
        service = self.registry.services.get(service_id)
        if not service:
            return False
        
        service.registration_status = ServiceRegistrationStatus.ARCHIVED
        service.last_updated = datetime.now(timezone.utc)
        
        archive_note = f"Service archived on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        if reason:
            archive_note += f": {reason}"
        service.compliance_notes.append(archive_note)
        
        self._record_lifecycle_event(service_id, "active", "archived", reason)
        self._save_registry()
        
        return True
    
    def _reactivate_service(self, service_id: str, reason: Optional[str] = None) -> bool:
        """Reactivate a deprecated or archived service."""
        service = self.registry.services.get(service_id)
        if not service:
            return False
        
        old_status = service.registration_status.value
        service.registration_status = ServiceRegistrationStatus.REGISTERED
        service.deprecation_date = None
        service.retirement_date = None
        service.last_updated = datetime.now(timezone.utc)
        
        # Reset quality level if it was deprecated
        if service.quality_level == QualityLevel.DEPRECATED:
            service.quality_level = QualityLevel.DEVELOPMENT
        
        reactivation_note = f"Service reactivated on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        if reason:
            reactivation_note += f": {reason}"
        service.compliance_notes.append(reactivation_note)
        
        self._record_lifecycle_event(service_id, old_status, "reactivated", reason)
        self._save_registry()
        
        return True
    
    def _generate_specific_documentation(self, service: ServiceMetadata, code_path: str, 
                                       doc_type: DocumentationType) -> Optional[ServiceDocumentation]:
        """Generate specific type of documentation."""
        try:
            doc_content = ""
            file_name = ""
            
            if doc_type == DocumentationType.API_DOCS:
                file_name = f"{service.name}_api_docs.md"
                doc_content = self._generate_api_documentation(service)
            elif doc_type == DocumentationType.USER_GUIDE:
                file_name = f"{service.name}_user_guide.md"
                doc_content = self._generate_user_guide(service)
            elif doc_type == DocumentationType.INTEGRATION_GUIDE:
                file_name = f"{service.name}_integration_guide.md"
                doc_content = self._generate_integration_guide(service)
            elif doc_type == DocumentationType.DEPLOYMENT:
                file_name = f"{service.name}_deployment.md"
                doc_content = self._generate_deployment_guide(service)
            else:
                return None
            
            # Save documentation to file
            doc_path = Path(self.config.documentation_base_path) / service.name / file_name
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            return ServiceDocumentation(
                doc_type=doc_type,
                title=f"{service.name} - {doc_type.value.replace('_', ' ').title()}",
                file_path=str(doc_path),
                auto_generated=True,
                review_required=True
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to generate {doc_type.value} documentation: {e}")
            return None
    
    def _generate_api_documentation(self, service: ServiceMetadata) -> str:
        """Generate API documentation."""
        doc = f"# {service.name} API Documentation\n\n"
        doc += f"{service.description}\n\n"
        
        if service.api_endpoints:
            doc += "## Endpoints\n\n"
            for endpoint in service.api_endpoints:
                doc += f"### {endpoint.method} {endpoint.path}\n\n"
                doc += f"{endpoint.description}\n\n"
                
                if endpoint.parameters:
                    doc += "**Parameters:**\n"
                    for param, details in endpoint.parameters.items():
                        doc += f"- `{param}`: {details}\n"
                    doc += "\n"
                
                if endpoint.authentication_required:
                    doc += "**Authentication:** Required\n\n"
        
        return doc
    
    def _generate_user_guide(self, service: ServiceMetadata) -> str:
        """Generate user guide documentation."""
        doc = f"# {service.name} User Guide\n\n"
        doc += f"{service.description}\n\n"
        
        doc += "## Installation\n\n"
        doc += "```bash\n"
        doc += f"# Install {service.name}\n"
        if service.programming_language.lower() == "python":
            doc += f"pip install {service.name}\n"
        elif service.programming_language.lower() == "javascript":
            doc += f"npm install {service.name}\n"
        doc += "```\n\n"
        
        doc += "## Basic Usage\n\n"
        doc += f"```{service.programming_language.lower()}\n"
        doc += f"# Example usage of {service.name}\n"
        doc += "# TODO: Add actual usage examples\n"
        doc += "```\n\n"
        
        return doc
    
    def _generate_integration_guide(self, service: ServiceMetadata) -> str:
        """Generate integration guide documentation."""
        doc = f"# {service.name} Integration Guide\n\n"
        doc += f"This guide explains how to integrate {service.name} into your system.\n\n"
        
        if service.dependencies:
            doc += "## Dependencies\n\n"
            for dep in service.dependencies:
                doc += f"- {dep.service_name} ({dep.version_requirement})\n"
            doc += "\n"
        
        if service.api_endpoints:
            doc += "## API Integration\n\n"
            doc += f"Base URL: `{service.api_endpoints[0].path if service.api_endpoints else 'TBD'}`\n\n"
        
        return doc
    
    def _generate_deployment_guide(self, service: ServiceMetadata) -> str:
        """Generate deployment guide documentation."""
        doc = f"# {service.name} Deployment Guide\n\n"
        doc += f"Instructions for deploying {service.name}.\n\n"
        
        doc += "## Requirements\n\n"
        if service.runtime_requirements:
            for runtime, reqs in service.runtime_requirements.items():
                doc += f"### {runtime.title()}\n"
                if isinstance(reqs, list):
                    for req in reqs:
                        doc += f"- {req}\n"
                else:
                    doc += f"- {reqs}\n"
                doc += "\n"
        
        doc += "## Deployment Steps\n\n"
        doc += "1. Clone the repository\n"
        doc += "2. Install dependencies\n"
        doc += "3. Configure environment variables\n"
        doc += "4. Start the service\n\n"
        
        return doc
    
    def _build_service_dependency_graph(self, service_id: str) -> Dict[str, Any]:
        """Build dependency graph for a specific service."""
        service = self.registry.services.get(service_id)
        if not service:
            return {"error": "Service not found"}
        
        graph = {
            "service_id": service_id,
            "service_name": service.name,
            "dependencies": [],
            "dependents": []
        }
        
        # Add direct dependencies
        for dep in service.dependencies:
            if dep.service_id != "external":
                dep_service = self.registry.services.get(dep.service_id)
                if dep_service:
                    graph["dependencies"].append({
                        "service_id": dep.service_id,
                        "service_name": dep_service.name,
                        "relationship": dep.relationship,
                        "version_requirement": dep.version_requirement
                    })
        
        # Find services that depend on this one
        for sid, svc in self.registry.services.items():
            for dep in svc.dependencies:
                if dep.service_id == service_id:
                    graph["dependents"].append({
                        "service_id": sid,
                        "service_name": svc.name,
                        "relationship": dep.relationship
                    })
        
        return graph
    
    def _build_ecosystem_dependency_graph(self) -> Dict[str, Any]:
        """Build full ecosystem dependency graph."""
        graph = {
            "nodes": [],
            "edges": [],
            "statistics": {
                "total_services": len(self.registry.services),
                "total_dependencies": 0
            }
        }
        
        # Add all services as nodes
        for service_id, service in self.registry.services.items():
            graph["nodes"].append({
                "id": service_id,
                "name": service.name,
                "type": service.service_type.value,
                "status": service.registration_status.value,
                "quality_level": service.quality_level.value
            })
        
        # Add dependencies as edges
        for service_id, service in self.registry.services.items():
            for dep in service.dependencies:
                if dep.service_id != "external" and dep.service_id in self.registry.services:
                    graph["edges"].append({
                        "source": dep.service_id,
                        "target": service_id,
                        "relationship": dep.relationship,
                        "type": dep.dependency_type
                    })
                    graph["statistics"]["total_dependencies"] += 1
        
        return graph
    
    def _generate_ecosystem_summary(self) -> Dict[str, Any]:
        """Generate high-level ecosystem summary."""
        total_services = len(self.registry.services)
        
        if total_services == 0:
            return {"total_services": 0, "message": "No services registered"}
        
        # Count by status
        status_counts = {}
        type_counts = {}
        quality_counts = {}
        
        for service in self.registry.services.values():
            status = service.registration_status.value
            service_type = service.service_type.value
            quality = service.quality_level.value
            
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[service_type] = type_counts.get(service_type, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        return {
            "total_services": total_services,
            "services_by_status": status_counts,
            "services_by_type": type_counts,
            "services_by_quality": quality_counts,
            "last_updated": self.registry.last_updated.isoformat()
        }
    
    def _generate_quality_report(self) -> Dict[str, Any]:
        """Generate quality assessment report."""
        quality_scores = []
        test_coverages = []
        security_scores = []
        compliant_services = 0
        
        for service in self.registry.services.values():
            if service.code_quality_score is not None:
                quality_scores.append(service.code_quality_score)
            
            if service.test_coverage is not None:
                test_coverages.append(service.test_coverage)
            
            if service.security_score is not None:
                security_scores.append(service.security_score)
            
            if service.compliance_status == ComplianceStatus.COMPLIANT:
                compliant_services += 1
        
        return {
            "average_code_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "average_test_coverage": sum(test_coverages) / len(test_coverages) if test_coverages else 0,
            "average_security_score": sum(security_scores) / len(security_scores) if security_scores else 0,
            "compliance_rate": (compliant_services / len(self.registry.services) * 100) if self.registry.services else 0,
            "quality_distribution": {level.value: 0 for level in QualityLevel}
        }
    
    def _generate_usage_report(self) -> Dict[str, Any]:
        """Generate usage analytics report."""
        total_usage = sum(service.usage_count for service in self.registry.services.values())
        most_popular = max(self.registry.services.values(), 
                          key=lambda s: s.popularity_score, 
                          default=None)
        
        # Recent access patterns
        recent_accesses = []
        for service in self.registry.services.values():
            if service.last_accessed:
                days_ago = (datetime.now(timezone.utc) - service.last_accessed).days
                if days_ago <= 30:
                    recent_accesses.append(service.name)
        
        return {
            "total_service_usage": total_usage,
            "most_popular_service": most_popular.name if most_popular else None,
            "services_accessed_recently": len(recent_accesses),
            "search_analytics": self.usage_analytics
        }
    
    def _generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance assessment report."""
        compliance_counts = {status.value: 0 for status in ComplianceStatus}
        
        for service in self.registry.services.values():
            compliance_counts[service.compliance_status.value] += 1
        
        total_services = len(self.registry.services)
        compliance_rate = (compliance_counts["compliant"] / total_services * 100) if total_services > 0 else 0
        
        return {
            "compliance_distribution": compliance_counts,
            "overall_compliance_rate": compliance_rate,
            "services_needing_review": compliance_counts["review_required"],
            "non_compliant_services": compliance_counts["non_compliant"]
        }
    
    def _generate_lifecycle_report(self) -> Dict[str, Any]:
        """Generate service lifecycle analysis report."""
        now = datetime.now(timezone.utc)
        deprecated_services = []
        soon_to_deprecate = []
        
        for service in self.registry.services.values():
            if service.registration_status == ServiceRegistrationStatus.DEPRECATED:
                deprecated_services.append({
                    "name": service.name,
                    "deprecation_date": service.deprecation_date.isoformat() if service.deprecation_date else None
                })
            
            # Services that haven't been updated in a while might need attention
            if service.last_updated:
                days_since_update = (now - service.last_updated).days
                if days_since_update > 365:  # Over a year
                    soon_to_deprecate.append({
                        "name": service.name,
                        "days_since_update": days_since_update
                    })
        
        return {
            "deprecated_services": deprecated_services,
            "services_needing_attention": soon_to_deprecate,
            "lifecycle_events_summary": len(self.lifecycle_events),
            "average_service_age_days": self._calculate_average_service_age()
        }
    
    def _calculate_average_service_age(self) -> float:
        """Calculate average age of services in days."""
        if not self.registry.services:
            return 0.0
        
        now = datetime.now(timezone.utc)
        ages = []
        
        for service in self.registry.services.values():
            age_days = (now - service.created_date).days
            ages.append(age_days)
        
        return sum(ages) / len(ages)


# Factory function for easy agent creation
def create_registrar_agent(config: Optional[RegistrarConfiguration] = None) -> RegistrarAgent:
    """
    Create a new Registrar Agent instance.
    
    Args:
        config: Optional registrar configuration
        
    Returns:
        Configured RegistrarAgent instance
    """
    return RegistrarAgent(config)


# Export main classes and functions
__all__ = [
    "RegistrarAgent",
    "RegistrarConfiguration",
    "ServiceMetadata",
    "RegistrationRequest",
    "ServiceSearchQuery",
    "ServiceSearchResult",
    "ServiceRegistrationStatus",
    "ServiceType",
    "QualityLevel",
    "DocumentationType",
    "create_registrar_agent"
]