"""
Router Agent for the AI-Galaxy Lower Meta-Layer.

This module implements the Router Agent, the intelligent navigation system of the
AI-Galaxy ecosystem. The Router Agent uses semantic analysis to route ideas to the
most appropriate departments and institutions, ensuring optimal resource allocation
and specialized handling of diverse project requirements.
"""

import json
import re
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field

from ...shared.models import Idea, IdeaStatus, Department, Institution
from ...shared.logger import get_logger, LogContext
from ..state_machine.router import StateMachineRouter, TransitionResult


class RoutingConfidence(str, Enum):
    """Confidence levels for routing decisions."""
    VERY_HIGH = "very_high"  # 0.9-1.0
    HIGH = "high"           # 0.8-0.89
    MEDIUM = "medium"       # 0.6-0.79
    LOW = "low"             # 0.4-0.59
    VERY_LOW = "very_low"   # 0.0-0.39


class RoutingPriority(str, Enum):
    """Priority levels for routing decisions."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class AnalysisType(str, Enum):
    """Types of semantic analysis."""
    KEYWORD_EXTRACTION = "keyword_extraction"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    DOMAIN_CLASSIFICATION = "domain_classification"
    CONTEXT_UNDERSTANDING = "context_understanding"
    REQUIREMENT_ANALYSIS = "requirement_analysis"


class DepartmentCapability(str, Enum):
    """Core capabilities of different departments."""
    MACHINE_LEARNING = "machine_learning"
    DATA_ENGINEERING = "data_engineering"
    WEB_DEVELOPMENT = "web_development"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    RESEARCH = "research"
    INTEGRATION = "integration"
    ANALYTICS = "analytics"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    COMPUTER_VISION = "computer_vision"
    BACKEND_DEVELOPMENT = "backend_development"
    FRONTEND_DEVELOPMENT = "frontend_development"
    DEVOPS = "devops"
    QUALITY_ASSURANCE = "quality_assurance"


@dataclass
class SemanticKeyword:
    """Represents a semantically extracted keyword with metadata."""
    term: str
    frequency: int
    importance_score: float
    domain_relevance: Dict[str, float] = field(default_factory=dict)
    context: List[str] = field(default_factory=list)


@dataclass
class DomainSignature:
    """Domain-specific signature for classification."""
    domain: str
    keywords: List[str]
    weight_multipliers: Dict[str, float] = field(default_factory=dict)
    capability_mapping: List[DepartmentCapability] = field(default_factory=list)
    exclusion_patterns: List[str] = field(default_factory=list)


@dataclass
class InstitutionProfile:
    """Profile of an institution with capabilities and metrics."""
    institution_id: str
    name: str
    department_id: str
    capabilities: List[DepartmentCapability]
    technology_stack: List[str]
    expertise_areas: List[str]
    current_workload: int
    max_capacity: int
    success_rate: float
    average_completion_time: float
    recent_projects: List[str] = field(default_factory=list)


@dataclass
class RoutingContext:
    """Context information for routing decisions."""
    idea_id: str
    semantic_keywords: List[SemanticKeyword]
    domain_scores: Dict[str, float]
    requirement_complexity: str
    integration_needs: List[str]
    performance_requirements: List[str]
    timeline_constraints: Optional[str] = None
    resource_constraints: Optional[str] = None


class RoutingDecision(BaseModel):
    """Comprehensive routing decision with reasoning and alternatives."""
    idea_id: str
    primary_department: str
    primary_institution: Optional[str] = None
    confidence_level: RoutingConfidence
    confidence_score: float = Field(ge=0.0, le=1.0)
    priority: RoutingPriority
    reasoning: str
    
    # Alternative options
    alternative_departments: List[str] = Field(default_factory=list)
    alternative_institutions: List[str] = Field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    semantic_keywords: List[str] = Field(default_factory=list)
    domain_classification: Dict[str, float] = Field(default_factory=dict)
    
    # Recommendations
    new_institution_needed: bool = False
    proposed_institution_name: Optional[str] = None
    proposed_institution_capabilities: List[str] = Field(default_factory=list)
    cross_department_collaboration: List[str] = Field(default_factory=list)
    
    # Validation
    escalation_needed: bool = False
    escalation_reason: Optional[str] = None
    validation_notes: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class SimilarityMatch(BaseModel):
    """Represents a similarity match with past projects."""
    idea_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    matched_keywords: List[str] = Field(default_factory=list)
    previous_routing: RoutingDecision
    outcome_success: Optional[bool] = None
    lessons_learned: List[str] = Field(default_factory=list)


class DepartmentWorkload(BaseModel):
    """Current workload information for a department."""
    department_id: str
    total_capacity: int
    current_load: int
    utilization_percentage: float = Field(ge=0.0, le=100.0)
    pending_ideas: int
    active_projects: int
    average_completion_time: float
    recent_success_rate: float = Field(ge=0.0, le=1.0)
    
    # Institution breakdown
    institution_workloads: Dict[str, int] = Field(default_factory=dict)
    available_institutions: List[str] = Field(default_factory=list)
    overloaded_institutions: List[str] = Field(default_factory=list)


class RoutingMetrics(BaseModel):
    """Comprehensive metrics for routing performance."""
    total_routings: int = 0
    successful_routings: int = 0
    failed_routings: int = 0
    escalated_routings: int = 0
    
    # Accuracy metrics
    routing_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    department_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Performance metrics
    average_analysis_time: float = 0.0
    average_confidence_score: float = 0.0
    
    # Learning metrics
    improvement_rate: float = 0.0
    adaptation_cycles: int = 0
    
    # Feedback integration
    feedback_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    learning_iterations: int = 0


class RouterConfiguration(BaseModel):
    """Configuration for the Router Agent."""
    # Semantic analysis settings
    min_keyword_frequency: int = 2
    keyword_importance_threshold: float = 0.3
    semantic_similarity_threshold: float = 0.7
    
    # Routing settings
    confidence_threshold: float = 0.6
    require_high_confidence: bool = True
    enable_cross_department_routing: bool = True
    enable_new_institution_suggestions: bool = True
    
    # Learning settings
    enable_adaptive_learning: bool = True
    feedback_weight: float = 0.3
    historical_weight: float = 0.7
    min_samples_for_learning: int = 5
    
    # Performance settings
    max_analysis_time: float = 30.0  # seconds
    enable_async_processing: bool = True
    cache_semantic_analysis: bool = True
    
    # Validation settings
    require_escalation_review: bool = True
    confidence_escalation_threshold: float = 0.4
    workload_balance_weight: float = 0.2


class RouterAgent:
    """
    The Router Agent - intelligent navigation system of AI-Galaxy.
    
    Uses semantic analysis to route ideas to the most appropriate departments
    and institutions, while learning and adapting from experience to improve
    routing decisions over time.
    """
    
    def __init__(self, config: Optional[RouterConfiguration] = None,
                 state_router: Optional[StateMachineRouter] = None):
        """
        Initialize the Router Agent.
        
        Args:
            config: Router configuration parameters
            state_router: State machine router for workflow transitions
        """
        self.logger = get_logger("router_agent")
        self.config = config or RouterConfiguration()
        self.state_router = state_router
        
        # Initialize semantic analysis engine
        self.domain_signatures = self._initialize_domain_signatures()
        self.semantic_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize department and institution knowledge base
        self.department_profiles: Dict[str, Dict[str, Any]] = {}
        self.institution_profiles: Dict[str, InstitutionProfile] = {}
        self.department_workloads: Dict[str, DepartmentWorkload] = {}
        
        # Routing history and learning
        self.routing_history: Dict[str, RoutingDecision] = {}
        self.similarity_index: Dict[str, List[str]] = defaultdict(list)
        self.feedback_data: Dict[str, Any] = {}
        
        # Performance metrics
        self.routing_metrics = RoutingMetrics()
        
        # Vector search integration (placeholder for future integration)
        self.vector_search_enabled = False
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        self.logger.agent_action("router_agent_initialized", "router_agent",
                                additional_context={
                                    "domain_signatures": len(self.domain_signatures),
                                    "adaptive_learning": self.config.enable_adaptive_learning,
                                    "cross_department_routing": self.config.enable_cross_department_routing,
                                    "vector_search": self.vector_search_enabled
                                })
    
    async def route_idea(self, idea: Idea) -> RoutingDecision:
        """
        Route an idea to the most appropriate department and institution.
        
        Args:
            idea: The idea to route
            
        Returns:
            Comprehensive routing decision with reasoning
        """
        start_time = datetime.now(timezone.utc)
        idea_id = str(idea.id)
        
        context = LogContext(
            agent_name="router_agent",
            idea_id=idea_id,
            additional_context={"routing_analysis_start": start_time.isoformat()}
        )
        
        self.logger.agent_action("starting_idea_routing", "router_agent", idea_id)
        
        try:
            # Step 1: Semantic analysis
            semantic_analysis = await self._analyze_idea_semantics(idea)
            
            # Step 2: Domain classification
            domain_classification = self._classify_domain(idea, semantic_analysis)
            
            # Step 3: Find similar past projects
            similar_projects = await self._find_similar_projects(idea, semantic_analysis)
            
            # Step 4: Assess department workloads
            workload_assessment = self._assess_department_workloads()
            
            # Step 5: Generate routing decision
            routing_decision = self._generate_routing_decision(
                idea, semantic_analysis, domain_classification, 
                similar_projects, workload_assessment
            )
            
            # Step 6: Validate routing decision
            validated_decision = self._validate_routing_decision(routing_decision, idea)
            
            # Step 7: Store decision and update learning
            self._store_routing_decision(validated_decision)
            self._update_similarity_index(idea_id, semantic_analysis)
            
            # Update metrics
            analysis_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_routing_metrics(validated_decision, analysis_time, success=True)
            
            self.logger.agent_action("idea_routing_completed", "router_agent", idea_id, {
                "primary_department": validated_decision.primary_department,
                "primary_institution": validated_decision.primary_institution,
                "confidence_level": validated_decision.confidence_level.value,
                "confidence_score": validated_decision.confidence_score,
                "analysis_time": analysis_time,
                "escalation_needed": validated_decision.escalation_needed
            })
            
            return validated_decision
            
        except Exception as e:
            self.logger.error(f"Idea routing failed: {e}", context, exc_info=True)
            
            # Create fallback routing decision
            fallback_decision = self._create_fallback_routing(idea, start_time)
            self._update_routing_metrics(fallback_decision, 
                                       (datetime.now(timezone.utc) - start_time).total_seconds(), 
                                       success=False)
            return fallback_decision
    
    async def _analyze_idea_semantics(self, idea: Idea) -> RoutingContext:
        """
        Perform comprehensive semantic analysis of the idea.
        
        Args:
            idea: The idea to analyze
            
        Returns:
            Routing context with semantic analysis results
        """
        idea_id = str(idea.id)
        
        # Check cache first
        if self.config.cache_semantic_analysis and idea_id in self.semantic_cache:
            cached_result = self.semantic_cache[idea_id]
            return RoutingContext(**cached_result)
        
        # Combine text for analysis
        text = f"{idea.title} {idea.description}"
        
        # Step 1: Extract semantic keywords
        semantic_keywords = self._extract_semantic_keywords(text)
        
        # Step 2: Calculate domain scores
        domain_scores = self._calculate_domain_scores(text, semantic_keywords)
        
        # Step 3: Analyze requirement complexity
        requirement_complexity = self._analyze_requirement_complexity(text)
        
        # Step 4: Identify integration needs
        integration_needs = self._identify_integration_needs(text)
        
        # Step 5: Extract performance requirements
        performance_requirements = self._extract_performance_requirements(text)
        
        # Step 6: Assess timeline and resource constraints
        timeline_constraints = self._assess_timeline_constraints(idea, text)
        resource_constraints = self._assess_resource_constraints(text)
        
        routing_context = RoutingContext(
            idea_id=idea_id,
            semantic_keywords=semantic_keywords,
            domain_scores=domain_scores,
            requirement_complexity=requirement_complexity,
            integration_needs=integration_needs,
            performance_requirements=performance_requirements,
            timeline_constraints=timeline_constraints,
            resource_constraints=resource_constraints
        )
        
        # Cache the result
        if self.config.cache_semantic_analysis:
            self.semantic_cache[idea_id] = {
                "idea_id": idea_id,
                "semantic_keywords": semantic_keywords,
                "domain_scores": domain_scores,
                "requirement_complexity": requirement_complexity,
                "integration_needs": integration_needs,
                "performance_requirements": performance_requirements,
                "timeline_constraints": timeline_constraints,
                "resource_constraints": resource_constraints
            }
        
        return routing_context
    
    def _extract_semantic_keywords(self, text: str) -> List[SemanticKeyword]:
        """Extract semantically meaningful keywords from text."""
        # Normalize text
        text = text.lower()
        
        # Define keyword patterns with domain relevance
        keyword_patterns = {
            # Machine Learning keywords
            "machine learning": {"machine_learning": 1.0, "research": 0.8},
            "ml": {"machine_learning": 1.0},
            "ai": {"machine_learning": 0.9, "research": 0.7},
            "neural network": {"machine_learning": 1.0},
            "deep learning": {"machine_learning": 1.0},
            "model training": {"machine_learning": 1.0},
            "prediction": {"machine_learning": 0.8, "analytics": 0.6},
            "classification": {"machine_learning": 0.9},
            "regression": {"machine_learning": 0.9},
            "clustering": {"machine_learning": 0.8, "analytics": 0.6},
            
            # Data Engineering keywords
            "data pipeline": {"data_engineering": 1.0},
            "etl": {"data_engineering": 1.0},
            "data processing": {"data_engineering": 0.9, "analytics": 0.7},
            "big data": {"data_engineering": 0.9, "analytics": 0.7},
            "streaming": {"data_engineering": 0.8, "infrastructure": 0.6},
            "batch processing": {"data_engineering": 0.9},
            
            # Web Development keywords
            "web application": {"web_development": 1.0},
            "frontend": {"web_development": 1.0},
            "backend": {"web_development": 0.9, "infrastructure": 0.6},
            "api": {"web_development": 0.8, "integration": 0.7},
            "rest": {"web_development": 0.8, "integration": 0.7},
            "graphql": {"web_development": 0.9},
            "react": {"web_development": 1.0},
            "vue": {"web_development": 1.0},
            "angular": {"web_development": 1.0},
            
            # Infrastructure keywords
            "microservice": {"infrastructure": 0.9, "web_development": 0.7},
            "docker": {"infrastructure": 1.0},
            "kubernetes": {"infrastructure": 1.0},
            "deployment": {"infrastructure": 0.9},
            "scalability": {"infrastructure": 0.8, "web_development": 0.6},
            "load balancing": {"infrastructure": 0.9},
            
            # Security keywords
            "authentication": {"security": 1.0, "web_development": 0.6},
            "authorization": {"security": 1.0},
            "encryption": {"security": 1.0},
            "security": {"security": 1.0},
            "vulnerability": {"security": 0.9},
            
            # Analytics keywords
            "dashboard": {"analytics": 1.0, "web_development": 0.6},
            "visualization": {"analytics": 0.9},
            "metrics": {"analytics": 0.8},
            "reporting": {"analytics": 0.8},
            "analytics": {"analytics": 1.0},
            
            # Integration keywords
            "integration": {"integration": 1.0},
            "connector": {"integration": 0.9},
            "webhook": {"integration": 0.8},
            "sync": {"integration": 0.7},
            "bridge": {"integration": 0.8}
        }
        
        # Extract keywords with frequency and importance
        keywords = []
        for pattern, domain_relevance in keyword_patterns.items():
            frequency = len(re.findall(pattern, text))
            if frequency >= self.config.min_keyword_frequency:
                # Calculate importance based on frequency and pattern significance
                importance_score = min(1.0, frequency * 0.2 + max(domain_relevance.values()) * 0.8)
                
                if importance_score >= self.config.keyword_importance_threshold:
                    # Extract context (surrounding words)
                    context = self._extract_keyword_context(text, pattern)
                    
                    keywords.append(SemanticKeyword(
                        term=pattern,
                        frequency=frequency,
                        importance_score=importance_score,
                        domain_relevance=domain_relevance,
                        context=context
                    ))
        
        # Sort by importance score
        keywords.sort(key=lambda k: k.importance_score, reverse=True)
        return keywords[:20]  # Limit to top 20 keywords
    
    def _extract_keyword_context(self, text: str, keyword: str, window_size: int = 3) -> List[str]:
        """Extract context words around a keyword."""
        words = text.split()
        contexts = []
        
        for i, word in enumerate(words):
            if keyword in word.lower():
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                context = ' '.join(words[start:end])
                contexts.append(context)
        
        return contexts
    
    def _calculate_domain_scores(self, text: str, 
                                semantic_keywords: List[SemanticKeyword]) -> Dict[str, float]:
        """Calculate relevance scores for different domains."""
        domain_scores = defaultdict(float)
        
        # Score based on semantic keywords
        for keyword in semantic_keywords:
            for domain, relevance in keyword.domain_relevance.items():
                domain_scores[domain] += keyword.importance_score * relevance
        
        # Apply domain signature matching
        for signature in self.domain_signatures.values():
            signature_score = 0.0
            matched_keywords = 0
            
            for sig_keyword in signature.keywords:
                if sig_keyword in text.lower():
                    weight = signature.weight_multipliers.get(sig_keyword, 1.0)
                    signature_score += weight
                    matched_keywords += 1
            
            # Check exclusion patterns
            excluded = any(pattern in text.lower() for pattern in signature.exclusion_patterns)
            
            if not excluded and matched_keywords > 0:
                # Normalize by number of signature keywords
                normalized_score = signature_score / len(signature.keywords)
                domain_scores[signature.domain] += normalized_score
        
        # Normalize scores to 0-1 range
        if domain_scores:
            max_score = max(domain_scores.values())
            if max_score > 0:
                domain_scores = {domain: score / max_score 
                               for domain, score in domain_scores.items()}
        
        return dict(domain_scores)
    
    def _analyze_requirement_complexity(self, text: str) -> str:
        """Analyze the complexity level of requirements."""
        complexity_indicators = {
            "simple": ["basic", "simple", "straightforward", "easy", "minimal"],
            "moderate": ["moderate", "standard", "typical", "normal", "average"],
            "complex": ["complex", "advanced", "sophisticated", "comprehensive", "enterprise"],
            "very_complex": ["highly complex", "enterprise-grade", "distributed", "scalable", "mission-critical"]
        }
        
        scores = {}
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text.lower())
            scores[level] = score
        
        # Also consider text length and specific complexity patterns
        if len(text) > 1000:
            scores["complex"] += 2
        elif len(text) > 500:
            scores["moderate"] += 1
        
        # Check for complexity patterns
        complex_patterns = [
            "real-time", "high availability", "fault tolerance", "distributed system",
            "microservice", "load balancing", "auto-scaling", "multi-tenant"
        ]
        
        for pattern in complex_patterns:
            if pattern in text.lower():
                scores["complex"] += 1
        
        # Return the complexity level with highest score
        if not scores or max(scores.values()) == 0:
            return "moderate"
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _identify_integration_needs(self, text: str) -> List[str]:
        """Identify integration requirements from the text."""
        integration_patterns = {
            "database": ["database", "db", "sql", "nosql", "mongodb", "postgresql"],
            "api_integration": ["api", "rest", "graphql", "webhook", "third-party"],
            "authentication": ["auth", "authentication", "login", "sso", "oauth"],
            "messaging": ["queue", "message", "event", "notification", "email"],
            "file_storage": ["file", "storage", "upload", "download", "s3", "blob"],
            "analytics": ["analytics", "tracking", "metrics", "logging", "monitoring"],
            "payment": ["payment", "billing", "subscription", "stripe", "paypal"],
            "search": ["search", "elasticsearch", "solr", "indexing"],
            "caching": ["cache", "redis", "memcached", "cdn"],
            "external_service": ["external", "third-party", "vendor", "saas"]
        }
        
        identified_integrations = []
        for integration_type, patterns in integration_patterns.items():
            if any(pattern in text.lower() for pattern in patterns):
                identified_integrations.append(integration_type)
        
        return identified_integrations
    
    def _extract_performance_requirements(self, text: str) -> List[str]:
        """Extract performance requirements from the text."""
        performance_patterns = {
            "real_time": ["real-time", "real time", "instant", "immediate"],
            "high_throughput": ["high throughput", "many requests", "concurrent", "bulk"],
            "low_latency": ["low latency", "fast response", "quick", "responsive"],
            "scalability": ["scalable", "scale", "growth", "elastic"],
            "availability": ["high availability", "uptime", "24/7", "reliable"],
            "performance": ["performance", "speed", "optimization", "efficient"]
        }
        
        requirements = []
        for req_type, patterns in performance_patterns.items():
            if any(pattern in text.lower() for pattern in patterns):
                requirements.append(req_type)
        
        return requirements
    
    def _assess_timeline_constraints(self, idea: Idea, text: str) -> Optional[str]:
        """Assess timeline constraints from idea priority and text."""
        # Priority-based timeline assessment
        if idea.priority >= 9:
            return "urgent"
        elif idea.priority >= 7:
            return "high_priority"
        elif idea.priority >= 5:
            return "normal"
        else:
            return "flexible"
    
    def _assess_resource_constraints(self, text: str) -> Optional[str]:
        """Assess resource constraints from the text."""
        constraint_patterns = {
            "limited": ["limited", "small budget", "minimal resources", "constraint"],
            "moderate": ["moderate", "standard", "typical"],
            "extensive": ["extensive", "large", "enterprise", "unlimited"]
        }
        
        for constraint_level, patterns in constraint_patterns.items():
            if any(pattern in text.lower() for pattern in patterns):
                return constraint_level
        
        return "moderate"  # Default
    
    def _classify_domain(self, idea: Idea, context: RoutingContext) -> Dict[str, float]:
        """Classify the idea into domain categories."""
        domain_classification = context.domain_scores.copy()
        
        # Enhance classification with additional analysis
        text = f"{idea.title} {idea.description}".lower()
        
        # Rule-based enhancements
        if any(keyword.term in ["machine learning", "ml", "ai", "neural"] 
               for keyword in context.semantic_keywords):
            domain_classification["machine_learning"] = max(
                domain_classification.get("machine_learning", 0), 0.8
            )
        
        if any(keyword.term in ["web", "frontend", "react", "vue"] 
               for keyword in context.semantic_keywords):
            domain_classification["web_development"] = max(
                domain_classification.get("web_development", 0), 0.8
            )
        
        if "data" in text and ("pipeline" in text or "processing" in text):
            domain_classification["data_engineering"] = max(
                domain_classification.get("data_engineering", 0), 0.8
            )
        
        return domain_classification
    
    async def _find_similar_projects(self, idea: Idea, 
                                   context: RoutingContext) -> List[SimilarityMatch]:
        """Find similar past projects for routing guidance."""
        if not self.routing_history:
            return []
        
        idea_text = f"{idea.title} {idea.description}".lower()
        similar_projects = []
        
        for hist_idea_id, past_decision in self.routing_history.items():
            # Calculate keyword overlap
            idea_keywords = {kw.term for kw in context.semantic_keywords}
            past_keywords = set(past_decision.semantic_keywords)
            
            if not idea_keywords or not past_keywords:
                continue
            
            keyword_overlap = len(idea_keywords.intersection(past_keywords))
            keyword_similarity = keyword_overlap / len(idea_keywords.union(past_keywords))
            
            # Calculate domain similarity
            domain_similarity = 0.0
            if context.domain_scores and past_decision.domain_classification:
                common_domains = set(context.domain_scores.keys()).intersection(
                    set(past_decision.domain_classification.keys())
                )
                
                if common_domains:
                    domain_diffs = [
                        abs(context.domain_scores.get(domain, 0) - 
                            past_decision.domain_classification.get(domain, 0))
                        for domain in common_domains
                    ]
                    domain_similarity = 1.0 - (sum(domain_diffs) / len(domain_diffs))
            
            # Combined similarity score
            combined_similarity = (keyword_similarity * 0.6 + domain_similarity * 0.4)
            
            if combined_similarity >= self.config.semantic_similarity_threshold:
                matched_keywords = list(idea_keywords.intersection(past_keywords))
                
                similar_match = SimilarityMatch(
                    idea_id=hist_idea_id,
                    similarity_score=combined_similarity,
                    matched_keywords=matched_keywords,
                    previous_routing=past_decision
                )
                similar_projects.append(similar_match)
        
        # Sort by similarity score and return top matches
        similar_projects.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_projects[:5]
    
    def _assess_department_workloads(self) -> Dict[str, DepartmentWorkload]:
        """Assess current workload across all departments."""
        # This would integrate with actual department monitoring
        # For now, we'll return mock data that shows the structure
        
        mock_workloads = {
            "department_of_ml": DepartmentWorkload(
                department_id="department_of_ml",
                total_capacity=100,
                current_load=75,
                utilization_percentage=75.0,
                pending_ideas=5,
                active_projects=15,
                average_completion_time=14.5,
                recent_success_rate=0.85,
                institution_workloads={"keras_institution": 30},
                available_institutions=["keras_institution"],
                overloaded_institutions=[]
            ),
            "department_of_web": DepartmentWorkload(
                department_id="department_of_web",
                total_capacity=80,
                current_load=45,
                utilization_percentage=56.25,
                pending_ideas=3,
                active_projects=8,
                average_completion_time=10.2,
                recent_success_rate=0.92,
                institution_workloads={},
                available_institutions=[],
                overloaded_institutions=[]
            ),
            "department_of_data": DepartmentWorkload(
                department_id="department_of_data",
                total_capacity=60,
                current_load=55,
                utilization_percentage=91.67,
                pending_ideas=7,
                active_projects=12,
                average_completion_time=18.0,
                recent_success_rate=0.78,
                institution_workloads={},
                available_institutions=[],
                overloaded_institutions=[]
            )
        }
        
        return mock_workloads
    
    def _generate_routing_decision(self, idea: Idea, context: RoutingContext,
                                 domain_classification: Dict[str, float],
                                 similar_projects: List[SimilarityMatch],
                                 workload_assessment: Dict[str, DepartmentWorkload]) -> RoutingDecision:
        """Generate the primary routing decision based on all analysis."""
        
        # Step 1: Determine primary department based on domain scores
        if not domain_classification:
            primary_department = "department_of_research"  # Default fallback
            confidence_score = 0.3
        else:
            primary_department = self._map_domain_to_department(
                max(domain_classification.items(), key=lambda x: x[1])
            )
            confidence_score = max(domain_classification.values())
        
        # Step 2: Adjust confidence based on similar projects
        if similar_projects:
            # Use feedback from similar projects to adjust confidence
            similar_success_rate = sum(
                1 for proj in similar_projects[:3] 
                if proj.outcome_success is not False
            ) / min(3, len(similar_projects))
            
            confidence_score = (confidence_score * 0.7 + similar_success_rate * 0.3)
        
        # Step 3: Consider workload in department selection
        if primary_department in workload_assessment:
            workload = workload_assessment[primary_department]
            if workload.utilization_percentage > 90:
                # Look for alternative department
                alternatives = self._find_alternative_departments(
                    domain_classification, workload_assessment
                )
                if alternatives:
                    primary_department = alternatives[0]
                    confidence_score *= 0.9  # Slight confidence reduction
        
        # Step 4: Determine institution
        primary_institution = self._select_institution(
            primary_department, context, workload_assessment
        )
        
        # Step 5: Calculate confidence level
        confidence_level = self._calculate_confidence_level(confidence_score)
        
        # Step 6: Determine priority
        priority = self._calculate_routing_priority(idea, context, confidence_score)
        
        # Step 7: Generate reasoning
        reasoning = self._generate_routing_reasoning(
            primary_department, primary_institution, context, 
            domain_classification, similar_projects
        )
        
        # Step 8: Find alternatives
        alternative_departments = self._find_alternative_departments(
            domain_classification, workload_assessment
        )
        alternative_institutions = self._find_alternative_institutions(
            primary_department, workload_assessment
        )
        
        # Step 9: Check if new institution is needed
        new_institution_needed = (
            primary_institution is None and 
            self.config.enable_new_institution_suggestions
        )
        
        proposed_institution_name = None
        proposed_capabilities = []
        if new_institution_needed:
            proposed_institution_name, proposed_capabilities = \
                self._propose_new_institution(primary_department, context)
        
        # Step 10: Check for cross-department collaboration
        cross_department_collaboration = self._identify_cross_department_needs(
            context, domain_classification
        )
        
        return RoutingDecision(
            idea_id=str(idea.id),
            primary_department=primary_department,
            primary_institution=primary_institution,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            priority=priority,
            reasoning=reasoning,
            alternative_departments=alternative_departments[:3],
            alternative_institutions=alternative_institutions[:3],
            semantic_keywords=[kw.term for kw in context.semantic_keywords[:10]],
            domain_classification=domain_classification,
            new_institution_needed=new_institution_needed,
            proposed_institution_name=proposed_institution_name,
            proposed_institution_capabilities=proposed_capabilities,
            cross_department_collaboration=cross_department_collaboration
        )
    
    def _map_domain_to_department(self, domain_score: Tuple[str, float]) -> str:
        """Map domain classification to department."""
        domain, score = domain_score
        
        domain_to_department = {
            "machine_learning": "department_of_ml",
            "data_engineering": "department_of_data",
            "web_development": "department_of_web",
            "infrastructure": "department_of_infrastructure",
            "security": "department_of_security",
            "research": "department_of_research",
            "integration": "department_of_integration",
            "analytics": "department_of_analytics"
        }
        
        return domain_to_department.get(domain, "department_of_research")
    
    def _select_institution(self, department: str, context: RoutingContext,
                          workload_assessment: Dict[str, DepartmentWorkload]) -> Optional[str]:
        """Select the best institution within a department."""
        if department not in workload_assessment:
            return None
        
        workload = workload_assessment[department]
        
        # For ML department, we have keras_institution
        if department == "department_of_ml":
            if "keras_institution" in workload.available_institutions:
                return "keras_institution"
        
        # For other departments, we might not have specific institutions yet
        # This is where new institution suggestions come in
        return None
    
    def _calculate_confidence_level(self, confidence_score: float) -> RoutingConfidence:
        """Convert confidence score to confidence level enum."""
        if confidence_score >= 0.9:
            return RoutingConfidence.VERY_HIGH
        elif confidence_score >= 0.8:
            return RoutingConfidence.HIGH
        elif confidence_score >= 0.6:
            return RoutingConfidence.MEDIUM
        elif confidence_score >= 0.4:
            return RoutingConfidence.LOW
        else:
            return RoutingConfidence.VERY_LOW
    
    def _calculate_routing_priority(self, idea: Idea, context: RoutingContext,
                                  confidence_score: float) -> RoutingPriority:
        """Calculate routing priority based on multiple factors."""
        # Base priority from idea
        base_priority = idea.priority
        
        # Adjust based on timeline constraints
        if context.timeline_constraints == "urgent":
            return RoutingPriority.CRITICAL
        elif context.timeline_constraints == "high_priority":
            if base_priority >= 8:
                return RoutingPriority.CRITICAL
            else:
                return RoutingPriority.HIGH
        
        # Adjust based on confidence and complexity
        if confidence_score >= 0.8 and base_priority >= 7:
            return RoutingPriority.HIGH
        elif confidence_score >= 0.6:
            return RoutingPriority.NORMAL
        else:
            return RoutingPriority.LOW
    
    def _generate_routing_reasoning(self, department: str, institution: Optional[str],
                                  context: RoutingContext, 
                                  domain_classification: Dict[str, float],
                                  similar_projects: List[SimilarityMatch]) -> str:
        """Generate human-readable reasoning for the routing decision."""
        reasoning_parts = []
        
        # Primary domain reasoning
        if domain_classification:
            top_domain = max(domain_classification.items(), key=lambda x: x[1])
            reasoning_parts.append(
                f"Primary domain classification: {top_domain[0]} (score: {top_domain[1]:.2f})"
            )
        
        # Keyword-based reasoning
        if context.semantic_keywords:
            top_keywords = [kw.term for kw in context.semantic_keywords[:3]]
            reasoning_parts.append(
                f"Key semantic indicators: {', '.join(top_keywords)}"
            )
        
        # Similar projects reasoning
        if similar_projects:
            reasoning_parts.append(
                f"Found {len(similar_projects)} similar past projects with "
                f"average similarity {sum(p.similarity_score for p in similar_projects) / len(similar_projects):.2f}"
            )
        
        # Institution reasoning
        if institution:
            reasoning_parts.append(f"Assigned to {institution} based on specialization match")
        else:
            reasoning_parts.append("No existing institution found - suggesting new institution creation")
        
        # Complexity reasoning
        if context.requirement_complexity:
            reasoning_parts.append(f"Requirement complexity: {context.requirement_complexity}")
        
        return ". ".join(reasoning_parts)
    
    def _find_alternative_departments(self, domain_classification: Dict[str, float],
                                    workload_assessment: Dict[str, DepartmentWorkload]) -> List[str]:
        """Find alternative departments for routing."""
        if not domain_classification:
            return []
        
        # Sort domains by score
        sorted_domains = sorted(domain_classification.items(), key=lambda x: x[1], reverse=True)
        
        alternatives = []
        for domain, score in sorted_domains[1:4]:  # Skip the top one, get next 3
            if score >= 0.4:  # Only consider reasonably relevant alternatives
                department = self._map_domain_to_department((domain, score))
                if department not in alternatives:
                    alternatives.append(department)
        
        return alternatives
    
    def _find_alternative_institutions(self, department: str,
                                     workload_assessment: Dict[str, DepartmentWorkload]) -> List[str]:
        """Find alternative institutions within a department."""
        if department not in workload_assessment:
            return []
        
        workload = workload_assessment[department]
        return workload.available_institutions
    
    def _propose_new_institution(self, department: str, 
                               context: RoutingContext) -> Tuple[Optional[str], List[str]]:
        """Propose a new institution name and capabilities."""
        if not self.config.enable_new_institution_suggestions:
            return None, []
        
        # Generate institution name based on top keywords and department
        top_keywords = [kw.term for kw in context.semantic_keywords[:2]]
        
        if not top_keywords:
            return None, []
        
        # Create institution name
        primary_keyword = top_keywords[0].replace(" ", "_")
        institution_name = f"{primary_keyword}_institution"
        
        # Determine capabilities
        capabilities = []
        for keyword in context.semantic_keywords[:5]:
            if keyword.importance_score >= 0.6:
                capabilities.append(keyword.term)
        
        # Add department-specific capabilities
        if department == "department_of_ml":
            capabilities.extend(["model_training", "inference", "data_preprocessing"])
        elif department == "department_of_web":
            capabilities.extend(["frontend_development", "backend_apis", "ui_ux"])
        elif department == "department_of_data":
            capabilities.extend(["data_pipelines", "etl_processing", "analytics"])
        
        return institution_name, capabilities
    
    def _identify_cross_department_needs(self, context: RoutingContext,
                                       domain_classification: Dict[str, float]) -> List[str]:
        """Identify if cross-department collaboration is needed."""
        if not self.config.enable_cross_department_routing:
            return []
        
        collaboration_needs = []
        
        # Check if multiple domains have high scores
        high_score_domains = [
            domain for domain, score in domain_classification.items() 
            if score >= 0.6
        ]
        
        if len(high_score_domains) > 1:
            # Map to departments
            departments = [
                self._map_domain_to_department((domain, 0))
                for domain in high_score_domains
            ]
            collaboration_needs = list(set(departments))
        
        # Check for specific integration patterns
        if "api" in [kw.term for kw in context.semantic_keywords]:
            if "department_of_integration" not in collaboration_needs:
                collaboration_needs.append("department_of_integration")
        
        if context.integration_needs and len(context.integration_needs) > 2:
            if "department_of_integration" not in collaboration_needs:
                collaboration_needs.append("department_of_integration")
        
        return collaboration_needs
    
    def _validate_routing_decision(self, decision: RoutingDecision, 
                                 idea: Idea) -> RoutingDecision:
        """Validate and potentially adjust the routing decision."""
        # Check if escalation is needed
        if (decision.confidence_score < self.config.confidence_escalation_threshold or
            decision.confidence_level == RoutingConfidence.VERY_LOW):
            
            decision.escalation_needed = True
            decision.escalation_reason = (
                f"Low confidence score ({decision.confidence_score:.2f}) below "
                f"threshold ({self.config.confidence_escalation_threshold})"
            )
        
        # Add validation notes
        validation_notes = []
        
        if decision.new_institution_needed:
            validation_notes.append(
                f"Recommends creating new institution: {decision.proposed_institution_name}"
            )
        
        if decision.cross_department_collaboration:
            validation_notes.append(
                f"Requires collaboration with: {', '.join(decision.cross_department_collaboration)}"
            )
        
        if len(decision.alternative_departments) > 0:
            validation_notes.append(
                f"Alternative departments available: {', '.join(decision.alternative_departments)}"
            )
        
        decision.validation_notes = validation_notes
        
        return decision
    
    def _store_routing_decision(self, decision: RoutingDecision):
        """Store routing decision in history for learning."""
        self.routing_history[decision.idea_id] = decision
        
        # Update department distribution metrics
        dept = decision.primary_department
        self.routing_metrics.department_distribution[dept] = \
            self.routing_metrics.department_distribution.get(dept, 0) + 1
    
    def _update_similarity_index(self, idea_id: str, context: RoutingContext):
        """Update similarity index for future matching."""
        for keyword in context.semantic_keywords:
            if keyword.importance_score >= 0.5:
                self.similarity_index[keyword.term].append(idea_id)
    
    def _update_routing_metrics(self, decision: RoutingDecision, 
                              analysis_time: float, success: bool):
        """Update routing performance metrics."""
        self.routing_metrics.total_routings += 1
        
        if success:
            self.routing_metrics.successful_routings += 1
        else:
            self.routing_metrics.failed_routings += 1
        
        if decision.escalation_needed:
            self.routing_metrics.escalated_routings += 1
        
        # Update average analysis time
        current_avg = self.routing_metrics.average_analysis_time
        total = self.routing_metrics.total_routings
        
        new_avg = ((current_avg * (total - 1)) + analysis_time) / total
        self.routing_metrics.average_analysis_time = new_avg
        
        # Update average confidence score
        current_conf_avg = self.routing_metrics.average_confidence_score
        new_conf_avg = ((current_conf_avg * (total - 1)) + decision.confidence_score) / total
        self.routing_metrics.average_confidence_score = new_conf_avg
        
        # Calculate routing accuracy (simplified)
        if total > 0:
            self.routing_metrics.routing_accuracy = (
                self.routing_metrics.successful_routings / total
            )
    
    def _create_fallback_routing(self, idea: Idea, start_time: datetime) -> RoutingDecision:
        """Create a fallback routing decision when analysis fails."""
        return RoutingDecision(
            idea_id=str(idea.id),
            primary_department="department_of_research",
            primary_institution=None,
            confidence_level=RoutingConfidence.VERY_LOW,
            confidence_score=0.2,
            priority=RoutingPriority.LOW,
            reasoning="Fallback routing due to analysis failure - assigned to research department for manual review",
            escalation_needed=True,
            escalation_reason="Routing analysis failed - requires manual review",
            new_institution_needed=False
        )
    
    def _initialize_domain_signatures(self) -> Dict[str, DomainSignature]:
        """Initialize domain-specific signatures for classification."""
        return {
            "machine_learning": DomainSignature(
                domain="machine_learning",
                keywords=[
                    "machine learning", "ml", "ai", "neural", "deep learning",
                    "model", "training", "prediction", "classification", "regression",
                    "supervised", "unsupervised", "tensorflow", "pytorch", "keras"
                ],
                weight_multipliers={
                    "machine learning": 2.0,
                    "neural": 1.8,
                    "deep learning": 2.0,
                    "model": 1.5
                },
                capability_mapping=[
                    DepartmentCapability.MACHINE_LEARNING,
                    DepartmentCapability.NATURAL_LANGUAGE_PROCESSING,
                    DepartmentCapability.COMPUTER_VISION
                ]
            ),
            
            "web_development": DomainSignature(
                domain="web_development",
                keywords=[
                    "web", "frontend", "backend", "api", "rest", "graphql",
                    "react", "vue", "angular", "node", "express", "django",
                    "flask", "html", "css", "javascript", "typescript"
                ],
                weight_multipliers={
                    "web": 1.5,
                    "frontend": 1.8,
                    "backend": 1.6,
                    "api": 1.4
                },
                capability_mapping=[
                    DepartmentCapability.WEB_DEVELOPMENT,
                    DepartmentCapability.FRONTEND_DEVELOPMENT,
                    DepartmentCapability.BACKEND_DEVELOPMENT
                ]
            ),
            
            "data_engineering": DomainSignature(
                domain="data_engineering",
                keywords=[
                    "data", "pipeline", "etl", "processing", "streaming",
                    "batch", "warehouse", "lake", "spark", "hadoop",
                    "kafka", "airflow", "big data"
                ],
                weight_multipliers={
                    "pipeline": 2.0,
                    "etl": 2.0,
                    "streaming": 1.8,
                    "data": 1.2
                },
                capability_mapping=[
                    DepartmentCapability.DATA_ENGINEERING,
                    DepartmentCapability.ANALYTICS
                ]
            ),
            
            "infrastructure": DomainSignature(
                domain="infrastructure",
                keywords=[
                    "infrastructure", "deployment", "docker", "kubernetes",
                    "cloud", "aws", "azure", "gcp", "microservice",
                    "scalability", "load balancing", "devops", "ci/cd"
                ],
                weight_multipliers={
                    "kubernetes": 2.0,
                    "docker": 1.8,
                    "microservice": 1.6,
                    "deployment": 1.5
                },
                capability_mapping=[
                    DepartmentCapability.INFRASTRUCTURE,
                    DepartmentCapability.DEVOPS
                ]
            ),
            
            "security": DomainSignature(
                domain="security",
                keywords=[
                    "security", "authentication", "authorization", "encryption",
                    "oauth", "jwt", "ssl", "vulnerability", "penetration",
                    "firewall", "compliance", "audit"
                ],
                weight_multipliers={
                    "security": 2.0,
                    "authentication": 1.8,
                    "encryption": 1.8,
                    "vulnerability": 1.6
                },
                capability_mapping=[DepartmentCapability.SECURITY]
            )
        }
    
    # Public API methods for external integration
    
    def get_routing_history(self, limit: Optional[int] = None) -> List[RoutingDecision]:
        """Get routing history with optional limit."""
        decisions = list(self.routing_history.values())
        decisions.sort(key=lambda d: d.analysis_timestamp, reverse=True)
        
        if limit:
            decisions = decisions[:limit]
        
        return decisions
    
    def get_routing_metrics(self) -> RoutingMetrics:
        """Get comprehensive routing performance metrics."""
        return self.routing_metrics
    
    def get_department_workloads(self) -> Dict[str, DepartmentWorkload]:
        """Get current department workload assessment."""
        return self._assess_department_workloads()
    
    async def provide_routing_feedback(self, idea_id: str, was_successful: bool,
                                     actual_department: Optional[str] = None,
                                     notes: Optional[str] = None):
        """
        Provide feedback on routing decision for learning.
        
        Args:
            idea_id: ID of the idea that was routed
            was_successful: Whether the routing was successful
            actual_department: The department that actually handled it (if different)
            notes: Additional feedback notes
        """
        if idea_id not in self.routing_history:
            self.logger.warning(f"No routing history found for idea {idea_id}")
            return
        
        # Store feedback
        self.feedback_data[idea_id] = {
            "successful": was_successful,
            "actual_department": actual_department,
            "notes": notes,
            "feedback_timestamp": datetime.now(timezone.utc),
            "original_decision": self.routing_history[idea_id]
        }
        
        # Update learning if adaptive learning is enabled
        if self.config.enable_adaptive_learning:
            await self._process_feedback_learning(idea_id)
        
        self.logger.agent_action("routing_feedback_received", "router_agent", idea_id, {
            "successful": was_successful,
            "actual_department": actual_department,
            "adaptive_learning": self.config.enable_adaptive_learning
        })
    
    async def _process_feedback_learning(self, idea_id: str):
        """Process feedback for adaptive learning."""
        feedback = self.feedback_data.get(idea_id)
        if not feedback:
            return
        
        original_decision = feedback["original_decision"]
        
        # Update routing metrics based on feedback
        if feedback["successful"]:
            # Positive feedback - reinforce similar routing patterns
            self._reinforce_routing_pattern(original_decision)
        else:
            # Negative feedback - adjust routing weights
            self._adjust_routing_weights(original_decision, feedback)
        
        # Update learning metrics
        self.routing_metrics.learning_iterations += 1
        self.routing_metrics.feedback_accuracy = self._calculate_feedback_accuracy()
    
    def _reinforce_routing_pattern(self, decision: RoutingDecision):
        """Reinforce successful routing patterns."""
        # This would update internal weights/models
        # For now, we'll just log the reinforcement
        self.logger.info(f"Reinforcing routing pattern: {decision.primary_department} "
                        f"for keywords: {decision.semantic_keywords[:3]}")
    
    def _adjust_routing_weights(self, decision: RoutingDecision, feedback: Dict[str, Any]):
        """Adjust routing weights based on negative feedback."""
        # This would update internal weights/models
        # For now, we'll just log the adjustment
        actual_dept = feedback.get("actual_department")
        if actual_dept:
            self.logger.info(f"Adjusting routing weights: {decision.primary_department} -> "
                           f"{actual_dept} for keywords: {decision.semantic_keywords[:3]}")
    
    def _calculate_feedback_accuracy(self) -> float:
        """Calculate accuracy based on feedback received."""
        if not self.feedback_data:
            return 0.0
        
        successful_count = sum(1 for f in self.feedback_data.values() if f["successful"])
        return successful_count / len(self.feedback_data)
    
    def export_routing_knowledge(self) -> Dict[str, Any]:
        """Export routing knowledge for backup or transfer."""
        return {
            "routing_history": {
                idea_id: decision.dict() 
                for idea_id, decision in self.routing_history.items()
            },
            "feedback_data": self.feedback_data,
            "routing_metrics": self.routing_metrics.dict(),
            "domain_signatures": {
                name: {
                    "domain": sig.domain,
                    "keywords": sig.keywords,
                    "weight_multipliers": sig.weight_multipliers,
                    "capability_mapping": [cap.value for cap in sig.capability_mapping],
                    "exclusion_patterns": sig.exclusion_patterns
                }
                for name, sig in self.domain_signatures.items()
            },
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def import_routing_knowledge(self, knowledge_data: Dict[str, Any]):
        """Import routing knowledge from backup or transfer."""
        try:
            # Import routing history
            if "routing_history" in knowledge_data:
                for idea_id, decision_data in knowledge_data["routing_history"].items():
                    self.routing_history[idea_id] = RoutingDecision(**decision_data)
            
            # Import feedback data
            if "feedback_data" in knowledge_data:
                self.feedback_data.update(knowledge_data["feedback_data"])
            
            # Import metrics
            if "routing_metrics" in knowledge_data:
                self.routing_metrics = RoutingMetrics(**knowledge_data["routing_metrics"])
            
            self.logger.agent_action("routing_knowledge_imported", "router_agent",
                                   additional_context={
                                       "routing_history_size": len(self.routing_history),
                                       "feedback_data_size": len(self.feedback_data)
                                   })
            
        except Exception as e:
            self.logger.error(f"Failed to import routing knowledge: {e}", exc_info=True)


# Factory function for easy agent creation
def create_router_agent(config: Optional[RouterConfiguration] = None,
                       state_router: Optional[StateMachineRouter] = None) -> RouterAgent:
    """
    Create a new Router Agent instance.
    
    Args:
        config: Optional router configuration
        state_router: Optional state machine router
        
    Returns:
        Configured RouterAgent instance
    """
    return RouterAgent(config, state_router)


# Export main classes and functions
__all__ = [
    "RouterAgent",
    "RouterConfiguration", 
    "RoutingDecision",
    "RoutingContext",
    "SimilarityMatch",
    "DepartmentWorkload",
    "RoutingMetrics",
    "RoutingConfidence",
    "RoutingPriority",
    "DepartmentCapability",
    "create_router_agent"
]