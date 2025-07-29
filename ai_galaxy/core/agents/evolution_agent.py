"""
Evolution Agent for the AI-Galaxy Lower Meta-Layer.

This module implements the Evolution Agent, responsible for monitoring system
adaptation and learning patterns, analyzing ecosystem health and performance
metrics, suggesting system-wide improvements and optimizations, tracking agent
performance, and implementing continuous improvement processes.
"""

import json
import statistics
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ...shared.models import Idea, IdeaStatus, SystemState, AgentMessage
from ...shared.logger import get_logger, LogContext


class EvolutionScope(str, Enum):
    """Scope of evolution analysis and recommendations."""
    AGENT_LEVEL = "agent_level"
    DEPARTMENT_LEVEL = "department_level"
    INSTITUTION_LEVEL = "institution_level"
    SYSTEM_LEVEL = "system_level"
    ECOSYSTEM_LEVEL = "ecosystem_level"


class AdaptationPattern(str, Enum):
    """Types of adaptation patterns detected in the system."""
    PERFORMANCE_DRIFT = "performance_drift"
    USAGE_SHIFT = "usage_shift"
    CAPACITY_STRAIN = "capacity_strain"
    EFFICIENCY_DECLINE = "efficiency_decline"
    QUALITY_DEGRADATION = "quality_degradation"
    INNOVATION_STAGNATION = "innovation_stagnation"
    RESOURCE_IMBALANCE = "resource_imbalance"
    WORKFLOW_BOTTLENECK = "workflow_bottleneck"


class HealthStatus(str, Enum):
    """Health status levels for system components."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class ImprovementType(str, Enum):
    """Types of improvements that can be suggested."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESOURCE_REALLOCATION = "resource_reallocation"
    WORKFLOW_ENHANCEMENT = "workflow_enhancement"
    CAPABILITY_EXPANSION = "capability_expansion"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    INNOVATION_BOOST = "innovation_boost"
    INTEGRATION_IMPROVEMENT = "integration_improvement"


class EvolutionPriority(str, Enum):
    """Priority levels for evolution recommendations."""
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXPERIMENTAL = "experimental"


@dataclass
class PerformanceMetric:
    """Individual performance metric tracking."""
    name: str
    current_value: float
    historical_values: List[float] = field(default_factory=list)
    target_value: Optional[float] = None
    unit: str = ""
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentPerformance:
    """Performance tracking for individual agents."""
    agent_name: str
    metrics: Dict[str, PerformanceMetric] = field(default_factory=dict)
    health_status: HealthStatus = HealthStatus.GOOD
    efficiency_score: float = 0.8
    adaptation_rate: float = 0.5
    error_rate: float = 0.05
    throughput: float = 1.0
    last_assessment: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EcosystemHealth(BaseModel):
    """Overall ecosystem health assessment."""
    overall_health: HealthStatus
    component_health: Dict[str, HealthStatus] = Field(default_factory=dict)
    performance_trends: Dict[str, str] = Field(default_factory=dict)
    bottlenecks: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    adaptation_patterns: List[AdaptationPattern] = Field(default_factory=list)
    assessment_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.8)


class EvolutionRecommendation(BaseModel):
    """Recommendation for system evolution and improvement."""
    id: str
    title: str
    description: str
    improvement_type: ImprovementType
    scope: EvolutionScope
    priority: EvolutionPriority
    expected_impact: Dict[str, float] = Field(default_factory=dict)
    implementation_steps: List[str] = Field(default_factory=list)
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    timeline_estimate: str
    success_metrics: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LearningPattern(BaseModel):
    """Detected learning pattern in the system."""
    pattern_type: str
    description: str
    frequency: float
    impact_score: float = Field(ge=0.0, le=1.0)
    components_affected: List[str] = Field(default_factory=list)
    trend_data: Dict[str, List[float]] = Field(default_factory=dict)
    first_detected: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_observed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(ge=0.0, le=1.0)


class EvolutionReport(BaseModel):
    """Comprehensive evolution analysis report."""
    report_id: str
    analysis_period: Tuple[datetime, datetime]
    ecosystem_health: EcosystemHealth
    learning_patterns: List[LearningPattern] = Field(default_factory=list)
    recommendations: List[EvolutionRecommendation] = Field(default_factory=list)
    agent_performance_summary: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    system_adaptation_score: float = Field(ge=0.0, le=1.0)
    innovation_index: float = Field(ge=0.0, le=1.0)
    efficiency_trends: Dict[str, float] = Field(default_factory=dict)
    key_insights: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    next_assessment_date: datetime
    confidence_level: float = Field(ge=0.0, le=1.0)


class EvolutionConfiguration(BaseModel):
    """Configuration for the Evolution Agent."""
    analysis_interval_hours: int = Field(default=168, ge=1, le=720)  # 1 week default
    performance_history_days: int = Field(default=30, ge=7, le=90)
    health_check_frequency_hours: int = Field(default=24, ge=1, le=168)
    adaptation_sensitivity: float = Field(default=0.3, ge=0.1, le=1.0)
    recommendation_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    auto_optimization_enabled: bool = False
    learning_pattern_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    trend_analysis_window_days: int = Field(default=14, ge=3, le=60)
    performance_decline_threshold: float = Field(default=0.15, ge=0.05, le=0.5)
    innovation_tracking_enabled: bool = True
    predictive_analysis_enabled: bool = True
    cross_agent_correlation_enabled: bool = True


class EvolutionAgent:
    """
    The Evolution Agent - adaptive intelligence for the AI-Galaxy ecosystem.
    
    Continuously monitors system performance, detects adaptation patterns,
    analyzes ecosystem health, and drives evolutionary improvements through
    data-driven recommendations and optimization strategies.
    """
    
    def __init__(self, config: Optional[EvolutionConfiguration] = None):
        """
        Initialize the Evolution Agent.
        
        Args:
            config: Configuration parameters for evolution monitoring
        """
        self.logger = get_logger("evolution_agent")
        self.config = config or EvolutionConfiguration()
        
        # Performance tracking
        self.agent_performances: Dict[str, AgentPerformance] = {}
        self.system_metrics: Dict[str, PerformanceMetric] = {}
        self.historical_health: List[EcosystemHealth] = []
        
        # Learning and adaptation tracking
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        self.innovation_metrics: Dict[str, float] = {}
        
        # Recommendations and improvements
        self.active_recommendations: Dict[str, EvolutionRecommendation] = {}
        self.implemented_improvements: List[EvolutionRecommendation] = []
        self.evolution_reports: List[EvolutionReport] = []
        
        # Analysis state
        self.last_analysis_time = datetime.now(timezone.utc)
        self.last_health_check = datetime.now(timezone.utc)
        self.baseline_metrics: Dict[str, float] = {}
        
        # Initialize baseline metrics
        self._initialize_baseline_metrics()
        
        self.logger.agent_action("evolution_agent_initialized", "evolution_agent",
                                additional_context={
                                    "analysis_interval": self.config.analysis_interval_hours,
                                    "auto_optimization": self.config.auto_optimization_enabled,
                                    "predictive_analysis": self.config.predictive_analysis_enabled
                                })
    
    def monitor_ecosystem_health(self) -> EcosystemHealth:
        """
        Monitor and assess overall ecosystem health.
        
        Returns:
            Current ecosystem health assessment
        """
        self.logger.agent_action("monitoring_ecosystem_health", "evolution_agent")
        
        try:
            # Assess component health
            component_health = self._assess_component_health()
            
            # Analyze performance trends
            performance_trends = self._analyze_performance_trends()
            
            # Identify bottlenecks and strengths
            bottlenecks = self._identify_bottlenecks()
            strengths = self._identify_strengths()
            risks = self._assess_risks()
            
            # Detect adaptation patterns
            adaptation_patterns = self._detect_adaptation_patterns()
            
            # Calculate overall health
            overall_health = self._calculate_overall_health(component_health)
            
            # Create health assessment
            health_assessment = EcosystemHealth(
                overall_health=overall_health,
                component_health=component_health,
                performance_trends=performance_trends,
                bottlenecks=bottlenecks,
                strengths=strengths,
                risks=risks,
                adaptation_patterns=adaptation_patterns,
                confidence_score=self._calculate_health_confidence(component_health)
            )
            
            # Store in history
            self.historical_health.append(health_assessment)
            
            # Limit history size
            if len(self.historical_health) > 100:
                self.historical_health = self.historical_health[-100:]
            
            self.last_health_check = datetime.now(timezone.utc)
            
            self.logger.agent_action("ecosystem_health_assessed", "evolution_agent",
                                    additional_context={
                                        "overall_health": overall_health.value,
                                        "bottlenecks_count": len(bottlenecks),
                                        "adaptation_patterns": len(adaptation_patterns)
                                    })
            
            return health_assessment
            
        except Exception as e:
            self.logger.error(f"Ecosystem health monitoring failed: {e}", exc_info=True)
            return EcosystemHealth(
                overall_health=HealthStatus.CRITICAL,
                confidence_score=0.0
            )
    
    def analyze_learning_patterns(self) -> List[LearningPattern]:
        """
        Analyze system learning and adaptation patterns.
        
        Returns:
            List of detected learning patterns
        """
        self.logger.agent_action("analyzing_learning_patterns", "evolution_agent")
        
        try:
            patterns = []
            
            # Analyze agent learning patterns
            agent_patterns = self._analyze_agent_learning_patterns()
            patterns.extend(agent_patterns)
            
            # Analyze workflow adaptation patterns
            workflow_patterns = self._analyze_workflow_patterns()
            patterns.extend(workflow_patterns)
            
            # Analyze performance improvement patterns
            improvement_patterns = self._analyze_improvement_patterns()
            patterns.extend(improvement_patterns)
            
            # Analyze innovation patterns
            if self.config.innovation_tracking_enabled:
                innovation_patterns = self._analyze_innovation_patterns()
                patterns.extend(innovation_patterns)
            
            # Filter by confidence threshold
            filtered_patterns = [p for p in patterns 
                               if p.confidence >= self.config.learning_pattern_min_confidence]
            
            # Update learning patterns storage
            for pattern in filtered_patterns:
                pattern_key = f"{pattern.pattern_type}_{hash(pattern.description) % 1000}"
                if pattern_key in self.learning_patterns:
                    # Update existing pattern
                    existing = self.learning_patterns[pattern_key]
                    existing.last_observed = pattern.last_observed
                    existing.frequency = (existing.frequency + pattern.frequency) / 2
                else:
                    self.learning_patterns[pattern_key] = pattern
            
            self.logger.agent_action("learning_patterns_analyzed", "evolution_agent",
                                    additional_context={
                                        "patterns_detected": len(filtered_patterns),
                                        "high_confidence_patterns": len([p for p in filtered_patterns if p.confidence > 0.8])
                                    })
            
            return filtered_patterns
            
        except Exception as e:
            self.logger.error(f"Learning pattern analysis failed: {e}", exc_info=True)
            return []
    
    def generate_evolution_recommendations(self) -> List[EvolutionRecommendation]:
        """
        Generate recommendations for system evolution and improvement.
        
        Returns:
            List of evolution recommendations
        """
        self.logger.agent_action("generating_evolution_recommendations", "evolution_agent")
        
        try:
            recommendations = []
            
            # Get current ecosystem health
            health = self.monitor_ecosystem_health()
            
            # Generate recommendations based on health issues
            health_recommendations = self._generate_health_based_recommendations(health)
            recommendations.extend(health_recommendations)
            
            # Generate performance-based recommendations
            performance_recommendations = self._generate_performance_recommendations()
            recommendations.extend(performance_recommendations)
            
            # Generate learning-based recommendations
            learning_recommendations = self._generate_learning_based_recommendations()
            recommendations.extend(learning_recommendations)
            
            # Generate innovation recommendations
            if self.config.innovation_tracking_enabled:
                innovation_recommendations = self._generate_innovation_recommendations()
                recommendations.extend(innovation_recommendations)
            
            # Generate predictive recommendations
            if self.config.predictive_analysis_enabled:
                predictive_recommendations = self._generate_predictive_recommendations()
                recommendations.extend(predictive_recommendations)
            
            # Filter and prioritize recommendations
            filtered_recommendations = self._filter_and_prioritize_recommendations(recommendations)
            
            # Store active recommendations
            for rec in filtered_recommendations:
                self.active_recommendations[rec.id] = rec
            
            self.logger.agent_action("evolution_recommendations_generated", "evolution_agent",
                                    additional_context={
                                        "total_recommendations": len(filtered_recommendations),
                                        "immediate_priority": len([r for r in filtered_recommendations 
                                                                 if r.priority == EvolutionPriority.IMMEDIATE]),
                                        "high_priority": len([r for r in filtered_recommendations 
                                                            if r.priority == EvolutionPriority.HIGH])
                                    })
            
            return filtered_recommendations
            
        except Exception as e:
            self.logger.error(f"Evolution recommendation generation failed: {e}", exc_info=True)
            return []
    
    def track_agent_performance(self, agent_name: str, performance_data: Dict[str, Any]) -> AgentPerformance:
        """
        Track and analyze individual agent performance.
        
        Args:
            agent_name: Name of the agent
            performance_data: Performance metrics and data
            
        Returns:
            Updated agent performance assessment
        """
        context = LogContext(
            agent_name="evolution_agent",
            additional_context={"target_agent": agent_name}
        )
        
        try:
            # Get or create agent performance record
            if agent_name not in self.agent_performances:
                self.agent_performances[agent_name] = AgentPerformance(agent_name=agent_name)
            
            agent_perf = self.agent_performances[agent_name]
            
            # Update metrics
            for metric_name, value in performance_data.items():
                if metric_name not in agent_perf.metrics:
                    agent_perf.metrics[metric_name] = PerformanceMetric(
                        name=metric_name,
                        current_value=value
                    )
                else:
                    metric = agent_perf.metrics[metric_name]
                    metric.historical_values.append(metric.current_value)
                    metric.current_value = value
                    metric.trend_direction = self._calculate_trend_direction(metric.historical_values)
                    metric.last_updated = datetime.now(timezone.utc)
                    
                    # Limit history size
                    if len(metric.historical_values) > 100:
                        metric.historical_values = metric.historical_values[-100:]
            
            # Calculate derived metrics
            agent_perf.efficiency_score = self._calculate_agent_efficiency(agent_perf)
            agent_perf.adaptation_rate = self._calculate_adaptation_rate(agent_perf)
            agent_perf.error_rate = performance_data.get("error_rate", agent_perf.error_rate)
            agent_perf.throughput = performance_data.get("throughput", agent_perf.throughput)
            
            # Assess health status
            agent_perf.health_status = self._assess_agent_health(agent_perf)
            agent_perf.last_assessment = datetime.now(timezone.utc)
            
            self.logger.debug(f"Agent performance updated: {agent_name}", context)
            
            return agent_perf
            
        except Exception as e:
            self.logger.error(f"Agent performance tracking failed for {agent_name}: {e}", 
                            context, exc_info=True)
            return self.agent_performances.get(agent_name, AgentPerformance(agent_name=agent_name))
    
    def implement_auto_optimization(self) -> List[str]:
        """
        Implement automatic optimizations based on current recommendations.
        
        Returns:
            List of optimizations that were implemented
        """
        if not self.config.auto_optimization_enabled:
            return []
        
        self.logger.agent_action("implementing_auto_optimizations", "evolution_agent")
        
        implemented = []
        
        try:
            # Get high-confidence, low-risk recommendations
            auto_implementable = [
                rec for rec in self.active_recommendations.values()
                if (rec.confidence_score >= 0.8 and 
                    rec.priority in [EvolutionPriority.HIGH, EvolutionPriority.IMMEDIATE] and
                    len(rec.risks) <= 2)
            ]
            
            for recommendation in auto_implementable:
                if self._can_auto_implement(recommendation):
                    success = self._implement_recommendation(recommendation)
                    if success:
                        implemented.append(recommendation.title)
                        self.implemented_improvements.append(recommendation)
                        del self.active_recommendations[recommendation.id]
            
            if implemented:
                self.logger.agent_action("auto_optimizations_implemented", "evolution_agent",
                                        additional_context={
                                            "optimizations_count": len(implemented),
                                            "optimizations": implemented
                                        })
            
            return implemented
            
        except Exception as e:
            self.logger.error(f"Auto-optimization implementation failed: {e}", exc_info=True)
            return []
    
    def generate_evolution_report(self, analysis_period_days: int = 30) -> EvolutionReport:
        """
        Generate comprehensive evolution analysis report.
        
        Args:
            analysis_period_days: Number of days to analyze
            
        Returns:
            Comprehensive evolution report
        """
        self.logger.agent_action("generating_evolution_report", "evolution_agent",
                                additional_context={"analysis_period_days": analysis_period_days})
        
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=analysis_period_days)
            
            # Get current ecosystem health
            ecosystem_health = self.monitor_ecosystem_health()
            
            # Analyze learning patterns
            learning_patterns = self.analyze_learning_patterns()
            
            # Generate recommendations
            recommendations = self.generate_evolution_recommendations()
            
            # Calculate system metrics
            system_adaptation_score = self._calculate_system_adaptation_score()
            innovation_index = self._calculate_innovation_index()
            efficiency_trends = self._calculate_efficiency_trends(analysis_period_days)
            
            # Generate insights and action items
            key_insights = self._generate_key_insights(ecosystem_health, learning_patterns)
            action_items = self._generate_action_items(recommendations)
            
            # Create agent performance summary
            agent_summary = self._create_agent_performance_summary()
            
            # Calculate next assessment date
            next_assessment = datetime.now(timezone.utc) + timedelta(hours=self.config.analysis_interval_hours)
            
            # Create report
            report = EvolutionReport(
                report_id=f"evolution_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                analysis_period=(start_time, end_time),
                ecosystem_health=ecosystem_health,
                learning_patterns=learning_patterns,
                recommendations=recommendations,
                agent_performance_summary=agent_summary,
                system_adaptation_score=system_adaptation_score,
                innovation_index=innovation_index,
                efficiency_trends=efficiency_trends,
                key_insights=key_insights,
                action_items=action_items,
                next_assessment_date=next_assessment,
                confidence_level=self._calculate_report_confidence(ecosystem_health, learning_patterns)
            )
            
            # Store report
            self.evolution_reports.append(report)
            
            # Limit report history
            if len(self.evolution_reports) > 50:
                self.evolution_reports = self.evolution_reports[-50:]
            
            self.last_analysis_time = datetime.now(timezone.utc)
            
            self.logger.agent_action("evolution_report_generated", "evolution_agent",
                                    additional_context={
                                        "report_id": report.report_id,
                                        "recommendations_count": len(recommendations),
                                        "learning_patterns_count": len(learning_patterns),
                                        "system_adaptation_score": system_adaptation_score
                                    })
            
            return report
            
        except Exception as e:
            self.logger.error(f"Evolution report generation failed: {e}", exc_info=True)
            
            # Return minimal report on failure
            return EvolutionReport(
                report_id=f"evolution_report_error_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                analysis_period=(datetime.now(timezone.utc) - timedelta(days=1), datetime.now(timezone.utc)),
                ecosystem_health=EcosystemHealth(overall_health=HealthStatus.CRITICAL),
                system_adaptation_score=0.0,
                innovation_index=0.0,
                next_assessment_date=datetime.now(timezone.utc) + timedelta(hours=24),
                confidence_level=0.0
            )
    
    def get_evolution_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive evolution metrics and statistics.
        
        Returns:
            Dictionary with evolution metrics
        """
        total_recommendations = len(self.active_recommendations) + len(self.implemented_improvements)
        
        current_health = self.historical_health[-1] if self.historical_health else None
        
        return {
            "current_ecosystem_health": current_health.overall_health.value if current_health else "unknown",
            "active_recommendations": len(self.active_recommendations),
            "implemented_improvements": len(self.implemented_improvements),
            "total_recommendations": total_recommendations,
            "learning_patterns_detected": len(self.learning_patterns),
            "agent_performance_tracked": len(self.agent_performances),
            "system_adaptation_score": self._calculate_system_adaptation_score(),
            "innovation_index": self._calculate_innovation_index(),
            "evolution_reports_generated": len(self.evolution_reports),
            "last_analysis": self.last_analysis_time.isoformat(),
            "last_health_check": self.last_health_check.isoformat(),
            "auto_optimization_enabled": self.config.auto_optimization_enabled,
            "recommendation_implementation_rate": self._calculate_implementation_rate(),
            "average_recommendation_confidence": self._calculate_average_recommendation_confidence(),
            "health_trend": self._calculate_health_trend(),
            "performance_improvement_rate": self._calculate_performance_improvement_rate(),
            "configuration": self.config.dict()
        }
    
    # Private helper methods
    
    def _initialize_baseline_metrics(self):
        """Initialize baseline metrics for comparison."""
        self.baseline_metrics = {
            "system_throughput": 1.0,
            "average_response_time": 100.0,  # milliseconds
            "error_rate": 0.05,
            "resource_utilization": 0.7,
            "innovation_rate": 0.1,
            "adaptation_speed": 0.5,
            "quality_score": 0.8,
            "efficiency_index": 0.75
        }
    
    def _assess_component_health(self) -> Dict[str, HealthStatus]:
        """Assess health of individual system components."""
        component_health = {}
        
        # Assess agent health
        for agent_name, performance in self.agent_performances.items():
            component_health[f"agent_{agent_name}"] = performance.health_status
        
        # Assess system-level components
        system_components = ["idea_pipeline", "council_process", "implementation_workflow", "feedback_loop"]
        
        for component in system_components:
            # Simplified health assessment based on metrics
            health = self._assess_system_component_health(component)
            component_health[component] = health
        
        return component_health
    
    def _assess_system_component_health(self, component: str) -> HealthStatus:
        """Assess health of a specific system component."""
        # Simplified assessment - in real implementation, this would use actual metrics
        if component in self.system_metrics:
            metric = self.system_metrics[component]
            if metric.current_value >= 0.9:
                return HealthStatus.EXCELLENT
            elif metric.current_value >= 0.8:
                return HealthStatus.GOOD
            elif metric.current_value >= 0.6:
                return HealthStatus.FAIR
            elif metric.current_value >= 0.4:
                return HealthStatus.POOR
            else:
                return HealthStatus.CRITICAL
        else:
            return HealthStatus.GOOD  # Default assumption
    
    def _analyze_performance_trends(self) -> Dict[str, str]:
        """Analyze performance trends across components."""
        trends = {}
        
        # Analyze agent performance trends
        for agent_name, performance in self.agent_performances.items():
            trend = self._calculate_agent_trend(performance)
            trends[f"agent_{agent_name}"] = trend
        
        # Analyze system metrics trends
        for metric_name, metric in self.system_metrics.items():
            trend = metric.trend_direction
            trends[metric_name] = trend
        
        return trends
    
    def _calculate_agent_trend(self, performance: AgentPerformance) -> str:
        """Calculate trend for agent performance."""
        if not performance.metrics:
            return "stable"
        
        # Analyze efficiency trend
        efficiency_values = []
        for metric in performance.metrics.values():
            if metric.historical_values:
                efficiency_values.extend(metric.historical_values[-5:])  # Last 5 values
        
        if len(efficiency_values) >= 3:
            recent_avg = statistics.mean(efficiency_values[-3:])
            older_avg = statistics.mean(efficiency_values[:3])
            
            if recent_avg > older_avg * 1.1:
                return "improving"
            elif recent_avg < older_avg * 0.9:
                return "declining"
        
        return "stable"
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify system bottlenecks."""
        bottlenecks = []
        
        # Check agent performance bottlenecks
        for agent_name, performance in self.agent_performances.items():
            if performance.efficiency_score < 0.6:
                bottlenecks.append(f"Low efficiency in {agent_name}")
            
            if performance.error_rate > 0.1:
                bottlenecks.append(f"High error rate in {agent_name}")
        
        # Check system-level bottlenecks
        if len(self.active_recommendations) > 20:
            bottlenecks.append("High number of unimplemented recommendations")
        
        # Check adaptation patterns for bottlenecks
        for pattern in self.learning_patterns.values():
            if pattern.pattern_type in ["workflow_bottleneck", "capacity_strain"]:
                bottlenecks.append(f"Detected {pattern.pattern_type}: {pattern.description}")
        
        return bottlenecks
    
    def _identify_strengths(self) -> List[str]:
        """Identify system strengths."""
        strengths = []
        
        # Check agent performance strengths
        high_performers = [name for name, perf in self.agent_performances.items() 
                          if perf.efficiency_score > 0.9]
        if high_performers:
            strengths.append(f"High-performing agents: {', '.join(high_performers)}")
        
        # Check system-level strengths
        if len(self.implemented_improvements) > 10:
            strengths.append("Strong track record of implementing improvements")
        
        # Check learning patterns for strengths
        positive_patterns = [p for p in self.learning_patterns.values() 
                           if p.impact_score > 0.8 and "improvement" in p.pattern_type]
        if positive_patterns:
            strengths.append("Strong learning and adaptation capabilities")
        
        # Check innovation
        if self._calculate_innovation_index() > 0.7:
            strengths.append("High innovation rate")
        
        return strengths
    
    def _assess_risks(self) -> List[str]:
        """Assess system risks."""
        risks = []
        
        # Performance risks
        declining_agents = [name for name, perf in self.agent_performances.items() 
                          if perf.health_status in [HealthStatus.POOR, HealthStatus.CRITICAL]]
        if declining_agents:
            risks.append(f"Declining agent performance: {', '.join(declining_agents)}")
        
        # Adaptation risks
        if self._calculate_system_adaptation_score() < 0.5:
            risks.append("Low system adaptation capability")
        
        # Innovation risks
        if self._calculate_innovation_index() < 0.3:
            risks.append("Stagnating innovation rate")
        
        # Resource risks
        if len(self.active_recommendations) > 50:
            risks.append("Recommendation backlog may indicate resource constraints")
        
        return risks
    
    def _detect_adaptation_patterns(self) -> List[AdaptationPattern]:
        """Detect adaptation patterns in the system."""
        patterns = []
        
        # Performance drift detection
        if self._detect_performance_drift():
            patterns.append(AdaptationPattern.PERFORMANCE_DRIFT)
        
        # Usage shift detection
        if self._detect_usage_shift():
            patterns.append(AdaptationPattern.USAGE_SHIFT)
        
        # Capacity strain detection
        if self._detect_capacity_strain():
            patterns.append(AdaptationPattern.CAPACITY_STRAIN)
        
        # Efficiency decline detection
        if self._detect_efficiency_decline():
            patterns.append(AdaptationPattern.EFFICIENCY_DECLINE)
        
        # Innovation stagnation detection
        if self._detect_innovation_stagnation():
            patterns.append(AdaptationPattern.INNOVATION_STAGNATION)
        
        return patterns
    
    def _detect_performance_drift(self) -> bool:
        """Detect gradual performance degradation."""
        if len(self.historical_health) < 5:
            return False
        
        recent_scores = [h.confidence_score for h in self.historical_health[-5:]]
        return statistics.mean(recent_scores) < 0.7
    
    def _detect_usage_shift(self) -> bool:
        """Detect shifts in system usage patterns."""
        # Simplified detection based on agent performance changes
        significant_changes = 0
        for performance in self.agent_performances.values():
            if abs(performance.efficiency_score - 0.8) > 0.2:  # Baseline is 0.8
                significant_changes += 1
        
        return significant_changes >= len(self.agent_performances) * 0.3
    
    def _detect_capacity_strain(self) -> bool:
        """Detect system capacity strain."""
        # Check if many agents are showing poor performance
        poor_performers = sum(1 for perf in self.agent_performances.values() 
                            if perf.health_status in [HealthStatus.POOR, HealthStatus.CRITICAL])
        
        return poor_performers >= len(self.agent_performances) * 0.4
    
    def _detect_efficiency_decline(self) -> bool:
        """Detect overall efficiency decline."""
        avg_efficiency = statistics.mean([perf.efficiency_score for perf in self.agent_performances.values()]) if self.agent_performances else 0.8
        return avg_efficiency < 0.6
    
    def _detect_innovation_stagnation(self) -> bool:
        """Detect innovation stagnation."""
        return self._calculate_innovation_index() < 0.4
    
    def _calculate_overall_health(self, component_health: Dict[str, HealthStatus]) -> HealthStatus:
        """Calculate overall ecosystem health."""
        if not component_health:
            return HealthStatus.FAIR
        
        health_scores = {
            HealthStatus.EXCELLENT: 5,
            HealthStatus.GOOD: 4,
            HealthStatus.FAIR: 3,
            HealthStatus.POOR: 2,
            HealthStatus.CRITICAL: 1
        }
        
        total_score = sum(health_scores[status] for status in component_health.values())
        avg_score = total_score / len(component_health)
        
        if avg_score >= 4.5:
            return HealthStatus.EXCELLENT
        elif avg_score >= 3.5:
            return HealthStatus.GOOD
        elif avg_score >= 2.5:
            return HealthStatus.FAIR
        elif avg_score >= 1.5:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def _calculate_health_confidence(self, component_health: Dict[str, HealthStatus]) -> float:
        """Calculate confidence in health assessment."""
        if not component_health:
            return 0.5
        
        # Higher confidence with more data points and consistent health
        data_confidence = min(1.0, len(component_health) / 10)
        
        # Consistency bonus
        health_values = list(component_health.values())
        most_common_health = max(set(health_values), key=health_values.count)
        consistency = health_values.count(most_common_health) / len(health_values)
        
        return (data_confidence + consistency) / 2
    
    def _analyze_agent_learning_patterns(self) -> List[LearningPattern]:
        """Analyze learning patterns specific to agents."""
        patterns = []
        
        for agent_name, performance in self.agent_performances.items():
            # Check for adaptation patterns
            if performance.adaptation_rate > 0.7:
                patterns.append(LearningPattern(
                    pattern_type="high_adaptation",
                    description=f"{agent_name} shows high adaptation rate",
                    frequency=performance.adaptation_rate,
                    impact_score=0.8,
                    components_affected=[agent_name],
                    confidence=0.8
                ))
            
            # Check for performance improvement patterns
            if performance.efficiency_score > 0.9:
                patterns.append(LearningPattern(
                    pattern_type="performance_excellence",
                    description=f"{agent_name} maintains excellent performance",
                    frequency=performance.efficiency_score,
                    impact_score=0.9,
                    components_affected=[agent_name],
                    confidence=0.9
                ))
        
        return patterns
    
    def _analyze_workflow_patterns(self) -> List[LearningPattern]:
        """Analyze workflow adaptation patterns."""
        patterns = []
        
        # Check recommendation implementation patterns
        if len(self.implemented_improvements) > 5:
            implementation_rate = len(self.implemented_improvements) / (len(self.implemented_improvements) + len(self.active_recommendations))
            
            if implementation_rate > 0.7:
                patterns.append(LearningPattern(
                    pattern_type="efficient_implementation",
                    description="High rate of recommendation implementation",
                    frequency=implementation_rate,
                    impact_score=0.8,
                    components_affected=["workflow"],
                    confidence=0.8
                ))
        
        return patterns
    
    def _analyze_improvement_patterns(self) -> List[LearningPattern]:
        """Analyze improvement and optimization patterns."""
        patterns = []
        
        # Check for continuous improvement pattern
        recent_improvements = [imp for imp in self.implemented_improvements 
                             if (datetime.now(timezone.utc) - imp.created_timestamp).days <= 30]
        
        if len(recent_improvements) > 3:
            patterns.append(LearningPattern(
                pattern_type="continuous_improvement",
                description="Consistent implementation of improvements",
                frequency=len(recent_improvements) / 30,  # Per day
                impact_score=0.7,
                components_affected=["system"],
                confidence=0.7
            ))
        
        return patterns
    
    def _analyze_innovation_patterns(self) -> List[LearningPattern]:
        """Analyze innovation and creativity patterns."""
        patterns = []
        
        innovation_index = self._calculate_innovation_index()
        
        if innovation_index > 0.6:
            patterns.append(LearningPattern(
                pattern_type="innovation_excellence",
                description="High innovation and creative solution generation",
                frequency=innovation_index,
                impact_score=0.9,
                components_affected=["ecosystem"],
                confidence=0.8
            ))
        
        return patterns
    
    def _generate_health_based_recommendations(self, health: EcosystemHealth) -> List[EvolutionRecommendation]:
        """Generate recommendations based on ecosystem health."""
        recommendations = []
        
        # Address bottlenecks
        for bottleneck in health.bottlenecks:
            rec = EvolutionRecommendation(
                id=f"bottleneck_{hash(bottleneck) % 1000}",
                title=f"Address Bottleneck: {bottleneck[:50]}",
                description=f"Resolve identified bottleneck: {bottleneck}",
                improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                scope=EvolutionScope.SYSTEM_LEVEL,
                priority=EvolutionPriority.HIGH,
                expected_impact={"performance": 0.3, "efficiency": 0.2},
                implementation_steps=[
                    "Analyze root cause of bottleneck",
                    "Design optimization solution",
                    "Implement and test solution",
                    "Monitor improvement"
                ],
                timeline_estimate="2-4 weeks",
                success_metrics=["Reduced bottleneck impact", "Improved throughput"],
                confidence_score=0.7
            )
            recommendations.append(rec)
        
        # Address critical health issues
        if health.overall_health in [HealthStatus.POOR, HealthStatus.CRITICAL]:
            rec = EvolutionRecommendation(
                id="critical_health_recovery",
                title="Critical Health Recovery Plan",
                description="Comprehensive plan to address critical ecosystem health issues",
                improvement_type=ImprovementType.QUALITY_ENHANCEMENT,
                scope=EvolutionScope.ECOSYSTEM_LEVEL,
                priority=EvolutionPriority.IMMEDIATE,
                expected_impact={"health": 0.5, "stability": 0.4},
                implementation_steps=[
                    "Emergency health assessment",
                    "Identify critical failure points",
                    "Implement emergency fixes",
                    "Establish monitoring",
                    "Plan comprehensive recovery"
                ],
                timeline_estimate="1-2 weeks",
                success_metrics=["Health status improvement", "Reduced critical issues"],
                confidence_score=0.8
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_performance_recommendations(self) -> List[EvolutionRecommendation]:
        """Generate performance-based recommendations."""
        recommendations = []
        
        # Address poor-performing agents
        poor_agents = [name for name, perf in self.agent_performances.items() 
                      if perf.efficiency_score < 0.6]
        
        for agent_name in poor_agents:
            rec = EvolutionRecommendation(
                id=f"improve_agent_{agent_name}",
                title=f"Improve {agent_name} Performance",
                description=f"Optimize performance and efficiency of {agent_name} agent",
                improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                scope=EvolutionScope.AGENT_LEVEL,
                priority=EvolutionPriority.HIGH,
                expected_impact={"agent_efficiency": 0.4, "system_performance": 0.2},
                implementation_steps=[
                    f"Analyze {agent_name} performance metrics",
                    "Identify optimization opportunities",
                    "Implement performance improvements",
                    "Monitor and validate improvements"
                ],
                timeline_estimate="1-3 weeks",
                success_metrics=[f"{agent_name} efficiency > 0.8", "Reduced error rate"],
                confidence_score=0.8
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_learning_based_recommendations(self) -> List[EvolutionRecommendation]:
        """Generate recommendations based on learning patterns."""
        recommendations = []
        
        # Leverage successful learning patterns
        high_impact_patterns = [p for p in self.learning_patterns.values() 
                              if p.impact_score > 0.8]
        
        for pattern in high_impact_patterns:
            rec = EvolutionRecommendation(
                id=f"leverage_pattern_{hash(pattern.pattern_type) % 1000}",
                title=f"Leverage {pattern.pattern_type.replace('_', ' ').title()}",
                description=f"Expand successful pattern: {pattern.description}",
                improvement_type=ImprovementType.CAPABILITY_EXPANSION,
                scope=EvolutionScope.SYSTEM_LEVEL,
                priority=EvolutionPriority.MEDIUM,
                expected_impact={"learning_rate": 0.3, "adaptation": 0.2},
                implementation_steps=[
                    "Analyze successful pattern components",
                    "Design expansion strategy",
                    "Implement pattern in new areas",
                    "Monitor effectiveness"
                ],
                timeline_estimate="2-6 weeks",
                success_metrics=["Pattern adoption rate", "Improved learning metrics"],
                confidence_score=pattern.confidence
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_innovation_recommendations(self) -> List[EvolutionRecommendation]:
        """Generate innovation-focused recommendations."""
        recommendations = []
        
        innovation_index = self._calculate_innovation_index()
        
        if innovation_index < 0.5:
            rec = EvolutionRecommendation(
                id="boost_innovation",
                title="Innovation Capability Enhancement",
                description="Enhance system innovation and creative problem-solving capabilities",
                improvement_type=ImprovementType.INNOVATION_BOOST,
                scope=EvolutionScope.ECOSYSTEM_LEVEL,
                priority=EvolutionPriority.MEDIUM,
                expected_impact={"innovation_rate": 0.4, "solution_quality": 0.3},
                implementation_steps=[
                    "Analyze innovation barriers",
                    "Design innovation enhancement mechanisms",
                    "Implement creative thinking processes",
                    "Establish innovation metrics"
                ],
                timeline_estimate="4-8 weeks",
                success_metrics=["Innovation index > 0.7", "Novel solution rate"],
                confidence_score=0.7
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_predictive_recommendations(self) -> List[EvolutionRecommendation]:
        """Generate predictive recommendations based on trend analysis."""
        recommendations = []
        
        # Predict future bottlenecks
        potential_bottlenecks = self._predict_future_bottlenecks()
        
        for bottleneck in potential_bottlenecks:
            rec = EvolutionRecommendation(
                id=f"prevent_bottleneck_{hash(bottleneck) % 1000}",
                title=f"Prevent Future Bottleneck: {bottleneck[:30]}",
                description=f"Proactively address predicted bottleneck: {bottleneck}",
                improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                scope=EvolutionScope.SYSTEM_LEVEL,
                priority=EvolutionPriority.LOW,
                expected_impact={"future_performance": 0.3, "stability": 0.2},
                implementation_steps=[
                    "Validate bottleneck prediction",
                    "Design preventive measures",
                    "Implement proactive solutions",
                    "Monitor for bottleneck emergence"
                ],
                timeline_estimate="3-6 weeks",
                success_metrics=["Bottleneck prevention", "Maintained performance"],
                confidence_score=0.6
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _predict_future_bottlenecks(self) -> List[str]:
        """Predict potential future bottlenecks."""
        predictions = []
        
        # Analyze trends in agent performance
        declining_trends = []
        for agent_name, performance in self.agent_performances.items():
            if self._calculate_agent_trend(performance) == "declining":
                declining_trends.append(f"Potential {agent_name} performance degradation")
        
        predictions.extend(declining_trends)
        
        # Predict capacity issues
        if len(self.active_recommendations) > 30:
            predictions.append("Recommendation processing capacity strain")
        
        return predictions
    
    def _filter_and_prioritize_recommendations(self, recommendations: List[EvolutionRecommendation]) -> List[EvolutionRecommendation]:
        """Filter and prioritize recommendations."""
        # Filter by confidence threshold
        filtered = [r for r in recommendations if r.confidence_score >= self.config.recommendation_threshold]
        
        # Sort by priority and confidence
        priority_order = {
            EvolutionPriority.IMMEDIATE: 5,
            EvolutionPriority.HIGH: 4,
            EvolutionPriority.MEDIUM: 3,
            EvolutionPriority.LOW: 2,
            EvolutionPriority.EXPERIMENTAL: 1
        }
        
        filtered.sort(key=lambda r: (priority_order[r.priority], r.confidence_score), reverse=True)
        
        # Limit to reasonable number
        return filtered[:20]
    
    def _can_auto_implement(self, recommendation: EvolutionRecommendation) -> bool:
        """Check if recommendation can be automatically implemented."""
        # Simple heuristics for auto-implementation
        auto_implementable_types = [
            ImprovementType.PERFORMANCE_OPTIMIZATION,
            ImprovementType.EFFICIENCY_IMPROVEMENT
        ]
        
        return (recommendation.improvement_type in auto_implementable_types and
                recommendation.scope == EvolutionScope.AGENT_LEVEL and
                len(recommendation.risks) <= 1)
    
    def _implement_recommendation(self, recommendation: EvolutionRecommendation) -> bool:
        """Implement a recommendation automatically."""
        # Placeholder for actual implementation logic
        # In real system, this would trigger actual changes
        self.logger.info(f"Auto-implementing recommendation: {recommendation.title}")
        return True
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from historical values."""
        if len(values) < 3:
            return "stable"
        
        recent = statistics.mean(values[-3:])
        older = statistics.mean(values[:3])
        
        if recent > older * 1.1:
            return "improving"
        elif recent < older * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _calculate_agent_efficiency(self, performance: AgentPerformance) -> float:
        """Calculate agent efficiency score."""
        if not performance.metrics:
            return 0.8  # Default
        
        # Simplified calculation based on available metrics
        efficiency_factors = []
        
        # Error rate factor (lower is better)
        error_factor = max(0, 1 - performance.error_rate * 2)
        efficiency_factors.append(error_factor)
        
        # Throughput factor
        throughput_factor = min(1.0, performance.throughput)
        efficiency_factors.append(throughput_factor)
        
        # Metrics-based factors
        for metric in performance.metrics.values():
            if metric.target_value:
                achievement = min(1.0, metric.current_value / metric.target_value)
                efficiency_factors.append(achievement)
            else:
                # Assume higher values are better for most metrics
                normalized = min(1.0, metric.current_value / 10.0)
                efficiency_factors.append(normalized)
        
        return statistics.mean(efficiency_factors) if efficiency_factors else 0.8
    
    def _calculate_adaptation_rate(self, performance: AgentPerformance) -> float:
        """Calculate agent adaptation rate."""
        # Simplified calculation based on performance changes
        adaptation_indicators = 0
        total_metrics = len(performance.metrics)
        
        if total_metrics == 0:
            return 0.5
        
        for metric in performance.metrics.values():
            if metric.trend_direction == "improving":
                adaptation_indicators += 1
        
        return adaptation_indicators / total_metrics
    
    def _assess_agent_health(self, performance: AgentPerformance) -> HealthStatus:
        """Assess individual agent health."""
        if performance.efficiency_score >= 0.9:
            return HealthStatus.EXCELLENT
        elif performance.efficiency_score >= 0.8:
            return HealthStatus.GOOD
        elif performance.efficiency_score >= 0.6:
            return HealthStatus.FAIR
        elif performance.efficiency_score >= 0.4:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def _calculate_system_adaptation_score(self) -> float:
        """Calculate overall system adaptation score."""
        if not self.agent_performances:
            return 0.5
        
        adaptation_rates = [perf.adaptation_rate for perf in self.agent_performances.values()]
        return statistics.mean(adaptation_rates)
    
    def _calculate_innovation_index(self) -> float:
        """Calculate innovation index based on various factors."""
        factors = []
        
        # Recommendation novelty factor
        if self.active_recommendations:
            innovation_recs = [r for r in self.active_recommendations.values() 
                             if r.improvement_type == ImprovementType.INNOVATION_BOOST]
            innovation_factor = len(innovation_recs) / len(self.active_recommendations)
            factors.append(innovation_factor)
        
        # Learning pattern diversity
        if self.learning_patterns:
            unique_types = len(set(p.pattern_type for p in self.learning_patterns.values()))
            diversity_factor = min(1.0, unique_types / 5.0)  # Normalize to max 5 types
            factors.append(diversity_factor)
        
        # Implementation creativity (based on successful improvements)
        if self.implemented_improvements:
            creative_improvements = [i for i in self.implemented_improvements 
                                   if i.improvement_type in [ImprovementType.INNOVATION_BOOST, 
                                                           ImprovementType.CAPABILITY_EXPANSION]]
            creativity_factor = len(creative_improvements) / len(self.implemented_improvements)
            factors.append(creativity_factor)
        
        return statistics.mean(factors) if factors else 0.5
    
    def _calculate_efficiency_trends(self, days: int) -> Dict[str, float]:
        """Calculate efficiency trends over specified period."""
        trends = {}
        
        # Agent efficiency trends
        for agent_name, performance in self.agent_performances.items():
            trends[f"agent_{agent_name}_efficiency"] = performance.efficiency_score
        
        # System-level efficiency
        if self.agent_performances:
            avg_efficiency = statistics.mean([p.efficiency_score for p in self.agent_performances.values()])
            trends["system_efficiency"] = avg_efficiency
        
        # Recommendation implementation efficiency
        if self.implemented_improvements and self.active_recommendations:
            impl_rate = len(self.implemented_improvements) / (len(self.implemented_improvements) + len(self.active_recommendations))
            trends["implementation_efficiency"] = impl_rate
        
        return trends
    
    def _generate_key_insights(self, health: EcosystemHealth, patterns: List[LearningPattern]) -> List[str]:
        """Generate key insights from analysis."""
        insights = []
        
        # Health insights
        if health.overall_health == HealthStatus.EXCELLENT:
            insights.append("Ecosystem is operating at peak performance")
        elif health.overall_health in [HealthStatus.POOR, HealthStatus.CRITICAL]:
            insights.append("Ecosystem requires immediate attention and intervention")
        
        # Pattern insights
        high_impact_patterns = [p for p in patterns if p.impact_score > 0.8]
        if high_impact_patterns:
            insights.append(f"Identified {len(high_impact_patterns)} high-impact learning patterns")
        
        # Performance insights
        if self.agent_performances:
            avg_efficiency = statistics.mean([p.efficiency_score for p in self.agent_performances.values()])
            if avg_efficiency > 0.9:
                insights.append("Agents are performing exceptionally well")
            elif avg_efficiency < 0.6:
                insights.append("Agent performance needs significant improvement")
        
        # Innovation insights
        innovation_index = self._calculate_innovation_index()
        if innovation_index > 0.8:
            insights.append("System shows strong innovation capabilities")
        elif innovation_index < 0.4:
            insights.append("Innovation capabilities need enhancement")
        
        return insights
    
    def _generate_action_items(self, recommendations: List[EvolutionRecommendation]) -> List[str]:
        """Generate action items from recommendations."""
        action_items = []
        
        # Immediate actions
        immediate_recs = [r for r in recommendations if r.priority == EvolutionPriority.IMMEDIATE]
        for rec in immediate_recs:
            action_items.append(f"IMMEDIATE: {rec.title}")
        
        # High priority actions
        high_priority_recs = [r for r in recommendations if r.priority == EvolutionPriority.HIGH]
        for rec in high_priority_recs[:3]:  # Top 3
            action_items.append(f"HIGH: {rec.title}")
        
        # System-level actions
        system_recs = [r for r in recommendations if r.scope == EvolutionScope.SYSTEM_LEVEL]
        if system_recs:
            action_items.append(f"Review {len(system_recs)} system-level recommendations")
        
        return action_items
    
    def _create_agent_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Create summary of agent performances."""
        summary = {}
        
        for agent_name, performance in self.agent_performances.items():
            summary[agent_name] = {
                "health_status": performance.health_status.value,
                "efficiency_score": performance.efficiency_score,
                "adaptation_rate": performance.adaptation_rate,
                "error_rate": performance.error_rate,
                "throughput": performance.throughput,
                "trend": self._calculate_agent_trend(performance),
                "last_assessment": performance.last_assessment.isoformat()
            }
        
        return summary
    
    def _calculate_report_confidence(self, health: EcosystemHealth, patterns: List[LearningPattern]) -> float:
        """Calculate confidence in the evolution report."""
        factors = []
        
        # Health assessment confidence
        factors.append(health.confidence_score)
        
        # Pattern detection confidence
        if patterns:
            pattern_confidences = [p.confidence for p in patterns]
            factors.append(statistics.mean(pattern_confidences))
        
        # Data availability factor
        data_factor = min(1.0, len(self.agent_performances) / 5.0)  # Assume 5 agents is good
        factors.append(data_factor)
        
        # Historical data factor
        history_factor = min(1.0, len(self.historical_health) / 10.0)  # 10 historical points is good
        factors.append(history_factor)
        
        return statistics.mean(factors)
    
    def _calculate_implementation_rate(self) -> float:
        """Calculate rate of recommendation implementation."""
        total = len(self.implemented_improvements) + len(self.active_recommendations)
        if total == 0:
            return 0.0
        return len(self.implemented_improvements) / total
    
    def _calculate_average_recommendation_confidence(self) -> float:
        """Calculate average confidence of recommendations."""
        all_recs = list(self.active_recommendations.values()) + self.implemented_improvements
        if not all_recs:
            return 0.0
        
        confidences = [rec.confidence_score for rec in all_recs]
        return statistics.mean(confidences)
    
    def _calculate_health_trend(self) -> str:
        """Calculate overall health trend."""
        if len(self.historical_health) < 3:
            return "insufficient_data"
        
        recent_health = self.historical_health[-3:]
        health_scores = {
            HealthStatus.EXCELLENT: 5,
            HealthStatus.GOOD: 4,
            HealthStatus.FAIR: 3,
            HealthStatus.POOR: 2,
            HealthStatus.CRITICAL: 1
        }
        
        scores = [health_scores[h.overall_health] for h in recent_health]
        
        if scores[-1] > scores[0]:
            return "improving"
        elif scores[-1] < scores[0]:
            return "declining"
        else:
            return "stable"
    
    def _calculate_performance_improvement_rate(self) -> float:
        """Calculate rate of performance improvement."""
        if not self.agent_performances:
            return 0.0
        
        improving_agents = sum(1 for perf in self.agent_performances.values() 
                             if self._calculate_agent_trend(perf) == "improving")
        
        return improving_agents / len(self.agent_performances)


# Factory function for easy agent creation
def create_evolution_agent(config: Optional[EvolutionConfiguration] = None) -> EvolutionAgent:
    """
    Create a new Evolution Agent instance.
    
    Args:
        config: Optional evolution configuration
        
    Returns:
        Configured EvolutionAgent instance
    """
    return EvolutionAgent(config)


# Export main classes and functions
__all__ = [
    "EvolutionAgent",
    "EvolutionConfiguration",
    "EvolutionReport",
    "EcosystemHealth",
    "EvolutionRecommendation",
    "LearningPattern",
    "AgentPerformance",
    "PerformanceMetric",
    "EvolutionScope",
    "AdaptationPattern",
    "HealthStatus",
    "ImprovementType",
    "EvolutionPriority",
    "create_evolution_agent"
]