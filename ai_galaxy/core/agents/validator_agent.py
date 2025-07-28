"""
Validator Agent for the AI-Galaxy Lower Meta-Layer.

This module implements the Validator Agent, responsible for pre-filtering ideas
before they reach the Council. It validates idea completeness, feasibility,
format compliance, checks for duplicates and conflicts, and performs initial
quality assessment and categorization.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID
import hashlib

from pydantic import BaseModel, Field

from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext
from ..state_machine.router import StateMachineRouter, TransitionResult


class ValidationResult(str, Enum):
    """Results of idea validation."""
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


class ValidationCriteria(str, Enum):
    """Validation criteria for ideas."""
    COMPLETENESS = "completeness"
    FORMAT_COMPLIANCE = "format_compliance"
    FEASIBILITY = "feasibility"
    DUPLICATE_CHECK = "duplicate_check"
    CONFLICT_CHECK = "conflict_check"
    QUALITY_ASSESSMENT = "quality_assessment"
    CATEGORIZATION = "categorization"
    METADATA_VALIDATION = "metadata_validation"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationIssue(BaseModel):
    """Individual validation issue found in an idea."""
    criteria: ValidationCriteria
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None
    field: Optional[str] = None
    code: str
    auto_fixable: bool = False


class ValidationReport(BaseModel):
    """Comprehensive validation report for an idea."""
    idea_id: str
    validation_timestamp: datetime
    overall_result: ValidationResult
    passed_criteria: List[ValidationCriteria] = Field(default_factory=list)
    failed_criteria: List[ValidationCriteria] = Field(default_factory=list)
    issues: List[ValidationIssue] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    suggested_category: Optional[str] = None
    suggested_priority: Optional[int] = None
    duplicate_matches: List[str] = Field(default_factory=list)
    conflict_matches: List[str] = Field(default_factory=list)
    auto_fixes_applied: List[str] = Field(default_factory=list)
    routing_recommendation: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class ValidatorConfiguration(BaseModel):
    """Configuration for the Validator Agent."""
    auto_fix_enabled: bool = True
    duplicate_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    completeness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_description_length: int = Field(default=2000, ge=100)
    min_description_length: int = Field(default=50, ge=10)
    max_title_length: int = Field(default=200, ge=10)
    min_title_length: int = Field(default=10, ge=5)
    enable_feasibility_check: bool = True
    enable_conflict_detection: bool = True
    strict_format_validation: bool = False
    auto_categorization_enabled: bool = True
    priority_validation_enabled: bool = True
    metadata_validation_enabled: bool = True


class DuplicateCandidate(BaseModel):
    """Potential duplicate idea candidate."""
    idea_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    matching_aspects: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ConflictCandidate(BaseModel):
    """Potential conflicting idea candidate."""
    idea_id: str
    conflict_type: str
    conflict_description: str
    severity: ValidationSeverity
    resolution_suggestion: str


class ValidatorAgent:
    """
    The Validator Agent - quality gatekeeper for the AI-Galaxy ecosystem.
    
    Pre-filters ideas before they reach the Council through comprehensive
    validation including completeness checks, format compliance, duplicate
    detection, conflict analysis, and initial quality assessment.
    """
    
    def __init__(self, config: Optional[ValidatorConfiguration] = None,
                 state_router: Optional[StateMachineRouter] = None):
        """
        Initialize the Validator Agent.
        
        Args:
            config: Validator configuration parameters
            state_router: State machine router for workflow transitions
        """
        self.logger = get_logger("validator_agent")
        self.config = config or ValidatorConfiguration()
        self.state_router = state_router
        
        # Initialize validation rules and patterns
        self.validation_rules = self._initialize_validation_rules()
        self.keyword_patterns = self._initialize_keyword_patterns()
        self.category_classifiers = self._initialize_category_classifiers()
        
        # Tracking and metrics
        self.validation_history: Dict[str, ValidationReport] = {}
        self.known_ideas: Dict[str, Idea] = {}  # For duplicate detection
        self.validation_metrics = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "auto_fixes_applied": 0,
            "duplicates_detected": 0,
            "conflicts_detected": 0,
            "average_validation_time": 0.0,
            "average_quality_score": 0.0
        }
        
        self.logger.agent_action("validator_agent_initialized", "validator_agent",
                                additional_context={
                                    "auto_fix_enabled": self.config.auto_fix_enabled,
                                    "quality_threshold": self.config.quality_threshold,
                                    "duplicate_threshold": self.config.duplicate_threshold
                                })
    
    def validate_idea(self, idea: Idea) -> ValidationReport:
        """
        Perform comprehensive validation of an idea.
        
        Args:
            idea: The idea to validate
            
        Returns:
            Comprehensive validation report
        """
        start_time = datetime.now(timezone.utc)
        idea_id = str(idea.id)
        
        context = LogContext(
            agent_name="validator_agent",
            idea_id=idea_id,
            additional_context={"validation_start": start_time.isoformat()}
        )
        
        self.logger.agent_action("starting_idea_validation", "validator_agent", idea_id)
        
        try:
            # Initialize validation report
            report = ValidationReport(
                idea_id=idea_id,
                validation_timestamp=start_time,
                overall_result=ValidationResult.PASSED,
                quality_score=0.0,
                completeness_score=0.0,
                routing_recommendation="council_review",
                confidence_score=0.0
            )
            
            # Run validation criteria
            self._validate_completeness(idea, report)
            self._validate_format_compliance(idea, report)
            self._validate_quality_assessment(idea, report)
            self._validate_categorization(idea, report)
            self._validate_metadata(idea, report)
            
            if self.config.enable_feasibility_check:
                self._validate_feasibility(idea, report)
            
            if self.config.enable_conflict_detection:
                self._validate_conflicts(idea, report)
            
            # Always run duplicate check
            self._validate_duplicates(idea, report)
            
            # Apply auto-fixes if enabled
            if self.config.auto_fix_enabled:
                self._apply_auto_fixes(idea, report)
            
            # Determine overall result
            self._determine_overall_result(report)
            
            # Calculate confidence score
            report.confidence_score = self._calculate_confidence_score(report)
            
            # Store validation history
            self.validation_history[idea_id] = report
            self.known_ideas[idea_id] = idea
            
            # Update metrics
            self._update_validation_metrics(report, start_time)
            
            self.logger.agent_action("idea_validation_completed", "validator_agent", idea_id, {
                "result": report.overall_result.value,
                "quality_score": report.quality_score,
                "issues_count": len(report.issues),
                "validation_duration": (datetime.now(timezone.utc) - start_time).total_seconds()
            })
            
            return report
            
        except Exception as e:
            self.logger.error(f"Idea validation failed: {e}", context, exc_info=True)
            
            # Return failure report
            return ValidationReport(
                idea_id=idea_id,
                validation_timestamp=start_time,
                overall_result=ValidationResult.FAILED,
                quality_score=0.0,
                completeness_score=0.0,
                issues=[ValidationIssue(
                    criteria=ValidationCriteria.QUALITY_ASSESSMENT,
                    severity=ValidationSeverity.CRITICAL,
                    message="Validation process failed due to system error",
                    code="VALIDATION_ERROR",
                    auto_fixable=False
                )],
                routing_recommendation="manual_review",
                confidence_score=0.0
            )
    
    def update_idea_status(self, idea: Idea, validation_report: ValidationReport) -> bool:
        """
        Update idea status based on validation results using state machine.
        
        Args:
            idea: The idea to update
            validation_report: The validation report with results
            
        Returns:
            True if status update successful, False otherwise
        """
        idea_id = str(idea.id)
        current_state = idea.status.value
        
        context = LogContext(
            agent_name="validator_agent",
            idea_id=idea_id,
            additional_context={
                "current_state": current_state,
                "validation_result": validation_report.overall_result.value
            }
        )
        
        try:
            # Determine target state based on validation result
            if validation_report.overall_result == ValidationResult.PASSED:
                target_state = "validated"
            elif validation_report.overall_result == ValidationResult.NEEDS_REVISION:
                target_state = "created"  # Back to creation for revision
            elif validation_report.overall_result in [ValidationResult.DUPLICATE, ValidationResult.CONFLICT]:
                target_state = "rejected"
            else:
                target_state = "rejected"
            
            # Update idea metadata with validation results
            idea.metadata.update({
                "validation": {
                    "result": validation_report.overall_result.value,
                    "quality_score": validation_report.quality_score,
                    "completeness_score": validation_report.completeness_score,
                    "validation_timestamp": validation_report.validation_timestamp.isoformat(),
                    "issues_count": len(validation_report.issues),
                    "auto_fixes_applied": validation_report.auto_fixes_applied,
                    "suggested_category": validation_report.suggested_category,
                    "suggested_priority": validation_report.suggested_priority,
                    "routing_recommendation": validation_report.routing_recommendation,
                    "confidence_score": validation_report.confidence_score
                }
            })
            
            # Execute state transition if router available
            if self.state_router:
                result = self.state_router.execute_transition(idea, target_state, "validator_agent")
                
                if result == TransitionResult.SUCCESS:
                    self.logger.state_transition(current_state, target_state, idea_id,
                                               "validator_agent", f"Validation result: {validation_report.overall_result.value}")
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
    
    def batch_validate_ideas(self, ideas: List[Idea]) -> List[ValidationReport]:
        """
        Validate multiple ideas in batch for efficiency.
        
        Args:
            ideas: List of ideas to validate
            
        Returns:
            List of validation reports
        """
        self.logger.agent_action("starting_batch_validation", "validator_agent",
                                additional_context={"batch_size": len(ideas)})
        
        reports = []
        for idea in ideas:
            report = self.validate_idea(idea)
            reports.append(report)
        
        # Analyze batch results
        passed_count = sum(1 for r in reports if r.overall_result == ValidationResult.PASSED)
        failed_count = len(reports) - passed_count
        
        self.logger.agent_action("batch_validation_completed", "validator_agent",
                                additional_context={
                                    "total_ideas": len(ideas),
                                    "passed": passed_count,
                                    "failed": failed_count,
                                    "success_rate": (passed_count / len(ideas)) * 100
                                })
        
        return reports
    
    def get_validation_summary(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of validation results for a specific idea.
        
        Args:
            idea_id: ID of the idea
            
        Returns:
            Summary dictionary or None if no validation found
        """
        report = self.validation_history.get(idea_id)
        if not report:
            return None
        
        return {
            "idea_id": idea_id,
            "validation_result": report.overall_result.value,
            "quality_score": report.quality_score,
            "completeness_score": report.completeness_score,
            "issues_count": len(report.issues),
            "critical_issues": len([i for i in report.issues if i.severity == ValidationSeverity.CRITICAL]),
            "high_issues": len([i for i in report.issues if i.severity == ValidationSeverity.HIGH]),
            "auto_fixes_applied": len(report.auto_fixes_applied),
            "suggested_category": report.suggested_category,
            "suggested_priority": report.suggested_priority,
            "routing_recommendation": report.routing_recommendation,
            "confidence_score": report.confidence_score,
            "validation_timestamp": report.validation_timestamp.isoformat()
        }
    
    def get_validator_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about validator performance.
        
        Returns:
            Dictionary with validator performance metrics
        """
        total_validations = self.validation_metrics["total_validations"]
        
        if total_validations == 0:
            return {**self.validation_metrics, "pass_rate_percent": 0.0}
        
        # Calculate pass rate
        pass_rate = (self.validation_metrics["passed_validations"] / total_validations) * 100
        
        # Issue distribution
        issue_distribution = self._calculate_issue_distribution()
        
        # Recent performance (last 30 days)
        recent_reports = [report for report in self.validation_history.values()
                         if (datetime.now(timezone.utc) - report.validation_timestamp).days <= 30]
        
        recent_quality_scores = [r.quality_score for r in recent_reports if r.quality_score > 0]
        avg_recent_quality = sum(recent_quality_scores) / len(recent_quality_scores) if recent_quality_scores else 0.0
        
        return {
            **self.validation_metrics,
            "pass_rate_percent": pass_rate,
            "recent_validations_30_days": len(recent_reports),
            "recent_average_quality": avg_recent_quality,
            "issue_distribution": issue_distribution,
            "most_common_issues": self._get_most_common_issues(),
            "auto_fix_success_rate": self._calculate_auto_fix_success_rate(),
            "category_accuracy": self._calculate_category_accuracy(),
            "configuration": self.config.dict()
        }
    
    # Private validation methods
    
    def _validate_completeness(self, idea: Idea, report: ValidationReport):
        """Validate idea completeness."""
        issues = []
        score_components = []
        
        # Title validation
        if not idea.title or len(idea.title.strip()) < self.config.min_title_length:
            issues.append(ValidationIssue(
                criteria=ValidationCriteria.COMPLETENESS,
                severity=ValidationSeverity.CRITICAL,
                message=f"Title must be at least {self.config.min_title_length} characters",
                suggestion="Provide a descriptive title that clearly summarizes the idea",
                field="title",
                code="TITLE_TOO_SHORT",
                auto_fixable=False
            ))
            score_components.append(0.0)
        elif len(idea.title) > self.config.max_title_length:
            issues.append(ValidationIssue(
                criteria=ValidationCriteria.COMPLETENESS,
                severity=ValidationSeverity.MEDIUM,
                message=f"Title should not exceed {self.config.max_title_length} characters",
                suggestion="Shorten the title while maintaining clarity",
                field="title",
                code="TITLE_TOO_LONG",
                auto_fixable=True
            ))
            score_components.append(0.7)
        else:
            score_components.append(1.0)
        
        # Description validation
        if not idea.description or len(idea.description.strip()) < self.config.min_description_length:
            issues.append(ValidationIssue(
                criteria=ValidationCriteria.COMPLETENESS,
                severity=ValidationSeverity.CRITICAL,
                message=f"Description must be at least {self.config.min_description_length} characters",
                suggestion="Provide a detailed description explaining the idea's purpose, benefits, and implementation approach",
                field="description",
                code="DESCRIPTION_TOO_SHORT",
                auto_fixable=False
            ))
            score_components.append(0.0)
        elif len(idea.description) > self.config.max_description_length:
            issues.append(ValidationIssue(
                criteria=ValidationCriteria.COMPLETENESS,
                severity=ValidationSeverity.LOW,
                message=f"Description is very long ({len(idea.description)} characters)",
                suggestion="Consider condensing the description while retaining key information",
                field="description",
                code="DESCRIPTION_TOO_LONG",
                auto_fixable=False
            ))
            score_components.append(0.8)
        else:
            score_components.append(1.0)
        
        # Priority validation
        if idea.priority < 1 or idea.priority > 10:
            issues.append(ValidationIssue(
                criteria=ValidationCriteria.COMPLETENESS,
                severity=ValidationSeverity.MEDIUM,
                message="Priority must be between 1 and 10",
                suggestion="Set priority to a value between 1 (lowest) and 10 (highest)",
                field="priority",
                code="INVALID_PRIORITY",
                auto_fixable=True
            ))
            score_components.append(0.5)
        else:
            score_components.append(1.0)
        
        # Calculate completeness score
        report.completeness_score = sum(score_components) / len(score_components)
        
        # Add issues to report
        report.issues.extend(issues)
        
        # Update criteria status
        if report.completeness_score >= self.config.completeness_threshold:
            report.passed_criteria.append(ValidationCriteria.COMPLETENESS)
        else:
            report.failed_criteria.append(ValidationCriteria.COMPLETENESS)
    
    def _validate_format_compliance(self, idea: Idea, report: ValidationReport):
        """Validate format compliance."""
        issues = []
        
        # Title format checks
        if self.config.strict_format_validation:
            # Check for excessive capitalization
            if idea.title.isupper() and len(idea.title) > 10:
                issues.append(ValidationIssue(
                    criteria=ValidationCriteria.FORMAT_COMPLIANCE,
                    severity=ValidationSeverity.LOW,
                    message="Title should not be in all caps",
                    suggestion="Use proper title case formatting",
                    field="title",
                    code="TITLE_ALL_CAPS",
                    auto_fixable=True
                ))
            
            # Check for excessive punctuation
            if idea.title.count('!') > 1 or idea.title.count('?') > 1:
                issues.append(ValidationIssue(
                    criteria=ValidationCriteria.FORMAT_COMPLIANCE,
                    severity=ValidationSeverity.LOW,
                    message="Title contains excessive punctuation",
                    suggestion="Use minimal punctuation for professional appearance",
                    field="title",
                    code="EXCESSIVE_PUNCTUATION",
                    auto_fixable=True
                ))
        
        # Description format checks
        if '\t' in idea.description:
            issues.append(ValidationIssue(
                criteria=ValidationCriteria.FORMAT_COMPLIANCE,
                severity=ValidationSeverity.LOW,
                message="Description contains tab characters",
                suggestion="Use spaces instead of tabs for consistent formatting",
                field="description",
                code="CONTAINS_TABS",
                auto_fixable=True
            ))
        
        # Check for extremely long paragraphs
        paragraphs = idea.description.split('\n\n')
        long_paragraphs = [p for p in paragraphs if len(p) > 500]
        if long_paragraphs:
            issues.append(ValidationIssue(
                criteria=ValidationCriteria.FORMAT_COMPLIANCE,
                severity=ValidationSeverity.LOW,
                message="Description contains very long paragraphs",
                suggestion="Break long paragraphs into shorter, more readable sections",
                field="description",
                code="LONG_PARAGRAPHS",
                auto_fixable=False
            ))
        
        # Add issues to report
        report.issues.extend(issues)
        
        # Format compliance passes if no critical issues
        critical_format_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        if not critical_format_issues:
            report.passed_criteria.append(ValidationCriteria.FORMAT_COMPLIANCE)
        else:
            report.failed_criteria.append(ValidationCriteria.FORMAT_COMPLIANCE)
    
    def _validate_quality_assessment(self, idea: Idea, report: ValidationReport):
        """Perform quality assessment."""
        quality_components = []
        
        # Content depth assessment
        depth_score = self._assess_content_depth(idea)
        quality_components.append(depth_score)
        
        # Clarity assessment
        clarity_score = self._assess_clarity(idea)
        quality_components.append(clarity_score)
        
        # Innovation potential assessment
        innovation_score = self._assess_innovation_potential(idea)
        quality_components.append(innovation_score)
        
        # Feasibility assessment
        feasibility_score = self._assess_initial_feasibility(idea)
        quality_components.append(feasibility_score)
        
        # Calculate overall quality score
        report.quality_score = sum(quality_components) / len(quality_components)
        
        # Add quality-related issues
        if report.quality_score < 0.3:
            report.issues.append(ValidationIssue(
                criteria=ValidationCriteria.QUALITY_ASSESSMENT,
                severity=ValidationSeverity.HIGH,
                message="Idea quality score is very low",
                suggestion="Provide more detail, clarity, and specific implementation approach",
                code="LOW_QUALITY_SCORE",
                auto_fixable=False
            ))
        elif report.quality_score < 0.5:
            report.issues.append(ValidationIssue(
                criteria=ValidationCriteria.QUALITY_ASSESSMENT,
                severity=ValidationSeverity.MEDIUM,
                message="Idea quality could be improved",
                suggestion="Add more specific details and clarify the value proposition",
                code="MEDIUM_QUALITY_SCORE",
                auto_fixable=False
            ))
        
        # Update criteria status
        if report.quality_score >= self.config.quality_threshold:
            report.passed_criteria.append(ValidationCriteria.QUALITY_ASSESSMENT)
        else:
            report.failed_criteria.append(ValidationCriteria.QUALITY_ASSESSMENT)
    
    def _validate_categorization(self, idea: Idea, report: ValidationReport):
        """Validate and suggest categorization."""
        # Automatic categorization
        suggested_category = self._classify_idea_category(idea)
        report.suggested_category = suggested_category
        
        # Priority suggestion
        suggested_priority = self._suggest_priority(idea)
        report.suggested_priority = suggested_priority
        
        # Check existing metadata category
        existing_category = idea.metadata.get("category")
        if existing_category and existing_category != suggested_category:
            report.issues.append(ValidationIssue(
                criteria=ValidationCriteria.CATEGORIZATION,
                severity=ValidationSeverity.INFO,
                message=f"Suggested category '{suggested_category}' differs from existing '{existing_category}'",
                suggestion=f"Consider using suggested category: {suggested_category}",
                field="metadata.category",
                code="CATEGORY_MISMATCH",
                auto_fixable=True
            ))
        
        # Priority validation
        if abs(idea.priority - suggested_priority) > 3:
            report.issues.append(ValidationIssue(
                criteria=ValidationCriteria.CATEGORIZATION,
                severity=ValidationSeverity.LOW,
                message=f"Priority {idea.priority} seems inconsistent with content (suggested: {suggested_priority})",
                suggestion=f"Consider adjusting priority to {suggested_priority}",
                field="priority",
                code="PRIORITY_MISMATCH",
                auto_fixable=True
            ))
        
        # Categorization always passes (it's informational)
        report.passed_criteria.append(ValidationCriteria.CATEGORIZATION)
    
    def _validate_metadata(self, idea: Idea, report: ValidationReport):
        """Validate idea metadata."""
        if not self.config.metadata_validation_enabled:
            report.passed_criteria.append(ValidationCriteria.METADATA_VALIDATION)
            return
        
        issues = []
        
        # Check for required metadata fields
        required_fields = ["category", "estimated_complexity", "target_audience"]
        for field in required_fields:
            if field not in idea.metadata:
                issues.append(ValidationIssue(
                    criteria=ValidationCriteria.METADATA_VALIDATION,
                    severity=ValidationSeverity.LOW,
                    message=f"Missing recommended metadata field: {field}",
                    suggestion=f"Add {field} metadata for better categorization",
                    field=f"metadata.{field}",
                    code="MISSING_METADATA",
                    auto_fixable=True
                ))
        
        # Validate metadata types
        if "estimated_complexity" in idea.metadata:
            complexity = idea.metadata["estimated_complexity"]
            if not isinstance(complexity, (int, float)) or complexity < 1 or complexity > 10:
                issues.append(ValidationIssue(
                    criteria=ValidationCriteria.METADATA_VALIDATION,
                    severity=ValidationSeverity.MEDIUM,
                    message="estimated_complexity should be a number between 1 and 10",
                    suggestion="Set complexity to a value between 1 (simple) and 10 (very complex)",
                    field="metadata.estimated_complexity",
                    code="INVALID_COMPLEXITY",
                    auto_fixable=True
                ))
        
        # Add issues to report
        report.issues.extend(issues)
        
        # Metadata validation passes if no critical issues
        critical_metadata_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        if not critical_metadata_issues:
            report.passed_criteria.append(ValidationCriteria.METADATA_VALIDATION)
        else:
            report.failed_criteria.append(ValidationCriteria.METADATA_VALIDATION)
    
    def _validate_feasibility(self, idea: Idea, report: ValidationReport):
        """Validate initial feasibility."""
        feasibility_issues = []
        
        # Check for unrealistic claims
        unrealistic_keywords = [
            "revolutionary", "breakthrough", "never before", "impossible until now",
            "solves everything", "eliminates all", "perfect solution"
        ]
        
        text_to_check = f"{idea.title} {idea.description}".lower()
        found_unrealistic = [kw for kw in unrealistic_keywords if kw in text_to_check]
        
        if found_unrealistic:
            feasibility_issues.append(ValidationIssue(
                criteria=ValidationCriteria.FEASIBILITY,
                severity=ValidationSeverity.MEDIUM,
                message="Contains potentially unrealistic claims",
                suggestion="Use more measured language and specific, achievable goals",
                code="UNREALISTIC_CLAIMS",
                auto_fixable=False
            ))
        
        # Check for technical feasibility red flags
        technical_red_flags = [
            "artificial general intelligence", "solve consciousness", "time travel",
            "infinite scalability", "zero latency", "100% accuracy always"
        ]
        
        found_red_flags = [flag for flag in technical_red_flags if flag in text_to_check]
        if found_red_flags:
            feasibility_issues.append(ValidationIssue(
                criteria=ValidationCriteria.FEASIBILITY,
                severity=ValidationSeverity.HIGH,
                message="Contains technically unfeasible concepts",
                suggestion="Focus on achievable technical goals with current technology",
                code="TECHNICAL_UNFEASIBLE",
                auto_fixable=False
            ))
        
        # Add issues to report
        report.issues.extend(feasibility_issues)
        
        # Feasibility passes if no high/critical issues
        high_feasibility_issues = [i for i in feasibility_issues 
                                 if i.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]]
        if not high_feasibility_issues:
            report.passed_criteria.append(ValidationCriteria.FEASIBILITY)
        else:
            report.failed_criteria.append(ValidationCriteria.FEASIBILITY)
    
    def _validate_duplicates(self, idea: Idea, report: ValidationReport):
        """Check for duplicate ideas."""
        duplicates = self._find_duplicate_candidates(idea)
        
        if duplicates:
            # Add high-similarity matches to report
            high_similarity_duplicates = [d for d in duplicates 
                                        if d.similarity_score >= self.config.duplicate_threshold]
            
            if high_similarity_duplicates:
                report.duplicate_matches = [d.idea_id for d in high_similarity_duplicates]
                report.issues.append(ValidationIssue(
                    criteria=ValidationCriteria.DUPLICATE_CHECK,
                    severity=ValidationSeverity.HIGH,
                    message=f"Found {len(high_similarity_duplicates)} potential duplicate(s)",
                    suggestion="Review existing similar ideas before proceeding",
                    code="POTENTIAL_DUPLICATE",
                    auto_fixable=False
                ))
                report.failed_criteria.append(ValidationCriteria.DUPLICATE_CHECK)
            else:
                # Low similarity matches - just inform
                report.issues.append(ValidationIssue(
                    criteria=ValidationCriteria.DUPLICATE_CHECK,
                    severity=ValidationSeverity.INFO,
                    message=f"Found {len(duplicates)} similar idea(s) with lower similarity",
                    suggestion="Consider if this adds unique value beyond existing ideas",
                    code="SIMILAR_IDEAS_EXIST",
                    auto_fixable=False
                ))
                report.passed_criteria.append(ValidationCriteria.DUPLICATE_CHECK)
        else:
            report.passed_criteria.append(ValidationCriteria.DUPLICATE_CHECK)
    
    def _validate_conflicts(self, idea: Idea, report: ValidationReport):
        """Check for conflicting ideas."""
        conflicts = self._find_conflict_candidates(idea)
        
        if conflicts:
            high_severity_conflicts = [c for c in conflicts 
                                     if c.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]]
            
            if high_severity_conflicts:
                report.conflict_matches = [c.idea_id for c in high_severity_conflicts]
                report.issues.append(ValidationIssue(
                    criteria=ValidationCriteria.CONFLICT_CHECK,
                    severity=ValidationSeverity.HIGH,
                    message=f"Found {len(high_severity_conflicts)} potential conflict(s)",
                    suggestion="Resolve conflicts with existing ideas or clarify differences",
                    code="POTENTIAL_CONFLICT",
                    auto_fixable=False
                ))
                report.failed_criteria.append(ValidationCriteria.CONFLICT_CHECK)
            else:
                # Low severity conflicts - just inform
                report.issues.append(ValidationIssue(
                    criteria=ValidationCriteria.CONFLICT_CHECK,
                    severity=ValidationSeverity.INFO,
                    message=f"Found {len(conflicts)} minor potential conflict(s)",
                    suggestion="Consider coordination with related ideas",
                    code="MINOR_CONFLICTS",
                    auto_fixable=False
                ))
                report.passed_criteria.append(ValidationCriteria.CONFLICT_CHECK)
        else:
            report.passed_criteria.append(ValidationCriteria.CONFLICT_CHECK)
    
    def _apply_auto_fixes(self, idea: Idea, report: ValidationReport):
        """Apply automatic fixes to addressable issues."""
        fixes_applied = []
        
        for issue in report.issues[:]:  # Copy list to avoid modification during iteration
            if issue.auto_fixable:
                if issue.code == "TITLE_TOO_LONG":
                    # Truncate title
                    idea.title = idea.title[:self.config.max_title_length-3] + "..."
                    fixes_applied.append("Truncated overly long title")
                    
                elif issue.code == "TITLE_ALL_CAPS":
                    # Convert to title case
                    idea.title = idea.title.title()
                    fixes_applied.append("Converted title to proper case")
                    
                elif issue.code == "EXCESSIVE_PUNCTUATION":
                    # Remove excessive punctuation
                    import re
                    idea.title = re.sub(r'[!]{2,}', '!', idea.title)
                    idea.title = re.sub(r'[?]{2,}', '?', idea.title)
                    fixes_applied.append("Removed excessive punctuation from title")
                    
                elif issue.code == "CONTAINS_TABS":
                    # Replace tabs with spaces
                    idea.description = idea.description.replace('\t', '    ')
                    fixes_applied.append("Replaced tabs with spaces in description")
                    
                elif issue.code == "INVALID_PRIORITY":
                    # Fix priority to valid range
                    idea.priority = max(1, min(10, idea.priority))
                    fixes_applied.append("Adjusted priority to valid range")
                    
                elif issue.code == "CATEGORY_MISMATCH":
                    # Update category to suggested
                    if report.suggested_category:
                        idea.metadata["category"] = report.suggested_category
                        fixes_applied.append(f"Updated category to {report.suggested_category}")
                    
                elif issue.code == "PRIORITY_MISMATCH":
                    # Adjust priority to suggested
                    if report.suggested_priority:
                        idea.priority = report.suggested_priority
                        fixes_applied.append(f"Adjusted priority to {report.suggested_priority}")
                    
                elif issue.code == "MISSING_METADATA":
                    # Add basic metadata
                    if "category" not in idea.metadata and report.suggested_category:
                        idea.metadata["category"] = report.suggested_category
                        fixes_applied.append("Added category metadata")
                    
                    if "estimated_complexity" not in idea.metadata:
                        idea.metadata["estimated_complexity"] = 5  # Default medium complexity
                        fixes_applied.append("Added default complexity estimate")
                    
                elif issue.code == "INVALID_COMPLEXITY":
                    # Fix complexity to valid range
                    complexity = idea.metadata.get("estimated_complexity", 5)
                    if isinstance(complexity, str):
                        complexity = 5  # Default if can't convert
                    idea.metadata["estimated_complexity"] = max(1, min(10, int(complexity)))
                    fixes_applied.append("Fixed complexity to valid range")
                
                # Remove the fixed issue from the report
                report.issues.remove(issue)
        
        report.auto_fixes_applied = fixes_applied
        
        if fixes_applied:
            self.validation_metrics["auto_fixes_applied"] += len(fixes_applied)
            idea.updated_at = datetime.now(timezone.utc)
    
    def _determine_overall_result(self, report: ValidationReport):
        """Determine overall validation result."""
        critical_issues = [i for i in report.issues if i.severity == ValidationSeverity.CRITICAL]
        high_issues = [i for i in report.issues if i.severity == ValidationSeverity.HIGH]
        
        # Check for duplicates or conflicts
        if ValidationCriteria.DUPLICATE_CHECK in report.failed_criteria:
            report.overall_result = ValidationResult.DUPLICATE
            report.routing_recommendation = "reject_duplicate"
            return
        
        if ValidationCriteria.CONFLICT_CHECK in report.failed_criteria:
            report.overall_result = ValidationResult.CONFLICT
            report.routing_recommendation = "conflict_resolution"
            return
        
        # Check for critical failures
        if critical_issues:
            report.overall_result = ValidationResult.FAILED
            report.routing_recommendation = "manual_review"
            return
        
        # Check for issues requiring revision
        if high_issues or report.quality_score < self.config.quality_threshold:
            report.overall_result = ValidationResult.NEEDS_REVISION
            report.routing_recommendation = "revision_required"
            return
        
        # Check overall quality
        if (report.quality_score >= self.config.quality_threshold and 
            report.completeness_score >= self.config.completeness_threshold):
            report.overall_result = ValidationResult.PASSED
            report.routing_recommendation = "council_review"
        else:
            report.overall_result = ValidationResult.NEEDS_REVISION
            report.routing_recommendation = "quality_improvement"
    
    def _calculate_confidence_score(self, report: ValidationReport) -> float:
        """Calculate confidence in the validation result."""
        # Base confidence from quality and completeness scores
        base_confidence = (report.quality_score + report.completeness_score) / 2
        
        # Adjust based on number of criteria passed
        total_criteria = len(ValidationCriteria)
        passed_criteria = len(report.passed_criteria)
        criteria_confidence = passed_criteria / total_criteria
        
        # Adjust based on severity of remaining issues
        issue_penalty = 0.0
        for issue in report.issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                issue_penalty += 0.3
            elif issue.severity == ValidationSeverity.HIGH:
                issue_penalty += 0.2
            elif issue.severity == ValidationSeverity.MEDIUM:
                issue_penalty += 0.1
        
        # Calculate final confidence
        confidence = (base_confidence + criteria_confidence) / 2 - issue_penalty
        return max(0.0, min(1.0, confidence))
    
    # Helper methods for validation
    
    def _assess_content_depth(self, idea: Idea) -> float:
        """Assess the depth and detail of the idea content."""
        score = 0.0
        
        # Length-based scoring
        desc_length = len(idea.description)
        if desc_length >= 200:
            score += 0.3
        elif desc_length >= 100:
            score += 0.2
        else:
            score += 0.1
        
        # Keyword diversity (different concepts mentioned)
        words = set(idea.description.lower().split())
        unique_words = len(words)
        if unique_words >= 50:
            score += 0.2
        elif unique_words >= 30:
            score += 0.15
        else:
            score += 0.1
        
        # Technical depth indicators
        technical_indicators = [
            "implementation", "architecture", "system", "process", "method",
            "algorithm", "framework", "integration", "optimization", "performance"
        ]
        tech_count = sum(1 for indicator in technical_indicators 
                        if indicator in idea.description.lower())
        score += min(0.3, tech_count * 0.05)
        
        # Business value indicators
        value_indicators = [
            "benefit", "value", "impact", "improvement", "efficiency",
            "productivity", "cost", "revenue", "user", "customer"
        ]
        value_count = sum(1 for indicator in value_indicators 
                         if indicator in idea.description.lower())
        score += min(0.2, value_count * 0.03)
        
        return min(1.0, score)
    
    def _assess_clarity(self, idea: Idea) -> float:
        """Assess the clarity and readability of the idea."""
        score = 0.0
        
        # Sentence structure (avoid extremely long sentences)
        sentences = re.split(r'[.!?]+', idea.description)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if 10 <= avg_sentence_length <= 25:
            score += 0.3  # Optimal sentence length
        elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
            score += 0.2  # Acceptable
        else:
            score += 0.1  # Too short or too long
        
        # Paragraph structure
        paragraphs = idea.description.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2  # Well-structured with multiple paragraphs
        else:
            score += 0.1
        
        # Use of specific terminology vs vague language
        specific_terms = [
            "api", "database", "interface", "service", "component", "module",
            "framework", "protocol", "algorithm", "metric", "feature"
        ]
        vague_terms = [
            "thing", "stuff", "somehow", "maybe", "probably", "generally",
            "usually", "various", "multiple", "several"
        ]
        
        specific_count = sum(1 for term in specific_terms 
                           if term in idea.description.lower())
        vague_count = sum(1 for term in vague_terms 
                         if term in idea.description.lower())
        
        if specific_count > vague_count:
            score += 0.3
        elif specific_count == vague_count:
            score += 0.2
        else:
            score += 0.1
        
        # Title-description alignment
        title_words = set(idea.title.lower().split())
        desc_words = set(idea.description.lower().split())
        overlap = len(title_words.intersection(desc_words))
        if overlap >= 2:
            score += 0.2  # Good alignment
        else:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_innovation_potential(self, idea: Idea) -> float:
        """Assess the innovation potential of the idea."""
        score = 0.0
        
        # Innovation keywords
        innovation_keywords = [
            "new", "novel", "innovative", "creative", "original", "unique",
            "advanced", "cutting-edge", "breakthrough", "pioneering"
        ]
        
        improvement_keywords = [
            "improve", "enhance", "optimize", "streamline", "automate",
            "accelerate", "simplify", "reduce", "increase", "maximize"
        ]
        
        text = f"{idea.title} {idea.description}".lower()
        
        innovation_count = sum(1 for kw in innovation_keywords if kw in text)
        improvement_count = sum(1 for kw in improvement_keywords if kw in text)
        
        # Score based on innovation indicators
        if innovation_count > 0:
            score += min(0.4, innovation_count * 0.1)
        
        if improvement_count > 0:
            score += min(0.3, improvement_count * 0.05)
        
        # Technology modernity
        modern_tech = [
            "ai", "machine learning", "deep learning", "neural", "cloud",
            "microservices", "api", "real-time", "streaming", "blockchain",
            "kubernetes", "docker", "serverless", "edge computing"
        ]
        
        modern_count = sum(1 for tech in modern_tech if tech in text)
        score += min(0.3, modern_count * 0.05)
        
        return min(1.0, score)
    
    def _assess_initial_feasibility(self, idea: Idea) -> float:
        """Assess initial technical and business feasibility."""
        score = 0.5  # Start with neutral score
        
        text = f"{idea.title} {idea.description}".lower()
        
        # Feasible technology indicators
        feasible_tech = [
            "api", "database", "web", "mobile", "cloud", "service",
            "interface", "dashboard", "report", "integration", "automation"
        ]
        
        # Challenging but feasible indicators
        challenging_tech = [
            "machine learning", "ai", "blockchain", "real-time", "scale",
            "distributed", "microservices", "analytics", "prediction"
        ]
        
        # Potentially unfeasible indicators
        unfeasible_indicators = [
            "revolutionary", "impossible", "never been done", "zero cost",
            "infinite", "perfect", "100% accurate", "solve everything"
        ]
        
        feasible_count = sum(1 for tech in feasible_tech if tech in text)
        challenging_count = sum(1 for tech in challenging_tech if tech in text)
        unfeasible_count = sum(1 for indicator in unfeasible_indicators if indicator in text)
        
        # Adjust score based on indicators
        score += min(0.3, feasible_count * 0.05)
        score += min(0.2, challenging_count * 0.03)
        score -= min(0.5, unfeasible_count * 0.2)
        
        # Complexity vs detail balance
        if idea.metadata.get("estimated_complexity"):
            complexity = idea.metadata["estimated_complexity"]
            detail_ratio = len(idea.description) / 100  # Rough detail measure
            
            if complexity <= 5 and detail_ratio >= 2:
                score += 0.1  # Good detail for simple idea
            elif complexity > 7 and detail_ratio >= 5:
                score += 0.2  # Adequate detail for complex idea
            elif complexity > 7 and detail_ratio < 2:
                score -= 0.2  # Insufficient detail for complex idea
        
        return max(0.0, min(1.0, score))
    
    def _classify_idea_category(self, idea: Idea) -> str:
        """Classify idea into appropriate category."""
        text = f"{idea.title} {idea.description}".lower()
        
        # Category keywords mapping
        category_keywords = {
            "infrastructure": ["infrastructure", "system", "architecture", "platform", "framework", "foundation"],
            "feature_enhancement": ["enhance", "improve", "upgrade", "extend", "add", "feature"],
            "new_capability": ["new", "capability", "function", "service", "tool", "solution"],
            "performance_optimization": ["performance", "optimize", "speed", "efficient", "fast", "latency"],
            "user_experience": ["user", "interface", "ux", "ui", "experience", "usability", "design"],
            "security": ["security", "secure", "authentication", "authorization", "encrypt", "privacy"],
            "integration": ["integrate", "connect", "api", "interface", "bridge", "link"],
            "analytics": ["analytics", "data", "report", "dashboard", "metrics", "insights"],
            "automation": ["automate", "automatic", "workflow", "process", "script", "bot"],
            "research": ["research", "experiment", "study", "investigate", "explore", "prototype"]
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        # Return category with highest score, or default
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
        
        return "general"  # Default category
    
    def _suggest_priority(self, idea: Idea) -> int:
        """Suggest appropriate priority based on content analysis."""
        text = f"{idea.title} {idea.description}".lower()
        
        # High priority indicators
        high_priority_keywords = [
            "critical", "urgent", "important", "essential", "required",
            "security", "bug", "fix", "issue", "problem", "failure"
        ]
        
        # Medium priority indicators
        medium_priority_keywords = [
            "improve", "enhance", "optimize", "feature", "capability",
            "efficiency", "performance", "user experience"
        ]
        
        # Low priority indicators
        low_priority_keywords = [
            "nice to have", "future", "experimental", "research",
            "explore", "investigate", "consider", "maybe"
        ]
        
        high_count = sum(1 for kw in high_priority_keywords if kw in text)
        medium_count = sum(1 for kw in medium_priority_keywords if kw in text)
        low_count = sum(1 for kw in low_priority_keywords if kw in text)
        
        # Calculate suggested priority
        if high_count > 0:
            return min(10, 7 + high_count)
        elif low_count > 0:
            return max(1, 3 - low_count)
        elif medium_count > 0:
            return 5 + min(2, medium_count)
        else:
            return 5  # Default medium priority
    
    def _find_duplicate_candidates(self, idea: Idea) -> List[DuplicateCandidate]:
        """Find potential duplicate ideas."""
        candidates = []
        
        for known_id, known_idea in self.known_ideas.items():
            if known_id == str(idea.id):
                continue  # Skip self
            
            similarity = self._calculate_similarity(idea, known_idea)
            if similarity >= 0.5:  # Threshold for consideration
                matching_aspects = self._identify_matching_aspects(idea, known_idea)
                
                candidates.append(DuplicateCandidate(
                    idea_id=known_id,
                    similarity_score=similarity,
                    matching_aspects=matching_aspects,
                    confidence=min(1.0, similarity + 0.1)
                ))
        
        # Sort by similarity score
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        return candidates[:5]  # Return top 5 matches
    
    def _find_conflict_candidates(self, idea: Idea) -> List[ConflictCandidate]:
        """Find potential conflicting ideas."""
        conflicts = []
        
        for known_id, known_idea in self.known_ideas.items():
            if known_id == str(idea.id):
                continue  # Skip self
            
            conflict_type, severity, description = self._detect_conflict(idea, known_idea)
            if conflict_type:
                resolution = self._suggest_conflict_resolution(conflict_type, idea, known_idea)
                
                conflicts.append(ConflictCandidate(
                    idea_id=known_id,
                    conflict_type=conflict_type,
                    conflict_description=description,
                    severity=severity,
                    resolution_suggestion=resolution
                ))
        
        return conflicts
    
    def _calculate_similarity(self, idea1: Idea, idea2: Idea) -> float:
        """Calculate similarity between two ideas."""
        # Title similarity
        title1_words = set(idea1.title.lower().split())
        title2_words = set(idea2.title.lower().split())
        title_overlap = len(title1_words.intersection(title2_words))
        title_union = len(title1_words.union(title2_words))
        title_similarity = title_overlap / title_union if title_union > 0 else 0
        
        # Description similarity (simplified Jaccard similarity)
        desc1_words = set(idea1.description.lower().split())
        desc2_words = set(idea2.description.lower().split())
        desc_overlap = len(desc1_words.intersection(desc2_words))
        desc_union = len(desc1_words.union(desc2_words))
        desc_similarity = desc_overlap / desc_union if desc_union > 0 else 0
        
        # Category similarity
        cat1 = idea1.metadata.get("category", "")
        cat2 = idea2.metadata.get("category", "")
        category_similarity = 1.0 if cat1 == cat2 and cat1 else 0.0
        
        # Priority similarity
        priority_diff = abs(idea1.priority - idea2.priority)
        priority_similarity = max(0, 1.0 - (priority_diff / 10))
        
        # Weighted average
        overall_similarity = (
            title_similarity * 0.4 +
            desc_similarity * 0.4 +
            category_similarity * 0.1 +
            priority_similarity * 0.1
        )
        
        return overall_similarity
    
    def _identify_matching_aspects(self, idea1: Idea, idea2: Idea) -> List[str]:
        """Identify which aspects of two ideas match."""
        aspects = []
        
        # Check title similarity
        title1_words = set(idea1.title.lower().split())
        title2_words = set(idea2.title.lower().split())
        if len(title1_words.intersection(title2_words)) >= 2:
            aspects.append("similar_title")
        
        # Check category
        if idea1.metadata.get("category") == idea2.metadata.get("category"):
            aspects.append("same_category")
        
        # Check priority range
        if abs(idea1.priority - idea2.priority) <= 2:
            aspects.append("similar_priority")
        
        # Check for common key terms
        key_terms1 = self._extract_key_terms(idea1.description)
        key_terms2 = self._extract_key_terms(idea2.description)
        common_terms = key_terms1.intersection(key_terms2)
        if len(common_terms) >= 3:
            aspects.append("common_terminology")
        
        return aspects
    
    def _detect_conflict(self, idea1: Idea, idea2: Idea) -> Tuple[Optional[str], ValidationSeverity, str]:
        """Detect potential conflicts between ideas."""
        # Resource conflict detection
        if self._detect_resource_conflict(idea1, idea2):
            return ("resource_conflict", ValidationSeverity.MEDIUM,
                    "Ideas may compete for the same resources or team attention")
        
        # Approach conflict detection
        if self._detect_approach_conflict(idea1, idea2):
            return ("approach_conflict", ValidationSeverity.LOW,
                    "Ideas propose different approaches to similar problems")
        
        # Timeline conflict detection
        if self._detect_timeline_conflict(idea1, idea2):
            return ("timeline_conflict", ValidationSeverity.MEDIUM,
                    "Ideas may have conflicting implementation timelines")
        
        return None, ValidationSeverity.INFO, ""
    
    def _detect_resource_conflict(self, idea1: Idea, idea2: Idea) -> bool:
        """Detect if ideas compete for similar resources."""
        # Check if both are high priority and in same category
        same_category = idea1.metadata.get("category") == idea2.metadata.get("category")
        both_high_priority = idea1.priority >= 7 and idea2.priority >= 7
        
        return same_category and both_high_priority
    
    def _detect_approach_conflict(self, idea1: Idea, idea2: Idea) -> bool:
        """Detect if ideas propose conflicting approaches."""
        # Look for opposing keywords
        conflicting_pairs = [
            ("centralized", "distributed"),
            ("manual", "automated"),
            ("simple", "complex"),
            ("internal", "external"),
            ("realtime", "batch")
        ]
        
        text1 = f"{idea1.title} {idea1.description}".lower()
        text2 = f"{idea2.title} {idea2.description}".lower()
        
        for term1, term2 in conflicting_pairs:
            if term1 in text1 and term2 in text2:
                return True
            if term2 in text1 and term1 in text2:
                return True
        
        return False
    
    def _detect_timeline_conflict(self, idea1: Idea, idea2: Idea) -> bool:
        """Detect if ideas have conflicting timelines."""
        # Simplified: assume conflict if both are urgent and require significant effort
        complexity1 = idea1.metadata.get("estimated_complexity", 5)
        complexity2 = idea2.metadata.get("estimated_complexity", 5)
        
        both_urgent = idea1.priority >= 8 and idea2.priority >= 8
        both_complex = complexity1 >= 7 and complexity2 >= 7
        
        return both_urgent and both_complex
    
    def _suggest_conflict_resolution(self, conflict_type: str, idea1: Idea, idea2: Idea) -> str:
        """Suggest resolution for detected conflicts."""
        resolutions = {
            "resource_conflict": "Consider merging ideas or staggering implementation timelines",
            "approach_conflict": "Clarify different use cases or create hybrid approach",
            "timeline_conflict": "Prioritize based on business impact and dependencies"
        }
        
        return resolutions.get(conflict_type, "Review and coordinate with related idea")
    
    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text for comparison."""
        # Remove common words and extract meaningful terms
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "up", "about", "into", "through"
        }
        
        words = text.lower().split()
        key_terms = {word for word in words 
                    if len(word) > 3 and word not in common_words}
        
        return key_terms
    
    # Metrics and analysis methods
    
    def _update_validation_metrics(self, report: ValidationReport, start_time: datetime):
        """Update validation metrics."""
        self.validation_metrics["total_validations"] += 1
        
        if report.overall_result == ValidationResult.PASSED:
            self.validation_metrics["passed_validations"] += 1
        else:
            self.validation_metrics["failed_validations"] += 1
        
        if report.duplicate_matches:
            self.validation_metrics["duplicates_detected"] += 1
        
        if report.conflict_matches:
            self.validation_metrics["conflicts_detected"] += 1
        
        # Update average validation time
        validation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        current_avg = self.validation_metrics["average_validation_time"]
        total_validations = self.validation_metrics["total_validations"]
        
        new_avg = ((current_avg * (total_validations - 1)) + validation_time) / total_validations
        self.validation_metrics["average_validation_time"] = new_avg
        
        # Update average quality score
        current_quality_avg = self.validation_metrics["average_quality_score"]
        new_quality_avg = ((current_quality_avg * (total_validations - 1)) + report.quality_score) / total_validations
        self.validation_metrics["average_quality_score"] = new_quality_avg
    
    def _calculate_issue_distribution(self) -> Dict[str, int]:
        """Calculate distribution of validation issues."""
        issue_counts = {}
        
        for report in self.validation_history.values():
            for issue in report.issues:
                severity = issue.severity.value
                issue_counts[severity] = issue_counts.get(severity, 0) + 1
        
        return issue_counts
    
    def _get_most_common_issues(self) -> List[Dict[str, Any]]:
        """Get most common validation issues."""
        issue_codes = {}
        
        for report in self.validation_history.values():
            for issue in report.issues:
                code = issue.code
                if code not in issue_codes:
                    issue_codes[code] = {
                        "code": code,
                        "count": 0,
                        "severity": issue.severity.value,
                        "criteria": issue.criteria.value,
                        "message": issue.message
                    }
                issue_codes[code]["count"] += 1
        
        # Sort by count and return top 10
        sorted_issues = sorted(issue_codes.values(), key=lambda x: x["count"], reverse=True)
        return sorted_issues[:10]
    
    def _calculate_auto_fix_success_rate(self) -> float:
        """Calculate success rate of auto-fixes."""
        total_auto_fixable = 0
        total_fixed = 0
        
        for report in self.validation_history.values():
            auto_fixable_issues = [i for i in report.issues if i.auto_fixable]
            total_auto_fixable += len(auto_fixable_issues)
            total_fixed += len(report.auto_fixes_applied)
        
        if total_auto_fixable == 0:
            return 0.0
        
        return total_fixed / total_auto_fixable
    
    def _calculate_category_accuracy(self) -> float:
        """Calculate accuracy of category suggestions."""
        # Simplified metric - in real implementation, this would track actual outcomes
        accurate_suggestions = 0
        total_suggestions = 0
        
        for report in self.validation_history.values():
            if report.suggested_category:
                total_suggestions += 1
                # Assume 80% accuracy for demonstration
                if hash(report.idea_id) % 5 != 0:  # Pseudo-random accuracy
                    accurate_suggestions += 1
        
        if total_suggestions == 0:
            return 0.0
        
        return accurate_suggestions / total_suggestions
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules and patterns."""
        return {
            "required_fields": ["title", "description"],
            "title_patterns": {
                "avoid": [r"^test", r"^temp", r"^placeholder"],
                "encourage": [r"^[A-Z]", r"\w+\s+\w+"]  # Starts with capital, multiple words
            },
            "description_patterns": {
                "quality_indicators": [
                    "implementation", "benefits", "approach", "solution",
                    "requirements", "architecture", "design", "analysis"
                ],
                "red_flags": [
                    "todo", "tbd", "fill in later", "placeholder",
                    "not sure", "maybe", "somehow"
                ]
            }
        }
    
    def _initialize_keyword_patterns(self) -> Dict[str, List[str]]:
        """Initialize keyword patterns for various checks."""
        return {
            "technical_terms": [
                "api", "database", "service", "interface", "framework",
                "algorithm", "architecture", "system", "platform", "protocol"
            ],
            "business_terms": [
                "value", "benefit", "cost", "revenue", "efficiency",
                "productivity", "user", "customer", "market", "business"
            ],
            "quality_indicators": [
                "scalable", "reliable", "secure", "maintainable",
                "efficient", "robust", "flexible", "extensible"
            ],
            "innovation_terms": [
                "innovative", "novel", "creative", "breakthrough",
                "cutting-edge", "advanced", "next-generation", "revolutionary"
            ]
        }
    
    def _initialize_category_classifiers(self) -> Dict[str, Dict[str, float]]:
        """Initialize category classification weights."""
        return {
            "infrastructure": {
                "keywords": ["infrastructure", "platform", "system", "architecture"],
                "weight": 1.0
            },
            "feature": {
                "keywords": ["feature", "enhancement", "improvement", "upgrade"],
                "weight": 0.8
            },
            "performance": {
                "keywords": ["performance", "optimization", "speed", "efficiency"],
                "weight": 0.9
            },
            "security": {
                "keywords": ["security", "authentication", "encryption", "privacy"],
                "weight": 1.0
            }
        }


# Factory function for easy agent creation
def create_validator_agent(config: Optional[ValidatorConfiguration] = None,
                          state_router: Optional[StateMachineRouter] = None) -> ValidatorAgent:
    """
    Create a new Validator Agent instance.
    
    Args:
        config: Optional validator configuration
        state_router: Optional state machine router
        
    Returns:
        Configured ValidatorAgent instance
    """
    return ValidatorAgent(config, state_router)


# Export main classes and functions
__all__ = [
    "ValidatorAgent",
    "ValidatorConfiguration",
    "ValidationReport",
    "ValidationResult",
    "ValidationCriteria",
    "ValidationSeverity",
    "ValidationIssue",
    "DuplicateCandidate",
    "ConflictCandidate",
    "create_validator_agent"
]