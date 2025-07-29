"""
Council Agent for the AI-Galaxy Lower Meta-Layer.

This module implements the Council Agent, the most critical component responsible
for evaluating ideas and making strategic decisions about what moves forward 
in the AI-Galaxy ecosystem. The Council simulates expert decision-making through
multi-criteria evaluation and consensus-building algorithms.
"""

import json
import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext
from ..state_machine.router import StateMachineRouter, TransitionResult


class EvaluationCriteria(str, Enum):
    """Core evaluation criteria for idea assessment."""
    FEASIBILITY = "feasibility"
    INNOVATION = "innovation"
    ALIGNMENT = "alignment"
    RESOURCE_REQUIREMENTS = "resource_requirements"
    STRATEGIC_VALUE = "strategic_value"
    MARKET_POTENTIAL = "market_potential"
    TECHNICAL_COMPLEXITY = "technical_complexity"
    RISK_ASSESSMENT = "risk_assessment"


class CouncilMemberType(str, Enum):
    """Types of virtual council members with different perspectives."""
    TECHNICAL_LEAD = "technical_lead"
    PRODUCT_STRATEGIST = "product_strategist"
    RESOURCE_MANAGER = "resource_manager"
    INNOVATION_ADVOCATE = "innovation_advocate"
    RISK_ASSESSOR = "risk_assessor"
    MARKET_ANALYST = "market_analyst"


class DecisionOutcome(str, Enum):
    """Possible outcomes from council evaluation."""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    DEFERRED = "deferred"
    APPEAL_REQUESTED = "appeal_requested"


class VotingResult(str, Enum):
    """Results of council voting process."""
    UNANIMOUS_APPROVAL = "unanimous_approval"
    MAJORITY_APPROVAL = "majority_approval"
    SPLIT_DECISION = "split_decision"
    MAJORITY_REJECTION = "majority_rejection"
    UNANIMOUS_REJECTION = "unanimous_rejection"


@dataclass
class EvaluationScore:
    """Individual evaluation score for a specific criteria."""
    criteria: EvaluationCriteria
    score: float  # 0.0 to 10.0
    weight: float  # Importance weight
    reasoning: str
    confidence: float  # 0.0 to 1.0


@dataclass
class CouncilMemberProfile:
    """Profile and characteristics of a virtual council member."""
    member_type: CouncilMemberType
    name: str
    expertise_areas: List[str]
    bias_factors: Dict[str, float]  # Inherent biases in scoring
    decision_threshold: float  # Personal approval threshold
    risk_tolerance: float  # 0.0 (risk-averse) to 1.0 (risk-seeking)


class CouncilVote(BaseModel):
    """Individual council member's vote and reasoning."""
    member_name: str
    member_type: CouncilMemberType
    vote: DecisionOutcome
    scores: List[EvaluationScore] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    concerns: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report for an idea."""
    idea_id: str
    evaluation_timestamp: datetime
    council_votes: List[CouncilVote] = Field(default_factory=list)
    final_decision: DecisionOutcome
    consensus_score: float = Field(ge=0.0, le=10.0)
    voting_result: VotingResult
    approval_percentage: float = Field(ge=0.0, le=100.0)
    key_strengths: List[str] = Field(default_factory=list)
    key_concerns: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    strategic_alignment_score: float = Field(ge=0.0, le=10.0)
    appeal_eligibility: bool = False
    implementation_priority: int = Field(ge=1, le=10)


class CouncilConfiguration(BaseModel):
    """Configuration for council evaluation parameters."""
    approval_threshold: float = Field(default=6.5, ge=0.0, le=10.0)
    rejection_threshold: float = Field(default=4.0, ge=0.0, le=10.0)
    consensus_requirement: float = Field(default=0.7, ge=0.0, le=1.0)
    appeal_eligibility_score: float = Field(default=5.5, ge=0.0, le=10.0)
    criteria_weights: Dict[EvaluationCriteria, float] = Field(default_factory=dict)
    enable_appeal_process: bool = True
    max_appeal_rounds: int = 2
    council_member_count: int = 6
    require_unanimous_for_high_impact: bool = True
    high_impact_threshold: float = 8.0


class AppealRequest(BaseModel):
    """Request for appealing a council decision."""
    idea_id: str
    original_decision: DecisionOutcome
    appeal_reason: str
    additional_evidence: Dict[str, Any] = Field(default_factory=dict)
    requested_by: str
    appeal_timestamp: datetime
    priority_escalation: bool = False


class CouncilAgent:
    """
    The Council Agent - critical decision-making component of AI-Galaxy.
    
    Evaluates ideas through multi-criteria scoring, simulates expert council
    decision-making, and manages the approval/rejection workflow with appeals
    process and strategic alignment assessment.
    """
    
    def __init__(self, config: Optional[CouncilConfiguration] = None, 
                 state_router: Optional[StateMachineRouter] = None):
        """
        Initialize the Council Agent.
        
        Args:
            config: Council configuration parameters
            state_router: State machine router for workflow transitions
        """
        self.logger = get_logger("council_agent")
        self.config = config or CouncilConfiguration()
        self.state_router = state_router
        
        # Initialize council members
        self.council_members = self._initialize_council_members()
        
        # Initialize criteria weights if not provided
        if not self.config.criteria_weights:
            self.config.criteria_weights = self._default_criteria_weights()
        
        # Tracking and metrics
        self.evaluation_history: Dict[str, List[EvaluationReport]] = {}
        self.appeal_history: Dict[str, List[AppealRequest]] = {}
        self.decision_metrics = {
            "total_evaluations": 0,
            "approvals": 0,
            "rejections": 0,
            "appeals": 0,
            "average_evaluation_time": 0.0
        }
        
        self.logger.agent_action("council_agent_initialized", "council_agent", 
                                additional_context={
                                    "approval_threshold": self.config.approval_threshold,
                                    "council_size": len(self.council_members),
                                    "appeal_process_enabled": self.config.enable_appeal_process
                                })
    
    def evaluate_idea(self, idea: Idea) -> EvaluationReport:
        """
        Comprehensive evaluation of an idea by the virtual council.
        
        Args:
            idea: The idea to evaluate
            
        Returns:
            Detailed evaluation report with decision and reasoning
        """
        start_time = datetime.now(timezone.utc)
        idea_id = str(idea.id)
        
        context = LogContext(
            agent_name="council_agent",
            idea_id=idea_id,
            additional_context={"evaluation_start": start_time.isoformat()}
        )
        
        self.logger.agent_action("starting_idea_evaluation", "council_agent", idea_id)
        
        try:
            # Gather council votes
            council_votes = []
            for member in self.council_members:
                vote = self._get_member_evaluation(idea, member)
                council_votes.append(vote)
                
                self.logger.debug(f"Council member {member.name} voted: {vote.vote}", context)
            
            # Analyze voting results
            voting_result = self._analyze_voting_results(council_votes)
            final_decision = self._determine_final_decision(council_votes, voting_result)
            
            # Calculate consensus metrics
            consensus_score = self._calculate_consensus_score(council_votes)
            approval_percentage = self._calculate_approval_percentage(council_votes)
            
            # Extract insights
            key_strengths = self._extract_key_strengths(council_votes)
            key_concerns = self._extract_key_concerns(council_votes)
            recommendations = self._extract_recommendations(council_votes)
            
            # Assess strategic alignment
            strategic_alignment_score = self._assess_strategic_alignment(idea, council_votes)
            
            # Determine resource requirements
            resource_requirements = self._estimate_resource_requirements(idea, council_votes)
            
            # Check appeal eligibility
            appeal_eligibility = self._check_appeal_eligibility(consensus_score, final_decision)
            
            # Set implementation priority
            implementation_priority = self._calculate_implementation_priority(
                consensus_score, strategic_alignment_score, idea.priority
            )
            
            # Create evaluation report
            report = EvaluationReport(
                idea_id=idea_id,
                evaluation_timestamp=start_time,
                council_votes=council_votes,
                final_decision=final_decision,
                consensus_score=consensus_score,
                voting_result=voting_result,
                approval_percentage=approval_percentage,
                key_strengths=key_strengths,
                key_concerns=key_concerns,
                recommendations=recommendations,
                resource_requirements=resource_requirements,
                strategic_alignment_score=strategic_alignment_score,
                appeal_eligibility=appeal_eligibility,
                implementation_priority=implementation_priority
            )
            
            # Store evaluation history
            if idea_id not in self.evaluation_history:
                self.evaluation_history[idea_id] = []
            self.evaluation_history[idea_id].append(report)
            
            # Update metrics
            self._update_decision_metrics(report, start_time)
            
            # Log decision
            self.logger.agent_action("idea_evaluation_completed", "council_agent", idea_id, {
                "decision": final_decision.value,
                "consensus_score": consensus_score,
                "voting_result": voting_result.value,
                "evaluation_duration": (datetime.now(timezone.utc) - start_time).total_seconds()
            })
            
            return report
            
        except Exception as e:
            self.logger.error(f"Idea evaluation failed: {e}", context, exc_info=True)
            
            # Return failure report
            return EvaluationReport(
                idea_id=idea_id,
                evaluation_timestamp=start_time,
                final_decision=DecisionOutcome.REJECTED,
                consensus_score=0.0,
                voting_result=VotingResult.UNANIMOUS_REJECTION,
                approval_percentage=0.0,
                key_concerns=["Evaluation process failed due to system error"],
                recommendations=["Technical review required before re-evaluation"],
                strategic_alignment_score=0.0,
                implementation_priority=1
            )
    
    def process_appeal(self, appeal_request: AppealRequest) -> EvaluationReport:
        """
        Process an appeal request for a previously rejected or deferred idea.
        
        Args:
            appeal_request: The appeal request with additional evidence
            
        Returns:
            New evaluation report based on appeal review
        """
        idea_id = appeal_request.idea_id
        
        context = LogContext(
            agent_name="council_agent",
            idea_id=idea_id,
            additional_context={
                "appeal_reason": appeal_request.appeal_reason,
                "appeal_timestamp": appeal_request.appeal_timestamp.isoformat()
            }
        )
        
        self.logger.agent_action("processing_appeal", "council_agent", idea_id)
        
        try:
            # Check if appeal is valid
            if not self._is_appeal_valid(appeal_request):
                self.logger.warning("Invalid appeal request", context)
                raise ValueError("Appeal request is not valid")
            
            # Get original evaluation
            original_evaluations = self.evaluation_history.get(idea_id, [])
            if not original_evaluations:
                raise ValueError("No original evaluation found for appeal")
            
            original_report = original_evaluations[-1]
            
            # Store appeal in history
            if idea_id not in self.appeal_history:
                self.appeal_history[idea_id] = []
            self.appeal_history[idea_id].append(appeal_request)
            
            # Enhanced evaluation with appeal context
            # Note: In a real implementation, we'd retrieve the actual idea
            # For now, we'll create a modified evaluation based on appeal evidence
            enhanced_votes = self._reevaluate_with_appeal_evidence(
                original_report.council_votes, appeal_request
            )
            
            # Create new evaluation report
            appeal_report = self._create_appeal_evaluation_report(
                idea_id, enhanced_votes, appeal_request, original_report
            )
            
            # Update evaluation history
            self.evaluation_history[idea_id].append(appeal_report)
            
            # Update metrics
            self.decision_metrics["appeals"] += 1
            
            self.logger.agent_action("appeal_processed", "council_agent", idea_id, {
                "original_decision": original_report.final_decision.value,
                "appeal_decision": appeal_report.final_decision.value,
                "decision_changed": original_report.final_decision != appeal_report.final_decision
            })
            
            return appeal_report
            
        except Exception as e:
            self.logger.error(f"Appeal processing failed: {e}", context, exc_info=True)
            raise
    
    def update_idea_status(self, idea: Idea, evaluation_report: EvaluationReport) -> bool:
        """
        Update idea status based on council decision using state machine.
        
        Args:
            idea: The idea to update
            evaluation_report: The evaluation report with decision
            
        Returns:
            True if status update successful, False otherwise
        """
        idea_id = str(idea.id)
        current_state = idea.status.value
        
        context = LogContext(
            agent_name="council_agent",
            idea_id=idea_id,
            additional_context={
                "current_state": current_state,
                "council_decision": evaluation_report.final_decision.value
            }
        )
        
        try:
            # Determine target state based on decision
            if evaluation_report.final_decision == DecisionOutcome.APPROVED:
                target_state = "approved"
            elif evaluation_report.final_decision == DecisionOutcome.REJECTED:
                target_state = "rejected"
            else:
                # For needs_revision, deferred, or appeal_requested, stay in council_review
                self.logger.info("Idea remains in council review for further action", context)
                return True
            
            # Update idea metadata with evaluation results
            idea.metadata.update({
                "council_evaluation": {
                    "decision": evaluation_report.final_decision.value,
                    "consensus_score": evaluation_report.consensus_score,
                    "strategic_alignment": evaluation_report.strategic_alignment_score,
                    "implementation_priority": evaluation_report.implementation_priority,
                    "evaluation_timestamp": evaluation_report.evaluation_timestamp.isoformat(),
                    "key_strengths": evaluation_report.key_strengths,
                    "key_concerns": evaluation_report.key_concerns,
                    "recommendations": evaluation_report.recommendations,
                    "resource_requirements": evaluation_report.resource_requirements,
                    "appeal_eligibility": evaluation_report.appeal_eligibility
                }
            })
            
            # Execute state transition if router available
            if self.state_router:
                result = self.state_router.execute_transition(idea, target_state, "council_agent")
                
                if result == TransitionResult.SUCCESS:
                    self.logger.state_transition(current_state, target_state, idea_id, 
                                               "council_agent", f"Council decision: {evaluation_report.final_decision.value}")
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
    
    def get_evaluation_summary(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of all evaluations for a specific idea.
        
        Args:
            idea_id: ID of the idea
            
        Returns:
            Summary dictionary or None if no evaluations found
        """
        evaluations = self.evaluation_history.get(idea_id, [])
        appeals = self.appeal_history.get(idea_id, [])
        
        if not evaluations:
            return None
        
        latest_evaluation = evaluations[-1]
        
        return {
            "idea_id": idea_id,
            "total_evaluations": len(evaluations),
            "latest_decision": latest_evaluation.final_decision.value,
            "latest_consensus_score": latest_evaluation.consensus_score,
            "strategic_alignment": latest_evaluation.strategic_alignment_score,
            "implementation_priority": latest_evaluation.implementation_priority,
            "appeal_count": len(appeals),
            "appeal_eligible": latest_evaluation.appeal_eligibility,
            "evaluation_timeline": [
                {
                    "timestamp": eval_report.evaluation_timestamp.isoformat(),
                    "decision": eval_report.final_decision.value,
                    "consensus_score": eval_report.consensus_score
                }
                for eval_report in evaluations
            ]
        }
    
    def get_council_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about council performance.
        
        Returns:
            Dictionary with council performance metrics
        """
        total_ideas = len(self.evaluation_history)
        
        if total_ideas == 0:
            return {**self.decision_metrics, "total_ideas_evaluated": 0}
        
        # Calculate additional metrics
        approval_rate = (self.decision_metrics["approvals"] / total_ideas) * 100
        rejection_rate = (self.decision_metrics["rejections"] / total_ideas) * 100
        appeal_rate = (self.decision_metrics["appeals"] / total_ideas) * 100
        
        # Council member performance
        member_performance = {}
        for member in self.council_members:
            member_performance[member.name] = {
                "type": member.member_type.value,
                "average_score": self._calculate_member_average_score(member.name),
                "approval_tendency": self._calculate_member_approval_tendency(member.name),
                "consistency": self._calculate_member_consistency(member.name)
            }
        
        return {
            **self.decision_metrics,
            "total_ideas_evaluated": total_ideas,
            "approval_rate_percent": approval_rate,
            "rejection_rate_percent": rejection_rate,
            "appeal_rate_percent": appeal_rate,
            "average_consensus_score": self._calculate_average_consensus_score(),
            "average_strategic_alignment": self._calculate_average_strategic_alignment(),
            "member_performance": member_performance,
            "configuration": {
                "approval_threshold": self.config.approval_threshold,
                "rejection_threshold": self.config.rejection_threshold,
                "consensus_requirement": self.config.consensus_requirement,
                "council_size": len(self.council_members)
            }
        }
    
    # Private helper methods
    
    def _initialize_council_members(self) -> List[CouncilMemberProfile]:
        """Initialize the virtual council members with diverse perspectives."""
        return [
            CouncilMemberProfile(
                member_type=CouncilMemberType.TECHNICAL_LEAD,
                name="Dr. Sarah Chen",
                expertise_areas=["system architecture", "scalability", "technical feasibility"],
                bias_factors={"technical_complexity": 1.2, "innovation": 1.1},
                decision_threshold=6.0,
                risk_tolerance=0.6
            ),
            CouncilMemberProfile(
                member_type=CouncilMemberType.PRODUCT_STRATEGIST,
                name="Marcus Rodriguez",
                expertise_areas=["product strategy", "user experience", "market fit"],
                bias_factors={"strategic_value": 1.3, "market_potential": 1.2},
                decision_threshold=6.5,
                risk_tolerance=0.7
            ),
            CouncilMemberProfile(
                member_type=CouncilMemberType.RESOURCE_MANAGER,
                name="Jennifer Kim",
                expertise_areas=["resource allocation", "project management", "cost optimization"],
                bias_factors={"resource_requirements": 1.4, "feasibility": 1.2},
                decision_threshold=7.0,
                risk_tolerance=0.4
            ),
            CouncilMemberProfile(
                member_type=CouncilMemberType.INNOVATION_ADVOCATE,
                name="Dr. Alex Okonkwo",
                expertise_areas=["innovation strategy", "emerging technologies", "research"],
                bias_factors={"innovation": 1.5, "strategic_value": 1.1},
                decision_threshold=5.5,
                risk_tolerance=0.8
            ),
            CouncilMemberProfile(
                member_type=CouncilMemberType.RISK_ASSESSOR,
                name="Elena Petrov",
                expertise_areas=["risk management", "compliance", "security"],
                bias_factors={"risk_assessment": 1.4, "feasibility": 1.1},
                decision_threshold=7.5,
                risk_tolerance=0.3
            ),
            CouncilMemberProfile(
                member_type=CouncilMemberType.MARKET_ANALYST,
                name="David Thompson",
                expertise_areas=["market analysis", "competitive intelligence", "business value"],
                bias_factors={"market_potential": 1.3, "strategic_value": 1.2},
                decision_threshold=6.2,
                risk_tolerance=0.5
            )
        ]
    
    def _default_criteria_weights(self) -> Dict[EvaluationCriteria, float]:
        """Default weights for evaluation criteria."""
        return {
            EvaluationCriteria.FEASIBILITY: 0.20,
            EvaluationCriteria.INNOVATION: 0.15,
            EvaluationCriteria.ALIGNMENT: 0.20,
            EvaluationCriteria.RESOURCE_REQUIREMENTS: 0.15,
            EvaluationCriteria.STRATEGIC_VALUE: 0.15,
            EvaluationCriteria.MARKET_POTENTIAL: 0.08,
            EvaluationCriteria.TECHNICAL_COMPLEXITY: 0.04,
            EvaluationCriteria.RISK_ASSESSMENT: 0.03
        }
    
    def _get_member_evaluation(self, idea: Idea, member: CouncilMemberProfile) -> CouncilVote:
        """Get evaluation from a specific council member."""
        scores = []
        
        # Evaluate each criteria with member-specific bias
        for criteria, base_weight in self.config.criteria_weights.items():
            # Apply member bias
            member_bias = member.bias_factors.get(criteria.value, 1.0)
            adjusted_weight = base_weight * member_bias
            
            # Simulate scoring based on member expertise and idea characteristics
            base_score = self._simulate_criteria_score(idea, criteria, member)
            
            score = EvaluationScore(
                criteria=criteria,
                score=base_score,
                weight=adjusted_weight,
                reasoning=self._generate_score_reasoning(idea, criteria, base_score, member),
                confidence=self._calculate_score_confidence(criteria, member)
            )
            scores.append(score)
        
        # Calculate overall weighted score
        overall_score = sum(score.score * score.weight for score in scores) / sum(score.weight for score in scores)
        
        # Determine vote based on member's threshold and risk tolerance
        vote = self._determine_member_vote(overall_score, member, idea)
        
        # Generate concerns and recommendations
        concerns = self._generate_member_concerns(scores, member, idea)
        recommendations = self._generate_member_recommendations(scores, member, idea)
        
        # Calculate confidence in decision
        confidence = self._calculate_decision_confidence(overall_score, member)
        
        return CouncilVote(
            member_name=member.name,
            member_type=member.member_type,
            vote=vote,
            scores=scores,
            overall_score=overall_score,
            confidence=confidence,
            reasoning=self._generate_vote_reasoning(overall_score, vote, member, idea),
            concerns=concerns,
            recommendations=recommendations
        )
    
    def _simulate_criteria_score(self, idea: Idea, criteria: EvaluationCriteria, 
                                member: CouncilMemberProfile) -> float:
        """Simulate scoring for a specific criteria."""
        # Base scoring logic with realistic variation
        base_score = 5.0  # Neutral starting point
        
        # Adjust based on idea characteristics
        if criteria == EvaluationCriteria.FEASIBILITY:
            # Higher priority ideas assumed more feasible
            base_score += (idea.priority - 5) * 0.5
            # Longer descriptions might indicate more thought
            if len(idea.description) > 200:
                base_score += 1.0
                
        elif criteria == EvaluationCriteria.INNOVATION:
            # Innovation harder to assess, more random
            import random
            base_score += random.uniform(-2, 3)
            
        elif criteria == EvaluationCriteria.ALIGNMENT:
            # Check for keywords that indicate alignment
            alignment_keywords = ["ai", "machine learning", "data", "automation", "intelligence"]
            keyword_count = sum(1 for keyword in alignment_keywords 
                              if keyword.lower() in idea.description.lower())
            base_score += keyword_count * 0.5
            
        elif criteria == EvaluationCriteria.RESOURCE_REQUIREMENTS:
            # Inverse scoring - lower requirements = higher score
            complexity_indicators = ["complex", "enterprise", "large-scale", "distributed"]
            complexity_count = sum(1 for indicator in complexity_indicators 
                                 if indicator.lower() in idea.description.lower())
            base_score += (3 - complexity_count) * 0.7
            
        elif criteria == EvaluationCriteria.STRATEGIC_VALUE:
            # Higher priority suggests higher strategic value
            base_score += (idea.priority - 3) * 0.8
            
        # Apply member-specific adjustments
        if criteria.value in member.expertise_areas:
            # Expert members are more discerning
            base_score *= 0.9  # Slight downward adjustment
        
        # Add some realistic variance
        import random
        variance = random.uniform(-0.5, 0.5)
        base_score += variance
        
        # Ensure score stays within bounds
        return max(0.0, min(10.0, base_score))
    
    def _generate_score_reasoning(self, idea: Idea, criteria: EvaluationCriteria, 
                                 score: float, member: CouncilMemberProfile) -> str:
        """Generate reasoning for a specific score."""
        reasoning_templates = {
            EvaluationCriteria.FEASIBILITY: [
                f"Based on the description complexity and scope, feasibility appears {'high' if score > 6 else 'moderate' if score > 4 else 'challenging'}.",
                f"Technical implementation seems {'straightforward' if score > 7 else 'complex' if score < 5 else 'achievable with effort'}.",
            ],
            EvaluationCriteria.INNOVATION: [
                f"Innovation level is {'exceptional' if score > 8 else 'good' if score > 6 else 'moderate' if score > 4 else 'limited'}.",
                f"The approach {'introduces novel concepts' if score > 7 else 'builds on existing ideas' if score > 5 else 'follows conventional patterns'}.",
            ],
            EvaluationCriteria.ALIGNMENT: [
                f"Strategic alignment with AI-Galaxy objectives is {'excellent' if score > 7 else 'good' if score > 5 else 'partial'}.",
                f"The idea {'strongly supports' if score > 6 else 'partially aligns with' if score > 4 else 'loosely connects to'} our core mission.",
            ]
        }
        
        template = reasoning_templates.get(criteria, [f"Score of {score:.1f} based on evaluation criteria."])
        return template[0] if template else f"Evaluated at {score:.1f} points."
    
    def _calculate_score_confidence(self, criteria: EvaluationCriteria, 
                                   member: CouncilMemberProfile) -> float:
        """Calculate confidence level for a score."""
        base_confidence = 0.7
        
        # Higher confidence in areas of expertise
        if criteria.value in member.expertise_areas:
            base_confidence += 0.2
        
        # Adjust based on criteria difficulty
        difficulty_adjustments = {
            EvaluationCriteria.FEASIBILITY: 0.1,
            EvaluationCriteria.INNOVATION: -0.1,  # Harder to assess
            EvaluationCriteria.ALIGNMENT: 0.05,
            EvaluationCriteria.MARKET_POTENTIAL: -0.05
        }
        
        adjustment = difficulty_adjustments.get(criteria, 0.0)
        confidence = base_confidence + adjustment
        
        return max(0.1, min(1.0, confidence))
    
    def _determine_member_vote(self, overall_score: float, member: CouncilMemberProfile, 
                              idea: Idea) -> DecisionOutcome:
        """Determine how a member votes based on their profile and the score."""
        # Adjust threshold based on risk tolerance and idea priority
        adjusted_threshold = member.decision_threshold
        
        # High-priority ideas get slightly lower threshold
        if idea.priority >= 8:
            adjusted_threshold -= 0.5
        
        # Risk-averse members have higher thresholds for risky ideas
        if member.risk_tolerance < 0.5 and overall_score < 6.0:
            adjusted_threshold += 1.0
        
        if overall_score >= adjusted_threshold:
            return DecisionOutcome.APPROVED
        elif overall_score >= adjusted_threshold - 1.5:
            # Borderline cases might need revision
            return DecisionOutcome.NEEDS_REVISION if overall_score >= 4.0 else DecisionOutcome.REJECTED
        else:
            return DecisionOutcome.REJECTED
    
    def _generate_member_concerns(self, scores: List[EvaluationScore], 
                                 member: CouncilMemberProfile, idea: Idea) -> List[str]:
        """Generate concerns based on low scores in member's areas of expertise."""
        concerns = []
        
        for score in scores:
            if score.score < 5.0 and score.criteria.value in member.expertise_areas:
                concern_templates = {
                    "feasibility": "Implementation complexity may exceed current capabilities",
                    "resource_requirements": "Resource demands appear higher than optimal",
                    "risk_assessment": "Potential risks need more thorough mitigation planning",
                    "technical_complexity": "Technical challenges may impact delivery timeline"
                }
                
                template = concern_templates.get(score.criteria.value, 
                                               f"Low {score.criteria.value} score requires attention")
                concerns.append(template)
        
        return concerns[:3]  # Limit to top 3 concerns
    
    def _generate_member_recommendations(self, scores: List[EvaluationScore], 
                                        member: CouncilMemberProfile, idea: Idea) -> List[str]:
        """Generate recommendations based on member expertise and scoring."""
        recommendations = []
        
        # Type-specific recommendations
        if member.member_type == CouncilMemberType.TECHNICAL_LEAD:
            recommendations.extend([
                "Conduct technical feasibility study",
                "Define clear architecture requirements",
                "Identify potential technical risks early"
            ])
        elif member.member_type == CouncilMemberType.RESOURCE_MANAGER:
            recommendations.extend([
                "Develop detailed resource allocation plan",
                "Establish clear project milestones",
                "Define success metrics and KPIs"
            ])
        elif member.member_type == CouncilMemberType.INNOVATION_ADVOCATE:
            recommendations.extend([
                "Explore additional innovation opportunities",
                "Consider broader ecosystem impact",
                "Plan for knowledge sharing and documentation"
            ])
        
        return recommendations[:2]  # Limit recommendations
    
    def _calculate_decision_confidence(self, overall_score: float, 
                                      member: CouncilMemberProfile) -> float:
        """Calculate member's confidence in their decision."""
        # Higher confidence when score is far from threshold
        distance_from_threshold = abs(overall_score - member.decision_threshold)
        base_confidence = 0.6 + (distance_from_threshold / 10.0) * 0.4
        
        return max(0.3, min(1.0, base_confidence))
    
    def _generate_vote_reasoning(self, overall_score: float, vote: DecisionOutcome, 
                                member: CouncilMemberProfile, idea: Idea) -> str:
        """Generate reasoning for the member's vote."""
        score_description = "strong" if overall_score > 7 else "moderate" if overall_score > 5 else "weak"
        
        if vote == DecisionOutcome.APPROVED:
            return f"The idea demonstrates {score_description} potential with a score of {overall_score:.1f}. " \
                   f"Aligns well with {member.member_type.value.replace('_', ' ')} priorities."
        elif vote == DecisionOutcome.REJECTED:
            return f"Score of {overall_score:.1f} falls below acceptance threshold. " \
                   f"Significant concerns from {member.member_type.value.replace('_', ' ')} perspective."
        else:
            return f"Score of {overall_score:.1f} suggests potential but requires refinement. " \
                   f"Addressing key concerns could improve viability."
    
    def _analyze_voting_results(self, votes: List[CouncilVote]) -> VotingResult:
        """Analyze the voting pattern to determine result type."""
        approval_votes = sum(1 for vote in votes if vote.vote == DecisionOutcome.APPROVED)
        rejection_votes = sum(1 for vote in votes if vote.vote == DecisionOutcome.REJECTED)
        total_votes = len(votes)
        
        approval_percentage = (approval_votes / total_votes) * 100
        
        if approval_votes == total_votes:
            return VotingResult.UNANIMOUS_APPROVAL
        elif rejection_votes == total_votes:
            return VotingResult.UNANIMOUS_REJECTION
        elif approval_percentage >= 70:
            return VotingResult.MAJORITY_APPROVAL
        elif approval_percentage <= 30:
            return VotingResult.MAJORITY_REJECTION
        else:
            return VotingResult.SPLIT_DECISION
    
    def _determine_final_decision(self, votes: List[CouncilVote], 
                                 voting_result: VotingResult) -> DecisionOutcome:
        """Determine final decision based on votes and configuration."""
        approval_votes = sum(1 for vote in votes if vote.vote == DecisionOutcome.APPROVED)
        total_votes = len(votes)
        approval_percentage = (approval_votes / total_votes) * 100
        
        # Calculate weighted consensus score
        weighted_score = sum(vote.overall_score * vote.confidence for vote in votes) / \
                        sum(vote.confidence for vote in votes)
        
        # High-impact ideas require unanimous approval if configured
        if (self.config.require_unanimous_for_high_impact and 
            weighted_score >= self.config.high_impact_threshold):
            if voting_result == VotingResult.UNANIMOUS_APPROVAL:
                return DecisionOutcome.APPROVED
            else:
                return DecisionOutcome.NEEDS_REVISION
        
        # Standard decision logic
        if weighted_score >= self.config.approval_threshold:
            if approval_percentage >= (self.config.consensus_requirement * 100):
                return DecisionOutcome.APPROVED
            else:
                return DecisionOutcome.NEEDS_REVISION
        elif weighted_score <= self.config.rejection_threshold:
            return DecisionOutcome.REJECTED
        else:
            return DecisionOutcome.NEEDS_REVISION
    
    def _calculate_consensus_score(self, votes: List[CouncilVote]) -> float:
        """Calculate consensus score based on vote similarity."""
        if not votes:
            return 0.0
        
        # Weight by confidence
        weighted_scores = [vote.overall_score * vote.confidence for vote in votes]
        weights = [vote.confidence for vote in votes]
        
        return sum(weighted_scores) / sum(weights)
    
    def _calculate_approval_percentage(self, votes: List[CouncilVote]) -> float:
        """Calculate percentage of approval votes."""
        if not votes:
            return 0.0
        
        approval_votes = sum(1 for vote in votes if vote.vote == DecisionOutcome.APPROVED)
        return (approval_votes / len(votes)) * 100
    
    def _extract_key_strengths(self, votes: List[CouncilVote]) -> List[str]:
        """Extract key strengths mentioned across votes."""
        strengths = []
        
        for vote in votes:
            for score in vote.scores:
                if score.score >= 7.0:
                    strength = f"Strong {score.criteria.value.replace('_', ' ')}"
                    if strength not in strengths:
                        strengths.append(strength)
        
        return strengths[:5]  # Top 5 strengths
    
    def _extract_key_concerns(self, votes: List[CouncilVote]) -> List[str]:
        """Extract key concerns mentioned across votes."""
        all_concerns = []
        
        for vote in votes:
            all_concerns.extend(vote.concerns)
        
        # Count frequency and return most common
        concern_counts = {}
        for concern in all_concerns:
            concern_counts[concern] = concern_counts.get(concern, 0) + 1
        
        # Sort by frequency and return top concerns
        sorted_concerns = sorted(concern_counts.items(), key=lambda x: x[1], reverse=True)
        return [concern for concern, count in sorted_concerns[:5]]
    
    def _extract_recommendations(self, votes: List[CouncilVote]) -> List[str]:
        """Extract recommendations from council votes."""
        all_recommendations = []
        
        for vote in votes:
            all_recommendations.extend(vote.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]  # Top 5 recommendations
    
    def _assess_strategic_alignment(self, idea: Idea, votes: List[CouncilVote]) -> float:
        """Assess strategic alignment based on council feedback."""
        alignment_scores = []
        
        for vote in votes:
            for score in vote.scores:
                if score.criteria == EvaluationCriteria.ALIGNMENT:
                    alignment_scores.append(score.score * score.weight)
        
        if not alignment_scores:
            return 5.0  # Default neutral score
        
        return sum(alignment_scores) / len(alignment_scores)
    
    def _estimate_resource_requirements(self, idea: Idea, votes: List[CouncilVote]) -> Dict[str, Any]:
        """Estimate resource requirements based on council analysis."""
        # Simplified resource estimation
        complexity_score = 5.0  # Default
        
        for vote in votes:
            for score in vote.scores:
                if score.criteria == EvaluationCriteria.RESOURCE_REQUIREMENTS:
                    complexity_score = 10.0 - score.score  # Inverse relationship
                    break
        
        # Map complexity to resource estimates
        if complexity_score <= 3:
            effort_level = "Low"
            estimated_hours = 40
            team_size = 1
        elif complexity_score <= 6:
            effort_level = "Medium"
            estimated_hours = 120
            team_size = 2
        else:
            effort_level = "High"
            estimated_hours = 300
            team_size = 4
        
        return {
            "effort_level": effort_level,
            "estimated_hours": estimated_hours,
            "recommended_team_size": team_size,
            "complexity_score": complexity_score,
            "priority_level": idea.priority
        }
    
    def _check_appeal_eligibility(self, consensus_score: float, decision: DecisionOutcome) -> bool:
        """Check if an idea is eligible for appeal."""
        if not self.config.enable_appeal_process:
            return False
        
        # Appeals allowed for close decisions
        if decision == DecisionOutcome.REJECTED and consensus_score >= self.config.appeal_eligibility_score:
            return True
        
        if decision == DecisionOutcome.NEEDS_REVISION:
            return True
        
        return False
    
    def _calculate_implementation_priority(self, consensus_score: float, 
                                          strategic_alignment: float, idea_priority: int) -> int:
        """Calculate implementation priority combining multiple factors."""
        # Weighted average of factors
        weighted_score = (consensus_score * 0.4 + 
                         strategic_alignment * 0.3 + 
                         idea_priority * 0.3)
        
        # Map to 1-10 scale
        if weighted_score >= 8.5:
            return 10
        elif weighted_score >= 7.5:
            return 8
        elif weighted_score >= 6.5:
            return 6
        elif weighted_score >= 5.5:
            return 4
        elif weighted_score >= 4.5:
            return 3
        else:
            return 1
    
    def _update_decision_metrics(self, report: EvaluationReport, start_time: datetime):
        """Update internal decision metrics."""
        self.decision_metrics["total_evaluations"] += 1
        
        if report.final_decision == DecisionOutcome.APPROVED:
            self.decision_metrics["approvals"] += 1
        elif report.final_decision == DecisionOutcome.REJECTED:
            self.decision_metrics["rejections"] += 1
        
        # Update average evaluation time
        evaluation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        current_avg = self.decision_metrics["average_evaluation_time"]
        total_evals = self.decision_metrics["total_evaluations"]
        
        new_avg = ((current_avg * (total_evals - 1)) + evaluation_time) / total_evals
        self.decision_metrics["average_evaluation_time"] = new_avg
    
    def _is_appeal_valid(self, appeal_request: AppealRequest) -> bool:
        """Validate if an appeal request is legitimate."""
        idea_id = appeal_request.idea_id
        
        # Check if idea exists in history
        if idea_id not in self.evaluation_history:
            return False
        
        # Check if last decision is appealable
        latest_evaluation = self.evaluation_history[idea_id][-1]
        if not latest_evaluation.appeal_eligibility:
            return False
        
        # Check appeal count limits
        appeal_count = len(self.appeal_history.get(idea_id, []))
        if appeal_count >= self.config.max_appeal_rounds:
            return False
        
        # Check time limits (appeals within 30 days)
        time_since_evaluation = datetime.now(timezone.utc) - latest_evaluation.evaluation_timestamp
        if time_since_evaluation > timedelta(days=30):
            return False
        
        return True
    
    def _reevaluate_with_appeal_evidence(self, original_votes: List[CouncilVote], 
                                        appeal_request: AppealRequest) -> List[CouncilVote]:
        """Reevaluate council votes considering appeal evidence."""
        enhanced_votes = []
        
        # Simulate how additional evidence might change votes
        for original_vote in original_votes:
            # Create modified vote considering appeal evidence
            adjusted_score = original_vote.overall_score
            
            # Moderate positive adjustment for appeal evidence
            if appeal_request.additional_evidence:
                adjustment = min(1.0, len(appeal_request.additional_evidence) * 0.3)
                adjusted_score = min(10.0, adjusted_score + adjustment)
            
            # Create new vote with adjusted score
            enhanced_vote = CouncilVote(
                member_name=original_vote.member_name,
                member_type=original_vote.member_type,
                vote=self._determine_appeal_vote(adjusted_score, original_vote),
                scores=original_vote.scores,  # Keep original detailed scores
                overall_score=adjusted_score,
                confidence=min(1.0, original_vote.confidence + 0.1),
                reasoning=f"Appeal review: {original_vote.reasoning} Additional evidence considered.",
                concerns=original_vote.concerns,
                recommendations=original_vote.recommendations
            )
            
            enhanced_votes.append(enhanced_vote)
        
        return enhanced_votes
    
    def _determine_appeal_vote(self, adjusted_score: float, original_vote: CouncilVote) -> DecisionOutcome:
        """Determine vote outcome for appeal based on adjusted score."""
        if adjusted_score >= 6.5:
            return DecisionOutcome.APPROVED
        elif adjusted_score >= 4.0:
            return DecisionOutcome.NEEDS_REVISION
        else:
            return DecisionOutcome.REJECTED
    
    def _create_appeal_evaluation_report(self, idea_id: str, enhanced_votes: List[CouncilVote],
                                        appeal_request: AppealRequest, 
                                        original_report: EvaluationReport) -> EvaluationReport:
        """Create evaluation report for appeal decision."""
        voting_result = self._analyze_voting_results(enhanced_votes)
        final_decision = self._determine_final_decision(enhanced_votes, voting_result)
        consensus_score = self._calculate_consensus_score(enhanced_votes)
        approval_percentage = self._calculate_approval_percentage(enhanced_votes)
        
        return EvaluationReport(
            idea_id=idea_id,
            evaluation_timestamp=datetime.now(timezone.utc),
            council_votes=enhanced_votes,
            final_decision=final_decision,
            consensus_score=consensus_score,
            voting_result=voting_result,
            approval_percentage=approval_percentage,
            key_strengths=self._extract_key_strengths(enhanced_votes),
            key_concerns=self._extract_key_concerns(enhanced_votes),
            recommendations=self._extract_recommendations(enhanced_votes),
            resource_requirements=original_report.resource_requirements,
            strategic_alignment_score=original_report.strategic_alignment_score + 0.5,  # Slight boost
            appeal_eligibility=False,  # No further appeals after appeal
            implementation_priority=self._calculate_implementation_priority(
                consensus_score, original_report.strategic_alignment_score + 0.5, 5
            )
        )
    
    # Metrics calculation helpers
    
    def _calculate_member_average_score(self, member_name: str) -> float:
        """Calculate average score given by a specific member."""
        all_scores = []
        
        for evaluations in self.evaluation_history.values():
            for evaluation in evaluations:
                for vote in evaluation.council_votes:
                    if vote.member_name == member_name:
                        all_scores.append(vote.overall_score)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    def _calculate_member_approval_tendency(self, member_name: str) -> float:
        """Calculate how often a member approves ideas."""
        approvals = 0
        total_votes = 0
        
        for evaluations in self.evaluation_history.values():
            for evaluation in evaluations:
                for vote in evaluation.council_votes:
                    if vote.member_name == member_name:
                        total_votes += 1
                        if vote.vote == DecisionOutcome.APPROVED:
                            approvals += 1
        
        return (approvals / total_votes) * 100 if total_votes > 0 else 0.0
    
    def _calculate_member_consistency(self, member_name: str) -> float:
        """Calculate scoring consistency for a member."""
        all_scores = []
        
        for evaluations in self.evaluation_history.values():
            for evaluation in evaluations:
                for vote in evaluation.council_votes:
                    if vote.member_name == member_name:
                        all_scores.append(vote.overall_score)
        
        if len(all_scores) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        std_dev = math.sqrt(variance)
        
        cv = std_dev / mean_score if mean_score > 0 else 0
        return max(0.0, 1.0 - cv)  # Convert to consistency score
    
    def _calculate_average_consensus_score(self) -> float:
        """Calculate average consensus score across all evaluations."""
        all_scores = []
        
        for evaluations in self.evaluation_history.values():
            for evaluation in evaluations:
                all_scores.append(evaluation.consensus_score)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    def _calculate_average_strategic_alignment(self) -> float:
        """Calculate average strategic alignment across all evaluations."""
        all_scores = []
        
        for evaluations in self.evaluation_history.values():
            for evaluation in evaluations:
                all_scores.append(evaluation.strategic_alignment_score)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0


# Factory function for easy agent creation
def create_council_agent(config: Optional[CouncilConfiguration] = None,
                        state_router: Optional[StateMachineRouter] = None) -> CouncilAgent:
    """
    Create a new Council Agent instance.
    
    Args:
        config: Optional council configuration
        state_router: Optional state machine router
        
    Returns:
        Configured CouncilAgent instance
    """
    return CouncilAgent(config, state_router)


# Export main classes and functions
__all__ = [
    "CouncilAgent",
    "CouncilConfiguration", 
    "EvaluationReport",
    "AppealRequest",
    "EvaluationCriteria",
    "DecisionOutcome",
    "VotingResult",
    "create_council_agent"
]