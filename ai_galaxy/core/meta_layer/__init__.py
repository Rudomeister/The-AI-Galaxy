"""
Higher Meta-Layer for the AI-Galaxy ecosystem.

This module contains the strategic intelligence components that operate at the
highest level of the system, analyzing patterns, detecting gaps, and suggesting
strategic improvements for optimal ecosystem evolution.
"""

from .meta_planner import (
    MetaPlannerAgent,
    MetaPlannerConfiguration,
    GapAnalysisResult,
    TrendAnalysis,
    DepartmentProposal,
    StrategicInsight,
    EcosystemHealthMetrics,
    GapType,
    TrendDirection,
    PlannerDecision,
    AnalysisConfidence,
    create_meta_planner_agent
)

__all__ = [
    "MetaPlannerAgent",
    "MetaPlannerConfiguration", 
    "GapAnalysisResult",
    "TrendAnalysis",
    "DepartmentProposal",
    "StrategicInsight",
    "EcosystemHealthMetrics",
    "GapType",
    "TrendDirection",
    "PlannerDecision",
    "AnalysisConfidence",
    "create_meta_planner_agent"
]