"""
State Machine module for the AI-Galaxy ecosystem.

This module provides the YAML-driven workflow system that routes ideas
through various stages and agents in the AI-Galaxy system.

The state machine handles:
- Idea workflow management
- State transitions and validation
- Agent assignment and coordination
- Progress tracking and monitoring
- Error handling and recovery

Main Components:
- StateMachineRouter: Core routing and workflow management
- YAML Configuration: Declarative workflow definition
- Logging Integration: Comprehensive audit trail
"""

from .router import (
    StateMachineRouter,
    RoutingDecision,
    TransitionResult,
    StateTransitionError,
    ConfigurationError,
    create_router
)

# Version information
__version__ = "1.0.0"
__author__ = "AI-Galaxy Team"

# Export main components
__all__ = [
    "StateMachineRouter",
    "RoutingDecision", 
    "TransitionResult",
    "StateTransitionError",
    "ConfigurationError",
    "create_router"
]