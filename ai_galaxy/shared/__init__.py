"""
Shared components for the AI-Galaxy ecosystem.

This package contains common models, utilities, and configurations
that are used across different parts of the AI-Galaxy system.
"""

from .models import (
    Idea,
    Department,
    Institution,
    Microservice,
    AgentMessage,
    SystemState,
    IdeaStatus,
    EntityStatus,
    MicroserviceStatus,
    MessageType,
    MessageStatus
)

__all__ = [
    "Idea",
    "Department",
    "Institution", 
    "Microservice",
    "AgentMessage",
    "SystemState",
    "IdeaStatus",
    "EntityStatus",
    "MicroserviceStatus",
    "MessageType",
    "MessageStatus"
]