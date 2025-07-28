"""
Core data models for the AI-Galaxy ecosystem.

This module defines the Pydantic models that represent the main entities
in the AI-Galaxy system, including ideas, departments, institutions,
microservices, and system communication structures.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class IdeaStatus(str, Enum):
    """Status enumeration for ideas flowing through the system."""
    CREATED = "created"
    VALIDATED = "validated"
    COUNCIL_REVIEW = "council_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    TEMPLATE_CREATED = "template_created"
    IMPLEMENTATION_STARTED = "implementation_started"
    CODING_IN_PROGRESS = "coding_in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class EntityStatus(str, Enum):
    """Generic status enumeration for system entities."""
    ACTIVE = "active"
    INACTIVE = "inactive"


class MicroserviceStatus(str, Enum):
    """Status enumeration for microservices."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"


class MessageType(str, Enum):
    """Types of messages for inter-agent communication."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessageStatus(str, Enum):
    """Status of agent messages."""
    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"


class Idea(BaseModel):
    """
    Represents an idea that flows through the AI-Galaxy system.
    
    Ideas are the core units of innovation that move through departments
    and institutions for validation, approval, and implementation.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the idea")
    title: str = Field(..., min_length=1, max_length=200, description="Brief title of the idea")
    description: str = Field(..., min_length=10, description="Detailed description of the idea")
    status: IdeaStatus = Field(default=IdeaStatus.CREATED, description="Current status of the idea")
    priority: int = Field(default=1, ge=1, le=10, description="Priority level (1-10, 10 being highest)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when idea was created")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when idea was last updated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the idea")
    department_assignment: Optional[UUID] = Field(None, description="ID of assigned department")
    institution_assignment: Optional[UUID] = Field(None, description="ID of assigned institution")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class Department(BaseModel):
    """
    Represents a main functional area in the AI-Galaxy system.
    
    Departments are top-level organizational units that contain
    specialized institutions and coordinate high-level operations.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the department")
    name: str = Field(..., min_length=1, max_length=100, description="Name of the department")
    description: str = Field(..., min_length=10, description="Detailed description of the department's purpose")
    status: EntityStatus = Field(default=EntityStatus.ACTIVE, description="Current operational status")
    manifest_path: str = Field(..., description="Path to the department's manifest file")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when department was created")
    institutions: List[UUID] = Field(default_factory=list, description="List of institution IDs under this department")
    microservices_count: int = Field(default=0, ge=0, description="Total count of microservices in this department")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class Institution(BaseModel):
    """
    Represents specialized sectors under departments.
    
    Institutions are specialized units that focus on specific
    technologies, frameworks, or problem domains within a department.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the institution")
    name: str = Field(..., min_length=1, max_length=100, description="Name of the institution")
    description: str = Field(..., min_length=10, description="Detailed description of the institution's specialization")
    department_id: UUID = Field(..., description="ID of the parent department")
    services_path: str = Field(..., description="Path to the institution's services directory")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when institution was created")
    microservices: List[UUID] = Field(default_factory=list, description="List of microservice IDs in this institution")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class Microservice(BaseModel):
    """
    Represents completed functional services in the AI-Galaxy system.
    
    Microservices are the atomic units of functionality that provide
    specific capabilities and can be composed into larger solutions.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the microservice")
    name: str = Field(..., min_length=1, max_length=100, description="Name of the microservice")
    description: str = Field(..., min_length=10, description="Detailed description of the microservice's functionality")
    institution_id: UUID = Field(..., description="ID of the parent institution")
    code_path: str = Field(..., description="Path to the microservice's code directory")
    api_endpoint: Optional[str] = Field(None, description="API endpoint URL if the service exposes an API")
    status: MicroserviceStatus = Field(default=MicroserviceStatus.ACTIVE, description="Current operational status")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when microservice was created")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class AgentMessage(BaseModel):
    """
    Represents messages for inter-agent communication in the AI-Galaxy system.
    
    AgentMessages facilitate communication between different agents,
    enabling coordination and information sharing across the ecosystem.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the message")
    sender_agent: str = Field(..., min_length=1, description="Name/ID of the sending agent")
    receiver_agent: str = Field(..., min_length=1, description="Name/ID of the receiving agent")
    message_type: MessageType = Field(..., description="Type of the message")
    payload: Dict[str, Any] = Field(..., description="Message content and data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when message was created")
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="Current status of the message")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class SystemState(BaseModel):
    """
    Tracks overall system health and metrics for the AI-Galaxy ecosystem.
    
    SystemState provides a high-level view of the system's operational
    status and key performance indicators.
    """
    active_departments: int = Field(default=0, ge=0, description="Number of currently active departments")
    active_institutions: int = Field(default=0, ge=0, description="Number of currently active institutions")
    total_microservices: int = Field(default=0, ge=0, description="Total number of microservices across all institutions")
    ideas_in_progress: int = Field(default=0, ge=0, description="Number of ideas currently being processed")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of last system state update")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# Export all models for easy importing
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