"""
Standardized logging system for the AI-Galaxy ecosystem.

This module provides consistent logging functionality across all agents,
services, and components in the AI-Galaxy system with structured output,
multiple log levels, and both file and console logging capabilities.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic import BaseModel


class LogLevel(str, Enum):
    """Log level enumeration for consistent logging across the system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogContext(BaseModel):
    """Structured log context for enhanced traceability."""
    agent_name: Optional[str] = None
    idea_id: Optional[str] = None
    department_id: Optional[str] = None
    institution_id: Optional[str] = None
    microservice_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_context: Dict[str, Any] = {}


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with context."""
    
    def format(self, record):
        # Base format
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Extract context if available
        context = getattr(record, 'context', {})
        
        # Build structured log entry
        log_entry = {
            'timestamp': timestamp,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context information
        if context:
            log_entry['context'] = context
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Format as structured string
        formatted_parts = [
            f"{timestamp}",
            f"[{record.levelname}]",
            f"{record.name}",
            f"- {record.getMessage()}"
        ]
        
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items() if v is not None])
            if context_str:
                formatted_parts.append(f"({context_str})")
        
        formatted_message = " ".join(formatted_parts)
        
        if record.exc_info:
            formatted_message += f"\n{self.formatException(record.exc_info)}"
            
        return formatted_message


class AIGalaxyLogger:
    """
    Centralized logger for the AI-Galaxy ecosystem.
    
    Provides structured logging with context awareness and multiple output targets.
    """
    
    def __init__(self, name: str, log_dir: Optional[str] = None):
        """
        Initialize logger with specified name and optional log directory.
        
        Args:
            name: Logger name (typically module or agent name)
            log_dir: Directory for log files (defaults to logs/ in project root)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
            
        # Set up log directory
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)
            
        log_dir.mkdir(exist_ok=True)
        
        # Set up formatters
        structured_formatter = StructuredFormatter()
        simple_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        all_logs_file = log_dir / "ai_galaxy.log"
        file_handler = logging.handlers.RotatingFileHandler(
            all_logs_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(structured_formatter)
        self.logger.addHandler(file_handler)
        
        # Agent-specific log file
        agent_log_file = log_dir / f"{name.replace('.', '_')}.log"
        agent_handler = logging.handlers.RotatingFileHandler(
            agent_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        agent_handler.setLevel(logging.DEBUG)
        agent_handler.setFormatter(structured_formatter)
        self.logger.addHandler(agent_handler)
        
        # Error-only log file
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(structured_formatter)
        self.logger.addHandler(error_handler)
    
    def _log_with_context(self, level: int, message: str, context: Optional[LogContext] = None, **kwargs):
        """Internal method to log with context."""
        extra = {}
        if context:
            # Convert context to dict, filtering out None values
            context_dict = {k: v for k, v in context.dict().items() if v is not None and k != 'additional_context'}
            if context.additional_context:
                context_dict.update(context.additional_context)
            extra['context'] = context_dict
        
        self.logger.log(level, message, extra=extra, **kwargs)
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log debug message with optional context."""
        self._log_with_context(logging.DEBUG, message, context, **kwargs)
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log info message with optional context."""
        self._log_with_context(logging.INFO, message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log warning message with optional context."""
        self._log_with_context(logging.WARNING, message, context, **kwargs)
    
    def error(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log error message with optional context."""
        self._log_with_context(logging.ERROR, message, context, **kwargs)
    
    def critical(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log critical message with optional context."""
        self._log_with_context(logging.CRITICAL, message, context, **kwargs)
    
    def agent_action(self, action: str, agent_name: str, idea_id: Optional[str] = None, 
                    additional_context: Optional[Dict[str, Any]] = None):
        """Log agent actions with standardized context."""
        context = LogContext(
            agent_name=agent_name,
            idea_id=idea_id,
            additional_context=additional_context or {}
        )
        self.info(f"Agent action: {action}", context)
    
    def state_transition(self, from_state: str, to_state: str, idea_id: str, 
                        agent_name: Optional[str] = None, reason: Optional[str] = None):
        """Log state transitions in the workflow."""
        context = LogContext(
            agent_name=agent_name,
            idea_id=idea_id,
            additional_context={
                'from_state': from_state,
                'to_state': to_state,
                'reason': reason
            }
        )
        self.info(f"State transition: {from_state} -> {to_state}", context)
    
    def system_event(self, event: str, additional_context: Optional[Dict[str, Any]] = None):
        """Log system-level events."""
        context = LogContext(additional_context=additional_context or {})
        self.info(f"System event: {event}", context)
    
    def router_decision(self, idea_id: str, current_state: str, next_agent: str, 
                       reason: Optional[str] = None):
        """Log routing decisions made by the state machine."""
        context = LogContext(
            idea_id=idea_id,
            agent_name=next_agent,
            additional_context={
                'current_state': current_state,
                'routing_reason': reason
            }
        )
        self.info(f"Routing decision: {current_state} -> {next_agent}", context)


def get_logger(name: str, log_dir: Optional[str] = None) -> AIGalaxyLogger:
    """
    Factory function to get a logger instance.
    
    Args:
        name: Logger name (typically module or agent name)
        log_dir: Optional custom log directory
        
    Returns:
        AIGalaxyLogger instance
    """
    return AIGalaxyLogger(name, log_dir)


# Export main classes and functions
__all__ = [
    "AIGalaxyLogger",
    "LogContext", 
    "LogLevel",
    "get_logger"
]