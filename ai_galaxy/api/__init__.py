"""
AI-Galaxy REST API Module.

This module provides a comprehensive REST API for the AI-Galaxy ecosystem,
enabling external access to agent management, idea processing, system monitoring,
and real-time communication with the autonomous development platform.
"""

from .main import create_app

__all__ = ["create_app"]