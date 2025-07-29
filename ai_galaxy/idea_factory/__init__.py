"""
Idea Factory module for autonomous idea generation in the AI-Galaxy ecosystem.

This module contains the Idea Factory Agent, which generates new ideas based on 
system needs, gaps, performance analysis, user patterns, and market trends.
"""

from .idea_factory import IdeaFactoryAgent, IdeaFactoryConfiguration, IdeaGenerationRequest

__all__ = [
    "IdeaFactoryAgent",
    "IdeaFactoryConfiguration", 
    "IdeaGenerationRequest"
]