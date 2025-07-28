"""
Vector Search Service Package for AI-Galaxy Ecosystem.

This package provides semantic search capabilities for ideas, departments,
institutions, microservices, and capabilities within the AI-Galaxy system.
"""

from .query_service import (
    VectorSearchService,
    SearchConfig,
    SearchResult,
    SearchQuery,
    IndexStats,
    create_vector_search_service
)

__all__ = [
    "VectorSearchService",
    "SearchConfig",
    "SearchResult",
    "SearchQuery", 
    "IndexStats",
    "create_vector_search_service"
]