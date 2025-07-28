"""
Vector Search Service for AI-Galaxy Ecosystem.

This module provides a comprehensive semantic search engine for ideas, services,
and capabilities within the AI-Galaxy ecosystem. It handles vector embedding
generation, storage, similarity search, ranking, content indexing, and retrieval
to enable intelligent discovery and matching of system components.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
import traceback

import numpy as np
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

from ...shared.logger import get_logger, LogContext
from ...shared.models import Idea, Department, Institution, Microservice


class SearchConfig(BaseModel):
    """Configuration for vector search service."""
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, description="ChromaDB port")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    max_results: int = Field(default=50, description="Maximum search results")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
    index_batch_size: int = Field(default=100, description="Batch size for indexing")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL for search results")


class SearchResult(BaseModel):
    """Represents a search result with metadata."""
    id: str = Field(..., description="Unique identifier of the result")
    content: str = Field(..., description="Text content that matched")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    entity_type: str = Field(..., description="Type of entity (idea, department, etc.)")
    entity_id: str = Field(..., description="ID of the source entity")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When result was indexed")


class SearchQuery(BaseModel):
    """Represents a search query with parameters."""
    query_text: str = Field(..., min_length=1, description="Search query text")
    entity_types: Optional[List[str]] = Field(default=None, description="Filter by entity types")
    max_results: Optional[int] = Field(default=10, description="Maximum results to return")
    similarity_threshold: Optional[float] = Field(default=0.7, description="Minimum similarity score")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")
    include_metadata: bool = Field(default=True, description="Include metadata in results")


class IndexStats(BaseModel):
    """Statistics about the search index."""
    total_documents: int = Field(default=0, description="Total indexed documents")
    documents_by_type: Dict[str, int] = Field(default_factory=dict, description="Count by entity type")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last index update")
    index_size_mb: float = Field(default=0.0, description="Approximate index size in MB")


class VectorSearchService:
    """
    Vector search service providing semantic search capabilities.
    
    This service enables intelligent discovery and matching of ideas, departments,
    institutions, and microservices using semantic similarity rather than just
    keyword matching. It provides the foundation for smart routing and discovery
    within the AI-Galaxy ecosystem.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize vector search service.
        
        Args:
            config: Search configuration. Uses defaults if not provided.
        """
        self.config = config or SearchConfig()
        self.logger = get_logger("vector_search.service")
        
        # ChromaDB client and collections
        self._client: Optional[chromadb.ClientAPI] = None
        self._collections: Dict[str, chromadb.Collection] = {}
        
        # Collection names for different entity types
        self.COLLECTIONS = {
            'ideas': 'ai_galaxy_ideas',
            'departments': 'ai_galaxy_departments', 
            'institutions': 'ai_galaxy_institutions',
            'microservices': 'ai_galaxy_microservices',
            'capabilities': 'ai_galaxy_capabilities',
            'general': 'ai_galaxy_general'
        }
        
        # Cache for search results and embeddings
        self._search_cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        self._embedding_cache: Dict[str, Tuple[List[float], datetime]] = {}
        
        # Statistics tracking
        self._stats = IndexStats()
        self._is_initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize vector search service and ChromaDB collections.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Initialize ChromaDB client
            settings = Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=self.config.chroma_host,
                chroma_server_http_port=self.config.chroma_port,
                anonymized_telemetry=False
            )
            
            self._client = chromadb.Client(settings)
            
            # Initialize collections for each entity type
            for entity_type, collection_name in self.COLLECTIONS.items():
                try:
                    # Try to get existing collection first
                    collection = self._client.get_collection(collection_name)
                    self.logger.info(f"Connected to existing collection: {collection_name}")
                except:
                    # Create new collection if it doesn't exist
                    collection = self._client.create_collection(
                        name=collection_name,
                        metadata={"description": f"AI-Galaxy {entity_type} search index"}
                    )
                    self.logger.info(f"Created new collection: {collection_name}")
                
                self._collections[entity_type] = collection
            
            # Load existing statistics
            await self._load_statistics()
            
            self._is_initialized = True
            self.logger.info("Vector search service initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector search service: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def shutdown(self):
        """Gracefully shutdown vector search service."""
        self.logger.info("Shutting down vector search service...")
        
        # Clear caches
        self._search_cache.clear()
        self._embedding_cache.clear()
        
        # Save statistics
        await self._save_statistics()
        
        self.logger.info("Vector search service shutdown completed")
    
    # === Document Indexing ===
    
    async def index_idea(self, idea: Idea) -> bool:
        """
        Index an idea for semantic search.
        
        Args:
            idea: Idea object to index
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self._is_initialized:
                await self.initialize()
            
            collection = self._collections['ideas']
            
            # Create searchable content
            content = f"{idea.title}\n{idea.description}"
            if idea.metadata:
                content += f"\nMetadata: {json.dumps(idea.metadata)}"
            
            # Prepare metadata
            metadata = {
                "entity_type": "idea",
                "entity_id": str(idea.id),
                "title": idea.title,
                "status": idea.status,
                "priority": idea.priority,
                "created_at": idea.created_at.isoformat(),
                "department_assignment": str(idea.department_assignment) if idea.department_assignment else None,
                "institution_assignment": str(idea.institution_assignment) if idea.institution_assignment else None
            }
            
            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[str(idea.id)]
            )
            
            # Update statistics
            self._stats.total_documents += 1
            self._stats.documents_by_type["idea"] = self._stats.documents_by_type.get("idea", 0) + 1
            self._stats.last_updated = datetime.now(timezone.utc)
            
            self.logger.debug(f"Indexed idea: {idea.title}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index idea {idea.id}: {str(e)}")
            return False
    
    async def index_department(self, department: Department) -> bool:
        """Index a department for semantic search."""
        try:
            if not self._is_initialized:
                await self.initialize()
            
            collection = self._collections['departments']
            
            # Create searchable content
            content = f"{department.name}\n{department.description}"
            
            # Prepare metadata
            metadata = {
                "entity_type": "department",
                "entity_id": str(department.id),
                "name": department.name,
                "status": department.status,
                "created_at": department.created_at.isoformat(),
                "manifest_path": department.manifest_path,
                "microservices_count": department.microservices_count
            }
            
            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[str(department.id)]
            )
            
            # Update statistics
            self._stats.total_documents += 1
            self._stats.documents_by_type["department"] = self._stats.documents_by_type.get("department", 0) + 1
            self._stats.last_updated = datetime.now(timezone.utc)
            
            self.logger.debug(f"Indexed department: {department.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index department {department.id}: {str(e)}")
            return False
    
    async def index_institution(self, institution: Institution) -> bool:
        """Index an institution for semantic search."""
        try:
            if not self._is_initialized:
                await self.initialize()
            
            collection = self._collections['institutions']
            
            # Create searchable content
            content = f"{institution.name}\n{institution.description}"
            
            # Prepare metadata
            metadata = {
                "entity_type": "institution",
                "entity_id": str(institution.id),
                "name": institution.name,
                "department_id": str(institution.department_id),
                "created_at": institution.created_at.isoformat(),
                "services_path": institution.services_path
            }
            
            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[str(institution.id)]
            )
            
            # Update statistics
            self._stats.total_documents += 1
            self._stats.documents_by_type["institution"] = self._stats.documents_by_type.get("institution", 0) + 1
            self._stats.last_updated = datetime.now(timezone.utc)
            
            self.logger.debug(f"Indexed institution: {institution.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index institution {institution.id}: {str(e)}")
            return False
    
    async def index_microservice(self, microservice: Microservice) -> bool:
        """Index a microservice for semantic search."""
        try:
            if not self._is_initialized:
                await self.initialize()
            
            collection = self._collections['microservices']
            
            # Create searchable content
            content = f"{microservice.name}\n{microservice.description}"
            
            # Prepare metadata
            metadata = {
                "entity_type": "microservice",
                "entity_id": str(microservice.id),
                "name": microservice.name,
                "institution_id": str(microservice.institution_id),
                "status": microservice.status,
                "created_at": microservice.created_at.isoformat(),
                "code_path": microservice.code_path,
                "api_endpoint": microservice.api_endpoint
            }
            
            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[str(microservice.id)]
            )
            
            # Update statistics
            self._stats.total_documents += 1
            self._stats.documents_by_type["microservice"] = self._stats.documents_by_type.get("microservice", 0) + 1
            self._stats.last_updated = datetime.now(timezone.utc)
            
            self.logger.debug(f"Indexed microservice: {microservice.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index microservice {microservice.id}: {str(e)}")
            return False
    
    async def index_capability(self, capability_id: str, name: str, description: str, 
                             tags: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Index a capability for semantic search.
        
        Args:
            capability_id: Unique identifier for the capability
            name: Capability name
            description: Detailed description
            tags: Optional tags for categorization
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self._is_initialized:
                await self.initialize()
            
            collection = self._collections['capabilities']
            
            # Create searchable content
            content = f"{name}\n{description}"
            if tags:
                content += f"\nTags: {', '.join(tags)}"
            
            # Prepare metadata
            meta = {
                "entity_type": "capability",
                "entity_id": capability_id,
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tags": tags or []
            }
            
            if metadata:
                meta.update(metadata)
            
            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[meta],
                ids=[capability_id]
            )
            
            # Update statistics
            self._stats.total_documents += 1
            self._stats.documents_by_type["capability"] = self._stats.documents_by_type.get("capability", 0) + 1
            self._stats.last_updated = datetime.now(timezone.utc)
            
            self.logger.debug(f"Indexed capability: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index capability {capability_id}: {str(e)}")
            return False
    
    # === Search Operations ===
    
    async def semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform semantic search across indexed content.
        
        Args:
            query: Search query with parameters
            
        Returns:
            List of search results sorted by relevance.
        """
        try:
            if not self._is_initialized:
                await self.initialize()
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_result = self._get_cached_search_result(cache_key)
            if cached_result:
                self.logger.debug(f"Returning cached search results for: {query.query_text}")
                return cached_result
            
            all_results = []
            
            # Determine which collections to search
            collections_to_search = self._get_collections_for_query(query)
            
            # Search each relevant collection
            for entity_type, collection in collections_to_search.items():
                try:
                    # Perform similarity search
                    results = collection.query(
                        query_texts=[query.query_text],
                        n_results=min(query.max_results or self.config.max_results, 50),
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    # Process results
                    if results['documents'] and results['documents'][0]:
                        for i, (doc, metadata, distance) in enumerate(zip(
                            results['documents'][0],
                            results['metadatas'][0], 
                            results['distances'][0]
                        )):
                            # Convert distance to similarity score (ChromaDB uses cosine distance)
                            similarity_score = 1.0 - distance
                            
                            # Apply similarity threshold
                            if similarity_score >= (query.similarity_threshold or self.config.similarity_threshold):
                                search_result = SearchResult(
                                    id=results['ids'][0][i],
                                    content=doc,
                                    similarity_score=round(similarity_score, 4),
                                    entity_type=metadata.get('entity_type', entity_type),
                                    entity_id=metadata.get('entity_id', results['ids'][0][i]),
                                    metadata=metadata if query.include_metadata else {},
                                    timestamp=datetime.fromisoformat(metadata.get('created_at', datetime.now(timezone.utc).isoformat()))
                                )
                                all_results.append(search_result)
                
                except Exception as e:
                    self.logger.error(f"Error searching collection {entity_type}: {str(e)}")
                    continue
            
            # Sort by similarity score (descending) and limit results
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = all_results[:query.max_results or self.config.max_results]
            
            # Cache results
            self._cache_search_result(cache_key, final_results)
            
            # Log search performance
            context = LogContext(additional_context={
                "query": query.query_text,
                "results_count": len(final_results),
                "collections_searched": list(collections_to_search.keys())
            })
            self.logger.info(f"Semantic search completed", context)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Failed to perform semantic search: {str(e)}")
            return []
    
    async def find_similar_ideas(self, idea: Idea, max_results: int = 5) -> List[SearchResult]:
        """Find ideas similar to the given idea."""
        query = SearchQuery(
            query_text=f"{idea.title} {idea.description}",
            entity_types=["idea"],
            max_results=max_results,
            similarity_threshold=0.6
        )
        
        results = await self.semantic_search(query)
        
        # Filter out the idea itself
        return [r for r in results if r.entity_id != str(idea.id)]
    
    async def find_relevant_departments(self, idea: Idea, max_results: int = 3) -> List[SearchResult]:
        """Find departments most relevant to an idea."""
        query = SearchQuery(
            query_text=f"{idea.title} {idea.description}",
            entity_types=["department"],
            max_results=max_results,
            similarity_threshold=0.5
        )
        
        return await self.semantic_search(query)
    
    async def find_relevant_institutions(self, idea: Idea, department_id: Optional[UUID] = None, 
                                       max_results: int = 3) -> List[SearchResult]:
        """Find institutions most relevant to an idea."""
        query = SearchQuery(
            query_text=f"{idea.title} {idea.description}",
            entity_types=["institution"],
            max_results=max_results,
            similarity_threshold=0.5
        )
        
        results = await self.semantic_search(query)
        
        # Filter by department if specified
        if department_id:
            results = [r for r in results if r.metadata.get('department_id') == str(department_id)]
        
        return results
    
    async def find_relevant_microservices(self, query_text: str, institution_id: Optional[UUID] = None,
                                        max_results: int = 10) -> List[SearchResult]:
        """Find microservices relevant to a query."""
        query = SearchQuery(
            query_text=query_text,
            entity_types=["microservice"],
            max_results=max_results,
            similarity_threshold=0.4
        )
        
        results = await self.semantic_search(query)
        
        # Filter by institution if specified
        if institution_id:
            results = [r for r in results if r.metadata.get('institution_id') == str(institution_id)]
        
        return results
    
    async def search_capabilities(self, capability_description: str, 
                                max_results: int = 10) -> List[SearchResult]:
        """Search for capabilities matching a description."""
        query = SearchQuery(
            query_text=capability_description,
            entity_types=["capability"],
            max_results=max_results,
            similarity_threshold=0.4
        )
        
        return await self.semantic_search(query)
    
    # === Index Management ===
    
    async def remove_from_index(self, entity_id: str, entity_type: str) -> bool:
        """
        Remove an entity from the search index.
        
        Args:
            entity_id: ID of the entity to remove
            entity_type: Type of entity (idea, department, etc.)
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if entity_type not in self._collections:
                self.logger.warning(f"Unknown entity type: {entity_type}")
                return False
            
            collection = self._collections[entity_type]
            collection.delete(ids=[entity_id])
            
            # Update statistics
            self._stats.total_documents = max(0, self._stats.total_documents - 1)
            self._stats.documents_by_type[entity_type] = max(0, self._stats.documents_by_type.get(entity_type, 0) - 1)
            self._stats.last_updated = datetime.now(timezone.utc)
            
            self.logger.debug(f"Removed {entity_type} {entity_id} from index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove {entity_type} {entity_id} from index: {str(e)}")
            return False
    
    async def update_index_entry(self, entity_id: str, entity_type: str, 
                               content: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing index entry."""
        try:
            # Remove old entry
            await self.remove_from_index(entity_id, entity_type)
            
            # Add updated entry
            if entity_type not in self._collections:
                return False
            
            collection = self._collections[entity_type]
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[entity_id]
            )
            
            # Update statistics (add back since we removed one)
            self._stats.total_documents += 1
            self._stats.documents_by_type[entity_type] = self._stats.documents_by_type.get(entity_type, 0) + 1
            self._stats.last_updated = datetime.now(timezone.utc)
            
            self.logger.debug(f"Updated {entity_type} {entity_id} in index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update {entity_type} {entity_id} in index: {str(e)}")
            return False
    
    async def get_index_statistics(self) -> IndexStats:
        """Get current index statistics."""
        try:
            # Update collection counts
            for entity_type, collection in self._collections.items():
                try:
                    count = collection.count()
                    self._stats.documents_by_type[entity_type] = count
                except:
                    pass
            
            # Update total
            self._stats.total_documents = sum(self._stats.documents_by_type.values())
            
            return self._stats
        except Exception as e:
            self.logger.error(f"Failed to get index statistics: {str(e)}")
            return self._stats
    
    async def clear_index(self, entity_type: Optional[str] = None) -> bool:
        """
        Clear the search index.
        
        Args:
            entity_type: Specific entity type to clear, or None for all
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if entity_type:
                # Clear specific entity type
                if entity_type not in self._collections:
                    return False
                
                collection = self._collections[entity_type]
                
                # Get all IDs and delete them
                results = collection.get()
                if results['ids']:
                    collection.delete(ids=results['ids'])
                
                # Update statistics
                self._stats.documents_by_type[entity_type] = 0
                
            else:
                # Clear all collections
                for entity_type, collection in self._collections.items():
                    results = collection.get()
                    if results['ids']:
                        collection.delete(ids=results['ids'])
                    self._stats.documents_by_type[entity_type] = 0
            
            # Update total count
            self._stats.total_documents = sum(self._stats.documents_by_type.values())
            self._stats.last_updated = datetime.now(timezone.utc)
            
            # Clear caches
            self._search_cache.clear()
            self._embedding_cache.clear()
            
            self.logger.info(f"Cleared index for: {entity_type or 'all entity types'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear index: {str(e)}")
            return False
    
    # === Helper Methods ===
    
    def _get_collections_for_query(self, query: SearchQuery) -> Dict[str, chromadb.Collection]:
        """Get collections to search based on query parameters."""
        if query.entity_types:
            # Search only specified entity types
            collections = {}
            for entity_type in query.entity_types:
                if entity_type in self._collections:
                    collections[entity_type] = self._collections[entity_type]
            return collections
        else:
            # Search all collections
            return self._collections
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        key_data = {
            "query": query.query_text,
            "entity_types": sorted(query.entity_types) if query.entity_types else None,
            "max_results": query.max_results,
            "threshold": query.similarity_threshold,
            "filters": sorted(query.filters.items()) if query.filters else None
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_search_result(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached search result if still valid."""
        if cache_key in self._search_cache:
            results, timestamp = self._search_cache[cache_key]
            if datetime.now(timezone.utc) - timestamp < timedelta(seconds=self.config.cache_ttl_seconds):
                return results
            else:
                # Remove expired cache entry
                del self._search_cache[cache_key]
        return None
    
    def _cache_search_result(self, cache_key: str, results: List[SearchResult]):
        """Cache search results."""
        self._search_cache[cache_key] = (results, datetime.now(timezone.utc))
        
        # Cleanup old cache entries (keep only last 100)
        if len(self._search_cache) > 100:
            oldest_keys = sorted(
                self._search_cache.keys(),
                key=lambda k: self._search_cache[k][1]
            )[:50]
            for key in oldest_keys:
                del self._search_cache[key]
    
    async def _load_statistics(self):
        """Load statistics from persistent storage."""
        try:
            # Load from each collection
            for entity_type, collection in self._collections.items():
                count = collection.count()
                self._stats.documents_by_type[entity_type] = count
            
            self._stats.total_documents = sum(self._stats.documents_by_type.values())
            self._stats.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Failed to load statistics: {str(e)}")
    
    async def _save_statistics(self):
        """Save statistics to persistent storage."""
        try:
            # Statistics are automatically persisted by ChromaDB
            # This could be extended to save to external storage
            pass
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {str(e)}")


# Factory function for easy instantiation
def create_vector_search_service(config: Optional[SearchConfig] = None) -> VectorSearchService:
    """
    Factory function to create a vector search service instance.
    
    Args:
        config: Optional search configuration
        
    Returns:
        VectorSearchService instance
    """
    return VectorSearchService(config)


# Export main classes
__all__ = [
    "VectorSearchService",
    "SearchConfig",
    "SearchResult", 
    "SearchQuery",
    "IndexStats",
    "create_vector_search_service"
]