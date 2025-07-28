# AI-Galaxy Core Services

This directory contains the essential core services that power the AI-Galaxy ecosystem. These services provide the foundational infrastructure for agent communication, data persistence, semantic search, and system orchestration.

## Services Overview

### 1. Redis Service (`redis/`)
**Location**: `ai_galaxy/services/redis/db.py`

The Redis service provides:
- **State Management**: Persistent storage for system state and agent data
- **Pub/Sub Messaging**: Real-time communication between agents
- **Session Management**: User and agent session tracking
- **Caching**: High-performance caching for frequently accessed data
- **Distributed Locking**: Coordination mechanisms for concurrent operations
- **Health Monitoring**: Service health checks and metrics

**Key Features**:
- Connection pooling for optimal performance
- Automatic retry and failover handling
- Comprehensive logging and error handling
- TTL support for automatic data expiration
- Atomic operations using Lua scripts

### 2. Vector Search Service (`vector_search/`)
**Location**: `ai_galaxy/services/vector_search/query_service.py`

The Vector Search service provides:
- **Semantic Search**: Find similar ideas, departments, and microservices
- **Vector Embeddings**: Generate and store semantic embeddings
- **Content Indexing**: Index all system entities for searchability
- **Similarity Ranking**: Rank results by semantic similarity
- **Multi-entity Search**: Search across different entity types
- **Smart Routing**: Help agents find relevant resources

**Key Features**:
- ChromaDB integration for vector storage
- Sentence transformer models for embeddings
- Configurable similarity thresholds
- Caching for improved performance
- Batch indexing capabilities
- Comprehensive search statistics

### 3. System Orchestrator
**Location**: `ai_galaxy/orchestrator.py`

The System Orchestrator provides:
- **Agent Registration**: Manage agent lifecycle and capabilities
- **Task Distribution**: Assign tasks to appropriate agents
- **Workflow Management**: Coordinate multi-step processes
- **Inter-agent Communication**: Message routing and delivery
- **Health Monitoring**: Track agent and system health
- **Metrics Collection**: Gather performance and usage statistics

**Key Features**:
- Async task processing with dependency management
- Agent heartbeat monitoring
- Automatic task retry and error handling
- Event-driven architecture with custom handlers
- Comprehensive logging and debugging
- Graceful shutdown and cleanup

## Integration Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main App      │    │  Orchestrator   │    │   Agents        │
│   (main.py)     │◄──►│                 │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Redis Service  │◄──►│ Vector Search   │    │   State         │
│                 │    │   Service       │    │  Management     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Usage Examples

### Starting the System
```python
from ai_galaxy.main import AIGalaxyApplication, AIGalaxyConfig

# Load configuration
config = AIGalaxyConfig.from_file('config.yaml')

# Create and run application
app = AIGalaxyApplication(config)
success = await app.initialize()
if success:
    await app.run()
```

### Using Redis Service
```python
from ai_galaxy.services.redis import create_redis_service

# Create Redis service
redis_service = create_redis_service()
await redis_service.initialize()

# Store state
await redis_service.set_state('agent:status', {'active': True})

# Publish message
await redis_service.publish_message('agent:router', {'task': 'route_idea'})
```

### Using Vector Search
```python
from ai_galaxy.services.vector_search import create_vector_search_service

# Create search service
search_service = create_vector_search_service()
await search_service.initialize()

# Index an idea
await search_service.index_idea(idea)

# Search for similar ideas
results = await search_service.find_similar_ideas(idea, max_results=5)
```

### Using Orchestrator
```python
from ai_galaxy.orchestrator import create_orchestrator

# Create orchestrator
orchestrator = create_orchestrator()
await orchestrator.initialize()

# Register an agent
await orchestrator.register_agent(
    name='my_agent',
    agent_type='processor',
    capabilities=['data_processing', 'analysis']
)

# Submit a task
task_id = await orchestrator.submit_task(
    task_type='process_data',
    agent_name='my_agent',
    payload={'data': 'example'}
)
```

## Configuration

All services are configurable through the main `config.yaml` file:

```yaml
redis:
  host: "localhost"
  port: 6379
  max_connections: 20

search:
  chroma_host: "localhost"
  chroma_port: 8000
  embedding_model: "all-MiniLM-L6-v2"

orchestrator:
  heartbeat_interval: 30
  task_timeout: 300
  max_concurrent_tasks: 10
```

## Health Monitoring

Each service provides health monitoring capabilities:

- **Redis**: Connection status, memory usage, active connections
- **Vector Search**: Index statistics, query performance, storage usage
- **Orchestrator**: Agent status, task throughput, system metrics

Health data is automatically collected and stored for monitoring and alerting.

## Error Handling

All services implement comprehensive error handling:

- Automatic retry with exponential backoff
- Graceful degradation when services are unavailable
- Detailed logging for debugging and monitoring
- Circuit breaker patterns for external dependencies
- Proper cleanup on shutdown

## Performance Considerations

- **Connection Pooling**: All services use connection pools for optimal performance
- **Async Operations**: Full async/await support for non-blocking operations
- **Caching**: Multi-level caching for frequently accessed data
- **Batch Processing**: Batch operations where possible to reduce overhead
- **Resource Limits**: Configurable limits to prevent resource exhaustion

## Development and Testing

Each service can be tested independently:

```python
# Test Redis service
python -m pytest tests/services/test_redis.py

# Test Vector Search service  
python -m pytest tests/services/test_vector_search.py

# Test Orchestrator
python -m pytest tests/test_orchestrator.py
```

## Security Considerations

- All Redis connections support authentication
- Vector search data is isolated by collection
- Agent communication uses secure message passing
- Sensitive data is not logged or cached unnecessarily
- Proper input validation and sanitization

## Scaling and Deployment

The services are designed for horizontal scaling:

- **Redis**: Can be deployed in cluster mode for high availability
- **Vector Search**: ChromaDB can be deployed as a distributed service
- **Orchestrator**: Multiple orchestrator instances can coordinate
- **Load Balancing**: Services support load balancing and failover

For production deployment, consider:
- Using Redis Sentinel or Cluster for high availability
- Deploying ChromaDB with persistent storage
- Setting up monitoring and alerting
- Implementing proper backup and recovery procedures