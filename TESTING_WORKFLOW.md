# Testing the AI-Galaxy Idea Workflow

This guide shows you how to test the complete idea processing workflow using the new REST API.

## Prerequisites

1. **AI-Galaxy System Running**:
   ```bash
   # Start infrastructure services
   docker-compose up -d
   
   # Activate conda environment
   conda activate Galaxian
   
   # Start AI-Galaxy with API
   python main.py --config config.yaml --verbose
   ```

2. **Verify System is Ready**:
   - AI-Galaxy logs show: `[SUCCESS] AI-Galaxy system initialized successfully`
   - API server started: `API server started on 0.0.0.0:8080`
   - All 8 agents are active and sending heartbeats

## Method 1: Using the API Documentation (Recommended)

1. **Open API Documentation**: http://localhost:8080/docs

2. **Check System Health**:
   - Go to `GET /health` endpoint
   - Click "Try it out" → "Execute"
   - Should show status: "healthy" with all services: true

3. **Submit a Test Idea**:
   - Go to `POST /api/v1/ideas/` endpoint
   - Click "Try it out"
   - Use this sample idea:

```json
{
  "title": "AI-Powered Code Review Assistant",
  "description": "Create an intelligent code review assistant that uses machine learning to automatically detect potential bugs, security vulnerabilities, and suggest code improvements. The system should integrate with Git repositories and provide real-time feedback to developers.",
  "priority": "high",
  "tags": [
    "machine-learning",
    "code-analysis", 
    "developer-tools",
    "automation"
  ],
  "metadata": {
    "estimated_complexity": "high",
    "target_languages": ["python", "javascript", "go"],
    "integration_requirements": ["git", "github", "gitlab"]
  },
  "department_hint": "machine_learning"
}
```

4. **Monitor Progress**:
   - Copy the returned `idea_id`
   - Use `GET /api/v1/ideas/{idea_id}` to check status
   - Use `GET /api/v1/agents/` to see agent activity
   - Use `GET /health/metrics` for system metrics

## Method 2: Using the Test Script

Run the automated test script:

```bash
# Install dependencies if needed
pip install aiohttp websockets

# Run the test
python test_idea_workflow.py
```

The script will:
- ✅ Check API health
- 🤖 Show current agent status  
- 💡 Submit a test idea
- 📡 Monitor via WebSocket
- 📈 Track progress updates
- 📊 Show final status

## Expected Workflow Sequence

When you submit an idea, you should observe:

### 1. Initial Creation
```bash
[INFO] Idea created with ID: abc-123-def
[INFO] Vector indexing completed
[INFO] Background workflow started
```

### 2. Agent Processing Chain
```bash
[INFO] validator_agent: Processing idea abc-123-def
[INFO] router_agent: Routing to machine_learning department
[INFO] council_agent: Strategic review initiated
[INFO] creator_agent: Template creation started
```

### 3. Status Transitions
- `created` → `validated` → `council_review` → `approved` → `in_development`

### 4. Real-time Updates
WebSocket events show:
- Agent assignments
- Status transitions  
- Processing milestones
- Error handling (if any)

## Monitoring Endpoints

### System Health
- `GET /health` - Basic health check
- `GET /health/metrics` - Comprehensive metrics
- `GET /health/status` - Detailed system status

### Agent Monitoring  
- `GET /api/v1/agents/` - All agent status
- `GET /api/v1/agents/{name}` - Specific agent details
- `POST /api/v1/agents/{name}/command` - Send agent commands

### Idea Management
- `GET /api/v1/ideas/` - List all ideas (with filters)
- `GET /api/v1/ideas/{id}` - Specific idea details
- `POST /api/v1/ideas/{id}/transition` - Manual state transitions

### Real-time Monitoring
- `ws://localhost:8080/ws/agents` - Agent events
- `ws://localhost:8080/ws/system` - System metrics
- `ws://localhost:8080/ws/workflows` - Workflow events

## Troubleshooting

### API Not Responding
```bash
# Check if AI-Galaxy is running
ps aux | grep python

# Check Docker services
docker-compose ps

# Check port availability
netstat -an | grep 8080
```

### Agents Not Processing
```bash
# Check agent status
curl http://localhost:8080/api/v1/agents/

# Restart specific agent
curl -X POST http://localhost:8080/api/v1/agents/validator_agent/restart
```

### WebSocket Connection Issues
```bash
# Test WebSocket manually
wscat -c ws://localhost:8080/ws/agents

# Check WebSocket stats
curl http://localhost:8080/ws/stats
```

## Sample Test Ideas

### Machine Learning Project
```json
{
  "title": "Predictive Maintenance System",
  "description": "AI system to predict equipment failures in manufacturing using sensor data and machine learning algorithms.",
  "priority": "high",
  "tags": ["machine-learning", "iot", "manufacturing"],
  "department_hint": "machine_learning"
}
```

### Web Development Project
```json
{
  "title": "Real-time Chat Application",
  "description": "Build a scalable real-time chat application with WebSocket support, message persistence, and user authentication.",
  "priority": "medium", 
  "tags": ["web-development", "real-time", "websockets"],
  "department_hint": "web_development"
}
```

### Data Engineering Project
```json
{
  "title": "Automated ETL Pipeline",
  "description": "Create an automated data pipeline for extracting, transforming, and loading data from multiple sources into a data warehouse.",
  "priority": "high",
  "tags": ["data-engineering", "etl", "automation"],
  "department_hint": "data_engineering"
}
```

## Success Indicators

✅ **API is working** when:
- Health endpoints return 200 status
- Agent endpoints show 8 active agents
- Ideas can be created and retrieved

✅ **Workflow is functioning** when:
- Ideas transition through multiple states
- Agents show activity in their task counts
- WebSocket events are being published
- Vector search indexes the ideas

✅ **System is healthy** when:
- All services show healthy status
- Redis and ChromaDB connections are stable
- Agent heartbeats are consistent
- No error logs in the console

Happy testing! 🚀