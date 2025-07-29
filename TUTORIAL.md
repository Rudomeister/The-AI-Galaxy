# AI-Galaxy Tutorial: Getting Started Guide

Welcome to AI-Galaxy! This tutorial will help you understand and use the AI-Galaxy autonomous development platform from scratch.

## 🌟 What is AI-Galaxy?

AI-Galaxy is a **self-expanding, agent-driven microservice civilization** - an autonomous AI system that:
- Creates microservices from ideas through collaboration between AI agents
- Uses semantic search to find similar projects and avoid duplication  
- Organizes itself into departments and institutions like a digital society
- Learns and evolves its capabilities over time

Think of it as a **digital ecosystem where AI agents work together to build software automatically**.

## 📋 Prerequisites

- **Python 3.8+** (we recommend Python 3.11+)
- **Docker & Docker Compose** (for Redis and ChromaDB)
- **Conda/Anaconda** (recommended for environment management)

## 🚀 Quick Start

### 1. Set Up Your Environment

```bash
# Clone the repository (if you haven't already)
git clone <your-repo-url>
cd The-AI-Galaxy

# Create and activate conda environment
conda create -n Galaxian python=3.11
conda activate Galaxian

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Infrastructure Services

AI-Galaxy needs Redis (for state management) and ChromaDB (for semantic search):

```bash
# Start both services with Docker Compose
docker-compose up -d

# Or use the convenient batch script (Windows)
scripts/start-services.bat

# Check that services are running
docker-compose ps
```

**Services will be available at:**
- **Redis**: `localhost:6379`
- **ChromaDB**: `localhost:8000` (has a web interface!)

### 3. Run AI-Galaxy

```bash
# Activate your conda environment
conda activate Galaxian

# Start the AI-Galaxy system
python main.py --config config.yaml --verbose
```

**You should see:**
```
[INIT] Initializing AI-Galaxy Ecosystem...
[SUCCESS] AI-Galaxy system initialized successfully
[RUNNING] AI-Galaxy system is now running...
Agent initialized and started: router_agent
Agent initialized and started: validator_agent
... (8 agents total)
```

## 🏗️ System Architecture

AI-Galaxy is organized like a digital civilization:

### 🏛️ **Hierarchical Structure**

```
HIGHER-META-LAYER
├── Decides which departments should exist
└── Initiates new ideas when gaps are found

LOWER-META-LAYER (8 Core Agents)
├── 🧭 Router Agent      → Routes ideas to correct departments
├── ✅ Validator Agent   → Validates idea feasibility  
├── 🏛️ Council Agent     → Strategic review and approval
├── 🏗️ Creator Agent     → Creates project templates
├── 📋 Implementer Agent → Manages implementation progress
├── 👨‍💻 Programmer Agent  → Writes actual code
├── 📚 Registrar Agent   → Archives completed services
└── 🧬 Evolution Agent   → System optimization

DEPARTMENTS (e.g., Machine Learning, Web Development)
└── INSTITUTIONS (e.g., Keras Institute, React Institute)
    └── MICROSERVICES (Final output - working software)
```

### 🔄 **Idea Processing Flow**

1. **Idea Creation** → Submit an idea to the system
2. **Validation** → Validator Agent checks feasibility
3. **Council Review** → Strategic evaluation and approval  
4. **Template Creation** → Creator Agent designs structure
5. **Implementation** → Programmer Agent writes code
6. **Routing** → Router Agent finds best department/institution
7. **Registration** → Registrar Agent archives the result

## 🌐 Available Interfaces

### 1. **ChromaDB Web Interface** 
- **URL**: http://localhost:8000
- **Purpose**: Explore the vector search database
- **Features**: 
  - View stored ideas, departments, institutions
  - See semantic search collections
  - Monitor search performance

### 2. **System Logs**
- **Real-time monitoring** of all agent activities
- **Location**: Console output when running `python main.py`
- **Types**: Agent heartbeats, task processing, system health

### 3. **Redis Data** (via CLI)
```bash
# Connect to Redis to see system state
docker exec -it ai-galaxy-redis redis-cli

# View stored keys
KEYS *

# Get agent information
GET agent:router_agent

# Monitor real-time activity  
MONITOR
```

### 4. **Future Interfaces** (Not Yet Implemented)
- **REST API**: Planned for `localhost:8080`
- **Web Dashboard**: Planned for `localhost:3000`

## 💡 How to Use AI-Galaxy

### Understanding the Current State

Right now, AI-Galaxy is in **foundation mode** - the core infrastructure and agents are running, but they need ideas to process. Here's what's happening:

1. **8 agents are active** and sending heartbeats
2. **Redis is storing** system state and agent communications
3. **ChromaDB is ready** to perform semantic searches
4. **The system is waiting** for ideas to process

### Next Steps (Development)

The system is architecturally complete but needs:

1. **Idea Input Interface** - Currently, ideas need to be programmatically inserted
2. **API Endpoints** - For external systems to submit ideas
3. **Web Dashboard** - For visual monitoring and interaction

## 🛠️ Configuration

### Main Configuration (`config.yaml`)

```yaml
# Core services
redis:
  host: "localhost" 
  port: 6379

search:  
  chroma_host: "localhost"
  chroma_port: 8000

# System behavior
orchestrator:
  heartbeat_interval: 30        # Agent heartbeat frequency
  agent_offline_threshold: 3600 # When agents are marked offline
  
system:
  log_level: "INFO"
  enable_api: true             # Enable REST API (when implemented)
  api_port: 8080
  enable_web_ui: true          # Enable web dashboard (when implemented)  
  web_ui_port: 3000
```

### Key Settings to Understand

- **`agent_offline_threshold`**: How long before agents are marked offline (currently 1 hour)
- **`heartbeat_interval`**: How often agents send "I'm alive" messages (30 seconds)
- **`log_level`**: Amount of detail in logs (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## 🔍 Monitoring Your System

### 1. **Agent Health**
```bash
# All agents should show as active in logs
[INFO] orchestrator - Agent status updated: router_agent
[INFO] orchestrator - Agent status updated: validator_agent
```

### 2. **Service Health**
```bash
# Check Docker services
docker-compose ps

# View service logs
docker-compose logs redis
docker-compose logs chromadb
```

### 3. **System Metrics**
The system automatically collects metrics about:
- Active agents and their status
- Task processing statistics  
- Redis and ChromaDB health
- Memory and performance metrics

## 🧪 Testing the System

### Basic Health Check

1. **Verify all services are running**:
```bash
docker-compose ps
# Should show both redis and chromadb as "Up"
```

2. **Check AI-Galaxy logs**:
```bash
python main.py --config config.yaml --verbose
# Should show 8 agents initialized and active
```

3. **Test ChromaDB interface**:
   - Open http://localhost:8000 in your browser
   - You should see the ChromaDB web interface

### Advanced Testing

1. **Monitor Redis activity**:
```bash
docker exec -it ai-galaxy-redis redis-cli monitor
# Shows real-time Redis commands (agent heartbeats, etc.)
```

2. **Check agent heartbeats**:
```bash
docker exec -it ai-galaxy-redis redis-cli
PSUBSCRIBE agent:*
# Shows heartbeat messages from all agents
```

## ❓ Troubleshooting

### Common Issues

#### 1. **Agents Going Offline**
```
[WARNING] orchestrator - Agent router_agent marked as offline
```
**Solution**: This was fixed in the tutorial setup. Agents now send proper heartbeats.

#### 2. **ChromaDB Connection Failed**  
```
ValueError: Could not connect to a Chroma server
```
**Solution**: 
```bash
docker-compose up -d chromadb
# Wait 10 seconds, then restart AI-Galaxy
```

#### 3. **Redis Connection Failed**
```
ConnectionError: Connection refused
```
**Solution**:
```bash
docker-compose up -d redis  
# Wait 5 seconds, then restart AI-Galaxy
```

#### 4. **Unicode Encoding Errors** (Windows)
```
UnicodeEncodeError: 'charmap' codec can't encode character
```
**Solution**: This was fixed - emoji characters replaced with text indicators.

#### 5. **DateTime Deprecation Warnings**
```
DeprecationWarning: datetime.datetime.utcnow() is deprecated
```
**Solution**: This was fixed - all datetime calls updated to modern timezone-aware approach.

### Getting Help

1. **Check logs first** - Most issues show up in the console output
2. **Verify Docker services** - Use `docker-compose ps` 
3. **Check configuration** - Review `config.yaml` settings
4. **Monitor Redis** - Use `redis-cli monitor` to see system activity

## 🔮 What's Next?

AI-Galaxy is a foundation for autonomous software development. Future enhancements could include:

### Short-term
- **REST API implementation** for idea submission
- **Web dashboard** for visual monitoring
- **Idea input interfaces** for easy interaction

### Medium-term  
- **Natural language idea processing** 
- **GitHub integration** for code generation
- **Automated testing and deployment**

### Long-term
- **Self-modifying code** capabilities
- **Multi-language support** (beyond Python)
- **Distributed agent networks**

## 📚 Key Concepts

### **Agents**
Autonomous AI entities that perform specific functions (routing, validation, programming, etc.)

### **Semantic Search**  
ChromaDB-powered system that finds similar ideas/projects to avoid duplication

### **State Machine**
Workflow engine that manages idea progression through different stages

### **Orchestrator**
Central coordinator that manages agent lifecycle and task distribution

### **Heartbeats**
Regular "I'm alive" messages agents send to stay active in the system

---

## 🎉 Congratulations!

You now have AI-Galaxy running and understand its architecture. The system is ready for enhancement and customization based on your specific needs.

**Remember**: AI-Galaxy is a foundation - it's designed to grow and evolve. Start experimenting and building upon this base to create your own autonomous development platform!

---

*Generated with AI-Galaxy Creator Agent* 🤖