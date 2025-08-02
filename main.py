"""
AI-Galaxy Main Application Entry Point.

This is the central entry point for the AI-Galaxy ecosystem. It initializes
all core services, starts the system orchestrator, registers agents, and
manages the overall lifecycle of the autonomous AI development platform.

The main application coordinates:
- System initialization and startup
- Agent registration and lifecycle management
- Health monitoring and metrics collection
- Graceful shutdown and cleanup
- Configuration management
- Logging and error handling

Usage:
    python main.py [--config config.yaml] [--verbose] [--daemon]
"""

import asyncio
import argparse
import signal
import sys
import os
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback
# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_galaxy.shared.logger import get_logger, LogContext
from ai_galaxy.shared.models import SystemState
from ai_galaxy.services.redis import RedisConfig, create_redis_service
from ai_galaxy.services.vector_search import SearchConfig, create_vector_search_service
from ai_galaxy.services.workflow_recovery import WorkflowRecoveryService, WorkflowRecoveryConfig, create_workflow_recovery_service
from ai_galaxy.orchestrator import (
    SystemOrchestrator, OrchestratorConfig, create_orchestrator,
    AgentStatus, TaskPriority
)

# Import agents
from ai_galaxy.core.agents.router_agent import RouterAgent
from ai_galaxy.core.agents.router_agent_enhanced import EnhancedRouterAgent
from ai_galaxy.core.agents.validator_agent import ValidatorAgent
from ai_galaxy.core.agents.validator_agent_enhanced import EnhancedValidatorAgent  # Enhanced validator agent
from ai_galaxy.core.agents.council_agent import CouncilAgent
from ai_galaxy.core.agents.council_agent_enhanced import EnhancedCouncilAgent
from ai_galaxy.core.agents.creator_agent import CreatorAgent
from ai_galaxy.core.agents.creator_agent_enhanced import EnhancedCreatorAgent
from ai_galaxy.core.agents.implementer_agent import ImplementerAgent
from ai_galaxy.core.agents.implementer_agent_enhanced import EnhancedImplementerAgent
from ai_galaxy.core.agents.programmer_agent import ProgrammerAgent
from ai_galaxy.core.agents.programmer_agent_enhanced import EnhancedProgrammerAgent
from ai_galaxy.core.agents.registrar_agent import RegistrarAgent
from ai_galaxy.core.agents.registrar_agent_enhanced import EnhancedRegistrarAgent
from ai_galaxy.core.agents.evolution_agent import EvolutionAgent
from ai_galaxy.core.agents.evolution_agent_enhanced import EnhancedEvolutionAgent


class AIGalaxyConfig:
    """Configuration container for the AI-Galaxy system."""
    
    def __init__(self):
        self.redis = RedisConfig()
        self.search = SearchConfig()
        self.orchestrator = OrchestratorConfig()
        self.system = {
            'log_level': 'INFO',
            'enable_api': True,
            'api_host': '0.0.0.0',
            'api_port': 8080,
            'enable_web_ui': True,
            'web_ui_port': 3000,
            'auto_start_agents': True,
            'agent_startup_delay': 2.0,
            'health_check_enabled': True,
            'metrics_collection_enabled': True,
            'workflow_recovery_enabled': True,
            'workflow_recovery_check_interval': 60,
            'workflow_recovery_stuck_threshold': 10
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AIGalaxyConfig':
        """Load configuration from YAML file."""
        config = cls()
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                # Update Redis config
                if 'redis' in yaml_config:
                    for key, value in yaml_config['redis'].items():
                        if hasattr(config.redis, key):
                            setattr(config.redis, key, value)
                
                # Override with environment variables if present
                if os.getenv('REDIS_HOST'):
                    config.redis.host = os.getenv('REDIS_HOST')
                if os.getenv('REDIS_PORT'):
                    config.redis.port = int(os.getenv('REDIS_PORT'))
                if os.getenv('REDIS_DB'):
                    config.redis.db = int(os.getenv('REDIS_DB'))
                if os.getenv('REDIS_PASSWORD'):
                    config.redis.password = os.getenv('REDIS_PASSWORD')
                
                # Update search config
                if 'search' in yaml_config:
                    for key, value in yaml_config['search'].items():
                        if hasattr(config.search, key):
                            setattr(config.search, key, value)
                
                # Override with environment variables if present
                if os.getenv('CHROMA_HOST'):
                    config.search.chroma_host = os.getenv('CHROMA_HOST')
                if os.getenv('CHROMA_PORT'):
                    config.search.chroma_port = int(os.getenv('CHROMA_PORT'))
                
                # Update orchestrator config
                if 'orchestrator' in yaml_config:
                    for key, value in yaml_config['orchestrator'].items():
                        if hasattr(config.orchestrator, key):
                            setattr(config.orchestrator, key, value)
                
                # Update system config
                if 'system' in yaml_config:
                    config.system.update(yaml_config['system'])
                    
            else:
                print(f"Config file {config_path} not found, using defaults")
                
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            print("Using default configuration")
        
        return config


class AIGalaxyApplication:
    """
    Main application class for the AI-Galaxy ecosystem.
    
    This class orchestrates the entire system lifecycle, from initialization
    through operation to graceful shutdown. It serves as the central coordinator
    ensuring all components work together harmoniously.
    """
    
    def __init__(self, config: AIGalaxyConfig):
        """
        Initialize the AI-Galaxy application.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = get_logger("ai_galaxy.main")
        
        # Core components
        self.orchestrator: Optional[SystemOrchestrator] = None
        self.redis_service = None
        self.vector_search_service = None
        self.workflow_recovery_service: Optional[WorkflowRecoveryService] = None
        
        # Agent instances
        self.agents: Dict[str, Any] = {}
        
        # Application state
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        self.shutdown_requested = False
        
        # Background tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def initialize(self) -> bool:
        """
        Initialize the entire AI-Galaxy system.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            self.logger.info("[INIT] Initializing AI-Galaxy Ecosystem...")
            
            # Create and initialize orchestrator
            self.orchestrator = create_orchestrator(
                redis_config=self.config.redis,
                search_config=self.config.search,
                orchestrator_config=self.config.orchestrator
            )
            
            orchestrator_success = await self.orchestrator.initialize()
            if not orchestrator_success:
                self.logger.error("Failed to initialize system orchestrator")
                return False
            
            # Get references to core services
            self.redis_service = self.orchestrator.redis_service
            self.vector_search_service = self.orchestrator.vector_search_service
            
            # Initialize and register agents
            if self.config.system['auto_start_agents']:
                await self._initialize_agents()
            
            # Initialize workflow recovery service
            if self.config.system['workflow_recovery_enabled']:
                await self._initialize_workflow_recovery()
            
            # Start monitoring tasks
            await self._start_monitoring()
            
            # Initialize API server if enabled
            if self.config.system['enable_api']:
                await self._start_api_server()
            
            self.is_running = True
            self.startup_time = datetime.now(timezone.utc)
            
            # Log system startup
            context = LogContext(additional_context={
                'startup_time': self.startup_time.isoformat(),
                'agent_count': len(self.agents),
                'config_summary': {
                    'redis_host': self.config.redis.host,
                    'search_host': self.config.search.chroma_host,
                    'api_enabled': self.config.system['enable_api']
                }
            })
            self.logger.info("[SUCCESS] AI-Galaxy system initialized successfully", context)
            
            # Register system startup event
            await self.orchestrator._emit_event('system_initialized', {
                'startup_time': self.startup_time.isoformat(),
                'agent_count': len(self.agents),
                'version': '1.0.0'
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI-Galaxy system: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def run(self):
        """
        Run the main application loop.
        
        This method keeps the application running and handles the main
        event loop until shutdown is requested.
        """
        try:
            self.logger.info("[RUNNING] AI-Galaxy system is now running...")
            
            # Main application loop
            while self.is_running and not self.shutdown_requested:
                try:
                    # Check system health
                    await self._check_system_health()
                    
                    # Process any pending system tasks
                    await self._process_system_tasks()
                    
                    # Brief sleep to prevent busy waiting
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(5)  # Longer delay on error
            
            self.logger.info("Main application loop exited")
            
        except Exception as e:
            self.logger.error(f"Fatal error in main application loop: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """
        Gracefully shutdown the AI-Galaxy system.
        
        This method ensures all components are properly stopped and
        resources are cleaned up.
        """
        if not self.is_running:
            return
        
        self.logger.info("[SHUTDOWN] Shutting down AI-Galaxy system...")
        self.is_running = False
        
        try:
            # Stop monitoring tasks
            for task in self._monitoring_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Unregister agents
            await self._shutdown_agents()
            
            # Stop workflow recovery service
            if self.workflow_recovery_service:
                await self.workflow_recovery_service.stop()
                self.logger.info("Workflow recovery service stopped")
            
            # Stop API server
            await self._stop_api_server()
            
            # Shutdown orchestrator
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            # Calculate uptime
            uptime = datetime.now(timezone.utc) - self.startup_time if self.startup_time else timedelta()
            
            context = LogContext(additional_context={
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'uptime_seconds': uptime.total_seconds(),
                'total_agents': len(self.agents)
            })
            self.logger.info("[COMPLETE] AI-Galaxy system shutdown completed", context)
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    async def _initialize_agents(self):
        """Initialize and register all core agents."""
        try:
            self.logger.info("Initializing core agents...")
            
            # Define agent configurations
            agent_configs = [
                {
                    'name': 'router_agent',
                    'class': EnhancedRouterAgent,
                    'capabilities': ['idea_routing', 'semantic_analysis', 'department_assignment', 'intelligent_routing', 'domain_classification', 'similarity_matching', 'workload_assessment', 'adaptive_learning']
                },
                {
                    'name': 'validator_agent', 
                    'class': EnhancedValidatorAgent,  # Use enhanced validator with message processing
                    'capabilities': ['idea_validation', 'feasibility_check', 'quality_assessment', 'completeness_check', 'format_compliance', 'duplicate_detection', 'conflict_detection']
                },
                {
                    'name': 'council_agent',
                    'class': EnhancedCouncilAgent,
                    'capabilities': ['strategic_review', 'resource_allocation', 'priority_setting', 'decision_making', 'appeal_processing']
                },
                {
                    'name': 'creator_agent',
                    'class': EnhancedCreatorAgent,
                    'capabilities': ['template_creation', 'structure_design', 'scaffolding', 'technology_selection', 'implementation_planning', 'department_routing']
                },
                {
                    'name': 'implementer_agent',
                    'class': EnhancedImplementerAgent,
                    'capabilities': ['implementation_planning', 'task_breakdown', 'coordination', 'implementation_orchestration', 'resource_allocation', 'progress_monitoring', 'risk_management']
                },
                {
                    'name': 'programmer_agent',
                    'class': EnhancedProgrammerAgent,
                    'capabilities': ['code_generation', 'programming', 'testing', 'debugging', 'architecture_design', 'code_optimization', 'documentation_generation', 'quality_assurance', 'multi_language_support']
                },
                {
                    'name': 'registrar_agent',
                    'class': EnhancedRegistrarAgent,
                    'capabilities': ['service_registration', 'metadata_management', 'quality_assessment', 'compliance_checking', 'documentation_generation', 'lifecycle_management', 'service_discovery', 'vector_search', 'analytics_reporting', 'governance_oversight', 'performance_monitoring', 'dependency_analysis', 'quality_gates', 'ecosystem_integration']
                },
                {
                    'name': 'evolution_agent',
                    'class': EnhancedEvolutionAgent,
                    'capabilities': ['ecosystem_health_monitoring', 'learning_pattern_analysis', 'evolution_recommendation_generation', 'agent_performance_tracking', 'auto_optimization', 'comprehensive_reporting', 'predictive_trend_analysis', 'innovation_tracking', 'system_adaptation_scoring', 'performance_improvement_orchestration', 'cross_agent_correlation_analysis', 'intelligent_bottleneck_prediction', 'adaptive_intelligence', 'evolutionary_optimization']
                }
            ]
            
            # Initialize each agent
            for agent_config in agent_configs:
                try:
                    agent_name = agent_config['name']
                    agent_class = agent_config['class']
                    capabilities = agent_config['capabilities']
                    
                    # Create and initialize agent instance
                    agent_instance = await self._create_agent_instance(agent_class, agent_name)
                    
                    if agent_instance:
                        # Register with orchestrator
                        success = await self.orchestrator.register_agent(
                            name=agent_name,
                            agent_type=agent_class.__name__,
                            capabilities=capabilities,
                            metadata={
                                'auto_started': True,
                                'initialization_time': datetime.now(timezone.utc).isoformat()
                            }
                        )
                        
                        if success:
                            self.agents[agent_name] = {
                                'instance': agent_instance,
                                'class': agent_class,
                                'capabilities': capabilities,
                                'status': 'active',
                                'heartbeat_task': None
                            }
                            
                            # Start agent heartbeat
                            heartbeat_task = asyncio.create_task(
                                self._agent_heartbeat_loop(agent_name)
                            )
                            self.agents[agent_name]['heartbeat_task'] = heartbeat_task
                            
                            self.logger.info(f"Agent initialized and started: {agent_name}")
                            
                            # Add startup delay between agents
                            await asyncio.sleep(self.config.system['agent_startup_delay'])
                        else:
                            self.logger.error(f"Failed to register agent: {agent_name}")
                    else:
                        self.logger.error(f"Failed to create agent instance: {agent_name}")
                
                except Exception as e:
                    self.logger.error(f"Error initializing agent {agent_config['name']}: {str(e)}")
            
            self.logger.info(f"Agent initialization completed. {len(self.agents)} agents registered.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {str(e)}")
    
    async def _create_agent_instance(self, agent_class, agent_name: str):
        """Create an agent instance with basic configuration."""
        try:
            # Special handling for enhanced agents
            if agent_class.__name__ == 'EnhancedValidatorAgent':
                # The enhanced validator agent requires async initialization
                agent_instance = agent_class(redis_config=self.config.redis)
                # Initialize the agent
                success = await agent_instance.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize enhanced validator agent")
                    return None
                self.logger.info(f"Enhanced validator agent initialized successfully")
                return agent_instance
            elif agent_class.__name__ == 'EnhancedCouncilAgent':
                # The enhanced council agent requires async initialization
                agent_instance = agent_class(redis_config=self.config.redis)
                # Initialize the agent
                success = await agent_instance.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize enhanced council agent")
                    return None
                self.logger.info(f"Enhanced council agent initialized successfully")
                return agent_instance
            elif agent_class.__name__ == 'EnhancedCreatorAgent':
                # The enhanced creator agent requires async initialization
                agent_instance = agent_class(redis_config=self.config.redis)
                # Initialize the agent
                success = await agent_instance.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize enhanced creator agent")
                    return None
                self.logger.info(f"Enhanced creator agent initialized successfully")
                return agent_instance
            elif agent_class.__name__ == 'EnhancedImplementerAgent':
                # The enhanced implementer agent requires async initialization
                agent_instance = agent_class(redis_config=self.config.redis)
                # Initialize the agent
                success = await agent_instance.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize enhanced implementer agent")
                    return None
                self.logger.info(f"Enhanced implementer agent initialized successfully")
                return agent_instance
            elif agent_class.__name__ == 'EnhancedProgrammerAgent':
                # The enhanced programmer agent requires async initialization
                agent_instance = agent_class(redis_config=self.config.redis)
                # Initialize the agent
                success = await agent_instance.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize enhanced programmer agent")
                    return None
                self.logger.info(f"Enhanced programmer agent initialized successfully")
                return agent_instance
            elif agent_class.__name__ == 'EnhancedRouterAgent':
                # The enhanced router agent requires async initialization
                agent_instance = agent_class(redis_config=self.config.redis)
                # Initialize the agent
                success = await agent_instance.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize enhanced router agent")
                    return None
                self.logger.info(f"Enhanced router agent initialized successfully")
                return agent_instance
            elif agent_class.__name__ == 'EnhancedRegistrarAgent':
                # The enhanced registrar agent requires async initialization
                agent_instance = agent_class(redis_config=self.config.redis)
                # Initialize the agent
                success = await agent_instance.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize enhanced registrar agent")
                    return None
                self.logger.info(f"Enhanced registrar agent initialized successfully")
                return agent_instance
            elif agent_class.__name__ == 'EnhancedEvolutionAgent':
                # The enhanced evolution agent requires async initialization
                agent_instance = agent_class(redis_config=self.config.redis)
                # Initialize the agent
                success = await agent_instance.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize enhanced evolution agent")
                    return None
                self.logger.info(f"Enhanced evolution agent initialized successfully")
                return agent_instance
            else:
                # For other agents, create instances with default configuration
                # This is a simplified approach - in production you'd want proper config injection
                agent_instance = agent_class()
                return agent_instance
        except Exception as e:
            self.logger.error(f"Error creating {agent_name} instance: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    async def _agent_heartbeat_loop(self, agent_name: str):
        """Send periodic heartbeats for an agent to the orchestrator."""
        try:
            # Check if this is an enhanced agent that handles its own heartbeats
            if agent_name in self.agents:
                agent_info = self.agents[agent_name]
                agent_class_name = agent_info['class'].__name__ if 'class' in agent_info else ''
                
                # Enhanced agents handle their own heartbeats, so we just monitor them
                if agent_class_name in ['EnhancedValidatorAgent', 'EnhancedCouncilAgent', 'EnhancedCreatorAgent', 'EnhancedImplementerAgent', 'EnhancedProgrammerAgent', 'EnhancedRouterAgent', 'EnhancedRegistrarAgent', 'EnhancedEvolutionAgent']:
                    self.logger.info(f"Enhanced agent {agent_name} handles its own heartbeats")
                    # Just keep the task running to maintain agent registry entry
                    while self.is_running and agent_name in self.agents:
                        await asyncio.sleep(60)  # Check every minute
                    return
            
            # For traditional agents, send heartbeats manually
            while self.is_running and agent_name in self.agents:
                try:
                    # Send heartbeat message to orchestrator
                    heartbeat_data = {
                        'type': 'heartbeat',
                        'agent_name': agent_name,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'status': 'active'
                    }
                    
                    # Send via Redis pub/sub to orchestrator
                    await self.redis_service.publish_message(
                        f"agent:{agent_name}",
                        heartbeat_data
                    )
                    
                    # Wait for next heartbeat interval (use half of orchestrator's interval)
                    await asyncio.sleep(self.config.orchestrator.heartbeat_interval // 2)
                    
                except Exception as e:
                    self.logger.error(f"Error sending heartbeat for {agent_name}: {str(e)}")
                    await asyncio.sleep(30)  # Wait before retrying
                    
        except asyncio.CancelledError:
            self.logger.debug(f"Heartbeat loop cancelled for {agent_name}")
        except Exception as e:
            self.logger.error(f"Fatal error in heartbeat loop for {agent_name}: {str(e)}")
    
    async def _initialize_workflow_recovery(self):
        """Initialize and start the workflow recovery service."""
        try:
            self.logger.info("Initializing workflow recovery service...")
            
            # Create recovery service configuration
            recovery_config = WorkflowRecoveryConfig(
                check_interval_seconds=self.config.system['workflow_recovery_check_interval'],
                stuck_threshold_minutes=self.config.system['workflow_recovery_stuck_threshold'],
                max_recovery_attempts=3,
                recovery_timeout_seconds=300
            )
            
            # Create and initialize the recovery service
            self.workflow_recovery_service = await create_workflow_recovery_service(
                redis_service=self.redis_service,
                config=recovery_config
            )
            
            # Start the recovery service
            await self.workflow_recovery_service.start()
            
            # Perform an immediate check for stuck ideas from potential restart
            self.logger.info("Performing initial check for stuck ideas after system restart...")
            initial_stats = await self.workflow_recovery_service.force_check_now()
            
            context = LogContext(additional_context=initial_stats)
            self.logger.info("Workflow recovery service initialized and initial check completed", context)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow recovery service: {str(e)}")
    
    async def _shutdown_agents(self):
        """Shutdown and unregister all agents."""
        try:
            self.logger.info("Shutting down agents...")
            
            for agent_name in list(self.agents.keys()):
                try:
                    agent_info = self.agents[agent_name]
                    
                    # Cancel heartbeat task
                    if 'heartbeat_task' in agent_info and agent_info['heartbeat_task']:
                        agent_info['heartbeat_task'].cancel()
                        try:
                            await agent_info['heartbeat_task']
                        except asyncio.CancelledError:
                            pass
                    
                    # Unregister from orchestrator
                    success = await self.orchestrator.unregister_agent(agent_name)
                    if success:
                        del self.agents[agent_name]
                        self.logger.info(f"Agent shutdown and unregistered: {agent_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error shutting down agent {agent_name}: {str(e)}")
            
            self.logger.info("Agent shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during agent shutdown: {str(e)}")
    
    async def _start_monitoring(self):
        """Start background monitoring tasks."""
        try:
            if self.config.system['health_check_enabled']:
                self._monitoring_tasks.append(
                    asyncio.create_task(self._health_monitoring_loop())
                )
            
            if self.config.system['metrics_collection_enabled']:
                self._monitoring_tasks.append(
                    asyncio.create_task(self._metrics_collection_loop())
                )
            
            self.logger.info(f"Started {len(self._monitoring_tasks)} monitoring tasks")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
    
    async def _start_api_server(self):
        """Start the REST API server."""
        try:
            import uvicorn
            from ai_galaxy.api import create_app
            
            # Create FastAPI app with dependency injection
            self.api_app = create_app(
                orchestrator=self.orchestrator,
                redis_service=self.redis_service,
                vector_search_service=self.vector_search_service
            )
            
            # Configure uvicorn
            api_config = uvicorn.Config(
                app=self.api_app,
                host=self.config.system['api_host'],
                port=self.config.system['api_port'],
                log_level="info",
                access_log=True
            )
            
            # Create and start server
            self.api_server = uvicorn.Server(api_config)
            
            # Start server in background task
            import asyncio
            self.api_server_task = asyncio.create_task(self.api_server.serve())
            
            self.logger.info(f"API server started on {self.config.system['api_host']}:{self.config.system['api_port']}")
            self.logger.info(f"API documentation available at http://{self.config.system['api_host']}:{self.config.system['api_port']}/docs")
            
        except ImportError:
            self.logger.error("uvicorn not installed. Install with: pip install uvicorn")
        except Exception as e:
            self.logger.error(f"Failed to start API server: {str(e)}")
    
    async def _stop_api_server(self):
        """Stop the REST API server."""
        try:
            if hasattr(self, 'api_server') and self.api_server:
                # Shutdown the API server
                self.api_server.should_exit = True
                
                # Wait for server task to complete
                if hasattr(self, 'api_server_task') and self.api_server_task:
                    try:
                        await asyncio.wait_for(self.api_server_task, timeout=30.0)
                    except asyncio.TimeoutError:
                        self.logger.warning("API server shutdown timed out")
                        self.api_server_task.cancel()
                
                self.logger.info("API server shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during API server shutdown: {str(e)}")
    
    async def _check_system_health(self):
        """Perform periodic system health checks."""
        try:
            if self.orchestrator:
                metrics = await self.orchestrator.get_system_metrics()
                
                # Check for critical issues
                active_agents = metrics.get('active_agents', 0)
                if active_agents == 0:
                    self.logger.warning("No active agents available")
                
                redis_health = metrics.get('redis_health', {})
                if not redis_health.get('is_healthy', False):
                    self.logger.error("Redis service is unhealthy")
                
                # Log health summary periodically
                if hasattr(self, '_last_health_log'):
                    if (datetime.now(timezone.utc) - self._last_health_log).total_seconds() > 300:  # Every 5 minutes
                        self._log_health_summary(metrics)
                        self._last_health_log = datetime.now(timezone.utc)
                else:
                    self._last_health_log = datetime.now(timezone.utc)
                    
        except Exception as e:
            self.logger.error(f"Error checking system health: {str(e)}")
    
    async def _process_system_tasks(self):
        """Process any pending system-level tasks."""
        try:
            # This could handle system-level maintenance tasks
            # For now, just ensure services are responsive
            if self.redis_service:
                # Ping Redis to ensure connectivity
                health = await self.redis_service.get_health_status()
                if not health.get('is_healthy', False):
                    self.logger.warning("Redis connectivity issue detected")
                    
        except Exception as e:
            self.logger.debug(f"Minor error in system tasks: {str(e)}")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.orchestrator:
                    metrics = await self.orchestrator.get_system_metrics()
                    
                    # Store health metrics in Redis for external monitoring
                    if self.redis_service:
                        await self.redis_service.cache_set(
                            'system_health_metrics',
                            metrics,
                            ttl_seconds=300
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {str(e)}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                # Collect and store system metrics
                if self.orchestrator:
                    metrics = await self.orchestrator.get_system_metrics()
                    
                    # Store timestamped metrics
                    timestamp = datetime.now(timezone.utc).isoformat()
                    metrics_record = {
                        'timestamp': timestamp,
                        'metrics': metrics
                    }
                    
                    if self.redis_service:
                        await self.redis_service.cache_set(
                            f'metrics:{timestamp}',
                            metrics_record,
                            ttl_seconds=86400  # Keep for 24 hours
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {str(e)}")
    
    def _log_health_summary(self, metrics: Dict[str, Any]):
        """Log a summary of system health."""
        try:
            summary = {
                'uptime_hours': round(metrics.get('uptime_seconds', 0) / 3600, 2),
                'total_agents': metrics.get('agent_count', 0),
                'active_agents': metrics.get('active_agents', 0),
                'active_tasks': metrics.get('active_tasks', 0),
                'active_workflows': metrics.get('active_workflows', 0),
                'total_tasks_processed': metrics.get('orchestrator_metrics', {}).get('total_tasks_processed', 0),
                'redis_healthy': metrics.get('redis_health', {}).get('is_healthy', False)
            }
            
            context = LogContext(additional_context=summary)
            self.logger.info("System health summary", context)
            
        except Exception as e:
            self.logger.error(f"Error logging health summary: {str(e)}")


async def main():
    """Main entry point for the AI-Galaxy application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI-Galaxy Autonomous Development Platform')
    parser.add_argument('--config', '-c', default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--daemon', '-d', action='store_true',
                       help='Run as daemon (placeholder)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = AIGalaxyConfig.from_file(args.config)
    
    # Adjust log level if verbose
    if args.verbose:
        config.system['log_level'] = 'DEBUG'
    
    # Create and run application
    app = AIGalaxyApplication(config)
    
    try:
        # Initialize the system
        success = await app.initialize()
        if not success:
            print("[ERROR] Failed to initialize AI-Galaxy system")
            sys.exit(1)
        
        # Run the main application
        await app.run()
        
    except KeyboardInterrupt:
        print("\n[STOP] Shutdown requested by user")
    except Exception as e:
        print(f"[ERROR] Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    print("[DONE] AI-Galaxy system shutdown complete")


if __name__ == "__main__":
    # Ensure we're running with Python 3.8+
    if sys.version_info < (3, 8):
        print("[ERROR] AI-Galaxy requires Python 3.8 or higher")
        sys.exit(1)
    
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOP] Application interrupted")
    except Exception as e:
        print(f"[ERROR] Application failed: {str(e)}")
        sys.exit(1)