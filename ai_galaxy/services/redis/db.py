"""
Redis Database Service for AI-Galaxy Ecosystem.

This module provides a comprehensive Redis client for state management, caching,
pub/sub messaging between agents, session management, and health monitoring.
It serves as the central nervous system for real-time communication and data
persistence across the AI-Galaxy ecosystem.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import UUID, uuid4
import traceback

import redis
import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from ...shared.logger import get_logger, LogContext
from ...shared.models import AgentMessage, MessageType, MessageStatus, SystemState


class RedisConfig(BaseModel):
    """Configuration for Redis connection."""
    host: str = Field(default="localhost", description="Redis server host")
    port: int = Field(default=6379, description="Redis server port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    max_connections: int = Field(default=20, description="Maximum connection pool size")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")
    connection_timeout: float = Field(default=10.0, description="Connection timeout in seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")


class MessageHandler:
    """Handler for processing pub/sub messages."""
    
    def __init__(self, callback: Callable[[str, Dict[str, Any]], None]):
        self.callback = callback
        
    async def handle_message(self, channel: str, message: Dict[str, Any]):
        """Handle incoming pub/sub message."""
        try:
            await self.callback(channel, message)
        except Exception as e:
            logger = get_logger("redis.message_handler")
            logger.error(f"Error handling message on channel {channel}: {str(e)}")


class RedisService:
    """
    Redis service providing state management, caching, and pub/sub messaging.
    
    This service acts as the central data layer for the AI-Galaxy ecosystem,
    enabling real-time communication between agents and persistent storage
    of system state and session data.
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis service with configuration.
        
        Args:
            config: Redis configuration. Uses defaults if not provided.
        """
        self.config = config or RedisConfig()
        self.logger = get_logger("redis.service")
        
        # Connection pools
        self._sync_pool: Optional[redis.ConnectionPool] = None
        self._async_pool: Optional[aioredis.ConnectionPool] = None
        self._sync_client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None
        
        # Pub/sub management
        self._pubsub_client: Optional[aioredis.Redis] = None
        self._subscriptions: Dict[str, MessageHandler] = {}
        self._pubsub_task: Optional[asyncio.Task] = None
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy: bool = False
        self._last_health_check: Optional[datetime] = None
        
        # Session tracking
        self._active_sessions: Dict[str, datetime] = {}
        
        # Key prefixes for organization
        self.KEY_PREFIXES = {
            'agent_message': 'msg:agent:',
            'session': 'sess:',
            'state': 'state:',
            'cache': 'cache:',
            'lock': 'lock:',
            'counter': 'counter:',
            'queue': 'queue:',
            'health': 'health:'
        }
    
    async def initialize(self) -> bool:
        """
        Initialize Redis connections and start background tasks.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Create connection pools
            self._sync_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connection_timeout,
                retry_on_timeout=self.config.retry_on_timeout
            )
            
            self._async_pool = aioredis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connection_timeout,
                retry_on_timeout=self.config.retry_on_timeout
            )
            
            # Create clients
            self._sync_client = redis.Redis(connection_pool=self._sync_pool)
            self._async_client = aioredis.Redis(connection_pool=self._async_pool)
            self._pubsub_client = aioredis.Redis(connection_pool=self._async_pool)
            
            # Test connections
            await self._async_client.ping()
            self._sync_client.ping()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self._is_healthy = True
            self.logger.info("Redis service initialized successfully")
            
            # Initialize system state
            await self._initialize_system_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis service: {str(e)}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown Redis service and cleanup resources."""
        self.logger.info("Shutting down Redis service...")
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass
        
        # Close connections
        if self._async_client:
            await self._async_client.close()
        if self._pubsub_client:
            await self._pubsub_client.close()
        if self._sync_client:
            self._sync_client.close()
        
        # Close pools
        if self._async_pool:
            await self._async_pool.disconnect()
        if self._sync_pool:
            self._sync_pool.disconnect()
        
        self.logger.info("Redis service shutdown completed")
    
    # === State Management ===
    
    async def set_state(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set state value with optional TTL.
        
        Args:
            key: State key
            value: Value to store (will be JSON serialized)
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            full_key = f"{self.KEY_PREFIXES['state']}{key}"
            serialized_value = json.dumps(value, default=str)
            
            if ttl_seconds:
                await self._async_client.setex(full_key, ttl_seconds, serialized_value)
            else:
                await self._async_client.set(full_key, serialized_value)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set state {key}: {str(e)}")
            return False
    
    async def get_state(self, key: str) -> Optional[Any]:
        """
        Get state value.
        
        Args:
            key: State key
            
        Returns:
            Deserialized value or None if not found.
        """
        try:
            full_key = f"{self.KEY_PREFIXES['state']}{key}"
            value = await self._async_client.get(full_key)
            
            if value is None:
                return None
            
            return json.loads(value.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"Failed to get state {key}: {str(e)}")
            return None
    
    async def delete_state(self, key: str) -> bool:
        """Delete state value."""
        try:
            full_key = f"{self.KEY_PREFIXES['state']}{key}"
            result = await self._async_client.delete(full_key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Failed to delete state {key}: {str(e)}")
            return False
    
    # === Caching ===
    
    async def cache_set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set cache value with TTL."""
        try:
            full_key = f"{self.KEY_PREFIXES['cache']}{key}"
            serialized_value = json.dumps(value, default=str)
            await self._async_client.setex(full_key, ttl_seconds, serialized_value)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set cache {key}: {str(e)}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value."""
        try:
            full_key = f"{self.KEY_PREFIXES['cache']}{key}"
            value = await self._async_client.get(full_key)
            
            if value is None:
                return None
            
            return json.loads(value.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"Failed to get cache {key}: {str(e)}")
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete cache value."""
        try:
            full_key = f"{self.KEY_PREFIXES['cache']}{key}"
            result = await self._async_client.delete(full_key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Failed to delete cache {key}: {str(e)}")
            return False
    
    # === Pub/Sub Messaging ===
    
    async def subscribe_to_channel(self, channel: str, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Subscribe to a pub/sub channel.
        
        Args:
            channel: Channel name to subscribe to
            callback: Async callback function to handle messages
        """
        try:
            handler = MessageHandler(callback)
            self._subscriptions[channel] = handler
            
            # Start pubsub listener if not already running
            if self._pubsub_task is None or self._pubsub_task.done():
                self._pubsub_task = asyncio.create_task(self._pubsub_listener())
            
            self.logger.info(f"Subscribed to channel: {channel}")
        except Exception as e:
            self.logger.error(f"Failed to subscribe to channel {channel}: {str(e)}")
    
    async def unsubscribe_from_channel(self, channel: str):
        """Unsubscribe from a pub/sub channel."""
        try:
            if channel in self._subscriptions:
                del self._subscriptions[channel]
                self.logger.info(f"Unsubscribed from channel: {channel}")
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from channel {channel}: {str(e)}")
    
    async def publish_message(self, channel: str, message: Dict[str, Any]) -> bool:
        """
        Publish message to a channel.
        
        Args:
            channel: Channel name
            message: Message data to publish
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            serialized_message = json.dumps(message, default=str)
            result = await self._async_client.publish(channel, serialized_message)
            
            self.logger.debug(f"Published message to channel {channel}, subscribers: {result}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish message to channel {channel}: {str(e)}")
            return False
    
    async def send_agent_message(self, message: AgentMessage) -> bool:
        """
        Send message between agents using pub/sub.
        
        Args:
            message: Agent message to send
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Store message in database
            message_key = f"{self.KEY_PREFIXES['agent_message']}{message.id}"
            await self.set_state(message_key, message.dict(), ttl_seconds=86400)  # 24 hours
            
            # Publish to agent's channel
            channel = f"agent:{message.receiver_agent}"
            notification = {
                "message_id": str(message.id),
                "sender": message.sender_agent,
                "type": message.message_type,
                "timestamp": message.timestamp.isoformat()
            }
            
            success = await self.publish_message(channel, notification)
            
            if success:
                # Update message status
                message.status = MessageStatus.DELIVERED
                await self.set_state(message_key, message.dict(), ttl_seconds=86400)
                
                context = LogContext(
                    agent_name=message.sender_agent,
                    additional_context={
                        "receiver": message.receiver_agent,
                        "message_type": message.message_type,
                        "message_id": str(message.id)
                    }
                )
                self.logger.info("Agent message sent successfully", context)
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to send agent message: {str(e)}")
            return False
    
    async def get_agent_message(self, message_id: UUID) -> Optional[AgentMessage]:
        """Retrieve agent message by ID."""
        try:
            message_key = f"{self.KEY_PREFIXES['agent_message']}{message_id}"
            message_data = await self.get_state(message_key)
            
            if message_data is None:
                return None
            
            return AgentMessage(**message_data)
        except Exception as e:
            self.logger.error(f"Failed to get agent message {message_id}: {str(e)}")
            return None
    
    # === Session Management ===
    
    async def create_session(self, session_id: str, session_data: Dict[str, Any], 
                           ttl_seconds: int = 3600) -> bool:
        """Create a new session with data."""
        try:
            session_key = f"{self.KEY_PREFIXES['session']}{session_id}"
            session_data['created_at'] = datetime.now(timezone.utc).isoformat()
            session_data['last_accessed'] = datetime.now(timezone.utc).isoformat()
            
            success = await self.set_state(session_key, session_data, ttl_seconds)
            
            if success:
                self._active_sessions[session_id] = datetime.now(timezone.utc)
                self.logger.info(f"Session created: {session_id}")
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to create session {session_id}: {str(e)}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data and update last accessed time."""
        try:
            session_key = f"{self.KEY_PREFIXES['session']}{session_id}"
            session_data = await self.get_state(session_key)
            
            if session_data is None:
                return None
            
            # Update last accessed time
            session_data['last_accessed'] = datetime.now(timezone.utc).isoformat()
            await self.set_state(session_key, session_data, ttl_seconds=3600)
            
            self._active_sessions[session_id] = datetime.now(timezone.utc)
            return session_data
        except Exception as e:
            self.logger.error(f"Failed to get session {session_id}: {str(e)}")
            return None
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data."""
        try:
            session_data = await self.get_session(session_id)
            if session_data is None:
                return False
            
            session_data.update(updates)
            session_data['last_accessed'] = datetime.now(timezone.utc).isoformat()
            
            session_key = f"{self.KEY_PREFIXES['session']}{session_id}"
            return await self.set_state(session_key, session_data, ttl_seconds=3600)
        except Exception as e:
            self.logger.error(f"Failed to update session {session_id}: {str(e)}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        try:
            session_key = f"{self.KEY_PREFIXES['session']}{session_id}"
            success = await self.delete_state(session_key)
            
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            
            if success:
                self.logger.info(f"Session deleted: {session_id}")
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {str(e)}")
            return False
    
    # === Health Monitoring ===
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        try:
            # Basic connectivity
            start_time = time.time()
            await self._async_client.ping()
            ping_time = (time.time() - start_time) * 1000  # ms
            
            # Memory info
            memory_info = await self._async_client.info('memory')
            
            # Connection info
            clients_info = await self._async_client.info('clients')
            
            # Database info
            db_info = await self._async_client.info('keyspace')
            
            # Server info
            server_info = await self._async_client.info('server')
            
            # Active sessions count
            active_sessions_count = len([
                s for s, t in self._active_sessions.items() 
                if datetime.now(timezone.utc) - t < timedelta(minutes=30)
            ])
            
            health_status = {
                'is_healthy': self._is_healthy,
                'last_check': self._last_health_check.isoformat() if self._last_health_check else None,
                'ping_time_ms': round(ping_time, 2),
                'memory_used_mb': round(memory_info.get('used_memory', 0) / 1024 / 1024, 2),
                'memory_peak_mb': round(memory_info.get('used_memory_peak', 0) / 1024 / 1024, 2),
                'connected_clients': clients_info.get('connected_clients', 0),
                'active_subscriptions': len(self._subscriptions),
                'active_sessions': active_sessions_count,
                'total_keys': sum([db_info.get(f'db{i}', {}).get('keys', 0) for i in range(16)]),
                'uptime_seconds': server_info.get('uptime_in_seconds', 0)
            }
            
            return health_status
        except Exception as e:
            self.logger.error(f"Failed to get health status: {str(e)}")
            return {'is_healthy': False, 'error': str(e)}
    
    # === Distributed Locking ===
    
    async def acquire_lock(self, lock_name: str, timeout_seconds: int = 10, 
                          ttl_seconds: int = 60) -> Optional[str]:
        """
        Acquire a distributed lock.
        
        Args:
            lock_name: Name of the lock
            timeout_seconds: How long to wait for the lock
            ttl_seconds: How long the lock is valid
            
        Returns:
            Lock token if acquired, None otherwise.
        """
        try:
            lock_key = f"{self.KEY_PREFIXES['lock']}{lock_name}"
            lock_token = str(uuid4())
            
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                # Try to set lock with NX (only if not exists)
                result = await self._async_client.set(
                    lock_key, lock_token, nx=True, ex=ttl_seconds
                )
                
                if result:
                    self.logger.debug(f"Acquired lock: {lock_name}")
                    return lock_token
                
                # Wait a bit before retrying
                await asyncio.sleep(0.1)
            
            self.logger.warning(f"Failed to acquire lock: {lock_name} (timeout)")
            return None
        except Exception as e:
            self.logger.error(f"Failed to acquire lock {lock_name}: {str(e)}")
            return None
    
    async def release_lock(self, lock_name: str, lock_token: str) -> bool:
        """Release a distributed lock."""
        try:
            lock_key = f"{self.KEY_PREFIXES['lock']}{lock_name}"
            
            # Use Lua script to ensure atomic check-and-delete
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            
            result = await self._async_client.eval(lua_script, 1, lock_key, lock_token)
            
            if result == 1:
                self.logger.debug(f"Released lock: {lock_name}")
                return True
            else:
                self.logger.warning(f"Failed to release lock: {lock_name} (token mismatch)")
                return False
        except Exception as e:
            self.logger.error(f"Failed to release lock {lock_name}: {str(e)}")
            return False
    
    # === Internal Methods ===
    
    async def _pubsub_listener(self):
        """Background task to listen for pub/sub messages."""
        try:
            pubsub = self._pubsub_client.pubsub()
            
            # Subscribe to all channels we're interested in
            for channel in self._subscriptions.keys():
                await pubsub.subscribe(channel)
            
            self.logger.info("Pub/sub listener started")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel'].decode('utf-8')
                    data = json.loads(message['data'].decode('utf-8'))
                    
                    if channel in self._subscriptions:
                        handler = self._subscriptions[channel]
                        await handler.handle_message(channel, data)
                    
                elif message['type'] == 'subscribe':
                    self.logger.debug(f"Subscribed to channel: {message['channel'].decode('utf-8')}")
        except asyncio.CancelledError:
            self.logger.info("Pub/sub listener cancelled")
        except Exception as e:
            self.logger.error(f"Pub/sub listener error: {str(e)}")
            # Try to restart listener after delay
            await asyncio.sleep(5)
            if not asyncio.current_task().cancelled():
                self._pubsub_task = asyncio.create_task(self._pubsub_listener())
    
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Perform health check
                start_time = time.time()
                await self._async_client.ping()
                ping_time = (time.time() - start_time) * 1000
                
                self._is_healthy = ping_time < 1000  # Consider healthy if ping < 1s
                self._last_health_check = datetime.now(timezone.utc)
                
                # Store health status
                health_key = f"{self.KEY_PREFIXES['health']}status"
                health_data = {
                    'is_healthy': self._is_healthy,
                    'ping_time_ms': ping_time,
                    'timestamp': self._last_health_check.isoformat()
                }
                await self.set_state(health_key, health_data, ttl_seconds=300)
                
                # Cleanup old sessions
                await self._cleanup_old_sessions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {str(e)}")
                self._is_healthy = False
    
    async def _cleanup_old_sessions(self):
        """Clean up expired sessions from tracking."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            expired_sessions = [
                session_id for session_id, last_access 
                in self._active_sessions.items()
                if last_access < cutoff_time
            ]
            
            for session_id in expired_sessions:
                del self._active_sessions[session_id]
            
            if expired_sessions:
                self.logger.debug(f"Cleaned up {len(expired_sessions)} expired session references")
        except Exception as e:
            self.logger.error(f"Session cleanup error: {str(e)}")
    
    async def _initialize_system_state(self):
        """Initialize system state tracking."""
        try:
            # Initialize system state if not exists
            system_state = await self.get_state('system_state')
            if system_state is None:
                initial_state = SystemState()
                await self.set_state('system_state', initial_state.dict())
                self.logger.info("Initialized system state")
        except Exception as e:
            self.logger.error(f"Failed to initialize system state: {str(e)}")


# Factory function for easy instantiation
def create_redis_service(config: Optional[RedisConfig] = None) -> RedisService:
    """
    Factory function to create a Redis service instance.
    
    Args:
        config: Optional Redis configuration
        
    Returns:
        RedisService instance
    """
    return RedisService(config)


# Export main classes
__all__ = [
    "RedisService",
    "RedisConfig", 
    "MessageHandler",
    "create_redis_service"
]