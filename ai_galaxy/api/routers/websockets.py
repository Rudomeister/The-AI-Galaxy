"""
WebSocket API Routes for Real-time Communication.

This module provides WebSocket endpoints for real-time
communication with AI-Galaxy agents and system monitoring.
"""

import json
import asyncio
from typing import Dict, Set
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.websockets import WebSocketState

from ...services.redis.db import RedisService
from ...orchestrator import SystemOrchestrator
from ...shared.logger import get_logger
from ..dependencies import get_redis_service, get_orchestrator

router = APIRouter()
logger = get_logger("api.websockets")


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "agents": set(),
            "system": set(),
            "workflows": set()
        }
        self.connection_info: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, channel: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        
        self.active_connections[channel].add(websocket)
        self.connection_info[websocket] = {
            "channel": channel,
            "connected_at": datetime.now(timezone.utc),
            "client_id": id(websocket)
        }
        
        logger.info(f"WebSocket connected to channel '{channel}' - Total: {len(self.active_connections[channel])}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.connection_info:
            channel = self.connection_info[websocket]["channel"]
            
            self.active_connections[channel].discard(websocket)
            del self.connection_info[websocket]
            
            logger.info(f"WebSocket disconnected from channel '{channel}' - Remaining: {len(self.active_connections[channel])}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")
            self.disconnect(websocket)
    
    async def broadcast_to_channel(self, message: dict, channel: str):
        """Broadcast a message to all connections in a channel."""
        if channel not in self.active_connections:
            return
        
        # Create a copy of connections to avoid modification during iteration
        connections = self.active_connections[channel].copy()
        
        for websocket in connections:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    self.disconnect(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to {channel}: {str(e)}")
                self.disconnect(websocket)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all active connections."""
        for channel in self.active_connections:
            await self.broadcast_to_channel(message, channel)
    
    def get_connection_stats(self) -> dict:
        """Get statistics about active connections."""
        return {
            "total_connections": sum(len(connections) for connections in self.active_connections.values()),
            "connections_by_channel": {
                channel: len(connections) 
                for channel, connections in self.active_connections.items()
            }
        }


# Global connection manager
manager = ConnectionManager()


@router.websocket("/agents")
async def websocket_agents_endpoint(
    websocket: WebSocket,
    redis_service: RedisService = Depends(get_redis_service),
    orchestrator: SystemOrchestrator = Depends(get_orchestrator)
):
    """
    WebSocket endpoint for real-time agent status updates.
    
    Provides live updates about agent status, tasks, and system events.
    """
    await manager.connect(websocket, "agents")
    
    # Start Redis listener task
    listener_task = asyncio.create_task(
        listen_to_agent_events(websocket, redis_service)
    )
    
    try:
        # Send initial agent status
        system_metrics = await orchestrator.get_system_metrics()
        initial_message = {
            "type": "agent_status",
            "data": system_metrics.get("agents", {}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await manager.send_personal_message(initial_message, websocket)
        
        # Handle incoming messages from client
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()},
                        websocket
                    )
                elif message.get("type") == "subscribe_agent":
                    # Handle agent-specific subscriptions
                    agent_name = message.get("agent_name")
                    if agent_name:
                        # Send agent-specific updates
                        agent_info = system_metrics.get("agents", {}).get(agent_name)
                        if agent_info:
                            await manager.send_personal_message({
                                "type": "agent_detail",
                                "agent_name": agent_name,
                                "data": agent_info,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }, websocket)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON format"},
                    websocket
                )
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {str(e)}")
                await manager.send_personal_message(
                    {"type": "error", "message": "Message processing error"},
                    websocket
                )
    
    except WebSocketDisconnect:
        pass
    finally:
        listener_task.cancel()
        manager.disconnect(websocket)


@router.websocket("/system")
async def websocket_system_endpoint(
    websocket: WebSocket,
    redis_service: RedisService = Depends(get_redis_service),
    orchestrator: SystemOrchestrator = Depends(get_orchestrator)
):
    """
    WebSocket endpoint for real-time system monitoring.
    
    Provides live updates about system metrics, health, and performance.
    """
    await manager.connect(websocket, "system")
    
    # Start system metrics update task
    metrics_task = asyncio.create_task(
        send_periodic_system_metrics(websocket, orchestrator, redis_service)
    )
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle system monitoring commands
                if message.get("type") == "get_metrics":
                    metrics = await orchestrator.get_system_metrics()
                    await manager.send_personal_message({
                        "type": "system_metrics",
                        "data": metrics,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, websocket)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in system WebSocket: {str(e)}")
    
    except WebSocketDisconnect:
        pass
    finally:
        metrics_task.cancel()
        manager.disconnect(websocket)


@router.websocket("/workflows")
async def websocket_workflows_endpoint(
    websocket: WebSocket,
    redis_service: RedisService = Depends(get_redis_service)
):
    """
    WebSocket endpoint for real-time workflow monitoring.
    
    Provides live updates about idea processing workflows and state transitions.
    """
    await manager.connect(websocket, "workflows")
    
    # Start workflow events listener
    workflow_task = asyncio.create_task(
        listen_to_workflow_events(websocket, redis_service)
    )
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle workflow monitoring commands
                if message.get("type") == "subscribe_idea":
                    idea_id = message.get("idea_id")
                    # TODO: Implement idea-specific workflow updates
                    await manager.send_personal_message({
                        "type": "workflow_subscription",
                        "idea_id": idea_id,
                        "status": "subscribed"
                    }, websocket)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in workflow WebSocket: {str(e)}")
    
    except WebSocketDisconnect:
        pass
    finally:
        workflow_task.cancel()
        manager.disconnect(websocket)


async def listen_to_agent_events(websocket: WebSocket, redis_service: RedisService):
    """Listen to Redis pub/sub for agent events and forward to WebSocket."""
    try:
        # Subscribe to agent events
        pubsub = await redis_service._async_client.pubsub()
        await pubsub.psubscribe("agent:*")
        
        async for message in pubsub.listen():
            if message["type"] == "pmessage":
                try:
                    # Parse and forward agent event
                    event_data = json.loads(message["data"])
                    websocket_message = {
                        "type": "agent_event",
                        "channel": message["channel"],
                        "data": event_data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await manager.send_personal_message(websocket_message, websocket)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error processing agent event: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error in agent events listener: {str(e)}")


async def send_periodic_system_metrics(
    websocket: WebSocket, 
    orchestrator: SystemOrchestrator, 
    redis_service: RedisService
):
    """Send periodic system metrics updates via WebSocket."""
    try:
        while True:
            try:
                # Get current system metrics
                metrics = await orchestrator.get_system_metrics()
                redis_health = await redis_service.get_health_status()
                
                # Send metrics update
                message = {
                    "type": "system_metrics_update",
                    "data": {
                        "system": metrics,
                        "redis": redis_health,
                        "connections": manager.get_connection_stats()
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await manager.send_personal_message(message, websocket)
                
                # Wait 30 seconds before next update
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error sending system metrics: {str(e)}")
                await asyncio.sleep(10)  # Shorter retry interval on error
                
    except asyncio.CancelledError:
        pass


async def listen_to_workflow_events(websocket: WebSocket, redis_service: RedisService):
    """Listen to Redis pub/sub for workflow events and forward to WebSocket."""
    try:
        # Subscribe to workflow events
        pubsub = await redis_service._async_client.pubsub()
        await pubsub.subscribe("workflow_events")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    # Parse and forward workflow event
                    event_data = json.loads(message["data"])
                    websocket_message = {
                        "type": "workflow_event",
                        "data": event_data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await manager.send_personal_message(websocket_message, websocket)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error processing workflow event: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error in workflow events listener: {str(e)}")


# Endpoint to get WebSocket connection statistics
@router.get("/stats")
async def get_websocket_stats():
    """Get statistics about active WebSocket connections."""
    return {
        "connection_stats": manager.get_connection_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }