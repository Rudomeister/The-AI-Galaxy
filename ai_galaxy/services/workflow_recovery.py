"""
Workflow Recovery Service for AI-Galaxy.

This service detects and recovers ideas that are stuck in the workflow
after system restarts or interruptions. It provides auto-kickstart
functionality to resume idea processing automatically.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..shared.logger import get_logger, LogContext
from ..shared.models import Idea, SystemState
from .redis import RedisService


@dataclass
class WorkflowRecoveryConfig:
    """Configuration for workflow recovery service."""
    check_interval_seconds: int = 60  # How often to check for stuck ideas
    stuck_threshold_minutes: int = 10  # How long before an idea is considered stuck
    max_recovery_attempts: int = 3    # Maximum attempts to recover a stuck idea
    recovery_timeout_seconds: int = 300  # Timeout for recovery operations


class WorkflowRecoveryService:
    """
    Service for detecting and recovering stuck workflow items.
    
    This service runs in the background and periodically checks for ideas
    that have been in a non-terminal state for too long without progress.
    It can automatically trigger the next workflow step to get ideas moving again.
    """
    
    def __init__(self, 
                 redis_service: RedisService,
                 config: Optional[WorkflowRecoveryConfig] = None):
        """
        Initialize the workflow recovery service.
        
        Args:
            redis_service: Redis service for data access
            config: Recovery service configuration
        """
        self.redis_service = redis_service
        self.config = config or WorkflowRecoveryConfig()
        self.logger = get_logger("ai_galaxy.workflow_recovery")
        
        # Service state
        self.is_running = False
        self.recovery_task: Optional[asyncio.Task] = None
        self.recovery_stats = {
            "total_checks": 0,
            "stuck_ideas_found": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "last_check": None
        }
        
        # Workflow state definitions based on state machine
        self.workflow_states = [
            "created", "validated", "council_review", "approved",
            "template_created", "implementation_started", "coding_in_progress"
        ]
        
        self.terminal_states = ["completed", "rejected", "archived"]
        
        # State transition mapping
        self.next_state_map = {
            "created": "validated",
            "validated": "council_review", 
            "council_review": "approved",
            "approved": "template_created",
            "template_created": "implementation_started",
            "implementation_started": "coding_in_progress",
            "coding_in_progress": "completed"
        }
        
        # Agent responsible for each transition
        self.transition_agents = {
            "created": "validator_agent",
            "validated": "council_agent",
            "council_review": "council_agent", 
            "approved": "creator_agent",
            "template_created": "implementer_agent",
            "implementation_started": "programmer_agent",
            "coding_in_progress": "programmer_agent"
        }
    
    async def start(self):
        """Start the workflow recovery service."""
        if self.is_running:
            self.logger.warning("Workflow recovery service is already running")
            return
        
        self.is_running = True
        self.recovery_task = asyncio.create_task(self._recovery_loop())
        
        context = LogContext(additional_context={
            "check_interval": self.config.check_interval_seconds,
            "stuck_threshold": self.config.stuck_threshold_minutes
        })
        self.logger.info("Workflow recovery service started", context)
    
    async def stop(self):
        """Stop the workflow recovery service."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.recovery_task and not self.recovery_task.done():
            self.recovery_task.cancel()
            try:
                await self.recovery_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Workflow recovery service stopped")
    
    async def _recovery_loop(self):
        """Main recovery loop that runs periodically."""
        while self.is_running:
            try:
                await self._check_and_recover_stuck_ideas()
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in recovery loop: {str(e)}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _check_and_recover_stuck_ideas(self):
        """Check for stuck ideas and attempt recovery."""
        try:
            self.recovery_stats["total_checks"] += 1
            self.recovery_stats["last_check"] = datetime.now(timezone.utc).isoformat()
            
            # Get all ideas from Redis using state key pattern
            redis_client = self.redis_service._async_client
            if not redis_client:
                await self.redis_service.initialize()
                redis_client = self.redis_service._async_client
            
            all_idea_keys = await redis_client.keys("state:idea:*")
            if not all_idea_keys:
                return
            
            stuck_ideas = []
            current_time = datetime.now(timezone.utc)
            
            # Fetch and check each idea for being stuck
            for key in all_idea_keys:
                # Convert bytes key to string if needed
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                # Remove the state: prefix since get_state adds it back
                clean_key = key_str.replace('state:', '', 1)
                idea_data = await self.redis_service.get_state(clean_key)
                
                if idea_data and await self._is_idea_stuck(idea_data, current_time):
                    stuck_ideas.append(idea_data)
            
            if stuck_ideas:
                self.recovery_stats["stuck_ideas_found"] += len(stuck_ideas)
                
                context = LogContext(additional_context={
                    "stuck_ideas_count": len(stuck_ideas),
                    "idea_ids": [idea["id"] for idea in stuck_ideas]
                })
                self.logger.info(f"Found {len(stuck_ideas)} stuck ideas", context)
                
                # Attempt recovery for each stuck idea
                for idea_data in stuck_ideas:
                    await self._recover_idea(idea_data)
        
        except Exception as e:
            self.logger.error(f"Error checking for stuck ideas: {str(e)}")
    
    async def _is_idea_stuck(self, idea_data: Dict[str, Any], current_time: datetime) -> bool:
        """
        Determine if an idea is stuck in the workflow.
        
        Args:
            idea_data: Idea data from Redis
            current_time: Current timestamp
            
        Returns:
            True if the idea appears to be stuck
        """
        try:
            # Skip terminal states
            status = idea_data.get("status", "")
            if status in self.terminal_states:
                return False
            
            # Skip if not in a trackable workflow state
            if status not in self.workflow_states:
                return False
            
            # Check if idea has been in current state too long
            updated_at_str = idea_data.get("updated_at", "")
            if not updated_at_str:
                return True  # No update time is suspicious
            
            try:
                updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
            except ValueError:
                # Handle different datetime formats
                updated_at = datetime.fromisoformat(updated_at_str)
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
            
            time_stuck = current_time - updated_at
            stuck_threshold = timedelta(minutes=self.config.stuck_threshold_minutes)
            
            is_stuck = time_stuck > stuck_threshold
            
            if is_stuck:
                context = LogContext(additional_context={
                    "idea_id": idea_data.get("id"),
                    "status": status,
                    "time_stuck_minutes": int(time_stuck.total_seconds() / 60),
                    "threshold_minutes": self.config.stuck_threshold_minutes
                })
                self.logger.debug("Idea identified as stuck", context)
            
            return is_stuck
            
        except Exception as e:
            self.logger.error(f"Error checking if idea is stuck: {str(e)}")
            return False
    
    async def _recover_idea(self, idea_data: Dict[str, Any]):
        """
        Attempt to recover a stuck idea by triggering the next workflow step.
        
        Args:
            idea_data: Stuck idea data
        """
        idea_id = idea_data.get("id", "")
        current_status = idea_data.get("status", "")
        
        try:
            # Get recovery attempt count
            recovery_key = f"recovery_attempts:{idea_id}"
            attempts = await self.redis_service.cache_get(recovery_key) or 0
            
            if attempts >= self.config.max_recovery_attempts:
                self.logger.warning(f"Max recovery attempts reached for idea {idea_id}")
                return
            
            # Increment attempt count
            await self.redis_service.cache_set(
                recovery_key, 
                attempts + 1, 
                ttl_seconds=86400  # Reset daily
            )
            
            # Determine next state and responsible agent
            next_state = self.next_state_map.get(current_status)
            responsible_agent = self.transition_agents.get(current_status)
            
            if not next_state or not responsible_agent:
                self.logger.error(f"No recovery path found for idea {idea_id} in state {current_status}")
                return
            
            context = LogContext(additional_context={
                "idea_id": idea_id,
                "current_status": current_status,
                "next_state": next_state,
                "responsible_agent": responsible_agent,
                "attempt": attempts + 1
            })
            self.logger.info(f"Attempting recovery for stuck idea", context)
            
            # Create recovery task message in the format expected by BaseAgentHandler
            recovery_task = {
                "type": "request",
                "task_id": f"recovery_{idea_id}_{int(datetime.now().timestamp())}",
                "task_type": f"process_{current_status}_to_{next_state}",
                "payload": {
                    "idea_id": idea_id,
                    "current_status": current_status,
                    "target_status": next_state,
                    "priority": "high",
                    "recovery_attempt": True,
                    "recovery_reason": "auto_kickstart_after_restart"
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
                "agent_name": responsible_agent
            }
            
            # Send task to appropriate agent
            agent_channel = f"agent:{responsible_agent}"
            await self.redis_service.publish_message(agent_channel, recovery_task)
            
            # Update the idea's updated_at timestamp to mark recovery attempt
            idea_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            if "recovery_attempts" not in idea_data.get("metadata", {}):
                if "metadata" not in idea_data:
                    idea_data["metadata"] = {}
                idea_data["metadata"]["recovery_attempts"] = 1
            else:
                idea_data["metadata"]["recovery_attempts"] += 1
            
            await self.redis_service.set_state(f"idea:{idea_id}", idea_data)
            
            self.recovery_stats["successful_recoveries"] += 1
            self.logger.info(f"Recovery task sent for idea {idea_id}")
            
        except Exception as e:
            self.recovery_stats["failed_recoveries"] += 1
            self.logger.error(f"Failed to recover idea {idea_id}: {str(e)}")
    
    async def manual_recover_idea(self, idea_id: str) -> bool:
        """
        Manually trigger recovery for a specific idea.
        
        Args:
            idea_id: ID of the idea to recover
            
        Returns:
            True if recovery was initiated successfully
        """
        try:
            # Get idea data using state key
            idea_data = await self.redis_service.get_state(f"idea:{idea_id}")
            if not idea_data:
                self.logger.error(f"Idea {idea_id} not found for manual recovery")
                return False
            
            await self._recover_idea(idea_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Manual recovery failed for idea {idea_id}: {str(e)}")
            return False
    
    async def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery service statistics."""
        return {
            **self.recovery_stats,
            "is_running": self.is_running,
            "config": {
                "check_interval_seconds": self.config.check_interval_seconds,
                "stuck_threshold_minutes": self.config.stuck_threshold_minutes,
                "max_recovery_attempts": self.config.max_recovery_attempts
            }
        }
    
    async def force_check_now(self) -> Dict[str, Any]:
        """
        Force an immediate check for stuck ideas.
        
        Returns:
            Summary of the check results
        """
        initial_stats = self.recovery_stats.copy()
        await self._check_and_recover_stuck_ideas()
        
        return {
            "checks_performed": self.recovery_stats["total_checks"] - initial_stats["total_checks"],
            "stuck_ideas_found": self.recovery_stats["stuck_ideas_found"] - initial_stats["stuck_ideas_found"],
            "recoveries_attempted": self.recovery_stats["successful_recoveries"] - initial_stats["successful_recoveries"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


async def create_workflow_recovery_service(
    redis_service: RedisService,
    config: Optional[WorkflowRecoveryConfig] = None
) -> WorkflowRecoveryService:
    """
    Factory function to create and initialize a workflow recovery service.
    
    Args:
        redis_service: Redis service instance
        config: Optional recovery configuration
        
    Returns:
        Initialized workflow recovery service
    """
    service = WorkflowRecoveryService(redis_service, config)
    return service