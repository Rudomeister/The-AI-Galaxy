"""
State Machine Router for the AI-Galaxy ecosystem.

This module implements the main routing logic that processes ideas through
the workflow states defined in the YAML configuration. It handles state
transitions, agent assignments, and workflow coordination.
"""

import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from pydantic import BaseModel, Field

from ...shared.models import Idea, IdeaStatus
from ...shared.logger import get_logger, LogContext


class TransitionResult(str, Enum):
    """Result of attempting a state transition."""
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    RETRY_NEEDED = "retry_needed"


class RoutingDecision(BaseModel):
    """Represents a routing decision made by the state machine."""
    idea_id: str
    current_state: str
    next_state: Optional[str] = None
    assigned_agent: Optional[str] = None
    actions: List[str] = Field(default_factory=list)
    conditions_met: bool = True
    reason: Optional[str] = None
    retry_count: int = 0
    estimated_completion: Optional[datetime] = None


class StateMachineConfig(BaseModel):
    """Parsed state machine configuration from YAML."""
    version: str
    description: str
    settings: Dict[str, Any]
    agents: Dict[str, Dict[str, Any]]
    states: Dict[str, Dict[str, Any]]
    transitions: Dict[str, Dict[str, Any]]
    routing_rules: Dict[str, Dict[str, Any]]
    error_handling: Dict[str, Dict[str, Any]]
    monitoring: Dict[str, Dict[str, Any]]


class StateTransitionError(Exception):
    """Exception raised when a state transition fails."""
    pass


class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""
    pass


class StateMachineRouter:
    """
    Main router that manages idea flow through the AI-Galaxy workflow.
    
    The router loads configuration from YAML, validates state transitions,
    assigns agents, and coordinates the overall workflow process.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the state machine router.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.logger = get_logger("state_machine.router")
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "state_machine.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize tracking
        self.active_ideas: Dict[str, Idea] = {}
        self.transition_history: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_workload: Dict[str, int] = {}
        
        self.logger.system_event("State machine router initialized", {
            "config_version": self.config.version,
            "total_states": len(self.config.states),
            "total_transitions": len(self.config.transitions)
        })
    
    def _load_config(self) -> StateMachineConfig:
        """Load and parse the YAML configuration file."""
        try:
            if not self.config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            # Validate required sections
            required_sections = ['states', 'transitions', 'agents']
            for section in required_sections:
                if section not in config_data:
                    raise ConfigurationError(f"Missing required section: {section}")
            
            config = StateMachineConfig(**config_data)
            
            self.logger.info("Configuration loaded successfully", 
                           LogContext(additional_context={"config_file": str(self.config_path)}))
            
            return config
            
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise ConfigurationError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def reload_config(self) -> bool:
        """Reload the configuration from disk."""
        try:
            old_version = self.config.version
            self.config = self._load_config()
            
            self.logger.system_event("Configuration reloaded", {
                "old_version": old_version,
                "new_version": self.config.version
            })
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")
            return False
    
    def register_idea(self, idea: Idea) -> bool:
        """
        Register an idea with the state machine for tracking.
        
        Args:
            idea: The idea to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            idea_id = str(idea.id)
            self.active_ideas[idea_id] = idea
            self.transition_history[idea_id] = []
            
            self.logger.agent_action("idea_registered", "router", idea_id, {
                "title": idea.title,
                "initial_state": idea.status.value,
                "priority": idea.priority
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register idea: {e}", 
                            LogContext(idea_id=str(idea.id) if idea else None))
            return False
    
    def get_next_step(self, idea: Idea) -> RoutingDecision:
        """
        Determine the next step for an idea based on its current state.
        
        Args:
            idea: The idea to route
            
        Returns:
            RoutingDecision with next state and agent assignment
        """
        idea_id = str(idea.id)
        current_state = idea.status.value
        
        context = LogContext(
            idea_id=idea_id,
            additional_context={"current_state": current_state}
        )
        
        try:
            # Find possible transitions from current state
            possible_transitions = self._get_possible_transitions(current_state)
            
            if not possible_transitions:
                self.logger.warning(f"No transitions available from state: {current_state}", context)
                return RoutingDecision(
                    idea_id=idea_id,
                    current_state=current_state,
                    conditions_met=False,
                    reason="No transitions available"
                )
            
            # Evaluate conditions for each possible transition
            for transition_key, transition_config in possible_transitions.items():
                if self._evaluate_transition_conditions(idea, transition_config):
                    next_state = transition_config["to"]
                    agent = transition_config["agent"]
                    actions = transition_config.get("actions", [])
                    
                    # Calculate estimated completion time
                    agent_timeout = self.config.agents.get(agent, {}).get("timeout", 
                                                                         self.config.settings.get("default_timeout", 300))
                    estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=agent_timeout)
                    
                    decision = RoutingDecision(
                        idea_id=idea_id,
                        current_state=current_state,
                        next_state=next_state,
                        assigned_agent=agent,
                        actions=actions,
                        conditions_met=True,
                        reason=f"Transition {transition_key} conditions met",
                        estimated_completion=estimated_completion
                    )
                    
                    self.logger.router_decision(idea_id, current_state, agent, decision.reason)
                    return decision
            
            # No conditions met for any transition
            self.logger.warning(f"No transition conditions met for idea in state: {current_state}", context)
            return RoutingDecision(
                idea_id=idea_id,
                current_state=current_state,
                conditions_met=False,
                reason="No transition conditions satisfied"
            )
            
        except Exception as e:
            self.logger.error(f"Error determining next step: {e}", context)
            return RoutingDecision(
                idea_id=idea_id,
                current_state=current_state,
                conditions_met=False,
                reason=f"Error in routing: {e}"
            )
    
    def execute_transition(self, idea: Idea, target_state: str, agent_name: str) -> TransitionResult:
        """
        Execute a state transition for an idea.
        
        Args:
            idea: The idea to transition
            target_state: The target state to transition to
            agent_name: Name of the agent executing the transition
            
        Returns:
            Result of the transition attempt
        """
        idea_id = str(idea.id)
        current_state = idea.status.value
        
        context = LogContext(
            agent_name=agent_name,
            idea_id=idea_id,
            additional_context={
                "from_state": current_state,
                "to_state": target_state
            }
        )
        
        try:
            # Validate transition is allowed
            if not self._is_transition_valid(current_state, target_state):
                self.logger.error(f"Invalid transition: {current_state} -> {target_state}", context)
                return TransitionResult.FAILED
            
            # Update idea state
            try:
                idea.status = IdeaStatus(target_state)
                idea.updated_at = datetime.now(timezone.utc)
            except ValueError as e:
                self.logger.error(f"Invalid target state: {target_state}", context)
                return TransitionResult.FAILED
            
            # Record transition in history
            transition_record = {
                "from_state": current_state,
                "to_state": target_state,
                "agent": agent_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
            
            if idea_id not in self.transition_history:
                self.transition_history[idea_id] = []
            self.transition_history[idea_id].append(transition_record)
            
            # Update active ideas tracking
            self.active_ideas[idea_id] = idea
            
            self.logger.state_transition(current_state, target_state, idea_id, agent_name, 
                                       "Transition executed successfully")
            
            return TransitionResult.SUCCESS
            
        except Exception as e:
            self.logger.error(f"Transition execution failed: {e}", context)
            
            # Record failed transition
            if idea_id in self.transition_history:
                failed_record = {
                    "from_state": current_state,
                    "to_state": target_state,
                    "agent": agent_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                    "error": str(e)
                }
                self.transition_history[idea_id].append(failed_record)
            
            return TransitionResult.FAILED
    
    def get_agent_workload(self, agent_name: str) -> int:
        """Get the current workload for a specific agent."""
        return self.agent_workload.get(agent_name, 0)
    
    def update_agent_workload(self, agent_name: str, delta: int):
        """Update the workload for a specific agent."""
        current = self.agent_workload.get(agent_name, 0)
        self.agent_workload[agent_name] = max(0, current + delta)
    
    def get_ideas_in_state(self, state: str) -> List[Idea]:
        """Get all ideas currently in a specific state."""
        return [idea for idea in self.active_ideas.values() if idea.status.value == state]
    
    def get_transition_history(self, idea_id: str) -> List[Dict[str, Any]]:
        """Get the transition history for a specific idea."""
        return self.transition_history.get(idea_id, [])
    
    def validate_workflow_integrity(self) -> List[str]:
        """
        Validate the workflow configuration for consistency and completeness.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Check that all states referenced in transitions exist
            for transition_name, transition_config in self.config.transitions.items():
                from_state = transition_config.get("from")
                to_state = transition_config.get("to")
                
                if from_state not in self.config.states:
                    errors.append(f"Transition {transition_name} references unknown from_state: {from_state}")
                
                if to_state not in self.config.states:
                    errors.append(f"Transition {transition_name} references unknown to_state: {to_state}")
            
            # Check that all agents referenced in transitions exist
            for transition_name, transition_config in self.config.transitions.items():
                agent = transition_config.get("agent")
                if agent not in self.config.agents:
                    errors.append(f"Transition {transition_name} references unknown agent: {agent}")
            
            # Check for orphaned states (states with no incoming transitions except initial)
            initial_states = [state for state, config in self.config.states.items() 
                            if config.get("is_initial", False)]
            
            reachable_states = set(initial_states)
            for transition_config in self.config.transitions.values():
                reachable_states.add(transition_config.get("to"))
            
            for state in self.config.states:
                if state not in reachable_states:
                    errors.append(f"State {state} is unreachable (no transitions lead to it)")
            
            # Check for terminal states with outgoing transitions
            for state, state_config in self.config.states.items():
                if state_config.get("is_terminal", False):
                    outgoing_transitions = [t for t in self.config.transitions.values() 
                                          if t.get("from") == state]
                    if outgoing_transitions:
                        errors.append(f"Terminal state {state} has outgoing transitions")
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return errors
    
    def _get_possible_transitions(self, from_state: str) -> Dict[str, Dict[str, Any]]:
        """Get all possible transitions from a given state."""
        return {
            name: config for name, config in self.config.transitions.items()
            if config.get("from") == from_state
        }
    
    def _evaluate_transition_conditions(self, idea: Idea, transition_config: Dict[str, Any]) -> bool:
        """
        Evaluate whether transition conditions are met for an idea.
        
        Args:
            idea: The idea to evaluate
            transition_config: Configuration for the transition
            
        Returns:
            True if all conditions are met, False otherwise
        """
        conditions = transition_config.get("conditions", [])
        
        try:
            for condition in conditions:
                if not self._evaluate_single_condition(idea, condition):
                    return False
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating transition conditions: {e}", 
                            LogContext(idea_id=str(idea.id)))
            return False
    
    def _evaluate_single_condition(self, idea: Idea, condition: str) -> bool:
        """
        Evaluate a single condition string.
        
        Args:
            idea: The idea to evaluate against
            condition: Condition string to evaluate
            
        Returns:
            True if condition is met, False otherwise
        """
        # Simple condition evaluation - in a real implementation, this would be more sophisticated
        try:
            # Replace idea placeholders with actual values
            condition = condition.replace("idea.description", f'"{idea.description}"')
            condition = condition.replace("idea.title", f'"{idea.title}"')
            condition = condition.replace("idea.priority", str(idea.priority))
            condition = condition.replace("len(idea.description)", str(len(idea.description)))
            
            # Simple condition checks
            if "is not null" in condition:
                return "None" not in condition and '""' not in condition
            
            if ">=" in condition:
                parts = condition.split(">=")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = int(parts[1].strip())
                    left_val = int(left) if left.isdigit() else 0
                    return left_val >= right
            
            # Default to True for unknown conditions
            return True
            
        except Exception:
            return False
    
    def _is_transition_valid(self, from_state: str, to_state: str) -> bool:
        """Check if a transition between two states is valid according to configuration."""
        state_config = self.config.states.get(from_state, {})
        allowed_transitions = state_config.get("allowed_transitions", [])
        return to_state in allowed_transitions


# Factory function for easy router creation
def create_router(config_path: Optional[str] = None) -> StateMachineRouter:
    """
    Create a new state machine router instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured StateMachineRouter instance
    """
    return StateMachineRouter(config_path)


# Export main classes and functions
__all__ = [
    "StateMachineRouter",
    "RoutingDecision",
    "TransitionResult",
    "StateTransitionError",
    "ConfigurationError",
    "create_router"
]