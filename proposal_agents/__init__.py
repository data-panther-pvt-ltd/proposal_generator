"""
Proposal Agents package for OpenAI SDK agents
Contains all proposal generation agents and related utilities
"""

from .sdk_agents import (
    # Core agent registry and utilities
    AGENT_REGISTRY,
    get_agent,
    get_all_agents,
    load_instructions,
    
    # Agent routing and handoff functions
    get_specialist_for_section,
    get_handoff_function_for_section,
    
    # Individual agent instances
    coordinator_agent,
    content_generator_agent,
    research_agent,
    budget_calculator_agent,
    quality_evaluator_agent,
    
    # Transfer functions for agent handoffs
    transfer_to_content_generator,
    transfer_to_researcher,
    transfer_to_budget_calculator,
    transfer_to_quality_evaluator,
    
    # Coordinator alias
    coordinator
)

__all__ = [
    # Core registry and utilities
    'AGENT_REGISTRY',
    'get_agent', 
    'get_all_agents',
    'load_instructions',
    
    # Routing functions
    'get_specialist_for_section',
    'get_handoff_function_for_section',
    
    # Agent instances
    'coordinator_agent',
    'content_generator_agent', 
    'research_agent',
    'budget_calculator_agent',
    'quality_evaluator_agent',
    
    # Transfer functions
    'transfer_to_content_generator',
    'transfer_to_researcher', 
    'transfer_to_budget_calculator',
    'transfer_to_quality_evaluator',
    
    # Coordinator alias
    'coordinator'
]