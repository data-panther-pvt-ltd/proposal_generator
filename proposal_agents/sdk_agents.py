"""
OpenAI Agents SDK Implementation
Defines all proposal generation agents using the SDK
"""

# Import OpenAI Agents SDK components
from agents import Agent, Runner, function_tool, ModelSettings
import os
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import pandas as pd

# Load agent instructions from markdown files
# def load_instructions(filename: str) -> str:
#     """Load agent instructions from markdown file"""
#     try:
#         path = Path(__file__).parent / filename
#         with open(path, 'r') as f:
#             return f.read()
#     except FileNotFoundError:
#         return f"Instructions for {filename} not found. Using default instructions for {filename.replace('.md', '')} agent."
from pathlib import Path

def load_instructions(filename: str) -> str:
    """Load agent instructions from markdown file"""
    try:
        path = Path(__file__).parent / filename
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Instructions for {filename} not found. Using default instructions for {filename.replace('.md', '')} agent."

# ============= AGENT DEFINITIONS =============

# Import the tools from proposal_tools
from tools.proposal_tools import get_rag_context as _original_get_rag_context, format_section

# Create wrapper for get_rag_context that ensures correct PDF path
@function_tool
def get_rag_context(query: str, pdf_path: str = None, max_tokens: int = 2000) -> str:
    """
    Get RAG context using the correct PDF path from settings
    
    Args:
        query: The query to search for
        pdf_path: PDF path (will be corrected if wrong)
        max_tokens: Maximum tokens to return
        
    Returns:
        Retrieved context string
    """
    # Load actual PDF path from settings
    try:
        config_path = Path(__file__).parent.parent / "config" / "settings.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        actual_pdf_path = config.get('rfp', {}).get('pdf_path', 'document.pdf')
        
        # If the provided path looks wrong (contains _RFP), use the actual path
        if pdf_path and '_RFP.pdf' in pdf_path:
            pdf_path = actual_pdf_path
        elif not pdf_path:
            pdf_path = actual_pdf_path
            
    except Exception:
        # Fallback to provided path
        pass
    
    return _original_get_rag_context(query, pdf_path, max_tokens)

# Content Generator Agent
content_generator_agent = Agent(
    name="ContentGenerator",
    instructions=load_instructions("content_generator.md"),
    tools=[get_rag_context, format_section],
    model_settings=ModelSettings(model="gpt-4o", temperature=0.3, max_tokens=2000)  # Reduced from 3000 to 2000
)

# Import research tools
from tools.proposal_tools import web_search
import yaml

# Load configuration to determine web search setup
def _get_web_search_tools():
    """Determine which web search tools to include based on configuration"""
    try:
        config_path = Path(__file__).parent.parent / "config" / "settings.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        web_search_config = config.get('web_search', {})
        provider = web_search_config.get('provider', 'tavily')
        
        # If using OpenAI browsing, don't include custom web_search tool
        # Let the SDK agent use its built-in browsing capabilities
        if provider == 'openai' and web_search_config.get('openai_browsing', False):
            print(f"🌐 Research Agent: Using OpenAI built-in browsing (no custom web_search tool)")
            return []  # No custom web search tool
        else:
            print(f"🔧 Research Agent: Using custom web_search tool with {provider} provider")
            return [web_search]
    except Exception as e:
        print(f"⚠️  Could not determine web search config: {e}, defaulting to custom tool")
        return [web_search]

# Research Agent with conditional web search tools
research_agent = Agent(
    name="ResearchAgent", 
    instructions=load_instructions("researcher.md"),
    tools=_get_web_search_tools(),
    model_settings=ModelSettings(
        model="gpt-4o", 
        temperature=0.1, 
        max_tokens=1500,  # Reduced from 2000 to 1500
        # Additional settings that may help with web browsing
        top_p=0.95,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
)

# Import budget tools
from tools.proposal_tools import calculate_project_costs, generate_timeline

# Budget Calculator Agent
budget_calculator_agent = Agent(
    name="BudgetCalculator", 
    instructions=load_instructions("budget_calculator.md"),
    tools=[calculate_project_costs, generate_timeline],
    model_settings=ModelSettings(model="gpt-4o", temperature=0.01, max_tokens=1200)  # Reduced from 2000 to 1200
)

# Import chart tools
from tools.proposal_tools import generate_gantt_chart, create_budget_chart, build_risk_matrix

# Chart Generator Agent - with minimal context requirements
chart_generator_agent = Agent(
    name="ChartGenerator",
    instructions=load_instructions("chart_generator.md"),
    tools=[generate_gantt_chart, create_budget_chart, build_risk_matrix],
    model_settings=ModelSettings(model="gpt-4o", temperature=0.01, max_tokens=500)  # Further reduced to 500 for charts
)

# Quality Evaluator Agent
quality_evaluator_agent = Agent(
    name="QualityEvaluator",
    instructions=load_instructions("quality_evaluator.md"),
    tools=[],  # No tools, just evaluation
    model_settings=ModelSettings(model="gpt-4o", temperature=0.05, max_tokens=1500)
)

# Import handoff tools
from tools.proposal_tools import analyze_risks

# Define handoff functions for the coordinator
def transfer_to_content_generator():
    """Transfer to content generator for standard content creation"""
    return content_generator_agent

def transfer_to_researcher():
    """Transfer to researcher for web search and research tasks"""
    return research_agent

def transfer_to_budget_calculator():
    """Transfer to budget calculator for cost calculations"""
    return budget_calculator_agent

def transfer_to_chart_generator():
    """Transfer to chart generator for visual content"""
    return chart_generator_agent

def transfer_to_quality_evaluator():
    """Transfer to quality evaluator for content review"""
    return quality_evaluator_agent

# Main Coordinator Agent (Hub in hub-and-spoke pattern)
coordinator_agent = Agent(
    name="ProposalCoordinator",
    instructions=load_instructions("coordinator.md"),
    tools=[
        analyze_risks,
        transfer_to_content_generator, 
        transfer_to_researcher,
        transfer_to_budget_calculator,
        transfer_to_chart_generator,
        transfer_to_quality_evaluator
    ],
    model_settings=ModelSettings(model="gpt-4o", temperature=0.1, max_tokens=1500)  # Reduced from 2000 to 1500
)

# ============= AGENT REGISTRY =============

AGENT_REGISTRY = {
    "coordinator": coordinator_agent,
    "content_generator": content_generator_agent,
    "researcher": research_agent,
    "budget_calculator": budget_calculator_agent,
    "quality_evaluator": quality_evaluator_agent,
    "chart_generator": chart_generator_agent
}

def get_agent(agent_name: str):
    """Get agent by name from registry"""
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Agent {agent_name} not found in registry")
    return AGENT_REGISTRY[agent_name]

def get_all_agents() -> Dict[str, Agent]:
    """Get all agents from registry"""
    return AGENT_REGISTRY.copy()

# Export coordinator for direct import
coordinator = coordinator_agent

def get_specialist_for_section(section_name: str, routing_config: Dict) -> Agent:
    """
    Get the appropriate specialist agent for a section based on routing config
    
    Args:
        section_name: Name of the proposal section
        routing_config: Section routing configuration from settings
    
    Returns:
        The appropriate specialist agent
    """
    section_config = routing_config.get(section_name, {})
    strategy = section_config.get('strategy', 'default')
    
    # Map strategies to specialist agents
    if strategy == 'calculation_based':
        return budget_calculator_agent
    elif strategy == 'search_based' or strategy == 'research_first':
        return research_agent
    elif strategy in ['timeline_with_gantt', 'chart_generation']:
        return chart_generator_agent
    else:
        return content_generator_agent

def get_handoff_function_for_section(section_name: str, routing_config: Dict):
    """
    Get the appropriate handoff function for a section based on routing config
    
    Args:
        section_name: Name of the proposal section
        routing_config: Section routing configuration from settings
    
    Returns:
        The appropriate handoff function
    """
    section_config = routing_config.get(section_name, {})
    strategy = section_config.get('strategy', 'default')
    
    # Map strategies to handoff functions
    if strategy == 'calculation_based':
        return transfer_to_budget_calculator
    elif strategy == 'search_based' or strategy == 'research_first':
        return transfer_to_researcher
    elif strategy in ['timeline_with_gantt', 'chart_generation']:
        return transfer_to_chart_generator
    else:
        return transfer_to_content_generator