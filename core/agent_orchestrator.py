"""
Agent Orchestrator for coordinating multiple agents
Fully integrated with OpenAI Agents SDK using hub-and-spoke pattern
"""

import json
import re
from typing import Dict, List, Any, Optional
import logging
import sys
import os
from datetime import datetime

# Add the parent directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SDK Components
from agents import Runner
from core.sdk_runner import ProposalRunner
from core.simple_cost_tracker import SimpleCostTracker

# SDK Agents - All specialist agents
from proposal_agents.sdk_agents import (
    get_agent,
    get_specialist_for_section,
    coordinator_agent,
    content_generator_agent,
    research_agent,
    budget_calculator_agent,
    chart_generator_agent,
    quality_evaluator_agent
)

# Legacy components for compatibility
from utils.agent_logger import agent_logger

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Orchestrates the interaction between different agents using OpenAI SDK
    Implements hub-and-spoke pattern with coordinator as central hub
    """
    
    def __init__(self, config: Dict, client: Optional[object] = None):
        self.config = config
        
        # Initialize cost tracker
        self.cost_tracker = SimpleCostTracker()
        
        # Initialize SDK-based ProposalRunner with cost tracker
        self.proposal_runner = ProposalRunner(config, cost_tracker=self.cost_tracker)
        logger.info("ProposalRunner initialized with SDK agents and cost tracking")
        
        # Get section routing configuration
        self.section_routing = config.get('section_routing', {})
        
        # SDK Agents Registry (hub-and-spoke pattern)
        self.coordinator = coordinator_agent
        self.specialists = {
            'content_generator': content_generator_agent,
            'researcher': research_agent,
            'budget_calculator': budget_calculator_agent,
            'chart_generator': chart_generator_agent,
            'quality_evaluator': quality_evaluator_agent
        }
        
        # Keep reference to RAG retriever for compatibility
        self.rag_retriever = self.proposal_runner.rag_retriever
        
        logger.info(f"Agent Orchestrator initialized with {len(self.specialists)} specialist agents")
    
    def _get_section_strategy(self, section_name: str) -> str:
        """Get the strategy for a section from routing configuration"""
        section_config = self.section_routing.get(section_name, {})
        return section_config.get('strategy', 'default')
    
    def _get_section_requirements(self, section_name: str) -> Dict[str, Any]:
        """Get section requirements from routing configuration"""
        return self.section_routing.get(section_name, {})
    
    def _should_route_through_coordinator(self, section_name: str) -> bool:
        """Determine if section should be routed through coordinator for complex orchestration"""
        section_config = self.section_routing.get(section_name, {})
        agents_list = section_config.get('agents', [])
        
        # Route through coordinator if multiple agents are involved
        return len(agents_list) > 1 or section_config.get('generate_chart', False)
    
    async def generate_section(
        self, 
        section_name: str, 
        routing: Dict, 
        context: Dict
    ) -> Dict:
        """
        Generate a section using SDK agents with hub-and-spoke pattern
        Routes through coordinator for complex sections, direct to specialist for simple ones
        """
        logger.info(f"Orchestrating section generation: {section_name}")
        agent_logger.log_task_start(f"generate_section_{section_name}")
        
        try:
            # Get section requirements and strategy
            section_config = self._get_section_requirements(section_name)
            strategy = section_config.get('strategy', 'default')
            
            # Determine routing approach
            use_coordinator = self._should_route_through_coordinator(section_name)
            
            if use_coordinator:
                # Complex sections: Route through coordinator (hub-and-spoke)
                result = await self._generate_section_via_coordinator(
                    section_name, section_config, context
                )
            else:
                # Simple sections: Direct to specialist
                result = await self._generate_section_direct(
                    section_name, section_config, context
                )
            
            # Add cost tracking and metadata
            result.update({
                'total_cost': self.cost_tracker.get_total_cost(),
                'cost_summary': self.cost_tracker.get_summary(),
                'generation_method': 'coordinator' if use_coordinator else 'direct'
            })
            
            # Update metadata
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata'].update({
                'strategy': strategy,
                'completed_at': datetime.now().isoformat()
            })
            
            logger.info(f"Section generation completed: {section_name} (${result['total_cost']:.4f})")
            agent_logger.log_task_completion(f"generate_section_{section_name}", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in section generation for {section_name}: {str(e)}")
            agent_logger.log_task_error(f"generate_section_{section_name}", str(e))
            
            # Return fallback result
            return {
                'content': f"[Section {section_name} - Generation failed: {str(e)}]",
                'metadata': {
                    'section_name': section_name,
                    'error': str(e),
                    'fallback': True,
                    'failed_at': datetime.now().isoformat()
                },
                'total_cost': self.cost_tracker.get_total_cost(),
                'cost_summary': self.cost_tracker.get_summary()
            }
    
    async def _generate_section_via_coordinator(
        self, 
        section_name: str, 
        section_config: Dict, 
        context: Dict
    ) -> Dict:
        """Generate section using coordinator agent (hub-and-spoke pattern)"""
        logger.info(f"Generating {section_name} via coordinator (complex routing)")
        
        # Prepare coordinator message
        coordinator_message = self._build_coordinator_message(
            section_name, section_config, context
        )
        
        # Set context variables for coordinator
        context_variables = self._build_context_variables(context, section_config)
        
        # Run coordinator agent
        response = Runner.run_sync(
            agent=self.coordinator,
            messages=coordinator_message, 
            context_variables=context_variables
        )
        
        # Track cost
        self.cost_tracker.track_completion(response, model="gpt-4o")
        
        # Extract content
        content = self._extract_content_from_response(response)
        
        return {
            'content': content,
            'metadata': {
                'section_name': section_name,
                'agent_used': 'coordinator',
                'routing_method': 'hub_and_spoke',
                'generated_at': datetime.now().isoformat(),
                'response': response,
                'word_count': len(content.split()) if content else 0
            }
        }
    
    async def _generate_section_direct(
        self, 
        section_name: str, 
        section_config: Dict, 
        context: Dict
    ) -> Dict:
        """Generate section using direct specialist agent"""
        # Get appropriate specialist
        specialist = get_specialist_for_section(section_name, self.section_routing)
        logger.info(f"Generating {section_name} via direct specialist: {specialist.name}")
        
        # Prepare specialist message
        specialist_message = self._build_specialist_message(
            section_name, section_config, context
        )
        
        # Set context variables
        context_variables = self._build_context_variables(context, section_config)
        
        # Run specialist agent
        response = Runner.run_sync(
            agent=specialist,
            messages=specialist_message, 
            context_variables=context_variables
        )
        
        # Track cost
        self.cost_tracker.track_completion(response, model="gpt-4o")
        
        # Extract content
        content = self._extract_content_from_response(response)
        
        return {
            'content': content,
            'metadata': {
                'section_name': section_name,
                'agent_used': specialist.name,
                'routing_method': 'direct_specialist',
                'generated_at': datetime.now().isoformat(),
                'response': response,
                'word_count': len(content.split()) if content else 0
            }
        }
    
    def _build_coordinator_message(
        self, 
        section_name: str, 
        section_config: Dict, 
        context: Dict
    ) -> str:
        """Build message for coordinator agent"""
        request = context.get('request')
        strategy = section_config.get('strategy', 'default')
        agents_needed = section_config.get('agents', [])
        
        message = f"""
Generate the '{section_name}' section for this proposal using the {strategy} strategy.

CLIENT: {getattr(request, 'client_name', 'Unknown') if request else 'Unknown'}
PROJECT: {getattr(request, 'project_name', 'Unknown') if request else 'Unknown'}
TIMELINE: {getattr(request, 'timeline', 'Unknown') if request else 'Unknown'}

SECTION REQUIREMENTS:
- Strategy: {strategy}
- Word count: {section_config.get('min_words', 300)}-{section_config.get('max_words', 800)} words
- Agents available: {', '.join(agents_needed)}
"""
        
        # Add chart generation requirement
        if section_config.get('generate_chart'):
            message += f"\n- Generate {section_config['generate_chart']} chart\n"
        
        # Add skills requirement
        if section_config.get('use_skills'):
            message += "\n- Include skills and resource information\n"
        
        message += "\nRoute to appropriate specialist agents as needed and coordinate their outputs."
        
        return message
    
    def _build_specialist_message(
        self, 
        section_name: str, 
        section_config: Dict, 
        context: Dict
    ) -> str:
        """Build message for specialist agent"""
        request = context.get('request')
        strategy = section_config.get('strategy', 'default')
        
        message = f"""
Generate the '{section_name}' section for this proposal.

CLIENT: {getattr(request, 'client_name', 'Unknown') if request else 'Unknown'}
PROJECT: {getattr(request, 'project_name', 'Unknown') if request else 'Unknown'}
TIMELINE: {getattr(request, 'timeline', 'Unknown') if request else 'Unknown'}

REQUIREMENTS:
- Strategy: {strategy}
- Word count: {section_config.get('min_words', 300)}-{section_config.get('max_words', 800)} words
"""
        
        # Add project requirements if available
        if request and hasattr(request, 'requirements'):
            requirements = getattr(request, 'requirements', {})
            if requirements:
                message += "\nPROJECT REQUIREMENTS:\n"
                for key, value in requirements.items():
                    if value and key != 'rfp_context':
                        message += f"- {key}: {value}\n"
        
        return message
    
    def _build_context_variables(self, context: Dict, section_config: Dict) -> Dict:
        """Build context variables for SDK agents"""
        context_variables = {
            'section_config': section_config,
            'company_profile': context.get('company_profile', {}),
            'skills_data': context.get('skills_data', {})
        }
        
        # Add request information
        request = context.get('request')
        if request:
            context_variables['request'] = {
                'client_name': getattr(request, 'client_name', 'Unknown'),
                'project_name': getattr(request, 'project_name', 'Unknown'),
                'project_type': getattr(request, 'project_type', 'general'),
                'timeline': getattr(request, 'timeline', 'Unknown'),
                'requirements': getattr(request, 'requirements', {})
            }
        
        # Add RAG context if required
        if section_config.get('requires_context', True):
            rag_context = self._get_rag_context(context)
            if rag_context:
                context_variables['rag_context'] = rag_context
        
        return context_variables
    
    def _get_rag_context(self, context: Dict) -> str:
        """Get RAG context for the section"""
        try:
            if hasattr(self.rag_retriever, 'retrieve'):
                request = context.get('request')
                query = f"Information for {getattr(request, 'project_name', 'project') if request else 'project'}"
                chunks = self.rag_retriever.retrieve(query, '')
                if chunks:
                    return "\n".join([chunk['text'][:500] for chunk in chunks[:3]])
        except Exception as e:
            logger.warning(f"Could not retrieve RAG context: {e}")
        return ""
    
    def _extract_content_from_response(self, response) -> str:
        """Extract content from SDK agent response with JSON parsing support"""
        try:
            # Step 1: Extract raw content using existing logic
            raw_content = None
            
            # Handle different response formats from the SDK
            if hasattr(response, 'final_output'):
                raw_content = response.final_output
            elif hasattr(response, 'messages') and response.messages:
                # Get the last message content
                last_message = response.messages[-1]
                if hasattr(last_message, 'content'):
                    raw_content = last_message.content
                else:
                    raw_content = str(last_message)
            elif hasattr(response, 'content'):
                raw_content = response.content
            elif hasattr(response, 'text'):
                raw_content = response.text
            else:
                raw_content = str(response)
            
            # Convert non-string responses to string (handles float, int, etc.)
            if raw_content is not None and not isinstance(raw_content, str):
                raw_content = str(raw_content)
            
            # Step 2: Check if content is wrapped in JSON markdown code blocks
            if raw_content and isinstance(raw_content, str):
                # Pattern to match JSON code blocks: ```json ... ``` or ``` ... ```
                json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                json_match = re.search(json_pattern, raw_content, re.DOTALL | re.IGNORECASE)
                
                if json_match:
                    json_content = json_match.group(1).strip()
                    try:
                        # Step 3: Parse the JSON and extract content field
                        parsed_json = json.loads(json_content)
                        
                        # Try to extract content from common field names
                        content_fields = ['content', 'text', 'body', 'section_content', 'message', 'response']
                        for field in content_fields:
                            if field in parsed_json and parsed_json[field]:
                                extracted_content = parsed_json[field]
                                logging.debug(f"Extracted content from JSON field '{field}': {len(extracted_content) if extracted_content else 0} chars")
                                return extracted_content
                        
                        # If no standard field found, check if the JSON itself is a string
                        if isinstance(parsed_json, str):
                            return parsed_json
                        
                        # If JSON is a dict but no content field, return JSON as string for backward compatibility
                        logging.warning("JSON found but no content field, returning JSON string")
                        return json.dumps(parsed_json, indent=2)
                        
                    except json.JSONDecodeError as json_err:
                        logging.warning(f"Invalid JSON in code block, falling back to raw content: {json_err}")
                        # Return the content inside code blocks even if JSON parsing fails
                        return json_content
                
                # Step 4: Check if entire content is valid JSON (without code blocks)
                try:
                    parsed_json = json.loads(raw_content)
                    
                    # Only try to extract fields if parsed_json is a dict
                    if isinstance(parsed_json, dict):
                        # Try to extract content from common field names
                        content_fields = ['content', 'text', 'body', 'section_content', 'message', 'response']
                        for field in content_fields:
                            if field in parsed_json and parsed_json[field]:
                                extracted_content = parsed_json[field]
                                logging.debug(f"Extracted content from direct JSON field '{field}': {len(extracted_content) if extracted_content else 0} chars")
                                return extracted_content
                    
                    # If JSON is a string, return it
                    elif isinstance(parsed_json, str):
                        return parsed_json
                        
                except json.JSONDecodeError:
                    # Not JSON, continue with raw content
                    pass
            
            # Step 5: Return raw content as fallback (backward compatibility)
            # Ensure we always return a string
            if raw_content is None:
                return ""
            elif isinstance(raw_content, str):
                return raw_content
            else:
                return str(raw_content)
            
        except Exception as e:
            logging.warning(f"Could not extract content from SDK response: {e}")
            # Graceful fallback - try to return something useful
            try:
                return str(response) if response else ""
            except:
                return ""
    
    async def _evaluate_quality(self, section_name: str, content: Dict) -> float:
        """Evaluate the quality of generated content using quality evaluator agent"""
        try:
            evaluator = self.specialists['quality_evaluator']
            
            eval_message = f"""
Evaluate the quality of this proposal section on a scale of 1-10:

Section: {section_name}
Content: {content.get('content', '')}

Rate based on:
- Relevance to requirements (25%)
- Technical accuracy (25%)
- Clarity and readability (20%)
- Completeness (20%)
- Professionalism (10%)

Return only a numeric score.
"""
            
            response = Runner.run_sync(
                agent=evaluator,
                messages=eval_message,
                context_variables={"section_name": section_name}
            )
            self.cost_tracker.track_completion(response, model="gpt-4o")
            
            score_text = self._extract_content_from_response(response)
            
            # Extract numeric score
            try:
                score = float(score_text.strip())
                return score
            except ValueError:
                logger.warning(f"Could not parse quality score for {section_name}: {score_text}")
                return 7.5
                
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return 7.5  # Default score if evaluation fails
    
    async def generate_chart(self, chart_type: str, data: Dict) -> Dict:
        """Generate a chart using the SDK chart generator agent"""
        logger.info(f"Generating {chart_type} chart using SDK chart generator")
        
        try:
            chart_generator = self.specialists['chart_generator']
            
            # Prepare chart generation message
            chart_message = f"""
Generate a {chart_type} chart with the following specifications:

Chart Type: {chart_type}
Data: {json.dumps(data, indent=2, default=str)}

Provide the chart in a format suitable for proposal inclusion.
"""
            
            # Set context variables
            context_variables = {
                'chart_type': chart_type,
                'data': data
            }
            
            # Run chart generator agent
            response = Runner.run_sync(
                agent=chart_generator,
                messages=chart_message, 
                context_variables=context_variables
            )
            self.cost_tracker.track_completion(response, model="gpt-4o")
            
            # Extract chart content
            chart_content = self._extract_content_from_response(response)
            
            result = {
                'content': chart_content,
                'metadata': {
                    'chart_type': chart_type,
                    'agent_used': 'chart_generator',
                    'generated_at': datetime.now().isoformat(),
                    'response': response
                }
            }
            
            logger.info(f"SDK chart generation complete: {chart_type}")
            return result
            
        except Exception as e:
            logger.error(f"SDK chart generation failed: {e}")
            
            # Return error result instead of fallback
            return {
                'error': str(e), 
                'chart_type': chart_type,
                'fallback_needed': True,
                'metadata': {
                    'failed_at': datetime.now().isoformat(),
                    'error': str(e)
                }
            }
    
    def _prepare_chart_data(self, chart_type: str, context: Dict) -> Dict:
        """Prepare data for chart generation based on chart type"""
        request = context.get('request')
        
        if 'gantt' in chart_type.lower() or 'timeline' in chart_type.lower():
            return self.proposal_runner._extract_timeline_data(None, context)
        elif 'budget' in chart_type.lower():
            skills_data = context.get('skills_data', {})
            return {
                'type': 'budget_breakdown',
                'categories': ['Development', 'Testing', 'Infrastructure', 'Management', 'Contingency'],
                'values': [45, 20, 15, 15, 5],
                'skills_data': skills_data
            }
        elif 'risk' in chart_type.lower():
            return self.proposal_runner._prepare_risk_matrix_data()
        else:
            return context.get('data', {})
    
    async def run_agent_by_name(self, agent_name: str, message: str, context: Dict = None) -> Dict:
        """Run a specific SDK agent by name"""
        if context is None:
            context = {}
            
        try:
            # Get agent from specialists registry
            if agent_name == 'coordinator':
                agent = self.coordinator
            elif agent_name in self.specialists:
                agent = self.specialists[agent_name]
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            logger.info(f"Running SDK agent: {agent_name}")
            
            # Prepare context variables
            context_variables = self._build_context_variables(context, {})
            
            # Run the agent
            response = Runner.run_sync(
                agent=agent,
                messages=message, 
                context_variables=context_variables
            )
            self.cost_tracker.track_completion(response, model="gpt-4o")
            
            # Extract content
            content = self._extract_content_from_response(response)
            
            return {
                'content': content,
                'metadata': {
                    'agent_used': agent_name,
                    'generated_at': datetime.now().isoformat(),
                    'response': response
                },
                'cost': self.cost_tracker.get_total_cost()
            }
            
        except Exception as e:
            logger.error(f"Error running agent {agent_name}: {str(e)}")
            return {
                'error': str(e),
                'agent': agent_name,
                'message': message
            }
    
    def _extract_timeline_data(self, content: Any, context: Dict) -> Dict:
        """Extract timeline data from content - delegated to SDK runner"""
        return self.proposal_runner._extract_timeline_data(content, context)
    
    def _prepare_risk_matrix_data(self) -> Dict:
        """Prepare risk matrix data - delegated to SDK runner"""
        return self.proposal_runner._prepare_risk_matrix_data()
    
    async def evaluate_proposal_quality(self, sections: Dict[str, Dict]) -> Dict[str, float]:
        """Evaluate the quality of all proposal sections"""
        quality_scores = {}
        
        for section_name, section_data in sections.items():
            if section_data.get('metadata', {}).get('fallback'):
                quality_scores[section_name] = 0.0  # Failed sections get 0 score
                continue
                
            try:
                score = await self._evaluate_quality(section_name, section_data)
                quality_scores[section_name] = score
            except Exception as e:
                logger.error(f"Failed to evaluate {section_name}: {e}")
                quality_scores[section_name] = 7.5  # Default score
        
        return quality_scores
    
    # ========== SDK INTEGRATION METHODS ==========
    
    def get_cost_summary(self) -> Dict:
        """Get cost tracking summary from cost tracker"""
        return self.cost_tracker.get_summary()
    
    def get_total_cost(self) -> float:
        """Get total execution cost from cost tracker"""
        return self.cost_tracker.get_total_cost()
    
    def get_sdk_agent(self, agent_name: str):
        """Get SDK agent instance for direct use"""
        try:
            if agent_name == 'coordinator':
                return self.coordinator
            elif agent_name in self.specialists:
                return self.specialists[agent_name]
            else:
                return get_agent(agent_name)
        except Exception as e:
            logger.error(f"Failed to get SDK agent {agent_name}: {str(e)}")
            raise ValueError(f"Agent {agent_name} not found or failed to load: {str(e)}")
    
    async def run_sdk_agent_directly(self, agent_name: str, message: str, context: Dict = None) -> Dict:
        """Run an SDK agent directly without orchestration"""
        return await self.run_agent_by_name(agent_name, message, context)
    
    def get_available_agents(self) -> List[str]:
        """Get list of available SDK agents"""
        return ['coordinator'] + list(self.specialists.keys())
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent"""
        try:
            agent = self.get_sdk_agent(agent_name)
            return {
                'name': agent.name,
                'model': agent.model,
                'tools': getattr(agent, 'tools', []),
                'available': True
            }
        except Exception as e:
            return {
                'name': agent_name,
                'available': False,
                'error': str(e)
            }
    
    def reset_cost_tracking(self):
        """Reset cost tracking for new proposal generation"""
        self.cost_tracker = SimpleCostTracker()
        self.proposal_runner.cost_tracker = self.cost_tracker
        logger.info("Cost tracking reset for new proposal generation")