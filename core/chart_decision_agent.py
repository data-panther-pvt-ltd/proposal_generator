"""
Lightweight Chart Decision Agent for Proposal Generation
Makes intelligent chart decisions using minimal context without SDK agents
Uses OpenAI SDK directly with Pydantic for structured output
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging
from openai import OpenAI

# Local imports
from utils.agent_logger import agent_logger


# Pydantic models for structured output
class ChartSpecification(BaseModel):
    """Individual chart specification"""
    section: str = Field(description="The proposal section name where this chart belongs")
    type: str = Field(description="Chart type (gantt, budget_breakdown, risk_matrix, bar, pie, line)")
    title: str = Field(description="Descriptive title for the chart")
    priority: int = Field(description="Priority level (1=required, 2=recommended, 3=optional)", ge=1, le=3)
    required: bool = Field(default=False, description="Whether this chart is required")
    rationale: Optional[str] = Field(default=None, description="Brief reason for suggesting this chart")


class ChartDecisionResponse(BaseModel):
    """Response containing all chart decisions"""
    charts: List[ChartSpecification] = Field(description="List of charts to generate")
    total_count: int = Field(description="Total number of charts")
    reasoning: str = Field(description="Brief explanation of chart selection strategy")


class ChartDecisionAgent:
    """
    Lightweight agent for making intelligent chart decisions for proposals.
    
    Features:
    - Uses minimal context to avoid token limits
    - Respects minimum chart requirements from configuration
    - Makes intelligent additional chart suggestions
    - Uses structured output with Pydantic models
    - Comprehensive logging and error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Chart Decision Agent
        
        Args:
            config: Configuration dictionary from settings.yml
        """
        self.config = config
        self.charts_config = config.get('charts', {})
        self.logger = agent_logger.get_agent_logger('chart_decision_agent')
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = self.config.get('openai', {}).get('model', 'gpt-4o')
        self.temperature = 0.2  # Low temperature for consistent decisions
        self.max_tokens = 1500
        
        # Supported chart types - all implemented in ChartGenerator
        self.supported_chart_types = [
            'gantt', 'budget_breakdown', 'risk_matrix',
            'bar', 'pie', 'line'
        ]
        
        self.logger.info("Chart Decision Agent initialized successfully")
        
    def decide_charts(self, section_summaries: Dict[str, str], project_type: str = "general") -> List[Dict[str, Any]]:
        """
        Make intelligent decisions about which charts to generate
        
        Args:
            section_summaries: Dict mapping section names to brief summaries (max 50 words each)
            project_type: Type of project (e.g., "software_development", "consulting", "research")
            
        Returns:
            List of chart specifications
        """
        self.logger.info(f"Making chart decisions for {project_type} project with {len(section_summaries)} sections")
        
        try:
            # Start with minimum required charts
            charts = self._get_minimum_required_charts()
            
            # Get additional chart suggestions if allowed
            if self._should_suggest_additional_charts():
                additional_charts = self._suggest_additional_charts(section_summaries, project_type)
                
                # Add additional charts up to max limit
                max_charts = self.charts_config.get('max_charts', 10)
                remaining_slots = max_charts - len(charts)
                
                for chart in additional_charts[:remaining_slots]:
                    charts.append(chart)
            
            # Convert to expected output format
            result = [self._format_chart_spec(chart) for chart in charts]
            
            self.logger.info(f"Final decision: {len(result)} charts selected")
            for chart in result:
                self.logger.debug(f"  - {chart['type']} chart for {chart['section']}: {chart['title']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in chart decision making: {str(e)}")
            # Fallback to minimum required charts only
            return [self._format_chart_spec(chart) for chart in self._get_minimum_required_charts()]
    
    def _get_minimum_required_charts(self) -> List[ChartSpecification]:
        """Get the minimum required charts from configuration"""
        min_charts = []
        
        for min_chart_config in self.charts_config.get('minimum_charts', []):
            chart = ChartSpecification(
                section=min_chart_config['section'],
                type=min_chart_config['type'],
                title=min_chart_config['title'],
                priority=1,
                required=True,
                rationale="Required by configuration"
            )
            min_charts.append(chart)
        
        self.logger.debug(f"Loaded {len(min_charts)} minimum required charts")
        return min_charts
    
    def _should_suggest_additional_charts(self) -> bool:
        """Determine if additional charts should be suggested"""
        mode = self.charts_config.get('mode', 'minimum')
        allow_additional = self.charts_config.get('allow_additional', False)
        
        return mode == 'dynamic' and allow_additional
    
    def _suggest_additional_charts(self, section_summaries: Dict[str, str], project_type: str) -> List[ChartSpecification]:
        """
        Use AI to suggest additional charts based on section content
        
        Args:
            section_summaries: Brief summaries of each section
            project_type: Type of project
            
        Returns:
            List of additional chart suggestions
        """
        try:
            # Format section summaries for prompt
            formatted_summaries = "\n".join([
                f"• {section}: {summary[:100]}..." if len(summary) > 100 else f"• {section}: {summary}"
                for section, summary in section_summaries.items()
            ])
            
            # Create prompt for chart suggestions
            prompt = f"""You are an expert data visualization consultant helping to enhance a {project_type} proposal.

Based on these section summaries, suggest 2-4 additional data visualizations that would make the proposal more compelling and easier to understand:

SECTION SUMMARIES:
{formatted_summaries}

GUIDELINES:
- Only suggest chart types from this list: {", ".join(self.supported_chart_types)}
- We already have required charts for: gantt (Project Plan), budget_breakdown (Budget), risk_matrix (Risk Analysis)
- Avoid duplicating these existing charts unless a different section would clearly benefit from similar visualization
- Use bar charts for comparisons, metrics, or phase-based data
- Use pie charts for distributions, allocations, or percentage breakdowns
- Use line charts for trends, progress over time, or performance metrics
- Each suggestion must add clear value to proposal understanding

Return a JSON response with the following structure:
{{
    "charts": [
        {{
            "section": "section name where this chart belongs",
            "type": "chart type (gantt, budget_breakdown, risk_matrix, bar, pie, or line)",
            "title": "descriptive title for the chart",
            "priority": 2,
            "required": false,
            "rationale": "brief reason for suggesting this chart"
        }}
    ],
    "total_count": number_of_charts,
    "reasoning": "brief explanation of chart selection strategy"
}}"""
            
            self.logger.debug(f"Requesting AI suggestions for additional charts")
            
            # Get AI response using OpenAI SDK with structured output
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data visualization consultant. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            response_json = json.loads(response.choices[0].message.content)
            
            # Convert to ChartDecisionResponse
            parsed_response = ChartDecisionResponse(**response_json)
            
            # Filter out charts that conflict with required ones
            required_sections = set(chart['section'] for chart in self.charts_config.get('minimum_charts', []))
            
            additional_charts = []
            for chart in parsed_response.charts:
                # Skip if it's a duplicate of required charts
                if chart.section not in required_sections:
                    chart.priority = 2  # Mark as recommended
                    chart.required = False
                    additional_charts.append(chart)
            
            self.logger.info(f"AI suggested {len(additional_charts)} additional charts")
            for chart in additional_charts:
                self.logger.debug(f"  - {chart.type} for {chart.section}: {chart.rationale}")
            
            return additional_charts
            
        except Exception as e:
            self.logger.warning(f"Could not get AI chart suggestions: {str(e)}")
            return self._get_fallback_additional_charts(section_summaries)
    
    def _get_fallback_additional_charts(self, section_summaries: Dict[str, str]) -> List[ChartSpecification]:
        """
        Provide fallback chart suggestions without AI
        
        Args:
            section_summaries: Brief summaries of each section
            
        Returns:
            List of fallback chart suggestions
        """
        # Since we only support gantt, budget_breakdown, and risk_matrix,
        # and these are already in minimum_charts, return empty list
        # to avoid suggesting unsupported chart types
        return []
    
    def _format_chart_spec(self, chart: ChartSpecification) -> Dict[str, Any]:
        """
        Format chart specification to expected output format
        
        Args:
            chart: ChartSpecification object
            
        Returns:
            Dictionary in expected format
        """
        return {
            'section': chart.section,
            'type': chart.type,
            'title': chart.title,
            'priority': chart.priority,
            'required': chart.required,
            'rationale': chart.rationale
        }
    
    def validate_chart_requirements(self, charts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that all required charts are included
        
        Args:
            charts: List of chart specifications
            
        Returns:
            Validation result with status and details
        """
        required_charts = self.charts_config.get('minimum_charts', [])
        chart_map = {(c['section'], c['type']): c for c in charts}
        
        missing_charts = []
        for req_chart in required_charts:
            key = (req_chart['section'], req_chart['type'])
            if key not in chart_map:
                missing_charts.append(req_chart)
        
        validation_result = {
            'valid': len(missing_charts) == 0,
            'missing_charts': missing_charts,
            'total_charts': len(charts),
            'required_charts': len(required_charts)
        }
        
        if not validation_result['valid']:
            self.logger.warning(f"Chart validation failed: {len(missing_charts)} missing required charts")
        else:
            self.logger.info("Chart requirements validation passed")
        
        return validation_result
    
    def get_chart_generation_summary(self, charts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of chart generation decisions for reporting
        
        Args:
            charts: List of chart specifications
            
        Returns:
            Summary information
        """
        chart_types = {}
        priorities = {1: 0, 2: 0, 3: 0}
        sections = set()
        
        for chart in charts:
            chart_type = chart['type']
            chart_types[chart_type] = chart_types.get(chart_type, 0) + 1
            priorities[chart['priority']] += 1
            sections.add(chart['section'])
        
        summary = {
            'total_charts': len(charts),
            'chart_types': dict(chart_types),
            'priority_distribution': dict(priorities),
            'sections_with_charts': len(sections),
            'chart_list': [{'section': c['section'], 'type': c['type'], 'title': c['title']} for c in charts]
        }
        
        self.logger.info(f"Chart generation summary: {summary['total_charts']} charts across {summary['sections_with_charts']} sections")
        return summary


# Convenience function for easy integration
def create_chart_decision_agent(config_path: str = "config/settings.yml") -> ChartDecisionAgent:
    """
    Create a ChartDecisionAgent with configuration loaded from file
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Configured ChartDecisionAgent instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return ChartDecisionAgent(config)


# Example usage for testing
if __name__ == "__main__":
    # Example usage
    agent = create_chart_decision_agent()
    
    sample_summaries = {
        "Executive Summary": "Overview of comprehensive digital transformation project requiring database development and system integration",
        "Budget": "Total project cost breakdown including development, testing, and deployment phases with resource allocation",
        "Implementation Strategy": "Phased approach with agile methodology covering analysis, design, development, and deployment over 12 months"
    }
    
    charts = agent.decide_charts(sample_summaries, "database_development")
    
    print("Chart Decisions:")
    for chart in charts:
        print(f"  - {chart['type']} chart for {chart['section']}: {chart['title']}")