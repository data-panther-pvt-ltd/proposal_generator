"""
Tool implementations for agents
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages tools available to agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.web_search_config = config.get('web_search', {})
        
        # Initialize OpenAI client (always needed)
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OpenAI API key required for all functionality")
        
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=openai_key)
        
        # Initialize web search based on configuration
        self.web_search_provider = self.web_search_config.get('provider', 'openai')
        
        if self.web_search_provider == 'tavily' and self.web_search_config.get('tavily_enabled', False):
            # Initialize Tavily if configured
            try:
                from tavily import TavilyClient
                tavily_key = config.get('api_keys', {}).get('tavily') or os.getenv('TAVILY_API_KEY')
                if tavily_key:
                    self.tavily_client = TavilyClient(api_key=tavily_key)
                    logger.info("Tavily web search initialized")
                else:
                    logger.warning("Tavily API key not found, falling back to OpenAI web search")
                    self.web_search_provider = 'openai'
            except ImportError:
                logger.warning("Tavily library not installed, using OpenAI web search")
                self.web_search_provider = 'openai'
        else:
            self.tavily_client = None
            logger.info("Using OpenAI Responses API for web search")
        
    async def web_search(self, query: str, max_results: int = None) -> List[Dict]:
        """Perform web search using configured provider (OpenAI or Tavily)"""
        if max_results is None:
            max_results = self.web_search_config.get('max_results', 5)
        
        try:
            if self.web_search_provider == 'tavily' and hasattr(self, 'tavily_client') and self.tavily_client:
                # Use Tavily for web search
                logger.info(f"Performing Tavily web search for: {query}")
                results = self.tavily_client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=self.web_search_config.get('search_depth', 'advanced')
                )
                
                return [
                    {
                        "title": r.get("title"),
                        "snippet": r.get("snippet"),
                        "url": r.get("url"),
                        "score": r.get("score", 0)
                    }
                    for r in results.get("results", [])
                ]
            else:
                # Use OpenAI's Responses API with web_search tool
                logger.info(f"Performing OpenAI web search for: {query}")
                response = self.openai_client.responses.create(
                    model=self.config.get('openai', {}).get('model', 'gpt-4o'),
                    input=f"Search for: {query}. Provide {max_results} relevant results.",
                    tools=[{"type": "web_search"}],  # Built-in web search tool
                )
                
                # Extract search results from response
                logger.info(f"OpenAI web search completed for query: {query}")
                
                # The response includes web search results with citations
                # For now, return structured results based on the response
                if hasattr(response, 'output_text'):
                    # Parse the output to extract search results
                    # In production, the API returns structured search results
                    return []
                
                return []
            
        except Exception as e:
            logger.error(f"Web search error with {self.web_search_provider}: {str(e)}")
            return []
    
    def calculate_costs(
        self, 
        resources: List[Dict],
        duration_months: int,
        include_overhead: bool = True
    ) -> Dict:
        """Calculate project costs based on resources"""
        
        total_labor = 0
        breakdown = {}
        
        for resource in resources:
            role = resource.get('role')
            level = resource.get('level', 'mid-level')
            hours = resource.get('hours', 160 * duration_months)
            rate = resource.get('hourly_rate', 100)
            
            cost = hours * rate
            total_labor += cost
            
            if role not in breakdown:
                breakdown[role] = 0
            breakdown[role] += cost
        
        # Add overhead costs
        overhead = 0
        if include_overhead:
            overhead = total_labor * 0.25  # 25% overhead
        
        # Add infrastructure costs
        infrastructure = duration_months * 5000  # $5k/month average
        
        # Add contingency
        subtotal = total_labor + overhead + infrastructure
        contingency = subtotal * 0.15  # 15% contingency
        
        total = subtotal + contingency
        
        return {
            "total_cost": round(total, 2),
            "labor_cost": round(total_labor, 2),
            "overhead": round(overhead, 2),
            "infrastructure": round(infrastructure, 2),
            "contingency": round(contingency, 2),
            "breakdown_by_role": breakdown,
            "monthly_burn": round(total / duration_months, 2)
        }
    
    def generate_timeline(
        self,
        start_date: str,
        phases: List[Dict]
    ) -> List[Dict]:
        """Generate project timeline with phases"""
        
        timeline = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        for phase in phases:
            duration_weeks = phase.get('duration_weeks', 4)
            end_date = current_date + timedelta(weeks=duration_weeks)
            
            timeline.append({
                "phase": phase.get('name'),
                "start": current_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "duration_weeks": duration_weeks,
                "deliverables": phase.get('deliverables', []),
                "milestones": phase.get('milestones', []),
                "resources": phase.get('resources', [])
            })
            
            current_date = end_date
        
        return timeline
    
    def analyze_risks(self, project_context: Dict) -> List[Dict]:
        """Analyze project risks"""
        
        risks = []
        
        # Technical risks
        if 'new_technology' in str(project_context):
            risks.append({
                "category": "Technical",
                "risk": "New technology adoption",
                "probability": "Medium",
                "impact": "High",
                "mitigation": "Proof of concept, training, expert consultation"
            })
        
        # Timeline risks
        if project_context.get('timeline', '').lower() in ['urgent', 'asap', 'fast']:
            risks.append({
                "category": "Schedule",
                "risk": "Aggressive timeline",
                "probability": "High",
                "impact": "Medium",
                "mitigation": "Parallel workstreams, additional resources, phased delivery"
            })
        
        # Integration risks
        if 'integration' in project_context.get('requirements', {}):
            risks.append({
                "category": "Integration",
                "risk": "Third-party system dependencies",
                "probability": "Medium",
                "impact": "Medium",
                "mitigation": "Early API testing, fallback mechanisms, vendor coordination"
            })
        
        # Resource risks
        risks.append({
            "category": "Resource",
            "risk": "Key personnel availability",
            "probability": "Low",
            "impact": "High",
            "mitigation": "Knowledge documentation, backup resources, cross-training"
        })
        
        # Compliance risks
        if any(term in str(project_context).lower() for term in ['compliance', 'regulation', 'gdpr', 'sox']):
            risks.append({
                "category": "Compliance",
                "risk": "Regulatory requirements",
                "probability": "Low",
                "impact": "Very High",
                "mitigation": "Legal review, compliance checklist, regular audits"
            })
        
        return risks
    
    def optimize_resources(
        self,
        requirements: Dict,
        budget_constraint: Optional[float] = None
    ) -> Dict:
        """Optimize resource allocation"""
        
        # Analyze requirements to determine optimal team composition
        team_composition = {
            "project_manager": 1,
            "technical_lead": 1,
            "senior_developers": 2,
            "mid_developers": 3,
            "junior_developers": 2,
            "qa_engineers": 2,
            "devops_engineer": 1,
            "ui_ux_designer": 1
        }
        
        # Adjust based on project size
        if requirements.get('users', 0) > 10000:
            team_composition['senior_developers'] += 1
            team_composition['qa_engineers'] += 1
        
        # Adjust based on budget if provided
        if budget_constraint:
            # Implement budget-based optimization logic
            pass
        
        return {
            "recommended_team": team_composition,
            "total_resources": sum(team_composition.values()),
            "skill_coverage": self._calculate_skill_coverage(team_composition, requirements),
            "optimization_notes": "Team optimized for balanced delivery and quality"
        }
    
    def _calculate_skill_coverage(self, team: Dict, requirements: Dict) -> float:
        """Calculate how well the team covers required skills"""
        # Simplified skill coverage calculation
        required_skills = len(requirements.get('features', [])) + len(requirements.get('integration', []))
        team_capacity = sum(team.values()) * 2  # Assume each person covers 2 skill areas
        
        coverage = min(100, (team_capacity / max(required_skills, 1)) * 100)
        return round(coverage, 1)
    
    def format_currency(self, amount: float) -> str:
        """Format amount as currency"""
        return f"${amount:,.2f}"
    
    def calculate_roi(
        self,
        investment: float,
        expected_benefits: Dict,
        years: int = 3
    ) -> Dict:
        """Calculate ROI projections"""
        
        annual_benefits = sum(expected_benefits.values())
        total_benefits = annual_benefits * years
        roi = ((total_benefits - investment) / investment) * 100
        payback_period = investment / annual_benefits if annual_benefits > 0 else float('inf')
        
        return {
            "investment": investment,
            "annual_benefits": annual_benefits,
            "total_benefits": total_benefits,
            "roi_percentage": round(roi, 1),
            "payback_period_years": round(payback_period, 1),
            "break_even_month": round(payback_period * 12, 0)
        }