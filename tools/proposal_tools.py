"""
Proposal Generation Tools for OpenAI Agents SDK
All tools are decorated with @function_tool for SDK integration
"""

from agents import function_tool
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import yaml
from pathlib import Path

# Import required components from core modules
from core.rag_retriever import RAGRetriever
from core.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)

# Global configuration - loaded once
_config = None
_rag_retriever = None

def _get_config():
    """Load configuration once and cache it"""
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent / "config" / "settings.yml"
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
    return _config

def _get_rag_retriever():
    """Get RAG retriever instance"""
    global _rag_retriever
    if _rag_retriever is None:
        config = _get_config()
        _rag_retriever = RAGRetriever(config)
    return _rag_retriever

def _get_chart_generator(output_format='static'):
    """Get chart generator instance with specified output format"""
    # Create a new instance each time to ensure proper format
    return ChartGenerator(output_format=output_format)

def _get_output_format():
    """Get the current output format from thread-local storage or default to static"""
    import threading
    return getattr(threading.current_thread(), '_chart_output_format', 'static')

def _set_output_format(format_type):
    """Set the output format for chart generation in current thread"""
    import threading
    threading.current_thread()._chart_output_format = format_type

# ============= WEB SEARCH TOOLS =============

@function_tool
def web_search(query: str, max_results: int = 5) -> list:
    """
    Perform web search using configured provider (OpenAI browsing or Tavily)
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of search results with title, snippet, url, and score
    """
    try:
        config = _get_config()
        web_search_config = config.get('web_search', {})
        provider = web_search_config.get('provider', 'tavily')
        
        logger.info(f"Web search provider: {provider}, query: {query}")
        
        if provider == 'openai' and web_search_config.get('openai_browsing', False):
            # When OpenAI browsing is enabled, this tool shouldn't be called
            # The research agent uses built-in browsing capability instead
            logger.info(f"OpenAI browsing is enabled - research agent uses built-in browsing")
            return []  # Return empty since agent should use built-in browsing
            
        elif provider == 'tavily' and web_search_config.get('tavily_enabled', False):
            # Use Tavily for web search
            try:
                from tavily import TavilyClient
                tavily_key = config.get('api_keys', {}).get('tavily') or os.getenv('TAVILY_API_KEY')
                if tavily_key:
                    tavily_client = TavilyClient(api_key=tavily_key)
                    logger.info(f"Performing Tavily web search for: {query}")
                    results = tavily_client.search(
                        query=query,
                        max_results=max_results,
                        search_depth=web_search_config.get('search_depth', 'advanced')
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
                    logger.warning("Tavily API key not found, falling back to search instructions")
            except ImportError:
                logger.warning("Tavily library not installed, falling back to search instructions")
        
        # Fallback: Return search instructions for manual research
        logger.info(f"Returning search instructions for: {query}")
        return [
            {
                "title": "Search Required",
                "snippet": f"Please search for current information about: {query}. Looking for {max_results} relevant results including recent developments, key facts, industry standards, and best practices related to this topic.",
                "url": "search_instruction",
                "score": 1.0
            }
        ]
        
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return [{
            "title": "Search Error",
            "snippet": f"Unable to perform web search: {str(e)}",
            "url": "error",
            "score": 0
        }]

# ============= BUDGET CALCULATION TOOLS =============

@function_tool
def calculate_project_costs(
    resources_json: str, 
    duration_months: int, 
    include_overhead: bool = True,
    project_type: str = "standard"
) -> str:
    """
    Calculate project costs based on resource requirements and duration
    
    Args:
        resources_json: JSON string containing list of resource requirements with roles, levels, and hours
                       Example: '[{"role": "Developer", "level": "Senior", "hours": 320, "source": "internal"}]'
        duration_months: Project duration in months
        include_overhead: Whether to include overhead costs (default: True)
        project_type: Type of project for cost adjustments (default: "standard")
        
    Returns:
        JSON string with detailed cost breakdown
    """
    try:
        # Parse resources from JSON
        resources = json.loads(resources_json)
        
        # Load skill data
        data_dir = Path(__file__).parent.parent / "data"
        internal_skills = pd.read_csv(data_dir / "skill_company.csv")
        external_skills = pd.read_csv(data_dir / "skill_external.csv")
        
        total_labor = 0
        breakdown = {}
        resource_details = []
        
        for resource in resources:
            role = resource.get('role', 'Developer')
            level = resource.get('level', 'Mid-level')
            hours = resource.get('hours', 160 * duration_months)  # Default: full-time
            source = resource.get('source', 'internal')  # internal or external
            
            # Find matching skill from appropriate CSV
            if source == 'internal':
                matching_skills = internal_skills[
                    (internal_skills['skill_name'].str.contains(role, case=False, na=False)) |
                    (internal_skills['skill_category'].str.contains(role, case=False, na=False))
                ]
                if not matching_skills.empty:
                    # Filter by experience level
                    level_match = matching_skills[matching_skills['experience_level'].str.contains(level, case=False, na=False)]
                    if not level_match.empty:
                        rate = level_match.iloc[0]['hourly_rate_usd']
                    else:
                        rate = matching_skills.iloc[0]['hourly_rate_usd']
                else:
                    # Default rates by level
                    rate = {'Junior': 50, 'Mid-level': 80, 'Senior': 120}.get(level, 80)
            else:
                # External resources
                matching_external = external_skills[
                    (external_skills['skill_name'].str.contains(role, case=False, na=False)) |
                    (external_skills['skill_category'].str.contains(role, case=False, na=False))
                ]
                if not matching_external.empty:
                    level_match = matching_external[matching_external['experience_level'].str.contains(level, case=False, na=False)]
                    if not level_match.empty:
                        rate = level_match.iloc[0]['hourly_rate_usd']
                    else:
                        rate = matching_external.iloc[0]['hourly_rate_usd']
                else:
                    # Default external rates (higher than internal)
                    rate = {'Junior': 70, 'Mid-level': 110, 'Senior': 160}.get(level, 110)
            
            cost = hours * rate
            total_labor += cost
            
            if role not in breakdown:
                breakdown[role] = 0
            breakdown[role] += cost
            
            resource_details.append({
                'role': role,
                'level': level,
                'hours': hours,
                'rate': rate,
                'cost': cost,
                'source': source
            })
        
        # Calculate additional costs
        overhead = 0
        if include_overhead:
            overhead = total_labor * 0.25  # 25% overhead
        
        # Infrastructure costs vary by project type
        infrastructure_multiplier = {
            'standard': 5000,
            'enterprise': 8000,
            'startup': 3000,
            'government': 7000
        }.get(project_type, 5000)
        
        infrastructure = duration_months * infrastructure_multiplier
        
        # Contingency based on project type
        contingency_rate = {
            'standard': 0.15,
            'enterprise': 0.20,
            'startup': 0.25,
            'government': 0.18
        }.get(project_type, 0.15)
        
        subtotal = total_labor + overhead + infrastructure
        contingency = subtotal * contingency_rate
        
        total = subtotal + contingency
        
        result = {
            "total_cost": round(total, 2),
            "labor_cost": round(total_labor, 2),
            "overhead": round(overhead, 2),
            "infrastructure": round(infrastructure, 2),
            "contingency": round(contingency, 2),
            "breakdown_by_role": {k: round(v, 2) for k, v in breakdown.items()},
            "monthly_burn": round(total / duration_months, 2),
            "resource_details": resource_details,
            "project_type": project_type,
            "contingency_rate": contingency_rate
        }
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Cost calculation error: {str(e)}")
        result = {
            "total_cost": 0,
            "error": str(e)
        }
        return json.dumps(result)

@function_tool
def generate_timeline(
    start_date: str,
    phases_json: str,
    project_type: str = "standard"
) -> str:
    """
    Generate project timeline with phases and milestones
    
    Args:
        start_date: Project start date in YYYY-MM-DD format
        phases_json: JSON string containing list of project phases with names, durations, and deliverables
                    Example: '[{"name": "Planning", "duration_weeks": 4, "deliverables": ["Requirements"], "milestones": ["Kickoff"]}]'
        project_type: Type of project for timeline adjustments
        
    Returns:
        JSON string with detailed timeline information
    """
    try:
        # Parse phases from JSON
        phases = json.loads(phases_json)
        
        timeline = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        for i, phase in enumerate(phases):
            duration_weeks = phase.get('duration_weeks', 4)
            
            # Adjust duration based on project type
            if project_type == 'enterprise':
                duration_weeks = int(duration_weeks * 1.2)  # 20% longer
            elif project_type == 'startup':
                duration_weeks = int(duration_weeks * 0.8)  # 20% shorter
            
            end_date = current_date + timedelta(weeks=duration_weeks)
            
            phase_info = {
                "phase": phase.get('name', f'Phase {i+1}'),
                "start": current_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "duration_weeks": duration_weeks,
                "deliverables": phase.get('deliverables', []),
                "milestones": phase.get('milestones', []),
                "resources": phase.get('resources', []),
                "dependencies": phase.get('dependencies', []),
                "risk_level": phase.get('risk_level', 'Medium')
            }
            
            timeline.append(phase_info)
            current_date = end_date
        
        # Calculate total project duration
        total_weeks = sum(phase['duration_weeks'] for phase in timeline)
        project_end = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(weeks=total_weeks)
        
        result = {
            "timeline": timeline,
            "total_duration_weeks": total_weeks,
            "project_start": start_date,
            "project_end": project_end.strftime("%Y-%m-%d"),
            "project_type": project_type,
            "total_phases": len(timeline)
        }
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Timeline generation error: {str(e)}")
        result = {
            "timeline": [],
            "error": str(e)
        }
        return json.dumps(result)

# ============= RISK ANALYSIS TOOLS =============

@function_tool
def analyze_risks(
    project_context: str,
    project_type: str = "standard"
) -> list:
    """
    Analyze project risks based on context and type
    
    Args:
        project_context: JSON string containing project requirements and context
        project_type: Type of project for risk assessment
        
    Returns:
        List of identified risks with mitigation strategies
    """
    try:
        # Parse project context from JSON string
        if project_context:
            try:
                context_dict = json.loads(project_context)
            except json.JSONDecodeError:
                context_dict = {}
        else:
            context_dict = {}
            
        risks = []
        
        # Technical risks
        requirements = context_dict.get('requirements', {})
        if any(tech in str(requirements).lower() for tech in ['ai', 'machine learning', 'blockchain', 'new technology']):
            risks.append({
                "category": "Technical",
                "risk": "New technology adoption",
                "probability": "Medium",
                "impact": "High",
                "risk_score": 3 * 4,
                "mitigation": "Proof of concept development, technical training, expert consultation",
                "owner": "Technical Lead",
                "timeline": "Phase 1"
            })
        
        # Timeline risks
        if context_dict.get('timeline', '').lower() in ['urgent', 'asap', 'fast', 'aggressive']:
            risks.append({
                "category": "Schedule",
                "risk": "Aggressive timeline constraints",
                "probability": "High",
                "impact": "Medium",
                "risk_score": 4 * 3,
                "mitigation": "Parallel development streams, additional resources, phased delivery approach",
                "owner": "Project Manager",
                "timeline": "Throughout project"
            })
        
        # Integration risks
        if 'integration' in str(requirements).lower() or 'third-party' in str(requirements).lower():
            risks.append({
                "category": "Integration",
                "risk": "Third-party system dependencies",
                "probability": "Medium",
                "impact": "Medium",
                "risk_score": 3 * 3,
                "mitigation": "Early API testing, fallback mechanisms, vendor coordination",
                "owner": "Integration Lead",
                "timeline": "Phase 2-3"
            })
        
        # Budget risks for different project types
        if project_type == 'startup':
            risks.append({
                "category": "Financial",
                "risk": "Budget constraints in startup environment",
                "probability": "High",
                "impact": "High",
                "risk_score": 4 * 4,
                "mitigation": "Flexible resource allocation, MVP approach, cost monitoring",
                "owner": "Project Manager",
                "timeline": "Throughout project"
            })
        elif project_type == 'enterprise':
            risks.append({
                "category": "Organizational",
                "risk": "Complex approval processes",
                "probability": "Medium",
                "impact": "Medium",
                "risk_score": 3 * 3,
                "mitigation": "Stakeholder alignment, clear decision matrix, escalation procedures",
                "owner": "Project Sponsor",
                "timeline": "Phase 1"
            })
        
        # Security risks
        if any(term in str(context_dict).lower() for term in ['security', 'privacy', 'compliance', 'gdpr', 'hipaa']):
            risks.append({
                "category": "Security",
                "risk": "Data security and compliance requirements",
                "probability": "Low",
                "impact": "Very High",
                "risk_score": 2 * 5,
                "mitigation": "Security audit, compliance framework implementation, regular testing",
                "owner": "Security Lead",
                "timeline": "Throughout project"
            })
        
        # Resource risks
        team_size = context_dict.get('team_size', 5)
        if team_size > 10:
            risks.append({
                "category": "Resource",
                "risk": "Large team coordination challenges",
                "probability": "Medium",
                "impact": "Medium",
                "risk_score": 3 * 3,
                "mitigation": "Clear communication protocols, regular standups, defined responsibilities",
                "owner": "Project Manager",
                "timeline": "Throughout project"
            })
        
        # Add default resource risk
        risks.append({
            "category": "Resource",
            "risk": "Key personnel availability",
            "probability": "Low",
            "impact": "High",
            "risk_score": 2 * 4,
            "mitigation": "Knowledge documentation, backup resources, cross-training programs",
            "owner": "Resource Manager",
            "timeline": "Phase 1"
        })
        
        return risks
        
    except Exception as e:
        logger.error(f"Risk analysis error: {str(e)}")
        return []

# ============= RAG RETRIEVAL TOOLS =============

@function_tool
def get_rag_context(
    query: str,
    pdf_path: str,
    max_tokens: int = 2000
) -> str:
    """
    Retrieve relevant context from RFP documents using RAG pipeline
    
    Args:
        query: The query to search for in the document
        pdf_path: Path to the PDF document to search
        max_tokens: Maximum tokens to return in context
        
    Returns:
        Formatted context string with relevant information
    """
    try:
        rag_retriever = _get_rag_retriever()
        
        # Process PDF if not already indexed
        if not rag_retriever.current_pdf_path == pdf_path:
            rag_retriever.process_and_index_pdf(pdf_path)
        
        # Get context for the query
        context = rag_retriever.get_context_for_agent(query, pdf_path, max_tokens)
        
        logger.info(f"Retrieved RAG context for query: {query[:50]}...")
        return context
        
    except Exception as e:
        logger.error(f"RAG context retrieval error: {str(e)}")
        return f"Error retrieving context: {str(e)}"

# ============= CHART GENERATION TOOLS =============

@function_tool
def generate_gantt_chart(timeline_data: str) -> str:
    """
    Generate Gantt chart for project timeline
    
    Args:
        timeline_data: JSON string containing timeline phases and information
        
    Returns:
        HTML string containing the generated Gantt chart
    """
    try:
        # Parse timeline data from JSON string
        if timeline_data:
            try:
                data_dict = json.loads(timeline_data)
            except json.JSONDecodeError:
                data_dict = {}
        else:
            data_dict = {}
            
        # Use thread-local output format or default to static
        output_format = _get_output_format()
        chart_generator = _get_chart_generator(output_format=output_format)
        
        # Transform timeline data to expected format
        phases = []
        for phase in data_dict.get('timeline', []):
            phases.append({
                'name': phase.get('phase', 'Unknown Phase'),
                'duration': phase.get('duration_weeks', 4),
                'color': None  # Let chart generator assign colors
            })
        
        chart_data = {
            'title': 'Project Timeline',
            'phases': phases
        }
        
        chart_html = chart_generator.generate_gantt_chart(chart_data)
        logger.info("Generated Gantt chart successfully")
        return chart_html
        
    except Exception as e:
        logger.error(f"Gantt chart generation error: {str(e)}")
        return f'<div class="chart-error">Error generating Gantt chart: {str(e)}</div>'

@function_tool
def create_budget_chart(budget_data: str) -> str:
    """
    Create budget breakdown visualization chart
    
    Args:
        budget_data: JSON string containing budget breakdown information
        
    Returns:
        HTML string containing the generated budget chart
    """
    try:
        # Parse budget data from JSON string
        if budget_data:
            try:
                data_dict = json.loads(budget_data)
            except json.JSONDecodeError:
                data_dict = {}
        else:
            data_dict = {}
            
        # Use thread-local output format or default to static
        output_format = _get_output_format()
        chart_generator = _get_chart_generator(output_format=output_format)
        chart_html = chart_generator.create_budget_chart(data_dict)
        logger.info("Generated budget chart successfully")
        return chart_html
        
    except Exception as e:
        logger.error(f"Budget chart generation error: {str(e)}")
        return f'<div class="chart-error">Error generating budget chart: {str(e)}</div>'

@function_tool
def build_risk_matrix(risks_json: str) -> str:
    """
    Build risk assessment matrix visualization
    
    Args:
        risks_json: JSON string containing list of risk dictionaries with probability and impact data
                   Example: '[{"risk": "Technical Risk", "probability": "Medium", "impact": "High"}]'
        
    Returns:
        HTML string containing the generated risk matrix chart
    """
    try:
        # Parse risks from JSON
        risks = json.loads(risks_json)
        
        # Use thread-local output format or default to static
        output_format = _get_output_format()
        chart_generator = _get_chart_generator(output_format=output_format)
        
        # Convert risk data to chart format
        chart_risks = []
        for risk in risks:
            # Map text values to numeric
            prob_map = {'Low': 1, 'Medium': 3, 'High': 4, 'Very High': 5}
            impact_map = {'Low': 1, 'Medium': 3, 'High': 4, 'Very High': 5}
            
            probability = prob_map.get(risk.get('probability', 'Medium'), 3)
            impact = impact_map.get(risk.get('impact', 'Medium'), 3)
            
            chart_risks.append({
                'name': risk.get('risk', 'Unknown Risk'),
                'probability': probability,
                'impact': impact
            })
        
        chart_html = chart_generator.build_risk_matrix(chart_risks)
        logger.info("Generated risk matrix successfully")
        return chart_html
        
    except Exception as e:
        logger.error(f"Risk matrix generation error: {str(e)}")
        return f'<div class="chart-error">Error generating risk matrix: {str(e)}</div>'

# ============= FORMATTING TOOLS =============

@function_tool
def format_section(
    content: str,
    section_title: str,
    formatting_style: str = "professional"
) -> str:
    """
    Format proposal section content with consistent styling
    
    Args:
        content: Raw content text to format
        section_title: Title of the section
        formatting_style: Style to apply (professional, technical, executive)
        
    Returns:
        Formatted HTML content string
    """
    try:
        if not content or not content.strip():
            return f'<div class="section-placeholder">Content for {section_title} will be generated.</div>'
        
        # Clean up content
        content = content.strip()
        
        # Apply formatting based on style
        if formatting_style == "professional":
            formatted_content = f"""
            <div class="section-content professional">
                <h2 class="section-title">{section_title}</h2>
                <div class="section-body">
                    {_format_paragraphs(content)}
                </div>
            </div>
            """
        elif formatting_style == "technical":
            formatted_content = f"""
            <div class="section-content technical">
                <h2 class="section-title technical-title">{section_title}</h2>
                <div class="section-body technical-body">
                    {_format_technical_content(content)}
                </div>
            </div>
            """
        elif formatting_style == "executive":
            formatted_content = f"""
            <div class="section-content executive">
                <h2 class="section-title executive-title">{section_title}</h2>
                <div class="section-body executive-body">
                    {_format_executive_content(content)}
                </div>
            </div>
            """
        else:
            # Default formatting
            formatted_content = f"""
            <div class="section-content">
                <h2 class="section-title">{section_title}</h2>
                <div class="section-body">
                    {_format_paragraphs(content)}
                </div>
            </div>
            """
        
        logger.info(f"Formatted section: {section_title}")
        return formatted_content
        
    except Exception as e:
        logger.error(f"Section formatting error: {str(e)}")
        return f'<div class="section-error">Error formatting {section_title}: {str(e)}</div>'

def _format_paragraphs(content: str) -> str:
    """Format content into HTML paragraphs"""
    paragraphs = content.split('\n\n')
    formatted = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            # Check if it's a list item
            if para.startswith('-') or para.startswith('•') or para.startswith('*'):
                # Convert to HTML list
                items = [item.strip().lstrip('-•*').strip() for item in para.split('\n') if item.strip()]
                formatted.append('<ul>' + ''.join(f'<li>{item}</li>' for item in items) + '</ul>')
            else:
                formatted.append(f'<p>{para}</p>')
    
    return '\n'.join(formatted)

def _format_technical_content(content: str) -> str:
    """Format technical content with code blocks and technical styling"""
    # Look for code blocks or technical terms
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Check for code-like content
            if any(tech in line.lower() for tech in ['python', 'javascript', 'sql', 'api', 'json', 'xml']):
                formatted_lines.append(f'<code class="inline-code">{line}</code>')
            else:
                formatted_lines.append(f'<p class="technical-para">{line}</p>')
    
    return '\n'.join(formatted_lines)

def _format_executive_content(content: str) -> str:
    """Format executive content with emphasis on key points"""
    paragraphs = content.split('\n\n')
    formatted = []
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if para:
            # Emphasize first paragraph
            if i == 0:
                formatted.append(f'<p class="executive-summary">{para}</p>')
            else:
                formatted.append(f'<p class="executive-detail">{para}</p>')
    
    return '\n'.join(formatted)

# ============= UTILITY TOOLS =============

@function_tool
def get_company_skills() -> str:
    """
    Get available company skills and resources
    
    Returns:
        JSON string containing internal and external skill information
    """
    try:
        data_dir = Path(__file__).parent.parent / "data"
        
        # Load skill data
        internal_skills = pd.read_csv(data_dir / "skill_company.csv")
        external_skills = pd.read_csv(data_dir / "skill_external.csv")
        
        # Process internal skills
        internal_summary = {
            'total_employees': internal_skills['employee_count'].sum(),
            'skill_categories': internal_skills['skill_category'].unique().tolist(),
            'skills_by_category': {},
            'experience_levels': internal_skills['experience_level'].unique().tolist(),
            'rate_ranges': {
                'min': internal_skills['hourly_rate_usd'].min(),
                'max': internal_skills['hourly_rate_usd'].max(),
                'avg': internal_skills['hourly_rate_usd'].mean()
            }
        }
        
        # Group by category
        for category in internal_summary['skill_categories']:
            category_skills = internal_skills[internal_skills['skill_category'] == category]
            internal_summary['skills_by_category'][category] = {
                'skills': category_skills['skill_name'].unique().tolist(),
                'total_employees': category_skills['employee_count'].sum(),
                'avg_rate': category_skills['hourly_rate_usd'].mean()
            }
        
        # Process external skills
        external_summary = {
            'total_vendors': len(external_skills),
            'skill_categories': external_skills['skill_category'].unique().tolist(),
            'availability': external_skills['availability'].value_counts().to_dict(),
            'rate_ranges': {
                'min': external_skills['hourly_rate_usd'].min(),
                'max': external_skills['hourly_rate_usd'].max(),
                'avg': external_skills['hourly_rate_usd'].mean()
            }
        }
        
        result = {
            'internal_skills': internal_summary,
            'external_skills': external_summary,
            'total_skill_categories': len(set(internal_summary['skill_categories'] + external_summary['skill_categories']))
        }
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error getting company skills: {str(e)}")
        result = {'error': str(e)}
        return json.dumps(result)

@function_tool
def validate_proposal_requirements(requirements: str) -> str:
    """
    Validate proposal requirements and suggest improvements
    
    Args:
        requirements: JSON string containing proposal requirements
        
    Returns:
        JSON string with validation results and suggestions
    """
    try:
        # Parse requirements from JSON string
        if requirements:
            try:
                req_dict = json.loads(requirements)
            except json.JSONDecodeError:
                req_dict = {}
        else:
            req_dict = {}
            
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'suggestions': [],
            'missing_fields': [],
            'score': 0
        }
        
        # Check required fields
        required_fields = ['project_name', 'client_name', 'timeline', 'budget_range']
        for field in required_fields:
            if field not in req_dict or not req_dict[field]:
                validation_results['missing_fields'].append(field)
                validation_results['is_valid'] = False
        
        # Check timeline reasonableness
        timeline = req_dict.get('timeline', '')
        if timeline and any(word in timeline.lower() for word in ['asap', 'urgent', 'immediately']):
            validation_results['warnings'].append('Aggressive timeline detected - consider risk mitigation')
        
        # Check budget information
        budget = req_dict.get('budget_range', '')
        if not budget or budget.lower() == 'tbd':
            validation_results['suggestions'].append('Specific budget range would help with accurate planning')
        
        # Calculate validation score
        total_checks = 10
        score = total_checks - len(validation_results['missing_fields']) - len(validation_results['warnings'])
        validation_results['score'] = max(0, (score / total_checks) * 100)
        
        return json.dumps(validation_results)
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        result = {'error': str(e), 'is_valid': False}
        return json.dumps(result)