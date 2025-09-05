"""
SDK-based Proposal Runner
Handles proposal generation using OpenAI SDK agents
"""

import asyncio
import json
import re
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.docx_exporter import DOCXExporter
import pandas as pd
# pandas removed - not used
import logging

logger = logging.getLogger(__name__)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed. Using approximate token counting. Install with: pip install tiktoken")

# Import Runner from OpenAI Agents SDK
from agents import Runner, set_tracing_disabled

# Disable OpenAI tracing/telemetry
set_tracing_disabled(True)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proposal_agents.sdk_agents import get_agent, get_specialist_for_section
from core.simple_cost_tracker import SimpleCostTracker
from core.html_generator import HTMLGenerator
from core.pdf_exporter import PDFExporter
from core.rag_retriever import RAGRetriever
from core.chart_decision_agent import ChartDecisionAgent
from core.chart_generator import ChartGenerator
from core.rfp_extractor import RFPExtractor
from utils.data_loader import DataLoader
from utils.validators import ProposalValidator
from utils.agent_logger import agent_logger

logger = logging.getLogger(__name__)

# Context size limits for different models
CONTEXT_LIMITS = {
    'gpt-4o': 128000
}

# Reserve tokens for response
RESPONSE_RESERVE_TOKENS = 2000

def count_tokens(text: str, model: str = 'gpt-4o') -> int:
    """Count tokens in text using tiktoken or fallback estimation"""
    if not text:
        return 0
        
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"tiktoken encoding failed: {e}, using fallback")
    
    # Fallback: estimate 4 characters per token (conservative estimate)
    return len(text) // 4

def truncate_text(text: str, max_tokens: int, model: str = 'gpt-4o') -> str:
    """Truncate text to fit within token limit"""
    if not text:
        return text
        
    current_tokens = count_tokens(text, model)
    if current_tokens <= max_tokens:
        return text
        
    # Binary search for optimal truncation point
    left, right = 0, len(text)
    best_text = text[:max_tokens * 4]  # Initial rough estimate
    
    while left < right:
        mid = (left + right) // 2
        candidate = text[:mid] + "...[truncated]"
        
        if count_tokens(candidate, model) <= max_tokens:
            best_text = candidate
            left = mid + 1
        else:
            right = mid
            
    return best_text

def prepare_minimal_chart_context(section_name: str, chart_type: str, context: Dict) -> Dict:
    """Prepare minimal context for chart generation to avoid token limits"""
    request = context.get('request')
    
    # Extract only essential information
    minimal_context = {
        'chart_type': chart_type,
        'section_name': section_name,
        'project_name': getattr(request, 'project_name', 'Project') if request else 'Project',
        'timeline': getattr(request, 'timeline', '3 months') if request else '3 months',
        'budget_range': getattr(request, 'budget_range', None) if request else None,
        'project_type': getattr(request, 'project_type', 'general') if request else 'general'
    }
    
    # Add minimal section data (first 200 words only)
    generated_sections = context.get('generated_sections', {})
    if section_name in generated_sections:
        content = generated_sections[section_name].get('content', '')
        words = content.split()[:200]  # First 200 words only
        minimal_context['section_summary'] = ' '.join(words)
    
    return minimal_context

@dataclass
class ProposalRequest:
    """Structure for proposal generation request"""
    client_name: str
    project_name: str
    project_type: str
    requirements: Dict[str, Any]
    timeline: str
    budget_range: Optional[str] = None
    special_requirements: Optional[List[str]] = None

class ProposalRunner:
    """SDK-based proposal runner using OpenAI agents"""
    def clear_vector_cache(self):
        """Clear all vector database caches to free memory"""
        if hasattr(self, 'rag_retriever'):
            self.rag_retriever.clear_all_caches()
            logger.info("Cleared all vector database caches")

    def switch_pdf(self, pdf_path: str):
        """Switch to a different PDF for RAG retrieval"""
        if hasattr(self, 'rag_retriever'):
            result = self.rag_retriever.process_and_index_pdf(pdf_path)
            logger.info(f"Switched to PDF: {pdf_path}")
            return result
        return None
    def __init__(self, config_or_path=None, cost_tracker: Optional[SimpleCostTracker] = None):
        """Initialize the proposal runner"""
        if isinstance(config_or_path, dict):
            self.config = config_or_path
        elif isinstance(config_or_path, str):
            self.config = self._load_config(config_or_path)
        else:
            self.config = self._load_config("config/settings.yml")
        self.cost_tracker = cost_tracker or SimpleCostTracker()
        
        # Initialize components
        self.html_generator = HTMLGenerator(self.config, output_format='interactive')
        self.html_generator_pdf = HTMLGenerator(self.config, output_format='static')
        self.pdf_exporter = PDFExporter(self.config)
        self.docx_exporter = DOCXExporter(self.config)
        self.data_loader = DataLoader(self.config)
        self.validator = ProposalValidator(self.config)
        self.rag_retriever = RAGRetriever(self.config)
        self.rfp_extractor = RFPExtractor(self.config)
        
        # Load data
        self.skills_data = self.data_loader.load_skills_data()
        self.company_profile = self.data_loader.load_company_profile()
        self.case_studies = self.data_loader.load_case_studies()
        # Prepare compact skills summary for prompts
        self.skills_summary = self._summarize_skills(self.skills_data)
        
        # Get section routing configuration
        self.section_routing = self.config.get('section_routing', {})
        
        # Cache for vector database to avoid reloading
        self._vector_db_cache = {}
        
        # Initialize chart generation components
        self.chart_decision_agent = ChartDecisionAgent(self.config)
        self.chart_generator = ChartGenerator(output_format='static')
        
        logger.info("SDK Proposal Runner initialized successfully")

    def _read_text_file(self, path: str) -> str:
        """Safely read and return UTF-8 text from a file path."""
        try:
            if path and os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Failed to read text file {path}: {e}")
        return ""

    def _extract_partnerships_and_testimonials(self, markdown_text: str) -> Dict[str, Any]:
        """Extract Partnerships table rows and Testimonials quotes from markdown."""
        result = {"partnerships": [], "testimonials": []}
        if not markdown_text:
            return result
        try:
            lines = markdown_text.splitlines()
            # Extract Partnerships markdown table rows (lines with 3+ pipes)
            in_table = False
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('## partnerships'):
                    in_table = True
                    continue
                if in_table:
                    if '|' in line and line.count('|') >= 3:
                        # Skip header/separator lines
                        if set(line.replace('|', '').strip()) <= set('-: '):
                            continue
                        cells = [c.strip().strip('*').strip() for c in line.strip().strip('|').split('|')]
                        if len(cells) >= 3:
                            client, partnership, year = cells[0], cells[1], cells[2]
                            if client and partnership:
                                result["partnerships"].append({
                                    "client": client.strip('*').strip(),
                                    "partnership": partnership,
                                    "year": year
                                })
                    else:
                        # End of table when a non-table line after starting
                        if result["partnerships"]:
                            in_table = False
                
            # Extract Testimonials blockquotes
            in_testimonials = False
            for line in lines:
                if line.strip().lower().startswith('## testimonials'):
                    in_testimonials = True
                    continue
                if in_testimonials:
                    if line.strip().startswith('>'):
                        content = line.lstrip('> ').strip()
                        # Next lines may include author lines starting with > —
                        if content:
                            # Collect author if present on same line after closing quotes, or separate line
                            result["testimonials"].append({"quote_or_note": content})
                    elif line.strip().startswith('---') or line.strip().startswith('## '):
                        break
            # Normalize testimonials into quote + author when pattern matches
            normalized = []
            current_quote = None
            for item in result["testimonials"]:
                text = item.get("quote_or_note", "")
                if text.startswith('**"') or text.startswith('"') or text.startswith('**“'):
                    current_quote = text.strip('*')
                elif text.startswith('—') or ('—' in text):
                    author = text.strip('—').strip()
                    normalized.append({"quote": (current_quote or "").strip('*'), "author": author})
                    current_quote = None
            if normalized:
                result["testimonials"] = normalized
            else:
                # Fallback: keep raw captured lines as notes
                result["testimonials"] = []
        except Exception as e:
            logger.warning(f"Markdown extraction failed: {e}")
        return result

    def _sanitize_generated_content(self, content: str) -> str:
        """Remove raw JSON/code fences and render JSON as readable text when possible."""
        try:
            if not content:
                return content
            text = content.strip()
            # Strip markdown code fences
            if text.startswith('```') and text.endswith('```'):
                text = text.strip('`')
                # Remove optional language tag line
                parts = text.split('\n', 1)
                if len(parts) == 2 and len(parts[0]) < 15:  # likely language tag
                    text = parts[1]
                text = text.strip()
            # If JSON, convert to readable lines
            import json as _json
            try:
                parsed = _json.loads(text)
                def render(obj, indent=0):
                    prefix = '  ' * indent
                    if isinstance(obj, dict):
                        lines = []
                        for k, v in obj.items():
                            if isinstance(v, (dict, list)):
                                lines.append(f"{prefix}- {k}:")
                                lines.extend(render(v, indent+1))
                            else:
                                lines.append(f"{prefix}- {k}: {v}")
                        return lines
                    elif isinstance(obj, list):
                        lines = []
                        for item in obj:
                            if isinstance(item, (dict, list)):
                                lines.append(f"{prefix}-")
                                lines.extend(render(item, indent+1))
                            else:
                                lines.append(f"{prefix}- {item}")
                        return lines
                    else:
                        return [f"{prefix}{obj}"]
                readable = '\n'.join(render(parsed))
                return readable.strip()
            except Exception:
                return text
        except Exception:
            return content

    def _build_timeline_table_html(self, request) -> str:
        """Build a fallback HTML timeline table using request timeline."""
        try:
            timeline = getattr(request, 'timeline', '12 weeks') if request else '12 weeks'
            weeks = 12
            try:
                import re
                m = re.search(r'(\d+)\s*(weeks?|week|months?|month)', timeline, re.IGNORECASE)
                if m:
                    val = int(m.group(1))
                    unit = m.group(2).lower()
                    weeks = val if 'week' in unit else val * 4
            except Exception:
                pass
            phases = [
                ("Initiation", "Kickoff, governance setup, requirements baseline", 1, min(2, weeks)),
                ("Discovery", "Stakeholder interviews, AS-IS analysis, backlog", 3, min(4, weeks)),
                ("Design", "UX/UI, solution architecture, acceptance criteria", 7, min(4, weeks)),
                ("Development", "Sprints, integrations, data migration", 11, max(4, weeks // 2)),
                ("Testing/UAT", "QA, performance, security, UAT sign-off", 11 + max(4, weeks // 2), min(4, weeks)),
                ("Deployment", "Go-live, hypercare, monitoring", weeks - 1, 1),
                ("Handover", "Docs, training, warranty transition", weeks, 1),
            ]
            rows = []
            for name, acts, start_wk, dur in phases:
                end_wk = start_wk + dur - 1
                rows.append(f"<tr><td>{name}</td><td>{acts}</td><td>Week {start_wk}</td><td>Week {end_wk}</td><td>{dur}</td><td>Owner</td></tr>")
            table = (
                "<h3>Project Timeline</h3>"
                "<table border=\"1\" cellspacing=\"0\" cellpadding=\"6\">"
                "<thead><tr><th>Phase</th><th>Key Activities/Deliverables</th><th>Start Date</th><th>End Date</th><th>Duration (weeks)</th><th>Owner/Team</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>"
                "</table>"
            )
            return table
        except Exception:
            return ""

    def _summarize_skills(self, skills_df: pd.DataFrame) -> str:
        """Create a compact textual summary of skills for grounding prompts."""
        try:
            if skills_df is None or skills_df.empty:
                return ""
            # Group by category and experience level to keep concise
            summary_lines = []
            cols = set(skills_df.columns)
            category_col = 'skill_category' if 'skill_category' in cols else None
            name_col = 'skill_name' if 'skill_name' in cols else None
            level_col = 'experience_level' if 'experience_level' in cols else None
            count_col = 'employee_count' if 'employee_count' in cols else None
            if category_col and name_col:
                grouped = skills_df.groupby([category_col, name_col]).size().reset_index(name='count')
                # Limit to top 30 entries for brevity
                for _, row in grouped.sort_values('count', ascending=False).head(30).iterrows():
                    parts = [str(row[category_col]), str(row[name_col])]
                    if level_col and level_col in skills_df.columns:
                        sample_level = skills_df[(skills_df[category_col] == row[category_col]) & (skills_df[name_col] == row[name_col])][level_col].mode()
                        if not sample_level.empty:
                            parts.append(str(sample_level.iloc[0]))
                    if count_col and count_col in skills_df.columns:
                        total_people = int(skills_df[(skills_df[category_col] == row[category_col]) & (skills_df[name_col] == row[name_col])][count_col].sum())
                        parts.append(f"people:{total_people}")
                    summary_lines.append(" - " + " | ".join(parts))
            return "Skills Overview:\n" + "\n".join(summary_lines)
        except Exception:
            return ""
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def generate_proposal(self, request: ProposalRequest) -> Dict[str, Any]:
        """
        Generate a complete proposal using SDK agents
        
        Args:
            request: ProposalRequest object with all requirements
            
        Returns:
            Dictionary containing the generated proposal
        """
        logger.info(f"Starting SDK proposal generation for {request.client_name}")
        
        # Pre-load vector database if PDF is provided
        if hasattr(request, 'requirements') and request.requirements.get('source_pdf'):
            pdf_path = request.requirements['source_pdf']
            logger.info(f"Pre-loading vector database for {pdf_path}")
            self.rag_retriever.process_and_index_pdf(pdf_path)
        
        # Initialize proposal context
        context = {
            "request": request,
            "skills_data": self.skills_data,
            "skills_summary": self.skills_summary,
            "company_profile": self.company_profile,
            "case_studies": self.case_studies,
            "generated_sections": {},
            "charts": {},
            "metadata": {
                "generation_started": datetime.now().isoformat(),
                "version": "2.0.0-SDK",
                "total_tokens": 0,
                "api_cost": 0.0
            }
        }
        
        # Get proposal outline from section routing
        outline = list(self.section_routing.keys())
        # If RFP requires a financial proposal, ensure the section exists
        try:
            rfp_req = request.requirements if hasattr(request, 'requirements') else {}
            need_financial = bool(rfp_req.get('financial_proposal_required'))
            if need_financial and 'Financial Proposal' not in outline:
                outline.append('Financial Proposal')
        except Exception:
            pass
        logger.info(f"Generating {len(outline)} sections using SDK agents")
        
        # Generate sections using coordinator agent
        coordinator = get_agent("coordinator")
        
        try:
            # Initialize session for workflow continuity
            session_context = {
                "request": context['request'].__dict__,
                "skills_data": context['skills_data'],
                "company_profile": context['company_profile'],
                "generated_sections": {}
            }
            
            # Batch sections for parallel execution (3-4 sections at a time)
            batch_size = 3
            for i in range(0, len(outline), batch_size):
                batch = outline[i:i+batch_size]
                logger.info(f"Generating batch of sections: {batch}")
                
                # Create tasks for parallel execution
                tasks = []
                section_names = []
                for section_name in batch:
                    logger.info(f"Creating task for section: {section_name}")
                    # Use handoff method for better agent coordination
                    task = self._generate_section_with_handoff(
                        section_name, 
                        context, 
                        coordinator
                    )
                    tasks.append(task)
                    section_names.append(section_name)
                
                # Execute batch in parallel using asyncio.gather
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for section_name, result in zip(section_names, results):
                        if isinstance(result, Exception):
                            logger.error(f"Failed to generate section {section_name}: {str(result)}")
                            context["generated_sections"][section_name] = {
                                "content": f"[Section {section_name} - Generation failed]",
                                "metadata": {"error": str(result), "fallback": True}
                            }
                        else:
                            context["generated_sections"][section_name] = result
                            session_context["generated_sections"][section_name] = result
                            
                            # Update cost tracking with SDK response
                            if result.get("metadata", {}).get("response"):
                                response = result["metadata"]["response"]
                                self.cost_tracker.track_completion(response, model="gpt-4o")
                except Exception as e:
                    logger.error(f"Batch execution failed: {str(e)}")
                    # Add fallback for all sections in batch
                    for section_name in section_names:
                        if section_name not in context["generated_sections"]:
                            context["generated_sections"][section_name] = {
                                "content": f"[Section {section_name} - Generation failed]",
                                "metadata": {"error": str(e), "fallback": True}
                            }
        
        except Exception as e:
            logger.error(f"Error in section generation: {str(e)}", exc_info=True)
            raise
        
        # Generate charts only if explicitly enabled in config
        try:
            include_charts = self.config.get('output', {}).get('include_charts', False)
        except Exception:
            include_charts = False
        if include_charts:
            await self._generate_charts(context)
        
        # Run quality evaluation
        await self._evaluate_quality(context)
        
        # Generate HTML and PDF outputs
        html_content = self.html_generator.generate(context)
        context["html"] = html_content
        
      
        #
        # Update final metadata
        cost_summary = self.cost_tracker.get_summary()
        context["metadata"].update({
            "generation_completed": datetime.now().isoformat(),
            "total_tokens": cost_summary["total_tokens"],
            "api_cost": cost_summary["total_cost"],
            "api_calls": cost_summary["api_calls"]
        })
        
        logger.info(f"SDK proposal generation completed. Cost: ${cost_summary['total_cost']:.4f}")
        
        return context

    def create_request_from_rfp(self, pdf_path: str) -> ProposalRequest:
        """Create ProposalRequest with auto-extracted info from RFP"""
        rfp_config = self.config.get('rfp', {})

        if rfp_config.get('auto_extract_info', True):
            # Extract information
            extracted_info = self.rfp_extractor.extract_rfp_info(pdf_path)

            # Create request with extracted info
            return ProposalRequest(
                client_name=extracted_info.get('client_name') or 'Unknown Client',
                project_name=extracted_info.get('project_name') or 'RFP Project',
                project_type=extracted_info.get('project_type', 'general'),
                requirements={
                    'source_pdf': pdf_path,
                    'rfp_extracted': extracted_info,
                    'financial_proposal_required': bool(extracted_info.get('financial_proposal_required', False)),
                    'evaluation_criteria': extracted_info.get('evaluation_criteria', []),
                    'language': extracted_info.get('language', 'en'),
                    'sla_requirements': extracted_info.get('sla_requirements', ''),
                    'data_residency': extracted_info.get('data_residency', ''),
                    'team_requirements': extracted_info.get('team_requirements', []),
                    'required_documents': extracted_info.get('required_documents', [])
                },
                timeline=extracted_info.get('timeline', 'As per RFP requirements'),
                budget_range=extracted_info.get('budget_range', 'As per RFP specifications'),
                special_requirements=rfp_config.get('special_requirements', [])
            )
        else:
            # Use manual config
            return ProposalRequest(
                client_name=rfp_config.get('client_name', 'Unknown Client'),
                project_name=rfp_config.get('project_name', 'RFP Project'),
                project_type=rfp_config.get('project_type', 'general'),
                requirements={'source_pdf': pdf_path},
                timeline=rfp_config.get('timeline', 'As per RFP requirements'),
                budget_range=rfp_config.get('budget_range', 'As per RFP specifications'),
                special_requirements=rfp_config.get('special_requirements', [])
            )
    
    async def _generate_section_with_handoff(self, section_name: str, context: Dict, coordinator) -> Dict[str, Any]:
        """Generate a single section using direct specialist agent based on strategy"""
        
        section_config = self.section_routing.get(section_name, {})
        strategy = section_config.get('strategy', 'default')
        
        # Get the appropriate specialist agent directly
        from proposal_agents.sdk_agents import get_specialist_for_section
        specialist = get_specialist_for_section(section_name, self.section_routing)
        
        # Build prompt for the specialist with grounding
        prompt_parts = [
            f"Generate the '{section_name}' section for this proposal.",
            f"Client: {context['request'].client_name}",
            f"Project: {context['request'].project_name}",
            f"Type: {context['request'].project_type}",
            f"Timeline: {context['request'].timeline}",
            "Language: Write this section in English",
            f"Use the {strategy} strategy. Make it professional, detailed, and tailored to the client's needs."
        ]

        # Add requirements but limit size
        if hasattr(context['request'], 'requirements'):
            try:
                req_str = json.dumps(context['request'].requirements, indent=2)[:1000]
                prompt_parts.append("\nRequirements:\n" + req_str)
            except Exception:
                pass

        # Add compact company and skills context
        company_ctx = context.get('company_profile', {})
        skills_summary = context.get('skills_summary', '')
        if company_ctx:
            prompt_parts.append("\nCompany Context (use only this, do not invent):\n" + str(company_ctx))
        # If generating Our Team/Company Profile, read markdown directly from config path
        if section_name.strip().lower() in {"our team/company profile", "our team", "company profile"}:
            profile_path = self.config.get('data', {}).get('company_profile', '')
            profile_md = self._read_text_file(profile_path)
            if profile_md:
                prompt_parts.append("\nUse ONLY the following company profile markdown as source. Convert into a polished narrative suitable for this section:\n" + profile_md)
        if skills_summary:
            prompt_parts.append("\nSkills Summary:\n" + skills_summary)

        # Grounding rules
        prompt_parts.append("\nGrounding Rules:\n- Use only the provided Company Context and Skills Summary.\n- If information is missing, write 'Not specified'.\n- Keep names and metrics consistent with the provided sources.")

        prompt = "\n".join(prompt_parts)
        
        # Check and truncate prompt if needed
        token_count = count_tokens(prompt, 'gpt-4o')
        if token_count > CONTEXT_LIMITS['gpt-4o'] - RESPONSE_RESERVE_TOKENS - 1000:  # Reserve space for context
            logger.warning(f"Section prompt too long ({token_count} tokens), truncating...")
            prompt = truncate_text(prompt, CONTEXT_LIMITS['gpt-4o'] - RESPONSE_RESERVE_TOKENS - 1000, 'gpt-4o')
        
        try:
            # Log agent execution start
            agent_name = specialist.name if hasattr(specialist, 'name') else 'specialist'
            agent_logger.log_agent_execution(
                agent_name,
                f"generate_{section_name}",
                {"section_name": section_name, "strategy": strategy},  # Minimal logging context
                {"status": "starting"}
            )
            
            # Use minimal context to avoid token limits
            minimal_section_context = {
                "section_name": section_name,
                "strategy": strategy,
                "client_name": getattr(context['request'], 'client_name', 'Unknown'),
                "project_name": getattr(context['request'], 'project_name', 'Unknown'),
                "project_type": getattr(context['request'], 'project_type', 'general'),
                "timeline": getattr(context['request'], 'timeline', '3 months')
            }
            
            # Use specialist agent directly for better performance
            response = await Runner.run(
                specialist,
                prompt,
                context=minimal_section_context,
                max_turns=6  # Allow more turns to avoid premature termination
            )
            
            content = self._extract_content_from_sdk_response(response)
            
            # Fallback: ensure timelines section has an HTML table if content is thin
            if section_name.strip().lower() in {"project plan and timelines", "project plan", "timelines"}:
                needs_table = (not content) or ("<table" not in content.lower())
                if needs_table:
                    table_html = self._build_timeline_table_html(context.get('request'))
                    if table_html:
                        content = (content or "") + "\n\n" + table_html

            # Log successful execution
            agent_logger.log_agent_execution(
                agent_name,
                f"generate_{section_name}",
                context,
                {"status": "success", "content": content[:500] if content else ""}
            )
            
            return {
                "content": content,
                "metadata": {
                    "section_name": section_name,
                    "strategy": strategy,
                    "agent_used": specialist.name if hasattr(specialist, 'name') else 'specialist',
                    "generated_at": datetime.now().isoformat(),
                    "response": response,
                    "word_count": len(content.split()) if content else 0
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate section {section_name} with specialist: {str(e)}")
            # Log error
            agent_logger.log_agent_execution(
                agent_name if 'agent_name' in locals() else 'specialist',
                f"generate_{section_name}",
                context,
                {"status": "error", "error": str(e)}
            )
            # Fallback to coordinator-based generation
            return await self._generate_section(section_name, context, coordinator)
    
    async def _generate_section(self, section_name: str, context: Dict, coordinator) -> Dict[str, Any]:
        """Generate a single section using appropriate specialist agent"""
        
        section_config = self.section_routing.get(section_name, {})
        strategy = section_config.get('strategy', 'default')
        requires_context = section_config.get('requires_context', True)
        
        # Get RAG context (use cached vector DB); default to configured RFP PDF if none provided
        rag_context = ""
        if hasattr(self.rag_retriever, 'retrieve'):
            try:
                # Get PDF path from context
                pdf_path = context['request'].requirements.get('source_pdf') or self.config.get('rfp', {}).get('pdf_path', '')
                
                # Use cached vector DB if available
                if pdf_path and pdf_path not in self._vector_db_cache:
                    # Load and cache the vector DB for this PDF
                    self._vector_db_cache[pdf_path] = True
                    logger.info(f"Caching vector DB for {pdf_path}")
                
                # Query for relevant context (specialize per section)
                if section_name.strip().lower() in {"risk analysis and mitigation", "risk analysis", "risks"}:
                    query = (
                        f"Risks, constraints, assumptions, SLAs, compliance, penalties, security requirements "
                        f"for {context['request'].project_name}"
                    )
                    chunks = self.rag_retriever.retrieve(query, pdf_path)
                    if chunks:
                        rag_context = "\n".join([chunk['text'][:800] for chunk in chunks[:5]])
                else:
                    query = f"Information about {section_name} for {context['request'].project_name}"
                    chunks = self.rag_retriever.retrieve(query, pdf_path)
                    if chunks:
                        rag_context = "\n".join([chunk['text'][:500] for chunk in chunks[:3]])
            except Exception as e:
                logger.warning(f"Could not retrieve RAG context for {section_name}: {e}")
        
        # Get appropriate specialist agent
        specialist = get_specialist_for_section(section_name, self.section_routing)
        
        # Prepare section generation prompt
        section_prompt = self._build_section_prompt(
            section_name, 
            context['request'], 
            section_config, 
            rag_context,
            context
        )
        
        try:
            # Create minimal session context to avoid token limits
            request_dict = context['request'].__dict__ if hasattr(context['request'], '__dict__') else {}
            
            # Truncate large fields
            truncated_request = {}
            for key, value in request_dict.items():
                if isinstance(value, str) and len(value) > 500:
                    truncated_request[key] = value[:500] + "...[truncated]"
                elif isinstance(value, dict):
                    # Limit dict size
                    truncated_request[key] = {k: v for k, v in list(value.items())[:10]}
                else:
                    truncated_request[key] = value
            
            # Truncate RAG context
            truncated_rag_context = truncate_text(rag_context, 2000, 'gpt-4o') if rag_context else ""
            
            session_context = {
                "section_name": section_name,
                "strategy": strategy,
                "request_summary": truncated_request,
                "rag_context": truncated_rag_context,
                "company_name": context.get('company_profile', {}).get('name', 'AzmX')[:100],
                "skills_available": str(list(context.get('skills_data', {}).keys())[:5])  # First 5 skills only
            }
            
            # Log agent execution start
            agent_logger.log_agent_execution(
                getattr(coordinator, 'name', 'coordinator'),
                f"generate_{section_name}_fallback",
                context,
                {"status": "starting", "reason": "specialist_fallback"}
            )
            
            # Check total context size before API call
            full_prompt = f"Generate the '{section_name}' section for the proposal:\n\n{section_prompt}"
            context_str = str(session_context)
            total_input = full_prompt + context_str
            
            token_count = count_tokens(total_input, 'gpt-4o')
            if token_count > CONTEXT_LIMITS['gpt-4o'] - RESPONSE_RESERVE_TOKENS:
                logger.warning(f"Total context too large ({token_count} tokens), using minimal context...")
                # Use absolute minimal context
                session_context = {
                    "section_name": section_name,
                    "strategy": strategy,
                    "client": getattr(context['request'], 'client_name', 'Unknown'),
                    "project": getattr(context['request'], 'project_name', 'Unknown')[:100]
                }
                section_prompt = truncate_text(section_prompt, CONTEXT_LIMITS['gpt-4o'] - RESPONSE_RESERVE_TOKENS - 500, 'gpt-4o')
            
            # Use SDK Runner to execute coordinator with minimal session management
            response = await Runner.run(
                coordinator,
                f"Generate the '{section_name}' section for the proposal. Write ONLY in English.\n\n{section_prompt}",
                context=session_context
            )
            
            # Extract and sanitize content from SDK response
            content = self._extract_content_from_sdk_response(response)
            content = self._sanitize_generated_content(content)
            
            # Log successful execution
            agent_logger.log_agent_execution(
                getattr(coordinator, 'name', 'coordinator'),
                f"generate_{section_name}_fallback",
                context,
                {"status": "success", "content_length": len(content) if content else 0}
            )
            
            return {
                "content": content,
                "metadata": {
                    "section_name": section_name,
                    "strategy": strategy,
                    "agent_used": coordinator.name,
                    "generated_at": datetime.now().isoformat(),
                    "response": response,
                    "word_count": len(content.split()) if content else 0,
                    "session_context": session_context
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate section {section_name}: {str(e)}")
            # Return fallback content
            return {
                "content": f"[Section {section_name} - Generation in progress]",
                "metadata": {
                    "section_name": section_name,
                    "fallback": True,
                    "error": str(e),
                    "generated_at": datetime.now().isoformat()
                }
            }
    
    def _build_section_prompt(self, section_name: str, request: ProposalRequest, 
                             section_config: Dict, rag_context: str, context: Dict) -> str:
        """Build a comprehensive prompt for section generation"""
        
        prompt_parts = [
            f"Generate the '{section_name}' section for a proposal.",
            f"Client: {request.client_name}",
            f"Project: {request.project_name}",
            f"Project Type: {request.project_type}",
            f"Timeline: {request.timeline}",
        ]
        
        if request.budget_range:
            prompt_parts.append(f"Budget: {request.budget_range}")
        
        # Add requirements
        if request.requirements:
            prompt_parts.append("\nProject Requirements:")
            for key, value in request.requirements.items():
                if value and key != 'rfp_context':
                    prompt_parts.append(f"- {key}: {value}")
        
        # Add RAG context
        # Add evaluation criteria guidance from RFP if present
        rfp_req = getattr(request, 'requirements', {}) or {}
        eval_criteria = rfp_req.get('evaluation_criteria', [])
        if isinstance(eval_criteria, list) and eval_criteria:
            try:
                criteria_str = "; ".join([f"{c.get('name', '')} ({c.get('weight_percent', '')}%)" for c in eval_criteria if isinstance(c, dict)])
                if criteria_str.strip():
                    prompt_parts.append(f"\nOptimize to score highly against these weighted criteria: {criteria_str}")
            except Exception:
                pass

        # Add SLAs, data residency and compliance cues if provided
        if rfp_req.get('sla_requirements'):
            prompt_parts.append(f"\nRFP SLA Requirements to address: {rfp_req.get('sla_requirements')}")
        if rfp_req.get('data_residency'):
            prompt_parts.append(f"\nData Residency: {rfp_req.get('data_residency')}")

        # Financial section specific cues
        if section_name.lower() == 'financial proposal':
            prompt_parts.append("\nCreate an itemized financial proposal consistent with the scope, including assumptions, taxes, and payment milestones. Present it as a clear table.")

        if rag_context:
            prompt_parts.append(f"\nRelevant Document Context:\n{rag_context}")
        
        # Add section-specific requirements
        min_words = section_config.get('min_words', 300)
        max_words = section_config.get('max_words', 800)
        prompt_parts.append(f"\nSection Requirements:")
        prompt_parts.append(f"- Word count: {min_words}-{max_words} words")
        prompt_parts.append(f"- Strategy: {section_config.get('strategy', 'default')}")
        
        # Language instruction: force English output regardless of RFP language
        prompt_parts.append("- Language: Write this section in English")
        
        # Always include compact company and skills context to anchor outputs
        company_ctx = context.get('company_profile', {})
        skills_summary = context.get('skills_summary', '')
        if company_ctx:
            prompt_parts.append(f"\nCompany Context (do not invent facts; use this):\n{company_ctx}")
        if skills_summary:
            prompt_parts.append(f"\nSkills Summary (internal + external):\n{skills_summary}")
        
        # Inject case studies when generating Success Stories/Case Studies
        if section_name.strip().lower() in {"success stories/case studies", "success stories", "case studies"}:
            cs_path = self.config.get('data', {}).get('case_studies', '')
            cs_md = self._read_text_file(cs_path)
            if cs_md:
                extracted = self._extract_partnerships_and_testimonials(cs_md)
                prompt_parts.append("\nSource Content (Markdown):\n" + cs_md)
                prompt_parts.append("\nStructured Data Extracted (use this to build concrete, data-driven cases; do not invent):\n" + json.dumps(extracted, indent=2))
                prompt_parts.append("\nInstructions: Create 2-4 concise case studies using ONLY the partnerships table (client, year, partnership) and the testimonials as supporting quotes where relevant. No conceptual placeholders. If details are missing, write 'Not specified'.")

        # Ensure timelines section includes a proper HTML table (even without charts)
        if section_name.strip().lower() in {"project plan and timelines", "project plan", "timelines"}:
            rfp_timeline = getattr(request, 'timeline', '') or self.config.get('rfp', {}).get('timeline', '')
            if rfp_timeline:
                prompt_parts.append(f"\nRFP Timeline Guidance: {rfp_timeline}")
            prompt_parts.append(
                "\nProduce a clean HTML <table> titled 'Project Timeline' with columns: Phase, Key Activities/Deliverables, Start Date, End Date, Duration (weeks), Owner/Team. Include 5-8 phases covering Initiation, Discovery, Design, Development, Testing/UAT, Deployment, Handover. Keep dates consistent; if exact dates are not specified, use relative week numbers."
            )

        # Enrich Proposed Solution with concrete structure
        if section_name.strip().lower() in {"proposed solution", "solution"}:
            prompt_parts.append(
                "\nAdd a concise architecture overview (components, integrations, data flows). Provide bullet lists for: Functional scope, Non-functional requirements (performance, security, scalability), Integrations (systems/APIs), Data protection & compliance. Close with a short paragraph on benefits measurable to the client."
            )

        # Enrich Technical Approach and Methodology with actionable plan
        if section_name.strip().lower() in {"technical approach and methodology", "technical approach", "methodology"}:
            prompt_parts.append(
                "\nDescribe delivery methodology (Agile/Scrum or hybrid), ceremonies, and artifacts. Detail environments (dev/test/stage/prod), CI/CD, code quality gates, testing strategy (unit/integration/security/performance), and security practices (OWASP, access control). Provide a step-by-step execution plan."
            )

        # Enrich Risk Analysis with a structured, data-driven risk register
        if section_name.strip().lower() in {"risk analysis and mitigation", "risk analysis", "risks"}:
            prompt_parts.append(
                "\nInclude an HTML <table> titled 'Risk Register' with columns: Risk, Category, Probability (Low/Med/High), Impact (Low/Med/High), Mitigation Strategy, Owner, Source. Provide 6-8 risks grounded ONLY in the RFP context and company capabilities. In the Source column, cite 'RFP p.X clause Y' or 'Company capability' as appropriate. If source unclear, write 'RFP (unspecified)'."
            )

        # Enrich Implementation Strategy with phases and acceptance criteria
        if section_name.strip().lower() in {"implementation strategy", "implementation"}:
            prompt_parts.append(
                "\nProvide a phased implementation plan with phase goals, entry/exit criteria, and acceptance criteria. Add a brief RACI-style responsibility summary (Client, Vendor) in bullets. Ensure dependencies and readiness checks are mentioned."
            )
        
        # Guardrail: enforce source grounding and USD currency
        prompt_parts.append(
            "\nGrounding Rules:\n"
            "- Use only the provided Company Context, Skills Summary, RAG context, and section-specific markdown.\n"
            "- If information is missing, state 'Not specified' instead of making assumptions.\n"
            "- Keep names and metrics consistent with the provided sources.\n"
            "- Currency: Express ALL costs and monetary values strictly in USD (use '$' and K/M abbreviations if needed)."
        )

        # Add skills data for technical sections
        if section_config.get('use_skills', False):
            prompt_parts.append(f"\nAvailable Skills/Resources: {context.get('skills_data', {})}")
        
        return "\n".join(prompt_parts)
    
    async def _generate_charts(self, context: Dict):
        """Generate charts using hybrid approach - ChartDecisionAgent + direct ChartGenerator"""
        logger.info("Starting hybrid chart generation")
        
        try:
            # Check if SDK agent should be used
            use_sdk_agent = self.config.get('charts', {}).get('use_sdk_agent', False)
            
            if use_sdk_agent:
                logger.info("Using legacy SDK agent approach for chart generation")
                await self._generate_charts_legacy_sdk(context)
            else:
                logger.info("Using hybrid approach: ChartDecisionAgent + direct ChartGenerator")
                await self._generate_charts_hybrid(context)
                
        except Exception as e:
            logger.error(f"Chart generation failed: {str(e)}")
            # Ensure charts dict exists even on failure
            if "charts" not in context:
                context["charts"] = {}
    
    async def _generate_charts_hybrid(self, context: Dict):
        """Generate charts using hybrid approach with ChartDecisionAgent and direct ChartGenerator"""
        generated_sections = context.get('generated_sections', {})
        
        # Step 1: Create section summaries for decision agent (minimal context)
        section_summaries = {}
        for section_name, section_data in generated_sections.items():
            content = section_data.get('content', '')
            # Extract only first 100 words as summary to minimize context
            words = content.split()[:100]
            summary = ' '.join(words) + '...' if len(words) == 100 else ' '.join(words)
            section_summaries[section_name] = summary
        
        # Step 2: Use ChartDecisionAgent to decide which charts to generate
        request = context.get('request')
        project_type = getattr(request, 'project_type', 'general') if request else 'general'
        
        try:
            chart_decisions = self.chart_decision_agent.decide_charts(
                section_summaries, 
                project_type
            )
            logger.info(f"Chart decision agent selected {len(chart_decisions)} charts")
        except Exception as e:
            logger.warning(f"Chart decision agent failed, using minimum required charts: {str(e)}")
            # Fallback to minimum required charts from config
            chart_decisions = []
            for min_chart in self.config.get('charts', {}).get('minimum_charts', []):
                chart_decisions.append({
                    'section': min_chart['section'],
                    'type': min_chart['type'], 
                    'title': min_chart['title'],
                    'required': True
                })
        
        # Step 3: Generate each chart directly using ChartGenerator
        for chart_spec in chart_decisions:
            section_name = chart_spec['section']
            chart_type = chart_spec['type']
            chart_title = chart_spec['title']
            
            logger.info(f"Generating {chart_type} chart for {section_name}")
            
            try:
                # Log chart generation start
                agent_logger.log_agent_execution(
                    'hybrid_chart_generator',
                    f"generate_chart_{section_name}",
                    {"section_name": section_name, "chart_type": chart_type},
                    {"status": "starting", "approach": "hybrid"}
                )
                
                # Generate chart directly
                chart_html = await self._generate_chart_direct(
                    chart_spec, 
                    context
                )
                
                if chart_html:
                    context["charts"][section_name] = {
                        "type": chart_type,
                        "data": chart_html,
                        "title": chart_title,
                        "generated_at": datetime.now().isoformat(),
                        "method": "hybrid_direct",
                        "html_length": len(chart_html)
                    }
                    
                    # Log successful chart generation
                    agent_logger.log_agent_execution(
                        'hybrid_chart_generator',
                        f"generate_chart_{section_name}",
                        {"section_name": section_name, "chart_type": chart_type},
                        {"status": "success", "method": "hybrid_direct"}
                    )
                else:
                    logger.warning(f"No chart HTML generated for {section_name}")
                    
            except Exception as e:
                logger.error(f"Failed to generate chart for {section_name}: {str(e)}")
                # Log chart generation failure
                agent_logger.log_agent_execution(
                    'hybrid_chart_generator',
                    f"generate_chart_{section_name}",
                    {"section_name": section_name, "chart_type": chart_type},
                    {"status": "error", "error": str(e), "method": "hybrid_direct"}
                )
                
                # Add fallback placeholder chart
                context["charts"][section_name] = {
                    "type": chart_type,
                    "data": f"<div>Chart generation failed for {section_name} ({chart_type})</div>",
                    "title": chart_title,
                    "generated_at": datetime.now().isoformat(),
                    "error": str(e),
                    "method": "hybrid_fallback"
                }
    
    def _prepare_chart_data(self, section_name: str, chart_type: str, context: Dict) -> str:
        """
        Prepare chart data based on section content and chart type
        
        Args:
            section_name: Name of the section requiring chart
            chart_type: Type of chart (gantt, budget_breakdown, risk_matrix)
            context: Full context with generated sections
            
        Returns:
            JSON string formatted for the specific chart type
        """
        generated_sections = context.get('generated_sections', {})
        request = context.get('request')
        
        try:
            if chart_type == 'gantt':
                # Extract timeline data from Project Plan or Technical Approach sections
                timeline_content = ""
                for section in ['Project Plan', 'Technical Approach', 'Project Scope']:
                    if section in generated_sections:
                        timeline_content += generated_sections[section].get('content', '')
                
                # Parse timeline from request
                timeline = getattr(request, 'timeline', '3 months') if request else '3 months'
                
                # Extract duration in months
                duration_months = 3
                try:
                    import re
                    match = re.search(r'(\d+)', timeline)
                    if match:
                        duration_months = int(match.group(1))
                except:
                    pass
                
                # Look for phases/milestones in content
                phases = []
                if 'phase' in timeline_content.lower() or 'milestone' in timeline_content.lower():
                    # Try to extract phases from content
                    phase_patterns = [
                        r'phase\s+(\d+)[:\-\s]*([^.!?]*)',
                        r'milestone[:\-\s]*([^.!?]*)',
                        r'(\d+[\.\)]\s*[^.!?]*(?:phase|milestone|stage)[^.!?]*)',
                    ]
                    
                    for pattern in phase_patterns:
                        matches = re.findall(pattern, timeline_content, re.IGNORECASE)
                        for match in matches[:5]:  # Limit to 5 phases
                            if isinstance(match, tuple):
                                phases.append({
                                    'name': match[1].strip() if len(match) > 1 else match[0].strip(),
                                    'duration': max(1, duration_months // 4)
                                })
                            else:
                                phases.append({
                                    'name': match.strip(),
                                    'duration': max(1, duration_months // 4)
                                })
                
                # Default phases if none extracted
                if not phases:
                    phase_duration = max(1, duration_months // 4)
                    phases = [
                        {'name': 'Requirements & Planning', 'duration': phase_duration},
                        {'name': 'Development & Implementation', 'duration': phase_duration * 2},
                        {'name': 'Testing & Quality Assurance', 'duration': phase_duration},
                        {'name': 'Deployment & Delivery', 'duration': phase_duration}
                    ]
                
                # Calculate start times
                current_start = 0
                for phase in phases:
                    phase['start'] = current_start
                    current_start += phase['duration']
                
                chart_data = {
                    'type': 'gantt',
                    'title': f'Project Timeline - {request.project_name if request else "Project"}',
                    'timeline': timeline,
                    'duration_months': duration_months,
                    'phases': phases
                }
                
            elif chart_type == 'budget_breakdown':
                # Extract budget data from Budget section
                budget_content = generated_sections.get('Budget', {}).get('content', '')
                budget_range = getattr(request, 'budget_range', None) if request else None
                
                # Default budget breakdown
                budget_items = []
                
                # Try to extract budget items from content
                if budget_content:
                    # Look for cost patterns in content
                    cost_patterns = [
                        r'(\$[\d,]+(?:\.\d{2})?)[^\w]*([^.!?\n]{1,50})',
                        r'([^.!?\n]{1,50})[^\w]*(\$[\d,]+(?:\.\d{2})?)',
                        r'(\w+(?:\s+\w+)*)[:\-\s]*(\$[\d,]+(?:\.\d{2})?)',
                        r'(\$[\d,]+(?:\.\d{2})?)[:\-\s]*(\w+(?:\s+\w+)*)'
                    ]
                    
                    for pattern in cost_patterns:
                        matches = re.findall(pattern, budget_content)
                        for match in matches[:6]:  # Limit to 6 items
                            if '$' in match[0]:
                                amount_str = match[0].replace('$', '').replace(',', '')
                                try:
                                    amount = float(amount_str)
                                    budget_items.append({
                                        'category': match[1].strip(),
                                        'amount': amount
                                    })
                                except ValueError:
                                    continue
                            elif '$' in match[1]:
                                amount_str = match[1].replace('$', '').replace(',', '')
                                try:
                                    amount = float(amount_str)
                                    budget_items.append({
                                        'category': match[0].strip(),
                                        'amount': amount
                                    })
                                except ValueError:
                                    continue
                
                # Default budget breakdown if none extracted
                if not budget_items:
                    total_budget = 100000  # Default
                    if budget_range:
                        # Try to extract number from budget range
                        try:
                            budget_match = re.search(r'(\d+(?:,\d+)*)', budget_range)
                            if budget_match:
                                total_budget = float(budget_match.group(1).replace(',', ''))
                        except:
                            pass
                    
                    budget_items = [
                        {'category': 'Development & Implementation', 'amount': total_budget * 0.40},
                        {'category': 'Project Management', 'amount': total_budget * 0.15},
                        {'category': 'Testing & QA', 'amount': total_budget * 0.20},
                        {'category': 'Infrastructure & Tools', 'amount': total_budget * 0.10},
                        {'category': 'Documentation & Training', 'amount': total_budget * 0.10},
                        {'category': 'Contingency', 'amount': total_budget * 0.05}
                    ]
                
                chart_data = {
                    'type': 'budget_breakdown',
                    'title': 'Project Budget Breakdown',
                    'budget_items': budget_items,
                    'total_budget': sum(item['amount'] for item in budget_items)
                }
                
            elif chart_type == 'risk_matrix':
                # Extract risk data from Risk Analysis section
                risk_content = generated_sections.get('Risk Analysis', {}).get('content', '')
                
                risks = []
                
                # Try to extract risks from content
                if risk_content:
                    # Look for risk patterns
                    risk_patterns = [
                        r'((?:risk|threat|issue)[^.!?\n]{1,100})',
                        r'(\d+[\.\)]\s*[^.!?\n]{1,100}(?:risk|threat|issue)[^.!?\n]*)',
                        r'([A-Z][^.!?\n]{1,100}(?:Risk|Threat|Issue))'
                    ]
                    
                    risk_categories = ['technical', 'schedule', 'resource', 'financial', 'operational']
                    
                    for pattern in risk_patterns:

                        matches = re.findall(pattern, risk_content, re.IGNORECASE)
                        for i, match in enumerate(matches[:8]):  # Limit to 8 risks
                            # Assign probability and impact based on keywords
                            risk_text = match.lower() if isinstance(match, str) else str(match).lower()
                            
                            # Determine probability (0.1 to 0.9)
                            if any(word in risk_text for word in ['high', 'likely', 'probable', 'common']):
                                probability = 0.7 + (i % 3) * 0.1
                            elif any(word in risk_text for word in ['medium', 'possible', 'moderate']):
                                probability = 0.4 + (i % 3) * 0.1
                            else:
                                probability = 0.2 + (i % 3) * 0.1
                            
                            # Determine impact (0.1 to 0.9)
                            if any(word in risk_text for word in ['critical', 'major', 'severe', 'significant']):
                                impact = 0.7 + (i % 3) * 0.1
                            elif any(word in risk_text for word in ['medium', 'moderate', 'important']):
                                impact = 0.4 + (i % 3) * 0.1
                            else:
                                impact = 0.2 + (i % 3) * 0.1
                            
                            # Determine category
                            category = 'technical'  # default
                            if any(word in risk_text for word in ['schedule', 'timeline', 'delay', 'time']):
                                category = 'schedule'
                            elif any(word in risk_text for word in ['resource', 'staff', 'team', 'personnel']):
                                category = 'resource'
                            elif any(word in risk_text for word in ['budget', 'cost', 'financial', 'money']):
                                category = 'financial'
                            elif any(word in risk_text for word in ['operational', 'process', 'workflow']):
                                category = 'operational'
                            
                            risks.append({
                                'name': match.strip()[:50] + ('...' if len(match.strip()) > 50 else ''),
                                'probability': min(0.9, max(0.1, probability)),
                                'impact': min(0.9, max(0.1, impact)),
                                'category': category
                            })
                
                # Default risks if none extracted
                if not risks:
                    risks = [
                        {'name': 'Technical Integration Risk', 'probability': 0.3, 'impact': 0.7, 'category': 'technical'},
                        {'name': 'Timeline Delays', 'probability': 0.4, 'impact': 0.6, 'category': 'schedule'},
                        {'name': 'Resource Availability', 'probability': 0.2, 'impact': 0.8, 'category': 'resource'},
                        {'name': 'Requirements Changes', 'probability': 0.5, 'impact': 0.5, 'category': 'technical'},
                        {'name': 'Budget Overrun', 'probability': 0.2, 'impact': 0.9, 'category': 'financial'}
                    ]
                
                chart_data = {
                    'type': 'risk_matrix',
                    'title': 'Risk Assessment Matrix',
                    'risks': risks
                }
            
            else:
                # Default chart data for unknown types
                chart_data = {
                    'type': chart_type,
                    'title': f'{section_name} Chart',
                    'data': 'No specific data structure defined for this chart type'
                }
            
            # Return JSON string
            return json.dumps(chart_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error preparing chart data for {section_name} ({chart_type}): {str(e)}")
            # Return minimal fallback data
            return json.dumps({
                'type': chart_type,
                'title': f'{section_name} Chart',
                'error': f'Data preparation failed: {str(e)}'
            }, indent=2)
    
    def _prepare_minimal_chart_data(self, section_name: str, chart_type: str, context: Dict) -> Dict:
        """
        Prepare minimal chart data without full section content to reduce context size
        
        Args:
            section_name: Name of the section requiring chart
            chart_type: Type of chart (gantt, budget_breakdown, risk_matrix)
            context: Full context with generated sections
            
        Returns:
            Dictionary with minimal essential chart data
        """
        request = context.get('request')
        
        try:
            if chart_type == 'gantt':
                # Use only timeline from request, create default phases
                timeline = getattr(request, 'timeline', '3 months') if request else '3 months'
                
                # Extract duration in months
                duration_months = 3
                try:
                    import re
                    match = re.search(r'(\d+)', timeline)
                    if match:
                        duration_months = int(match.group(1))
                except:
                    pass
                
                # Create default phases based on project type
                phase_duration = max(1, duration_months // 4)
                phases = [
                    {'name': 'Requirements & Planning', 'duration': phase_duration, 'start': 0},
                    {'name': 'Development & Implementation', 'duration': phase_duration * 2, 'start': phase_duration},
                    {'name': 'Testing & Quality Assurance', 'duration': phase_duration, 'start': phase_duration * 3},
                    {'name': 'Deployment & Delivery', 'duration': phase_duration, 'start': phase_duration * 4}
                ]
                
                return {
                    'type': 'gantt',
                    'title': f'Project Timeline - {request.project_name if request else "Project"}',
                    'timeline': timeline,
                    'duration_months': duration_months,
                    'phases': phases
                }
                
            elif chart_type == 'budget_breakdown':
                # Use budget range from request if available
                budget_range = getattr(request, 'budget_range', None) if request else None
                
                # Default total budget
                total_budget = 100000
                if budget_range:
                    try:
                        import re
                        budget_match = re.search(r'(\d+(?:,\d+)*)', budget_range)
                        if budget_match:
                            total_budget = float(budget_match.group(1).replace(',', ''))
                    except:
                        pass
                
                # Create standard budget breakdown
                budget_items = [
                    {'category': 'Development & Implementation', 'amount': total_budget * 0.40},
                    {'category': 'Project Management', 'amount': total_budget * 0.15},
                    {'category': 'Testing & QA', 'amount': total_budget * 0.20},
                    {'category': 'Infrastructure & Tools', 'amount': total_budget * 0.10},
                    {'category': 'Documentation & Training', 'amount': total_budget * 0.10},
                    {'category': 'Contingency', 'amount': total_budget * 0.05}
                ]
                
                return {
                    'type': 'budget_breakdown',
                    'title': 'Project Budget Breakdown',
                    'budget_items': budget_items,
                    'total_budget': total_budget
                }
                
            elif chart_type == 'risk_matrix':
                # Create standard project risks without parsing section content
                risks = [
                    {'name': 'Technical Integration Risk', 'probability': 0.3, 'impact': 0.7, 'category': 'technical'},
                    {'name': 'Timeline Delays', 'probability': 0.4, 'impact': 0.6, 'category': 'schedule'},
                    {'name': 'Resource Availability', 'probability': 0.2, 'impact': 0.8, 'category': 'resource'},
                    {'name': 'Requirements Changes', 'probability': 0.5, 'impact': 0.5, 'category': 'technical'},
                    {'name': 'Budget Overrun', 'probability': 0.2, 'impact': 0.9, 'category': 'financial'}
                ]
                
                return {
                    'type': 'risk_matrix',
                    'title': 'Risk Assessment Matrix',
                    'risks': risks
                }
            
            else:
                # Default minimal chart data
                return {
                    'type': chart_type,
                    'title': f'{section_name} Chart',
                    'message': f'Standard {chart_type} visualization'
                }
            
        except Exception as e:
            logger.error(f"Error preparing minimal chart data for {section_name} ({chart_type}): {str(e)}")
            # Return minimal fallback data
            return {
                'type': chart_type,
                'title': f'{section_name} Chart',
                'error': f'Data preparation failed: {str(e)}'
            }
    
    def _extract_chart_html_from_response(self, response) -> str:
        """
        Extract HTML chart content from SDK response, specifically handling chart generation
        
        Args:
            response: SDK response from chart generation agent
            
        Returns:
            String containing the HTML chart content
        """
        try:
            # First, try the standard content extraction
            raw_content = self._extract_content_from_sdk_response(response)
            
            if not raw_content:
                return ""
            
            # Look for HTML content specifically
            html_patterns = [
                # HTML code blocks in markdown
                r'```html\s*\n?(.*?)\n?```',
                r'```\s*\n?(<!DOCTYPE html.*?</html>)\s*\n?```',
                r'```\s*\n?(<html.*?</html>)\s*\n?```',
                r'```\s*\n?(<div.*?</div>)\s*\n?```',
                # HTML without code blocks
                r'(<!DOCTYPE html.*?</html>)',
                r'(<html.*?</html>)',
                # More flexible div matching for charts
                r'(<div[^>]*class[^>]*chart[^>]*>.*?</div>)',
                r'(<div[^>]*>.*?</div>)',
                # Plotly specific patterns
                r'(<div[^>]*id=["\'][^"\']*plotly[^"\']*["\'][^>]*>.*?</div>)',
                r'(<div[^>]*>.*?Plotly\.newPlot.*?</script>.*?</div>)',
                # Base64 image patterns
                r'(<img[^>]*src="data:image/[^"]*"[^>]*>)',
                # Any HTML-like structure
                r'(<[^>]+>.*?</[^>]+>)',
            ]
            
            for pattern in html_patterns:
                matches = re.findall(pattern, raw_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    html_content = matches[0]
                    # Clean up the HTML
                    html_content = html_content.strip()
                    
                    # Basic validation - ensure it looks like HTML
                    if '<' in html_content and '>' in html_content:
                        # Additional validation for chart content
                        if self._is_valid_chart_html(html_content):
                            logger.debug(f"Extracted valid chart HTML content: {len(html_content)} characters")
                            return html_content
                        elif len(html_content) > 50:  # Reasonable length even if not perfectly valid
                            logger.debug(f"Extracted HTML content (may need cleaning): {len(html_content)} characters")
                            return html_content
            
            # If no HTML pattern found, check if the raw content itself is HTML
            if raw_content.strip().startswith('<') and raw_content.strip().endswith('>'):
                return raw_content.strip()
            
            # Check if content contains HTML-like structures
            if any(tag in raw_content.lower() for tag in ['<html', '<div', '<script', 'plotly']):
                # Try to extract the largest HTML-like block
                html_start = raw_content.find('<')
                html_end = raw_content.rfind('>') + 1
                
                if html_start != -1 and html_end > html_start:
                    potential_html = raw_content[html_start:html_end]
                    if len(potential_html) > 50:  # Reasonable HTML content
                        return potential_html
            
            # Fallback - return raw content if it contains some HTML indicators
            if any(indicator in raw_content.lower() for indicator in ['plotly', 'chart', 'svg', 'canvas']):
                return raw_content
            
            # Last resort - wrap content in a div if it might be chart-related
            if any(keyword in raw_content.lower() for keyword in ['error', 'chart', 'data', 'visualization']):
                return f"<div>{raw_content}</div>"
            
            return raw_content
            
        except Exception as e:
            logger.error(f"Error extracting chart HTML from response: {str(e)}")
            return ""
    
    def _is_valid_chart_html(self, html_content: str) -> bool:
        """Check if HTML content appears to be a valid chart"""
        chart_indicators = [
            'plotly', 'chart', 'svg', 'canvas', 'graph',
            'data:image/', 'visualization', 'figure',
            'Plotly.newPlot', 'plotly-graph-div'
        ]
        content_lower = html_content.lower()
        return any(indicator in content_lower for indicator in chart_indicators)
    
    def _set_chart_output_format(self, format_type):
        """Set the output format for chart generation tools in current thread"""
        import threading
        threading.current_thread()._chart_output_format = format_type
    
    async def _generate_chart_direct(self, chart_spec: Dict[str, Any], context: Dict) -> str:
        """Generate a chart directly using ChartGenerator class"""
        chart_type = chart_spec['type']
        section_name = chart_spec['section']
        
        try:
            # Extract relevant data from section content
            chart_data = self._extract_chart_data_from_section(
                section_name, 
                chart_type, 
                context
            )
            
            # Generate chart using direct ChartGenerator methods
            if chart_type == 'gantt':
                return self.chart_generator.generate_gantt_chart(chart_data)
            elif chart_type == 'budget_breakdown':
                return self.chart_generator.create_budget_chart(chart_data)
            elif chart_type == 'risk_matrix':
                return self.chart_generator.build_risk_matrix(chart_data.get('risks', []))
            elif chart_type == 'bar':
                return self.chart_generator.generate_bar_chart(chart_data)
            elif chart_type == 'pie':
                return self.chart_generator.generate_pie_chart(chart_data)
            elif chart_type == 'line':
                return self.chart_generator.generate_line_chart(chart_data)
            else:
                logger.warning(f"Unsupported chart type: {chart_type}")
                return f"<div>Unsupported chart type: {chart_type}</div>"
                
        except Exception as e:
            logger.error(f"Direct chart generation failed for {section_name}: {str(e)}")
            return f"<div>Chart generation error: {str(e)}</div>"
    
    def _extract_chart_data_from_section(self, section_name: str, chart_type: str, context: Dict) -> Dict[str, Any]:
        """Extract minimal chart data from section content using smart extractors"""
        generated_sections = context.get('generated_sections', {})
        section_data = generated_sections.get(section_name, {})
        section_content = section_data.get('content', '')
        request = context.get('request')
        
        try:
            if chart_type == 'gantt':
                return self._extract_gantt_data(section_content, request)
            elif chart_type == 'budget_breakdown': 
                return self._extract_budget_data(section_content, request)
            elif chart_type == 'risk_matrix':
                return self._extract_risk_data(section_content, request)
            elif chart_type == 'bar':
                return self._extract_bar_chart_data(section_content, section_name)
            elif chart_type == 'pie':
                return self._extract_pie_chart_data(section_content, section_name)
            elif chart_type == 'line':
                return self._extract_line_chart_data(section_content, section_name, request)
            else:
                return {'title': f'{section_name} Chart', 'data': {}}
                
        except Exception as e:
            logger.error(f"Data extraction failed for {section_name}: {str(e)}")
            return {'title': f'{section_name} Chart', 'error': str(e)}
    
    def _extract_gantt_data(self, content: str, request) -> Dict[str, Any]:
        """Extract timeline data for Gantt chart"""
        # Get timeline from request
        timeline = getattr(request, 'timeline', '3 months') if request else '3 months'
        project_name = getattr(request, 'project_name', 'Project') if request else 'Project'
        
        # Extract duration in weeks (prefer explicit weeks; fallback to months * 4)
        duration_weeks = 12
        try:
            import re as _re
            def _normalize_digits(s: str) -> str:
                trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
                return s.translate(trans)
            tl = _normalize_digits(timeline or '')
            mw = _re.search(r'(\d+)\s*(weeks?|week)', tl, _re.IGNORECASE)
            if mw:
                duration_weeks = int(mw.group(1))
            else:
                mm = _re.search(r'(\d+)', tl)
                if mm:
                    duration_weeks = int(mm.group(1)) * 4
        except Exception:
            pass
        
        # Look for phases in content using regex
        phases = []
        content_lower = content.lower()
        
        if 'phase' in content_lower or 'milestone' in content_lower or 'stage' in content_lower:
            # Try to extract phases from content
            phase_patterns = [
                r'phase\s+(\d+)[:\-\s]*([^.!?\n]{1,80})',
                r'milestone[:\-\s]*([^.!?\n]{1,80})',
                r'(\d+[\.)\s]*[^.!?\n]*(?:phase|milestone|stage)[^.!?\n]{0,50})',
                r'([A-Z][^.!?\n]{1,80}(?:phase|stage|milestone))',
            ]
            
            phase_duration = max(1, duration_weeks // 4)
            for pattern in phase_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for i, match in enumerate(matches[:5]):  # Limit to 5 phases
                    if isinstance(match, tuple):
                        phase_name = match[1].strip() if len(match) > 1 else match[0].strip()
                    else:
                        phase_name = match.strip()
                        
                    # Clean up phase name
                    phase_name = re.sub(r'^\d+[\.)\s]*', '', phase_name)
                    phase_name = phase_name.strip()[:50]  # Limit length
                    
                    if phase_name and len(phase_name) > 3:
                        phases.append({
                            'name': phase_name,
                            'duration': phase_duration,
                            'start': i * phase_duration
                        })
                
                if phases:
                    break  # Use first successful pattern
        
        # Default phases if none extracted: split total weeks by weights 20%/50%/20%/10%
        if not phases:
            total_weeks = max(4, int(duration_weeks))
            p1 = max(1, round(total_weeks * 0.2))
            p2 = max(1, round(total_weeks * 0.5))
            p3 = max(1, round(total_weeks * 0.2))
            # Ensure sum equals total
            p4 = max(1, total_weeks - (p1 + p2 + p3))
            starts = [0, p1, p1 + p2, p1 + p2 + p3]
            phases = [
                {'name': 'Requirements & Planning', 'duration': p1, 'start': starts[0]},
                {'name': 'Development & Implementation', 'duration': p2, 'start': starts[1]},
                {'name': 'Testing & Quality Assurance', 'duration': p3, 'start': starts[2]},
                {'name': 'Deployment & Delivery', 'duration': p4, 'start': starts[3]}
            ]
        
        # Try to derive a start date from request (requirements.start_date or timeline text)
        start_date_iso = None
        try:
            import re
            from datetime import datetime
            # 1) explicit field
            req = getattr(request, 'requirements', {}) or {}
            candidate = req.get('start_date') or timeline
            # Try multiple formats
            date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%b %Y', '%B %Y']
            parsed = None
            for fmt in date_formats:
                try:
                    parsed = datetime.strptime(candidate.strip(), fmt)
                    break
                except Exception:
                    continue
            if not parsed:
                # Search for Month YYYY pattern
                m = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})', candidate, re.IGNORECASE)
                if m:
                    try:
                        parsed = datetime.strptime(f"{m.group(1)[:3].title()} {m.group(2)}", '%b %Y')
                    except Exception:
                        parsed = None
            if parsed:
                start_date_iso = parsed.date().isoformat()
        except Exception:
            start_date_iso = None

        return {
            'title': f'Project Timeline - {project_name}',
            'timeline': timeline,
            'duration_weeks': duration_weeks,
            'phases': phases,
            'start_date': start_date_iso
        }
    
    def _extract_budget_data(self, content: str, request) -> Dict[str, Any]:
        """Extract budget data for budget breakdown chart"""
        budget_range = getattr(request, 'budget_range', None) if request else None
        
        # Try to extract budget items from content
        budget_items = {}
        
        # Look for cost patterns in content
        cost_patterns = [
            r'(\$[\d,]+(?:\.\d{2})?)[^\w]*([^.!?\n]{1,50})',
            r'([^.!?\n]{1,50})[^\w]*(\$[\d,]+(?:\.\d{2})?)',
            r'(\w+(?:\s+\w+)*)[:\-\s]*(\$[\d,]+(?:\.\d{2})?)',
        ]
        
        for pattern in cost_patterns:
            matches = re.findall(pattern, content)
            for match in matches[:6]:  # Limit to 6 items
                if '$' in match[0]:
                    amount_str = match[0].replace('$', '').replace(',', '')
                    category = match[1].strip()[:30]  # Limit category name
                elif '$' in match[1]:
                    amount_str = match[1].replace('$', '').replace(',', '')
                    category = match[0].strip()[:30]
                else:
                    continue
                    
                try:
                    amount = float(amount_str)
                    if category and amount > 0:
                        budget_items[category] = amount
                except ValueError:
                    continue
        
        # Default budget breakdown if none extracted
        if not budget_items:
            total_budget = 100000  # Default
            if budget_range:
                try:
                    budget_match = re.search(r'(\d+(?:,\d+)*)', budget_range)
                    if budget_match:
                        total_budget = float(budget_match.group(1).replace(',', ''))
                except:
                    pass
            
            budget_items = {
                'Development & Implementation': total_budget * 0.40,
                'Project Management': total_budget * 0.15,
                'Testing & QA': total_budget * 0.20,
                'Infrastructure & Tools': total_budget * 0.10,
                'Documentation & Training': total_budget * 0.10,
                'Contingency': total_budget * 0.05
            }
        
        return {
            'breakdown_by_role': budget_items,
            'total_cost': sum(budget_items.values()),
            'categories': list(budget_items.keys()),
            'values': list(budget_items.values())
        }
    
    def _extract_risk_data(self, content: str, request) -> Dict[str, Any]:
        """Extract risk data for risk matrix chart"""
        risks = []
        
        # Look for risks in content
        content_lower = content.lower()
        
        if 'risk' in content_lower or 'threat' in content_lower:
            # Risk extraction patterns
            risk_patterns = [
                r'((?:risk|threat|issue)[^.!?\n]{1,100})',
                r'(\d+[\.)\s]*[^.!?\n]{1,100}(?:risk|threat|issue)[^.!?\n]*)',
                r'([A-Z][^.!?\n]{1,100}(?:Risk|Threat|Issue))',
                r'([^.!?\n]{1,100}(?:probability|impact|likelihood)[^.!?\n]{1,100})'
            ]
            
            risk_names = set()  # Avoid duplicates
            for pattern in risk_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    risk_text = match.strip()
                    if len(risk_text) > 10 and len(risk_text) < 100:
                        # Clean up risk text
                        risk_text = re.sub(r'^\d+[\.)\s]*', '', risk_text)
                        risk_text = risk_text.strip()
                        
                        if risk_text not in risk_names:
                            risk_names.add(risk_text)
                            
                            # Assign probability and impact based on keywords
                            risk_lower = risk_text.lower()
                            
                            # Determine probability (0.2 to 0.9)
                            if any(word in risk_lower for word in ['high', 'likely', 'probable', 'common']):
                                probability = 0.8
                            elif any(word in risk_lower for word in ['medium', 'possible', 'moderate']):
                                probability = 0.5
                            else:
                                probability = 0.3
                            
                            # Determine impact (0.2 to 0.9)
                            if any(word in risk_lower for word in ['critical', 'major', 'severe', 'significant']):
                                impact = 0.8
                            elif any(word in risk_lower for word in ['medium', 'moderate', 'important']):
                                impact = 0.5
                            else:
                                impact = 0.3
                            
                            risks.append({
                                'name': risk_text[:50],  # Limit name length
                                'probability': probability,
                                'impact': impact
                            })
                            
                            if len(risks) >= 8:  # Limit to 8 risks
                                break
                
                if risks:
                    break  # Use first successful pattern
        
        # Default risks if none extracted
        if not risks:
            risks = [
                {'name': 'Technical Integration Risk', 'probability': 0.3, 'impact': 0.7},
                {'name': 'Timeline Delays', 'probability': 0.4, 'impact': 0.6},
                {'name': 'Resource Availability', 'probability': 0.2, 'impact': 0.8},
                {'name': 'Requirements Changes', 'probability': 0.5, 'impact': 0.5},
                {'name': 'Budget Overrun', 'probability': 0.2, 'impact': 0.9}
            ]
        
        return {'risks': risks}
    
    def _extract_bar_chart_data(self, content: str, section_name: str) -> Dict[str, Any]:
        """Extract data for bar chart from section content"""
        # Try to extract comparative data from content
        categories = []
        values = []
        
        # Look for numbered items or comparisons in content
        content_lower = content.lower()
        
        # Try to extract metrics or comparisons
        if any(keyword in content_lower for keyword in ['phase', 'stage', 'step', 'milestone', 'component']):
            # Extract phase/stage data
            phase_patterns = [
                r'(\d+[\.)\s]*[^.!?\n]{1,50})',
                r'phase\s+(\d+)[:\-\s]*([^.!?\n]{1,50})',
                r'([A-Z][^.!?\n]{1,50}(?:phase|stage|step))'
            ]
            
            for pattern in phase_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for i, match in enumerate(matches[:6]):  # Limit to 6 items
                    if isinstance(match, tuple):
                        category = match[1].strip() if len(match) > 1 else match[0].strip()
                    else:
                        category = match.strip()
                    
                    # Clean category name
                    category = re.sub(r'^\d+[\.)\s]*', '', category)[:30]
                    if category and len(category) > 3:
                        categories.append(category)
                        # Generate reasonable values
                        values.append(20 + (i * 10) + (i % 3) * 5)
                
                if categories:
                    break
        
        # Default data if none extracted - make it section-aware
        if not categories:
            name_lower = section_name.lower()
            if 'support' in name_lower or 'maintenance' in name_lower:
                categories = ['SLA Support', 'Monitoring & Alerts', 'Bug Fixes', 'Security Patching', 'Minor Enhancements']
                values = [30, 15, 20, 15, 20]
                title = 'Support & Maintenance Allocation (%)'
            elif 'solution' in name_lower or 'features' in name_lower:
                categories = ['Core Features', 'Integrations', 'Security', 'Reporting']
                values = [40, 25, 20, 15]
                title = 'Solution Capability Emphasis'
            elif 'implementation' in name_lower or 'work plan' in name_lower:
                categories = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']
                values = [25, 35, 30, 40]
                title = 'Implementation Phases'
            else:
                categories = ['Planning', 'Execution', 'Testing', 'Handover']
                values = [20, 50, 20, 10]
                title = f'{section_name} Breakdown'
        else:
            title = f'{section_name} Breakdown'
        
        return {
            'title': title,
            'categories': categories[:6],  # Limit to 6 items
            'values': values[:6],
            'x_label': 'Categories',
            'y_label': 'Value'
        }
    
    def _extract_pie_chart_data(self, content: str, section_name: str) -> Dict[str, Any]:
        """Extract data for pie chart from section content"""
        # Try to extract distribution data from content
        labels = []
        values = []
        
        content_lower = content.lower()
        
        # Look for percentage or distribution patterns
        if any(keyword in content_lower for keyword in ['percent', '%', 'distribution', 'allocation', 'breakdown']):
            # Try to extract percentages
            percent_patterns = [
                r'(\d+(?:\.\d+)?)\s*%[^\w]*([^.!?\n]{1,40})',
                r'([^.!?\n]{1,40})[^\w]*(\d+(?:\.\d+)?)\s*%'
            ]
            
            for pattern in percent_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches[:6]:  # Limit to 6 items
                    if '%' in match[0] or match[0].replace('.', '').isdigit():
                        try:
                            value = float(match[0].replace('%', '').strip())
                            label = match[1].strip()[:25]
                        except:
                            continue
                    else:
                        try:
                            value = float(match[1].replace('%', '').strip())
                            label = match[0].strip()[:25]
                        except:
                            continue
                    
                    if label and value > 0:
                        labels.append(label)
                        values.append(value)
                
                if labels:
                    break
        
        # Attempt to infer deliverables list from bullets/lines when section mentions deliverables
        if not labels and 'deliverable' in section_name.lower():
            item_lines = []
            for line in content.split('\n'):
                line_stripped = line.strip().lstrip('-*').strip()
                if not line_stripped:
                    continue
                if any(k in line_stripped.lower() for k in ['deliverable', 'video', 'print', 'report', 'module', 'feature', 'milestone', 'document', 'training', 'deployment']):
                    item_lines.append(line_stripped)
            # Deduplicate and cap
            unique = []
            for it in item_lines:
                key = it[:25]
                if key and key not in unique:
                    unique.append(key)
                if len(unique) >= 6:
                    break
            if len(unique) >= 2:
                labels = unique
                # Even split values
                per = round(100 / len(labels), 2)
                values = [per] * len(labels)

        # Try to infer items for deliverables/solution sections
        if not labels and any(k in section_name.lower() for k in ['deliverable', 'solution', 'proposed solution']):
            candidates = []
            for line in content.split('\n'):
                raw = line.strip().lstrip('-*').strip()
                if not raw:
                    continue
                raw_l = raw.lower()
                if any(w in raw_l for w in ['feature', 'module', 'service', 'component', 'capability', 'deliverable', 'video', 'print', 'report', 'dashboard']):
                    candidates.append(raw[:25])
            # de-duplicate preserving order
            seen = set()
            inferred = []
            for item in candidates:
                if item and item not in seen:
                    seen.add(item)
                    inferred.append(item)
                if len(inferred) >= 6:
                    break
            if len(inferred) >= 2:
                labels = inferred
                share = round(100 / len(labels), 2)
                values = [share] * len(labels)

        # Default data if none extracted
        if not labels:
            if 'success' in section_name.lower() or 'case' in section_name.lower():
                labels = ['Healthcare', 'Finance', 'Retail', 'Technology', 'Manufacturing']
                values = [25, 20, 15, 30, 10]
                title = 'Industry Distribution'
            elif 'implementation' in section_name.lower():
                labels = ['Planning', 'Development', 'Testing', 'Deployment', 'Support']
                values = [15, 40, 20, 15, 10]
                title = 'Resource Allocation'
            else:
                labels = ['Category A', 'Category B', 'Category C', 'Category D']
                values = [30, 25, 20, 25]
                title = f'{section_name} Distribution'
        else:
            title = f'{section_name} Breakdown'
        
        return {
            'title': title,
            'labels': labels[:6],  # Limit to 6 items
            'values': values[:6]
        }
    
    def _extract_line_chart_data(self, content: str, section_name: str, request) -> Dict[str, Any]:
        """Extract data for line chart from section content"""
        # Try to extract trend or timeline data
        x_data = []
        y_data = []
        
        content_lower = content.lower()
        
        # Look for temporal or sequential patterns
        if any(keyword in content_lower for keyword in ['month', 'week', 'quarter', 'year', 'progress', 'growth']):
            # Extract timeline references
            time_patterns = [
                r'(month|week|quarter|year)\s+(\d+)',
                r'(\d+)\s+(months?|weeks?|quarters?|years?)',
                r'(Q\d+|Week\s+\d+|Month\s+\d+)'
            ]
            
            for pattern in time_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if len(matches) >= 3:  # Need at least 3 points for a line
                    for i, match in enumerate(matches[:8]):  # Limit to 8 points
                        if isinstance(match, tuple):
                            x_label = f"{match[0]} {match[1]}" if len(match) > 1 else match[0]
                        else:
                            x_label = match
                        x_data.append(x_label)
                        # Generate progressive values
                        y_data.append(10 + (i * 15) + (i ** 1.5) * 5)
                    
                    if x_data:
                        break
        
        # Default data if none extracted - adapt to RFP timeline
        if not x_data:
            name_lower = section_name.lower()
            if 'technical' in name_lower or 'methodology' in name_lower or 'project plan' in name_lower or 'timeline' in name_lower:
                # Derive number of weeks from request timeline if available
                weeks = 8
                try:
                    import re
                    # Helper to normalize Eastern Arabic numerals
                    def normalize_digits(s: str) -> str:
                        trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
                        return s.translate(trans)

                    # 1) Prefer explicit weeks in section content
                    content_norm = normalize_digits(content or '')
                    mw = re.search(r'(\d+)\s*(weeks?|week)', content_norm, re.IGNORECASE)
                    if mw:
                        weeks = int(mw.group(1))
                    else:
                        # 2) Fallback to months in section content
                        mm = re.search(r'(\d+)\s*(months?|month)', content_norm, re.IGNORECASE)
                        if mm:
                            weeks = int(mm.group(1)) * 4
                        else:
                            # 3) Use request timeline (months or weeks)
                            tl = normalize_digits(getattr(request, 'timeline', '') or '') if request else ''
                            mw2 = re.search(r'(\d+)\s*(weeks?|week)', tl, re.IGNORECASE)
                            if mw2:
                                weeks = int(mw2.group(1))
                            else:
                                m2 = re.search(r'(\d+)', tl)
                                if m2:
                                    months = int(m2.group(1))
                                    weeks = months * 4
                                else:
                                    # 4) Check rfp_extracted timeline field
                                    reqs = getattr(request, 'requirements', {}) or {}
                                    ext = reqs.get('rfp_extracted', {}) if isinstance(reqs, dict) else {}
                                    tl2 = normalize_digits(str(ext.get('timeline', '')))
                                    mw3 = re.search(r'(\d+)\s*(weeks?|week)', tl2, re.IGNORECASE)
                                    if mw3:
                                        weeks = int(mw3.group(1))
                                    else:
                                        m3 = re.search(r'(\d+)', tl2)
                                        if m3:
                                            weeks = int(m3.group(1)) * 4
                except Exception:
                    pass
                # Clamp to sensible bounds
                weeks = max(4, min(24, int(weeks)))
                x_data = [f'Week {i+1}' for i in range(weeks)]
                # Create a smooth S-curve progress trend
                y_data = []
                for i in range(weeks):
                    pct = int(5 + (i / max(1, weeks - 1)) * 95)
                    y_data.append(pct)
                title = 'Development Progress'
                x_label = 'Timeline'
                y_label = 'Completion %'
            else:
                x_data = ['Q1', 'Q2', 'Q3', 'Q4']
                y_data = [20, 35, 55, 80]
                title = f'{section_name} Progress'
                x_label = 'Quarter'
                y_label = 'Progress'
        else:
            title = f'{section_name} Trend Analysis'
            x_label = 'Timeline'
            y_label = 'Value'
        
        return {
            'title': title,
            'x_data': x_data[:8],  # Limit to 8 points
            'y_data': y_data[:8],
            'x_label': x_label,
            'y_label': y_label
        }
    
    async def _generate_charts_legacy_sdk(self, context: Dict):
        """Legacy SDK agent approach for chart generation (fallback)"""
        chart_generator = get_agent("chart_generator")
        
        # Set output format for chart generation tools (default to static)
        self._set_chart_output_format('static')
        
        for section_name, section_config in self.section_routing.items():
            if section_config.get('generate_chart'):
                chart_type = section_config['generate_chart']
                logger.info(f"Generating {chart_type} chart for {section_name} (legacy SDK)")
                
                try:
                    # Log chart generation start
                    agent_logger.log_agent_execution(
                        chart_generator.name if hasattr(chart_generator, 'name') else 'chart_generator',
                        f"generate_chart_{section_name}_legacy",
                        {"section_name": section_name, "chart_type": chart_type},
                        {"status": "starting", "method": "legacy_sdk"}
                    )
                    
                    # Prepare minimal chart data - only essential extracted data
                    chart_data = self._prepare_minimal_chart_data(section_name, chart_type, context)
                    
                    # Create simple, direct prompts for each chart type
                    if chart_type == 'gantt':
                        chart_prompt = f"""Use the generate_gantt_chart tool to create a Gantt chart.

Project: {chart_data.get('title', 'Project Timeline')}
Phases: {chart_data.get('phases', [])}
Duration: {chart_data.get('duration_months', 3)} months

Call the generate_gantt_chart tool now."""

                    elif chart_type == 'budget_breakdown':
                        chart_prompt = f"""Use the create_budget_chart tool to create a budget breakdown.

Budget Items: {chart_data.get('budget_items', [])}
Total: ${chart_data.get('total_budget', 100000):,.2f}

Call the create_budget_chart tool now."""

                    elif chart_type == 'risk_matrix':
                        chart_prompt = f"""Use the build_risk_matrix tool to create a risk matrix.

Risks: {chart_data.get('risks', [])}

Call the build_risk_matrix tool now."""

                    else:
                        chart_prompt = f"""Create a {chart_type} chart using the appropriate tool.

Data: {chart_data}

Use the most suitable chart generation tool."""
                    
                    # Prepare minimal context for chart generation
                    minimal_context = prepare_minimal_chart_context(section_name, chart_type, context)
                    
                    # Check token count before API call
                    total_prompt = chart_prompt + str(minimal_context)
                    token_count = count_tokens(total_prompt, 'gpt-4o')
                    
                    if token_count > CONTEXT_LIMITS['gpt-4o'] - RESPONSE_RESERVE_TOKENS:
                        logger.warning(f"Chart prompt too long ({token_count} tokens), truncating...")
                        chart_prompt = truncate_text(chart_prompt, CONTEXT_LIMITS['gpt-4o'] - RESPONSE_RESERVE_TOKENS - 500, 'gpt-4o')
                        minimal_context = {"chart_type": chart_type, "basic_data": str(chart_data)[:500]}
                    
                    # Use minimal context only
                    response = await Runner.run(
                        chart_generator,
                        chart_prompt,
                        context=minimal_context
                    )
                    
                    # Extract and validate chart HTML
                    chart_html = self._extract_chart_html_from_response(response)
                    
                    # Validate HTML content
                    if not chart_html or len(chart_html.strip()) < 50:
                        logger.warning(f"Generated chart HTML for {section_name} seems too short or empty")
                        chart_html = f"<div>Chart generation incomplete for {section_name} ({chart_type})</div>"
                    
                    # Check if HTML contains expected chart elements
                    if not any(keyword in chart_html.lower() for keyword in ['plotly', 'chart', 'svg', 'canvas', '<div']):
                        logger.warning(f"Chart HTML for {section_name} may not contain valid chart content")
                        chart_html = f"<div>Chart content validation failed for {section_name}</div>"
                    
                    context["charts"][section_name] = {
                        "type": chart_type,
                        "data": chart_html,
                        "generated_at": datetime.now().isoformat(),
                        "data_used": chart_data,
                        "html_length": len(chart_html),
                        "method": "legacy_sdk"
                    }
                    
                    # Log successful chart generation
                    agent_logger.log_agent_execution(
                        chart_generator.name if hasattr(chart_generator, 'name') else 'chart_generator',
                        f"generate_chart_{section_name}_legacy",
                        {"section_name": section_name, "chart_type": chart_type},
                        {"status": "success", "chart_type": chart_type, "method": "legacy_sdk"}
                    )
                    
                    # Track cost with SDK response
                    self.cost_tracker.track_completion(response, model="gpt-4o")
                    
                except Exception as e:
                    logger.error(f"Failed to generate chart for {section_name} (legacy): {str(e)}")
                    # Log chart generation failure
                    agent_logger.log_agent_execution(
                        chart_generator.name if hasattr(chart_generator, 'name') else 'chart_generator',
                        f"generate_chart_{section_name}_legacy",
                        {"section_name": section_name, "chart_type": chart_type},
                        {"status": "error", "error": str(e), "method": "legacy_sdk"}
                    )
    
    async def _evaluate_quality(self, context: Dict):
        """Evaluate the quality of generated content"""
        evaluator = get_agent("quality_evaluator")
        
        total_score = 0
        section_count = 0
        
        for section_name, section_data in context["generated_sections"].items():
            if section_data.get("metadata", {}).get("fallback"):
                continue  # Skip fallback sections
            
            try:
                eval_prompt = f"""
                Evaluate the quality of this proposal section on a scale of 1-10:
                
                Section: {section_name}
                Content: {section_data.get('content', '')}
                
                Rate based on:
                - Relevance to requirements (25%)
                - Technical accuracy (25%) 
                - Clarity and readability (20%)
                - Completeness (20%)
                - Professionalism (10%)
                
                Return only a numeric score.
                """
                
                # Log evaluation start
                agent_logger.log_agent_execution(
                    evaluator.name if hasattr(evaluator, 'name') else 'quality_evaluator',
                    f"evaluate_{section_name}",
                    {"section_name": section_name},
                    {"status": "starting"}
                )
                
                response = await Runner.run(
                    evaluator,
                    eval_prompt,
                    context={"section_name": section_name}
                )
                
                score_text = self._extract_content_from_sdk_response(response)
                
                # Extract numeric score
                try:
                    score = float(score_text.strip())
                    section_data["metadata"]["quality_score"] = score
                    total_score += score
                    section_count += 1
                    
                    # Log successful evaluation
                    agent_logger.log_agent_execution(
                        evaluator.name if hasattr(evaluator, 'name') else 'quality_evaluator',
                        f"evaluate_{section_name}",
                        {"section_name": section_name},
                        {"status": "success", "score": score}
                    )
                except ValueError:
                    logger.warning(f"Could not parse quality score for {section_name}: {score_text}")
                
                # Track cost with SDK response
                self.cost_tracker.track_completion(response, model="gpt-4o")
                
            except Exception as e:
                logger.error(f"Failed to evaluate section {section_name}: {str(e)}")
                # Log evaluation failure
                agent_logger.log_agent_execution(
                    evaluator.name if hasattr(evaluator, 'name') else 'quality_evaluator',
                    f"evaluate_{section_name}",
                    {"section_name": section_name},
                    {"status": "error", "error": str(e)}
                )
        
        # Calculate overall quality score
        if section_count > 0:
            overall_score = total_score / section_count
            context["metadata"]["quality_score"] = overall_score
            logger.info(f"Overall proposal quality score: {overall_score:.1f}/10")
    
    async def generate_section(self, section_name: str, routing: Dict = None, context: Dict = None) -> Dict:
        """Public interface for generating a single section (used by orchestrator)"""
        # Create a ProposalRequest from context if not provided
        if 'request' not in context:
            context['request'] = ProposalRequest(
                client_name=context.get('client_name', 'Unknown'),
                project_name=context.get('project_name', 'Unknown'),
                project_type=context.get('project_type', 'general'),
                requirements=context.get('requirements', {}),
                timeline=context.get('timeline', '3 months')
            )
        
        # Get coordinator agent
        coordinator = get_agent("coordinator")
        
        # Generate the section with updated context handling
        result = await self._generate_section(section_name, context, coordinator)
        
        return result
    
    async def evaluate_single_section(self, section_name: str, content: Dict = None) -> float:
        """Evaluate quality of a single section (used by orchestrator)"""
        evaluator = get_agent("quality_evaluator")
        
        try:
            # Truncate content for evaluation
            section_content = content.get('content', '') if content else ''
            truncated_content = truncate_text(section_content, 3000, 'gpt-4o')
            
            eval_prompt = f"""
            Evaluate the quality of this proposal section on a scale of 1-10:
            
            Section: {section_name}
            Content: {truncated_content}
            
            Rate based on:
            - Relevance to requirements (25%)
            - Technical accuracy (25%) 
            - Clarity and readability (20%)
            - Completeness (20%)
            - Professionalism (10%)
            
            Return only a numeric score.
            """
            
            # Log evaluation start
            agent_logger.log_agent_execution(
                evaluator.name if hasattr(evaluator, 'name') else 'quality_evaluator',
                f"evaluate_single_{section_name}",
                {"section_name": section_name},
                {"status": "starting"}
            )
            
            response = await Runner.run(
                evaluator,
                eval_prompt,
                context={"section_name": section_name, "word_count": len(section_content.split()) if section_content else 0}
            )
            
            score_text = self._extract_content_from_sdk_response(response)
            
            # Extract numeric score
            try:
                score = float(score_text.strip())
                # Log successful evaluation
                agent_logger.log_agent_execution(
                    evaluator.name if hasattr(evaluator, 'name') else 'quality_evaluator',
                    f"evaluate_single_{section_name}",
                    {"section_name": section_name},
                    {"status": "success", "score": score}
                )
                return score
            except ValueError:
                logger.warning(f"Could not parse quality score for {section_name}: {score_text}")
                return 7.5  # Default score
                
        except Exception as e:
            logger.error(f"Failed to evaluate section {section_name}: {str(e)}")
            return 7.5  # Default score
    
    def _extract_content_from_sdk_response(self, response) -> str:
        """Extract content from SDK response object with JSON parsing support"""
        
        try:
            # Step 1: Extract raw content using existing logic
            raw_content = None
            
            # Handle RunResult from Runner.run_sync()
            if hasattr(response, 'final_output'):
                raw_content = response.final_output
            # Handle different response formats from the SDK
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
                                logger.debug(f"Extracted content from JSON field '{field}': {len(extracted_content) if extracted_content else 0} chars")
                                return extracted_content
                        
                        # If no standard field found, check if the JSON itself is a string
                        if isinstance(parsed_json, str):
                            return parsed_json
                        
                        # If JSON is a dict but no content field, return JSON as string for backward compatibility
                        logger.warning("JSON found but no content field, returning JSON string")
                        return json.dumps(parsed_json, indent=2)
                        
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"Invalid JSON in code block, falling back to raw content: {json_err}")
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
                                logger.debug(f"Extracted content from direct JSON field '{field}': {len(extracted_content) if extracted_content else 0} chars")
                                return extracted_content
                        
                        # Special handling for ResearchAgent response format
                        if 'topic' in parsed_json and ('summary' in parsed_json or 'findings' in parsed_json):
                            logger.debug("Detected ResearchAgent JSON format, converting to readable content")
                            content_parts = []
                            
                            # Don't add topic as title for section content - it's redundant
                            # The section already has a proper heading
                            
                            # Add summary as main content if available
                            if 'summary' in parsed_json and parsed_json['summary'].strip():
                                content_parts.append(parsed_json['summary'])
                            
                            # Add findings as main content if no summary or summary is incomplete
                            if 'findings' in parsed_json and isinstance(parsed_json['findings'], list):
                                if not content_parts:  # No summary, so make findings the main content
                                    # Convert findings to narrative format
                                    findings_text = []
                                    for i, finding in enumerate(parsed_json['findings']):
                                        if isinstance(finding, dict) and 'insight' in finding:
                                            insight = finding['insight']
                                            findings_text.append(insight)
                                    
                                    if findings_text:
                                        # Create a cohesive narrative from findings
                                        content_parts.append(" ".join(findings_text))
                                else:
                                    # We have summary, so add findings as supporting details
                                    content_parts.append("\n## Key Supporting Findings:")
                                    for finding in parsed_json['findings']:
                                        if isinstance(finding, dict) and 'insight' in finding:
                                            insight = finding['insight']
                                            content_parts.append(f"- {insight}")
                            
                            return "\n\n".join(content_parts)
                        
                        # Special handling for structured section content
                        if 'section_title' in parsed_json and 'content' in parsed_json:
                            logger.debug("Detected ContentGenerator JSON format with section_title")
                            return parsed_json['content']
                    
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
            logger.warning(f"Could not extract content from SDK response: {e}")
            # Graceful fallback - try to return something useful
            try:
                return str(response) if response else ""
            except:
                return ""
    
    async def _run_sdk_agent(self, agent, message: str, context: Dict = None) -> Dict:
        """Run an SDK agent with a message and context using proper Runner"""
        if context is None:
            context = {}
            
        try:
            # Set minimal context variables for the agent to avoid token limits
            request_data = context.get('request', {})
            if hasattr(request_data, '__dict__'):
                # Truncate request data
                request_dict = {k: (v[:200] + "...[truncated]" if isinstance(v, str) and len(v) > 200 else v) for k, v in request_data.__dict__.items()}
            else:
                request_dict = request_data
                
            context_variables = {
                'request_summary': request_dict,
                'company_name': context.get('company_profile', self.company_profile).get('name', 'AzmX')[:100],
                'company_context': context.get('company_profile', self.company_profile),
                'skills_summary': context.get('skills_summary', self.skills_summary),
            }
            
            # Check message size before API call
            total_input = message + str(context_variables)
            token_count = count_tokens(total_input, 'gpt-4o')
            
            if token_count > CONTEXT_LIMITS['gpt-4o'] - RESPONSE_RESERVE_TOKENS:
                logger.warning(f"Agent input too large ({token_count} tokens), truncating message...")
                message = truncate_text(message, CONTEXT_LIMITS['gpt-4o'] - RESPONSE_RESERVE_TOKENS - 200, 'gpt-4o')
                context_variables = {'agent': agent.name if hasattr(agent, 'name') else 'agent'}
            
            # Use SDK Runner for execution
            response = await Runner.run(
                agent,
                message,
                context=context_variables
            )
            
            # Extract content from SDK response
            content = self._extract_content_from_sdk_response(response)
            
            # Track cost with SDK response
            self.cost_tracker.track_completion(response, model="gpt-4o")
            
            return {
                'content': content,
                'response': response,
                'cost': self.cost_tracker.get_total_cost(),
                'agent_name': agent.name
            }
            
        except Exception as e:
            logger.error(f"Error running SDK agent {agent.name}: {str(e)}")
            return {
                'error': str(e),
                'agent': agent.name if hasattr(agent, 'name') else 'unknown',
                'message': message
            }
        
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost tracking summary"""
        return self.cost_tracker.get_summary()
    
    def get_total_cost(self) -> float:
        """Get total cost"""
        return self.cost_tracker.get_total_cost()