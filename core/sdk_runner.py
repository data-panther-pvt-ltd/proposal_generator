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
from utils.config_loader import load_config_with_profile

logger = logging.getLogger(__name__)
# tiktoken removed - using exact token counts from API responses

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
from core.rfp_extractor import RFPExtractor
from utils.data_loader import DataLoader
from utils.validators import ProposalValidator
from utils.agent_logger import agent_logger

logger = logging.getLogger(__name__)

# Context size limits for different models
CONTEXT_LIMITS = {
    # GPT-5 series
    'gpt-5': 200000,
    'gpt-5-mini': 128000,

    # GPT-4o series
    'gpt-4o': 128000,
    'gpt-4o-mini': 128000,

    # Default fallback
    'default': 128000
}

def get_context_limit(model: str) -> int:
    """Get context limit for a given model"""
    return CONTEXT_LIMITS.get(model, CONTEXT_LIMITS['default'])

# Reserve tokens for response
RESPONSE_RESERVE_TOKENS = 2000

def estimate_tokens(text: str) -> int:
    """Estimate tokens in text using character count
    
    This is a rough estimation used only for truncation purposes.
    Actual token counts come from API responses.
    """
    if not text:
        return 0
    # Rough estimate: 4 characters per token (conservative)
    return len(text) // 4

def truncate_text(text: str, max_completion_tokens: int, model: str = 'gpt-4o') -> str:
    """Truncate text to fit within token limit"""
    if not text:
        return text
        
    current_tokens = estimate_tokens(text)
    if current_tokens <= max_completion_tokens:
        return text
        
    # Binary search for optimal truncation point
    left, right = 0, len(text)
    best_text = text[:max_completion_tokens * 4]  # Initial rough estimate
    
    while left < right:
        mid = (left + right) // 2
        candidate = text[:mid] + "...[truncated]"
        
        if estimate_tokens(candidate) <= max_completion_tokens:
            best_text = candidate
            left = mid + 1
        else:
            right = mid
            
    return best_text



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
        self.cost_tracker = cost_tracker or SimpleCostTracker(self.config)
        
        # Initialize components
        self.html_generator = HTMLGenerator(self.config, output_format='interactive')
        self.html_generator_pdf = HTMLGenerator(self.config, output_format='static')
        self.pdf_exporter = PDFExporter(self.config)
        self.docx_exporter = DOCXExporter(self.config)
        self.data_loader = DataLoader(self.config)
        self.validator = ProposalValidator(self.config)
        self.rag_retriever = RAGRetriever(self.config, self.cost_tracker)
        self.rfp_extractor = RFPExtractor(self.config, self.cost_tracker)
        
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
        """Load configuration from YAML file with company profile from markdown"""
        return load_config_with_profile(config_path)

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
            "metadata": {}
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
                                self.cost_tracker.track_completion(response)
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
        model_used = self.config.get('openai', {}).get('model', 'gpt-4o')
        context_limit = get_context_limit(model_used)
        token_count = estimate_tokens(prompt)
        if token_count > context_limit - RESPONSE_RESERVE_TOKENS - 1000:  # Reserve space for context
            logger.warning(f"Section prompt too long ({token_count} tokens), truncating...")
            prompt = truncate_text(prompt, context_limit - RESPONSE_RESERVE_TOKENS - 1000, model_used)
        
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
            
            # Track cost with SDK response
            self.cost_tracker.track_completion(response)
            
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
            model_used = self.config.get('openai', {}).get('model', 'gpt-4o')
            truncated_rag_context = truncate_text(rag_context, 2000, model_used) if rag_context else ""
            
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
            
            token_count = estimate_tokens(total_input)
            context_limit = get_context_limit(model_used)
            if token_count > context_limit - RESPONSE_RESERVE_TOKENS:
                logger.warning(f"Total context too large ({token_count} tokens), using minimal context...")
                # Use absolute minimal context
                session_context = {
                    "section_name": section_name,
                    "strategy": strategy,
                    "client": getattr(context['request'], 'client_name', 'Unknown'),
                    "project": getattr(context['request'], 'project_name', 'Unknown')[:100]
                }
                section_prompt = truncate_text(section_prompt, context_limit - RESPONSE_RESERVE_TOKENS - 500, model_used)
            
            # Use SDK Runner to execute coordinator with minimal session management
            response = await Runner.run(
                coordinator,
                f"Generate the '{section_name}' section for the proposal. Write ONLY in English.\n\n{section_prompt}",
                context=session_context
            )
            
            # Track cost with SDK response
            self.cost_tracker.track_completion(response)
            
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

        # Ensure timelines section includes a proper HTML table
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
                
                # Track cost with SDK response
                self.cost_tracker.track_completion(response)
                
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
                self.cost_tracker.track_completion(response)
                
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
                        
                        # If JSON is a dict but no content field, extract readable content
                        logger.warning("JSON found but no content field, extracting readable content")
                        return self._extract_readable_from_json(parsed_json)
                        
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
            token_count = estimate_tokens(total_input)

            # Get model from config for dynamic context limit
            model_used = self.config.get('openai', {}).get('model', 'gpt-4o')
            context_limit = get_context_limit(model_used)

            if token_count > context_limit - RESPONSE_RESERVE_TOKENS:
                logger.warning(f"Agent input too large ({token_count} tokens), truncating message...")
                message = truncate_text(message, context_limit - RESPONSE_RESERVE_TOKENS - 200, model_used)
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
            self.cost_tracker.track_completion(response)
            
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
    
    def _extract_readable_from_json(self, parsed_json: dict) -> str:
        """Extract readable content from structured JSON responses"""
        try:
            # Handle ResearchAgent response format
            if 'topic' in parsed_json and ('summary' in parsed_json or 'findings' in parsed_json):
                logger.debug("Detected ResearchAgent JSON format, converting to readable content")
                content_parts = []
                
                # Add summary as main content if available
                if 'summary' in parsed_json and parsed_json['summary'].strip():
                    content_parts.append(parsed_json['summary'].strip())
                
                # Add findings as bullet points if available
                if 'findings' in parsed_json and isinstance(parsed_json['findings'], list):
                    findings_content = []
                    for finding in parsed_json['findings']:
                        if isinstance(finding, dict):
                            if 'insight' in finding:
                                findings_content.append(f"- {finding['insight']}")
                            elif 'data_point' in finding:
                                findings_content.append(f"- {finding['data_point']}")
                        elif isinstance(finding, str):
                            findings_content.append(f"- {finding}")
                    
                    if findings_content:
                        content_parts.append("\n".join(findings_content))
                
                return "\n\n".join(content_parts) if content_parts else "[No readable content extracted]"
            
            # Handle ContentGenerator response format with nested structure
            if 'section_title' in parsed_json and 'content' in parsed_json:
                return parsed_json['content']
            
            # Handle responses with key_points, suggestions, recommendations
            content_parts = []
            if 'summary' in parsed_json:
                content_parts.append(parsed_json['summary'])
            
            if 'key_points' in parsed_json and isinstance(parsed_json['key_points'], list):
                content_parts.append("\n".join([f"- {point}" for point in parsed_json['key_points']]))
            
            if 'recommendations' in parsed_json and isinstance(parsed_json['recommendations'], list):
                content_parts.append("**Recommendations:**")
                content_parts.append("\n".join([f"- {rec}" for rec in parsed_json['recommendations']]))
                
            if 'suggested_visuals' in parsed_json and isinstance(parsed_json['suggested_visuals'], list):
                content_parts.append("**Suggested Visuals:**")
                content_parts.append("\n".join([f"- {visual}" for visual in parsed_json['suggested_visuals']]))
            
            # If we found readable parts, return them
            if content_parts:
                return "\n\n".join(content_parts)
            
            # Final fallback - check for any string values in the JSON
            text_values = []
            for key, value in parsed_json.items():
                if isinstance(value, str) and len(value.strip()) > 10:  # Only meaningful text
                    if key not in ['topic', 'confidence_level', 'model_used']:  # Skip metadata
                        text_values.append(value.strip())
            
            if text_values:
                return "\n\n".join(text_values)
            
            # If all else fails, return a placeholder
            return "[Complex structured response - unable to extract readable content]"
            
        except Exception as e:
            logger.error(f"Error extracting readable content from JSON: {str(e)}")
            return "[Error processing structured response]"

    def get_total_cost(self) -> float:
        """Get total cost"""
        return self.cost_tracker.get_total_cost()