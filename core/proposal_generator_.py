"""
SDK-integrated Proposal Generator using OpenAI Agents
AzmX - Enterprise Proposal Generation System
Version 2.0 - SDK Integration
"""

import os
import json
import yaml
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from docx_exporter import DOCXExporter

from dataclasses import dataclass
import pandas as pd
import logging

# SDK imports
from proposal_agents.sdk_agents import AGENT_REGISTRY, get_agent, get_specialist_for_section
from core.sdk_runner import ProposalRunner, ProposalRequest
from core.simple_cost_tracker import SimpleCostTracker
from core.rag_retriever import RAGRetriever
from core.html_generator import HTMLGenerator
from core.pdf_exporter import PDFExporter
from utils.data_loader import DataLoader
from utils.validators import ProposalValidator
from utils.agent_logger import agent_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ProposalRequest is now imported from core.sdk_runner
# Removed duplicate dataclass definition

class ProposalGenerator:
    """SDK-integrated proposal generator class"""

    def __init__(self, config_path: str = "config/settings.yml", cost_tracker: Optional[SimpleCostTracker] = None):
        """Initialize the SDK-integrated proposal generator"""
        self.config = self._load_config(config_path)
        self.cost_tracker = cost_tracker or SimpleCostTracker()

        # Initialize SDK runner
        self.sdk_runner = ProposalRunner(config_path, self.cost_tracker)

        # Initialize document processing components
        self.rag_retriever = RAGRetriever(self.config)
        self.html_generator = HTMLGenerator(self.config, output_format='interactive')
        self.html_generator_pdf = HTMLGenerator(self.config, output_format='static')
        self.pdf_exporter = PDFExporter(self.config)

        # ADD THIS LINE - Initialize DOCXExporter
        self.docx_exporter = DOCXExporter(self.config)

        self.data_loader = DataLoader(self.config)
        self.validator = ProposalValidator(self.config)

        # Load data
        self.skills_data = self.data_loader.load_skills_data()
        self.company_profile = self.data_loader.load_company_profile()

        # Validate SDK agents
        self._validate_sdk_agents()
        logger.info("SDK-integrated Proposal Generator initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def generate_proposal(self, request: ProposalRequest) -> Dict[str, Any]:
        """
        Generate a complete proposal using SDK agents and RAG pipeline
        
        Args:
            request: ProposalRequest object with all requirements
            
        Returns:
            Dictionary containing the generated proposal
        """
        logger.info(f"Starting SDK proposal generation for {request.client_name}")
        
        # Process documents if source PDF is provided
        if 'source_pdf' in request.requirements:
            await self._process_rfp_documents(request)
        
        # Use SDK runner for proposal generation
        proposal = await self.sdk_runner.generate_proposal(request)
        
        # Enhance with additional processing
        enhanced_proposal = await self._enhance_proposal(proposal, request)
        
        # Run final validation and scoring
        final_proposal = await self._finalize_proposal(enhanced_proposal, request)
        
        logger.info(f"SDK proposal generation completed for {request.client_name}")
        return final_proposal
        
        # Removed old sequential loop - now using SDK runner
        
        # Final output generation handled by SDK runner and enhancement methods
    
    def _validate_sdk_agents(self):
        """Validate that all required SDK agents are available"""
        required_agents = ["coordinator", "content_generator", "researcher", "budget_calculator"]
        
        for agent_name in required_agents:
            try:
                agent = get_agent(agent_name)
                logger.info(f"✓ Agent '{agent_name}' validated")
            except Exception as e:
                logger.error(f"✗ Agent '{agent_name}' validation failed: {str(e)}")
                raise ValueError(f"Required agent '{agent_name}' not available")
    
    async def _process_rfp_documents(self, request: ProposalRequest):
        """Process RFP documents using RAG pipeline"""
        source_pdf = request.requirements.get('source_pdf')
        if not source_pdf:
            return
        
        logger.info(f"Processing RFP document: {source_pdf}")
        
        try:
            # Process and index the PDF
            result = self.rag_retriever.process_and_index_pdf(source_pdf, force_reindex=False)
            logger.info(f"Indexed {result['num_chunks']} chunks from RFP document")
            
            # Update request with enhanced context
            if 'rfp_context' not in request.requirements:
                request.requirements['rfp_context'] = {}
            
            request.requirements['rfp_context']['indexed_chunks'] = result['num_chunks']
            request.requirements['rfp_context']['processing_completed'] = True
            
        except Exception as e:
            logger.error(f"Failed to process RFP document: {str(e)}")
            # Continue without RFP processing
    
    async def _enhance_proposal(self, proposal: Dict[str, Any], request: ProposalRequest) -> Dict[str, Any]:
        """Enhance the proposal with additional processing and validation"""
        logger.info("Enhancing proposal with additional processing...")
        
        # Add enhanced metadata
        if 'metadata' not in proposal or not isinstance(proposal.get('metadata'), dict):
            proposal['metadata'] = {}
        proposal['metadata']['enhancement_completed'] = datetime.now().isoformat()
        proposal['metadata']['processor_version'] = "SDK-2.0"
        
        # Validate sections
        enhanced_sections = {}
        for section_name, section_data in proposal.get('generated_sections', {}).items():
            
            # Validate section content
            validation = self.validator.validate_section(section_name, section_data)
            
            if validation['is_valid']:
                enhanced_sections[section_name] = section_data
                if 'metadata' not in enhanced_sections[section_name] or not isinstance(enhanced_sections[section_name].get('metadata'), dict):
                    enhanced_sections[section_name]['metadata'] = {}
                enhanced_sections[section_name]['metadata']['validation'] = validation
            else:
                logger.warning(f"Section '{section_name}' failed validation: {validation.get('feedback', 'Unknown issue')}")
                enhanced_sections[section_name] = section_data
                if 'metadata' not in enhanced_sections[section_name] or not isinstance(enhanced_sections[section_name].get('metadata'), dict):
                    enhanced_sections[section_name]['metadata'] = {}
                enhanced_sections[section_name]['metadata']['validation'] = validation
                enhanced_sections[section_name]['metadata']['requires_review'] = True
        
        proposal['generated_sections'] = enhanced_sections
        
        # Add cost tracking information
        cost_summary = self.cost_tracker.get_summary()
        proposal['metadata']['cost_tracking'] = cost_summary
        
        logger.info("Proposal enhancement completed")
        return proposal

    async def _generate_docx_async(self, html_content: str, client_name: str) -> str:
        """Generate DOCX file asynchronously to avoid blocking"""
        import asyncio

        def _sync_docx_generation():
            """Synchronous DOCX generation to run in thread"""
            try:
                return self.docx_exporter.export(html_content, client_name)
            except Exception as e:
                logger.error(f"Sync DOCX generation failed: {str(e)}")
                raise

        # Run the synchronous DOCX generation in a thread pool
        loop = asyncio.get_event_loop()
        docx_path = await loop.run_in_executor(None, _sync_docx_generation)
        return docx_path

    async def _finalize_proposal(self, proposal: Dict[str, Any], request: ProposalRequest) -> Dict[str, Any]:
        """Finalize the proposal with outputs and quality scoring"""
        logger.info("Finalizing proposal with outputs and quality scoring...")

        if 'metadata' not in proposal or not isinstance(proposal.get('metadata'), dict):
            proposal['metadata'] = {}

        # Calculate overall quality score
        sections = proposal.get('generated_sections', {})
        if sections:
            overall_score = self.validator.calculate_overall_score(sections)
            proposal['metadata']['quality_score'] = overall_score
            logger.info(f"Overall proposal quality score: {overall_score:.1f}/10")

        # Generate HTML output
        html_output = self._generate_html_output(proposal, request)
        proposal['html'] = html_output

        # Generate PDF if enabled
        if self.config.get('output', {}).get('enable_pdf', True):
            try:
                # Generate static HTML for PDF export
                static_html_output = self.html_generator_pdf.generate(proposal)
                pdf_path = self.pdf_exporter.export(static_html_output, request.client_name)
                proposal['pdf_path'] = pdf_path
                logger.info(f"PDF generated: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to generate PDF: {str(e)}")

        # Generate DOCX if enabled - THIS IS THE MISSING PART!
        if self.config.get('output', {}).get('enable_docx', True):
            logger.info("=== STARTING DOCX GENERATION ===")
            try:
                # Debug information
                logger.info(f"HTML content length: {len(html_output) if html_output else 0}")
                logger.info(f"Client name: '{request.client_name}'")
                logger.info(f"DOCX exporter available: {hasattr(self, 'docx_exporter')}")

                # Generate DOCX using async wrapper
                docx_path = await self._generate_docx_async(html_output, request.client_name)
                proposal['docx_path'] = docx_path

                # Verify file was created
                if os.path.exists(docx_path):
                    file_size = os.path.getsize(docx_path)
                    logger.info(f"✓ DOCX generated successfully: {docx_path} (Size: {file_size} bytes)")
                else:
                    logger.error(f"✗ DOCX file not found at: {docx_path}")

            except Exception as e:
                logger.error(f"✗ Failed to generate DOCX: {str(e)}", exc_info=True)
                proposal['docx_error'] = str(e)
        else:
            logger.info("DOCX generation is disabled in configuration")

        # Add completion timestamp
        if 'metadata' not in proposal or not isinstance(proposal.get('metadata'), dict):
            proposal['metadata'] = {}
        proposal['metadata']['finalization_completed'] = datetime.now().isoformat()
        logger.info("Proposal finalization completed")
        return proposal

    def _generate_html_output(self, proposal: Dict[str, Any], request: ProposalRequest) -> str:
        """Generate HTML output from proposal data"""
        try:
            return self.html_generator.generate(proposal)
        except Exception as e:
            logger.error(f"Failed to generate HTML output: {str(e)}")
            return self._generate_fallback_html(proposal, request)
    
    def _generate_fallback_html(self, proposal: Dict[str, Any], request: ProposalRequest) -> str:
        """Generate basic HTML fallback when main generator fails"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>Proposal Document</title></head><body>",
            f"<h1>Proposal for {request.client_name}</h1>",
            f"<h2>Project: {request.project_name}</h2>",
            "<hr>"
        ]
        
        sections = proposal.get('generated_sections', {})
        for section_name, section_data in sections.items():
            content = section_data.get('content', '') if isinstance(section_data, dict) else str(section_data)
            html_parts.extend([
                f"<h3>{section_name}</h3>",
                f"<div>{content}</div>",
                "<br>"
            ])
        
        html_parts.extend(["</body></html>"])
        
        return "\n".join(html_parts)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        return self.cost_tracker.get_summary()
    
    def _get_proposal_outline(self, project_type: str) -> List[str]:
        """Get the proposal outline based on project type"""
        # Standard outline for most projects
        standard_outline = [
            "Problem or Need Statement",
            "Project Scope",
            "Proposed Solution",
            "List of Deliverables",
            "Technical Approach and Methodology",
            "Project Plan and Timelines",
            "Budget",
            "Risk Analysis and Mitigation",
            "Our Team/Company Profile",
            "Success Stories/Case Studies",
            "Implementation Strategy",
            "Support and Maintenance",
            "Terms and Conditions",
            "Conclusion",
            "Executive Summary"  # Generated last but placed first
        ]
        
        # Customize based on project type
        if project_type == "web_development":
            standard_outline.insert(5, "UI/UX Design Approach")
        elif project_type == "mobile_app":
            standard_outline.insert(5, "Platform Strategy")
        elif project_type == "data_analytics":
            standard_outline.insert(5, "Data Architecture")
        
        return standard_outline
    
    # Legacy methods removed - now handled by SDK runner and agent
    # are all managed by the SDK agents through the coordinator

# SDK-integrated main function for testing
async def main():
    """Test the SDK-integrated proposal generator"""
    
    # Initialize cost tracker
    cost_tracker = SimpleCostTracker()
    
    # Initialize SDK-integrated generator
    generator = ProposalGenerator(cost_tracker=cost_tracker)
    
    # Sample request using imported ProposalRequest
    request = ProposalRequest(
        client_name="ACME Corporation",
        project_name="Digital Transformation Initiative",
        project_type="web_development",
        requirements={
            "platform": "Cloud-based SaaS",
            "users": 10000,
            "features": ["Dashboard", "Analytics", "Reporting", "API"],
            "integration": ["SAP", "Salesforce"],
            "security": "SOC 2 Type II compliant",
            "technical": "Modern web application with microservices architecture",
            "scope": "Full-stack development including frontend, backend, and API services"
        },
        timeline="6 months",
        budget_range="$300,000 - $500,000",
        special_requirements=["Arabic localization", "24/7 support", "Cloud deployment"]
    )
    
    logger.info("="*60)
    logger.info("TESTING SDK-INTEGRATED PROPOSAL GENERATOR")
    logger.info("="*60)
    
    # Generate proposal using SDK agents
    proposal = await generator.generate_proposal(request)
    
    # Save proposal
    output_dir = "generated_proposals"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/SDK_{request.client_name}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(proposal, f, indent=2, default=str)
    
    # Display results
    logger.info("="*60)
    logger.info("PROPOSAL GENERATION COMPLETED")
    logger.info("="*60)
    logger.info(f"Output saved to: {output_file}")
    
    if 'html' in proposal:
        html_file = f"{output_dir}/SDK_{request.client_name}_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(proposal['html'])
        logger.info(f"HTML saved to: {html_file}")
    
    if 'pdf_path' in proposal:
        logger.info(f"PDF exported to: {proposal['pdf_path']}")
    
    # Show cost summary
    cost_summary = generator.get_cost_summary()
    logger.info(f"Total API Cost: ${cost_summary['total_cost']:.4f}")
    logger.info(f"Total Tokens: {cost_summary['total_tokens']}")
    logger.info(f"API Calls: {cost_summary['total_calls']}")
    
    # Show quality score
    if 'metadata' in proposal and 'quality_score' in proposal['metadata']:
        logger.info(f"Quality Score: {proposal['metadata']['quality_score']:.1f}/10")

if __name__ == "__main__":
    asyncio.run(main())