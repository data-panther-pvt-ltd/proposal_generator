"""
Main entry point for the Proposal Generator with SDK Agents and RAG Pipeline
Uses OpenAI SDK agents for intelligent proposal generation
Uses FAISS vector database for document retrieval
"""

import os
# Fix OpenMP conflict with PyTorch/NumPy libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
# Add this import at the top
from core.rfp_extractor import RFPExtractor
from dataclasses import dataclass
import pandas as pd
import logging

# SDK imports - fully integrated
from proposal_agents.sdk_agents import get_all_agents, coordinator, get_agent, get_specialist_for_section
from core.sdk_runner import ProposalRunner, ProposalRequest
from core.simple_cost_tracker import SimpleCostTracker
from core.rag_retriever import RAGRetriever
from core.html_generator import HTMLGenerator
from core.pdf_exporter import PDFExporter
from core.docx_exporter import DOCXExporter
from utils.data_loader import DataLoader
from utils.validators import ProposalValidator
from utils.logging_config import setup_logging, get_logger
from utils.agent_logger import agent_logger

def load_config(config_path: str = "config/settings.yml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ProposalGenerator:
    """SDK-integrated proposal generator with full OpenAI Agents SDK integration"""
    
    def __init__(self, config_path: str = "config/settings.yml", cost_tracker: Optional[SimpleCostTracker] = None):
        """Initialize the SDK-integrated proposal generator"""
        self.config = self._load_config(config_path)
        self.cost_tracker = cost_tracker or SimpleCostTracker()
        
        # Initialize SDK runner with cost tracker instead of OpenAI client
        self.sdk_runner = ProposalRunner(config_path, self.cost_tracker)
        
        # Initialize document processing components - keeping existing functionality
        self.rag_retriever = RAGRetriever(self.config)
        self.html_generator = HTMLGenerator(self.config, output_format='interactive')
        self.html_generator_pdf = HTMLGenerator(self.config, output_format='static')
        self.pdf_exporter = PDFExporter(self.config)
        self.docx_exporter = DOCXExporter(self.config)
        self.data_loader = DataLoader(self.config)
        self.validator = ProposalValidator(self.config)
        
        # Load data
        self.skills_data = self.data_loader.load_skills_data()
        self.company_profile = self.data_loader.load_company_profile()
        
        # Get all SDK agents and validate
        self.agents = get_all_agents()
        self.coordinator_agent = coordinator
        self._validate_sdk_agents()
        
        logger = get_logger(__name__)
        logger.info("SDK-integrated Proposal Generator initialized successfully")
        logger.info(f"Available SDK agents: {list(self.agents.keys())}")
    
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
        logger = get_logger(__name__)
        logger.info(f"Starting SDK proposal generation for {request.client_name}")
        
        # Process RFP documents using RAG pipeline if source PDF is provided
        if 'source_pdf' in request.requirements:
            await self._process_rfp_documents(request)
        
        # Use SDK runner for proposal generation with agents
        proposal = await self.sdk_runner.generate_proposal(request)
        
        # Enhance with additional processing while keeping all existing functionality
        enhanced_proposal = await self._enhance_proposal(proposal, request)
        
        # Run final validation and scoring
        final_proposal = await self._finalize_proposal(enhanced_proposal, request)
        
        logger.info(f"SDK proposal generation completed for {request.client_name}")
        return final_proposal
    
    def _validate_sdk_agents(self):
        """Validate that all required SDK agents are available"""
        required_agents = ["coordinator", "content_generator", "researcher", "budget_calculator", "chart_generator"]
        logger = get_logger(__name__)
        
        for agent_name in required_agents:
            try:
                agent = get_agent(agent_name)
                logger.info(f"✓ Agent '{agent_name}' validated")
            except Exception as e:
                logger.error(f"✗ Agent '{agent_name}' validation failed: {str(e)}")
                raise ValueError(f"Required agent '{agent_name}' not available")
    
    async def _process_rfp_documents(self, request: ProposalRequest):
        """Process RFP documents using RAG pipeline - keeping existing functionality"""
        source_pdf = request.requirements.get('source_pdf')
        if not source_pdf:
            return
        
        logger = get_logger(__name__)
        logger.info(f"Processing RFP document: {source_pdf}")
        
        try:
            # Process and index the PDF using FAISS vector database
            result = self.rag_retriever.process_and_index_pdf(source_pdf, force_reindex=False)
            logger.info(f"Indexed {result['num_chunks']} chunks from RFP document")
            
            # Update request with enhanced context
            if 'rfp_context' not in request.requirements:
                request.requirements['rfp_context'] = {}
            
            request.requirements['rfp_context']['indexed_chunks'] = result['num_chunks']
            request.requirements['rfp_context']['processing_completed'] = True
            
        except Exception as e:
            logger.error(f"Failed to process RFP document: {str(e)}")
            # Continue without RFP processing - no fallback as per instructions
    
    async def _enhance_proposal(self, proposal: Dict[str, Any], request: ProposalRequest) -> Dict[str, Any]:
        """Enhance the proposal with additional processing and validation"""
        logger = get_logger(__name__)
        logger.info("Enhancing proposal with additional processing...")
        
        # Add enhanced metadata
        if 'metadata' not in proposal:
            proposal['metadata'] = {}
        proposal['metadata']['enhancement_completed'] = datetime.now().isoformat()
        proposal['metadata']['processor_version'] = "SDK-2.0"
        proposal['metadata']['sdk_agents_used'] = list(self.agents.keys())
        
        # Validate sections using existing validation
        enhanced_sections = {}
        for section_name, section_data in proposal.get('generated_sections', {}).items():
            
            # Validate section content
            validation = self.validator.validate_section(section_name, section_data)
            
            if validation['is_valid']:
                enhanced_sections[section_name] = section_data
                if isinstance(enhanced_sections[section_name], dict):
                    if 'metadata' not in enhanced_sections[section_name]:
                        enhanced_sections[section_name]['metadata'] = {}
                    enhanced_sections[section_name]['metadata']['validation'] = validation
            else:
                logger.warning(f"Section '{section_name}' failed validation: {validation.get('feedback', 'Unknown issue')}")
                enhanced_sections[section_name] = section_data
                if isinstance(enhanced_sections[section_name], dict):
                    if 'metadata' not in enhanced_sections[section_name]:
                        enhanced_sections[section_name]['metadata'] = {}
                    enhanced_sections[section_name]['metadata']['validation'] = validation
                    enhanced_sections[section_name]['metadata']['requires_review'] = True
        
        proposal['generated_sections'] = enhanced_sections
        
        # Add cost tracking information from SimpleCostTracker
        cost_summary = self.cost_tracker.get_summary()
        proposal['metadata']['cost_tracking'] = cost_summary
        
        logger.info("Proposal enhancement completed")
        return proposal

    async def _finalize_proposal(self, proposal: Dict[str, Any], request: ProposalRequest) -> Dict[str, Any]:
        """Finalize the proposal with outputs and quality scoring"""
        logger = get_logger(__name__)
        logger.info("Finalizing proposal with outputs and quality scoring...")

        # Calculate overall quality score
        sections = proposal.get('generated_sections', {})
        if sections:
            overall_score = self.validator.calculate_overall_score(sections)
            proposal['metadata']['quality_score'] = overall_score
            logger.info(f"Overall proposal quality score: {overall_score:.1f}/10")

        # Generate HTML output
        html_output = self._generate_html_output(proposal, request)
        proposal['html'] = html_output

        # SINGLE PDF generation point - check if already exists
        if 'pdf_path' not in proposal and self.config.get('output', {}).get('enable_pdf', True):
            try:
                logger.info("Generating PDF from static HTML...")
                static_html_output = self._generate_static_html_output(proposal, request)

                # Validate HTML content before PDF generation
                if len(static_html_output.strip()) < 1000:  # Less than 1KB suggests empty content
                    logger.warning("Static HTML content seems too small, checking...")
                    logger.debug(f"HTML preview: {static_html_output[:500]}")

                pdf_path = self.pdf_exporter.export(static_html_output, request.client_name)
                if pdf_path:
                    proposal['pdf_path'] = pdf_path
                    logger.info(f"PDF generated successfully: {pdf_path}")
                else:
                    logger.error("PDF generation returned None")

            except Exception as e:
                logger.error(f"Failed to generate PDF: {str(e)}")
        elif 'pdf_path' in proposal:
            logger.info(f"PDF already exists: {proposal['pdf_path']}")

        # Generate DOCX if enabled and not already present
        if self.config.get('output', {}).get('enable_docx', True) and 'docx_path' not in proposal:
            try:
                logger.info("Generating DOCX from HTML...")
                docx_path = self.docx_exporter.export(html_output, request.client_name)
                if docx_path:
                    proposal['docx_path'] = docx_path
                    logger.info(f"DOCX generated successfully: {docx_path}")
                else:
                    logger.error("DOCX generation returned None")
            except Exception as e:
                logger.error(f"Failed to generate DOCX: {str(e)}")

        proposal['metadata']['finalization_completed'] = datetime.now().isoformat()
        return proposal
    
    def _generate_html_output(self, proposal: Dict[str, Any], request: ProposalRequest) -> str:
        """Generate HTML output from proposal data - keeping existing functionality"""
        logger = get_logger(__name__)
        try:
            # Map generated_sections to sections for HTML generator compatibility
            html_proposal = proposal.copy()
            if 'generated_sections' in html_proposal:
                html_proposal['sections'] = html_proposal['generated_sections']
            
            # Add project and client info
            html_proposal['project'] = request.project_name
            html_proposal['client'] = request.client_name
            html_proposal['timeline'] = request.timeline
            
            # Ensure charts are included
            if 'charts' not in html_proposal and 'charts' in proposal:
                html_proposal['charts'] = proposal['charts']
            
            return self.html_generator.generate(html_proposal)
        except Exception as e:
            logger.error(f"Failed to generate HTML output: {str(e)}")
            return self._generate_fallback_html(proposal, request)
    
    def _generate_static_html_output(self, proposal: Dict[str, Any], request: ProposalRequest) -> str:
        """Generate static HTML output for PDF export"""
        logger = get_logger(__name__)
        try:
            # Map generated_sections to sections for HTML generator compatibility
            html_proposal = proposal.copy()
            if 'generated_sections' in html_proposal:
                html_proposal['sections'] = html_proposal['generated_sections']
            
            # Add project and client info
            html_proposal['project'] = request.project_name
            html_proposal['client'] = request.client_name
            html_proposal['timeline'] = request.timeline
            
            # Ensure charts are included
            if 'charts' not in html_proposal and 'charts' in proposal:
                html_proposal['charts'] = proposal['charts']
            
            return self.html_generator_pdf.generate(html_proposal)
        except Exception as e:
            logger.error(f"Failed to generate static HTML output: {str(e)}")
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
        """Get cost tracking summary from SimpleCostTracker"""
        return self.cost_tracker.get_summary()


async def main():
    """Main function with automatic RFP extraction"""

    # Load configuration from settings.yml
    config = load_config()

    # Setup logging
    log_manager = setup_logging(config)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("PROPOSAL GENERATOR v2.0-SDK - Starting")
    logger.info("Using OpenAI SDK Agents with Auto RFP Extraction")
    logger.info("=" * 60)

    # Check OpenAI API key
    import os
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logger.error("CRITICAL: OpenAI API key not found!")
        return 1

    # Get RFP configuration
    rfp_config = config.get('rfp', {})
    pdf_path = rfp_config.get('pdf_path')

    if not pdf_path:
        logger.error("No RFP PDF path configured in settings.yml")
        return 1

    if not Path(pdf_path).exists():
        logger.error(f"RFP PDF not found: {pdf_path}")
        return 1

    logger.info(f"Processing RFP PDF: {pdf_path}")

    # Initialize cost tracker
    cost_tracker = SimpleCostTracker()

    # Initialize generator
    try:
        generator = ProposalGenerator(cost_tracker=cost_tracker)
    except Exception as e:
        logger.error(f"Generator initialization failed: {str(e)}")
        return 1

    # Auto-extract info and create request
    try:
        request = await create_request_from_rfp(pdf_path, rfp_config, config, logger)
    except Exception as e:
        logger.error(f"Failed to process RFP: {str(e)}")
        return 1

    # Display extracted configuration
    logger.info("=" * 60)
    logger.info("AUTO-EXTRACTED RFP INFORMATION:")
    logger.info(f"Client: {request.client_name}")
    logger.info(f"Project: {request.project_name}")
    logger.info(f"Type: {request.project_type}")
    logger.info(f"Timeline: {request.timeline}")
    logger.info(f"Budget: {request.budget_range}")
    logger.info("=" * 60)

    # Generate proposal
    logger.info("Starting proposal generation...")
    start_time = datetime.now()

    try:
        proposal = await generator.generate_proposal(request)

        # Rest of your existing code for saving and displaying results...
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Save outputs (keep existing code)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output_dir = Path(config['output']['output_directory'])
        artifacts_dir = base_output_dir / config['output'].get('artifacts_directory', 'artifacts')

        base_output_dir.mkdir(exist_ok=True)
        artifacts_dir.mkdir(exist_ok=True)

        clean_client = request.client_name.replace(' ', '_').replace('/', '_')

        json_file = artifacts_dir / f"proposal_{clean_client}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(proposal, f, indent=2, default=str)
        logger.info(f"JSON saved to: {json_file}")

        # Save HTML in artifacts folder
        html_file = artifacts_dir / f"proposal_{clean_client}_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(proposal.get('html', ''))
        logger.info(f"HTML saved to: {html_file}")

        if 'pdf_path' in proposal:
            logger.info(f"PDF saved to: {proposal['pdf_path']}")

        logger.info("=" * 60)
        logger.info("PROPOSAL GENERATION COMPLETED SUCCESSFULLY")
        logger.info(f"Generation Time: {duration:.2f} seconds")

        # Display results (keep existing code)
        if 'metadata' in proposal and 'quality_score' in proposal['metadata']:
            score = proposal['metadata']['quality_score']
            logger.info(f"Quality Score: {score:.1f}/10")

        cost_summary = generator.get_cost_summary()
        logger.info(f"Total Cost: ${cost_summary['total_cost']:.4f}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error generating proposal: {str(e)}")
        return 1

    return 0


async def create_request_from_rfp(pdf_path: str, rfp_config: dict, config: dict, logger) -> ProposalRequest:
    """Create ProposalRequest with auto-extracted information"""

    # Initialize RFP extractor
    extractor = RFPExtractor(config)

    # Extract information from PDF
    logger.info("Auto-extracting client information from RFP...")
    extracted_info = extractor.extract_rfp_info(pdf_path)

    logger.info("Extracted information:")
    for key, value in extracted_info.items():
        logger.info(f"  - {key}: {value}")

    # Process PDF for RAG (keep existing functionality)
    logger.info("Processing PDF for RAG retrieval...")
    rag_retriever = RAGRetriever(config)
    result = rag_retriever.process_and_index_pdf(pdf_path, force_reindex=False)
    logger.info(f"Indexed {result['num_chunks']} chunks from PDF")

    # Extract requirements using RAG
    queries = {
        'technical': "What are the technical requirements?",
        'scope': "What is the project scope?",
        'deliverables': "What are the deliverables?",
    }

    requirements = {'source_pdf': pdf_path}
    for key, query in queries.items():
        chunks = rag_retriever.retrieve(query, pdf_path)
        if chunks:
            context = "\n".join([chunk['text'][:300] for chunk in chunks[:2]])
            requirements[key] = context

    # Create request with extracted info
    request = ProposalRequest(
        client_name=extracted_info.get('client_name') or 'Unknown Client',
        project_name=extracted_info.get('project_name') or 'RFP Project',
        project_type=extracted_info.get('project_type', 'general'),
        requirements=requirements,
        timeline=extracted_info.get('timeline', 'As per RFP requirements'),
        budget_range=extracted_info.get('budget_range', 'As per RFP specifications'),
        special_requirements=rfp_config.get('special_requirements', [])
    )

    return request


if __name__ == "__main__":
    print("\nProposal Generator - Auto RFP Processing...")
    print("Configuration: config/settings.yml")
    print("PDF path will be read from config")
    print("-" * 40)


    exit_code = asyncio.run(main())  # No hardcoded parameters

    if exit_code == 0:
        print("\n✅ Proposal generation completed successfully!")
    else:
        print("\n❌ Proposal generation failed. Check logs for details.")

    exit(exit_code)