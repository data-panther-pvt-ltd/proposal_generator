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
                logger.warning(
                    f"Section '{section_name}' failed validation: {validation.get('feedback', 'Unknown issue')}")
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
        """Finalize the proposal with outputs and quality scoring - keeping existing functionality"""
        logger = get_logger(__name__)
        logger.info("Finalizing proposal with outputs and quality scoring...")

        # Calculate overall quality score using existing evaluation system
        sections = proposal.get('generated_sections', {})
        if sections:
            overall_score = self.validator.calculate_overall_score(sections)
            proposal['metadata']['quality_score'] = overall_score
            logger.info(f"Overall proposal quality score: {overall_score:.1f}/10")

        # Generate HTML output using existing HTML generator (interactive)
        html_output = self._generate_html_output(proposal, request)
        proposal['html'] = html_output

        # Generate PDF if enabled using existing PDF exporter
        if self.config.get('output', {}).get('enable_pdf', True):
            try:
                # Generate static HTML for PDF export
                static_html_output = self._generate_static_html_output(proposal, request)
                pdf_path = self.pdf_exporter.export(static_html_output, request.client_name)
                proposal['pdf_path'] = pdf_path
                logger.info(f"PDF generated: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to generate PDF: {str(e)}")

        # Add completion timestamp
        proposal['metadata']['finalization_completed'] = datetime.now().isoformat()

        logger.info("Proposal finalization completed")
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


async def main(
        pdf_path: str,
        client_name: str,
        project_name: str,
        project_type: str
):
    """Main function to run the proposal generator with full SDK integration"""

    # Load config
    config = load_config()

    # Setup logging
    log_manager = setup_logging(config)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("PROPOSAL GENERATOR v2.0-SDK - Starting")
    logger.info("=" * 60)

    # Validate API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logger.error("CRITICAL: OpenAI API key not found!")
        return 1

    # Process RFP PDF
    try:
        request = await process_rfp_pdf(
            pdf_path=pdf_path,
            config=config,
            logger=logger,
            client_name=client_name,
            project_name=project_name,
            project_type=project_type
        )
    except Exception as e:
        logger.error(f"Failed to process RFP PDF: {str(e)}", exc_info=True)
        return 1

    # Initialize generator
    cost_tracker = SimpleCostTracker()
    generator = ProposalGenerator(cost_tracker=cost_tracker)

    logger.info("=" * 60)
    logger.info(f"Client: {client_name}")
    logger.info(f"Project: {project_name}")
    logger.info(f"Type: {project_type}")
    logger.info(f"Timeline: {request.timeline}")
    logger.info(f"Budget Range: {request.budget_range}")
    logger.info("=" * 60)

    # Generate proposal
    start_time = datetime.now()
    proposal = await generator.generate_proposal(request)
    duration = (datetime.now() - start_time).total_seconds()

    logger.info(f"Proposal generated in {duration:.2f} seconds")
    return 0


async def process_rfp_pdf(
        pdf_path: str,
        config: dict,
        logger,
        client_name: str,
        project_name: str,
        project_type: str
) -> ProposalRequest:
    """Process RFP PDF using RAG pipeline and create proposal request"""

    logger.info(f"Processing RFP PDF with RAG Pipeline: {pdf_path}")
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"RFP PDF not found: {pdf_path}")

    # RAG retriever
    rag_retriever = RAGRetriever(config)
    rag_retriever.process_and_index_pdf(pdf_path, force_reindex=False)

    # Extract with RAG
    extracted_info = {}
    queries = {
        'requirements': "What are all the technical and functional requirements?",
        'timeline': "What is the project timeline, deadline, and key milestones?",
        'budget': "What is the budget range or cost constraints?",
        'deliverables': "What are the key deliverables and outputs?",
        'evaluation': "What are the evaluation criteria for proposals?",
        'submission': "What are the submission requirements and guidelines?",
        'organization': "What is the client organization and project background?",
        'scope': "What is the project scope and objectives?"
    }

    for key, query in queries.items():
        chunks = rag_retriever.retrieve(query, pdf_path)
        extracted_info[key] = "\n".join([chunk['text'][:500] for chunk in chunks[:3]]) if chunks else ""

    # Build ProposalRequest directly
    return ProposalRequest(
        client_name=client_name,
        project_name=project_name,
        project_type=project_type,
        requirements={
            'technical': extracted_info.get('requirements', ''),
            'scope': extracted_info.get('scope', ''),
            'deliverables': extracted_info.get('deliverables', ''),
            'evaluation_criteria': extracted_info.get('evaluation', ''),
            'submission_guidelines': extracted_info.get('submission', ''),
            'source_pdf': pdf_path,
            'rfp_context': extracted_info
        },
        timeline=extracted_info.get('timeline', "As per RFP"),
        budget_range=extracted_info.get('budget', "As per RFP"),
        special_requirements=[
            "Address all RFP requirements",
            "Include compliance matrix",
            "Match evaluation criteria"
        ]
    )

    # Save RAG extraction results for reference
    extraction_dir = Path('rag_extractions')
    extraction_dir.mkdir(exist_ok=True)
    extraction_file = extraction_dir / f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    extraction_data = {
        'pdf_path': str(pdf_path),
        'timestamp': datetime.now().isoformat(),
        'chunks_indexed': result['num_chunks'],
        'extracted_info': extracted_info,
        'vector_db_stats': rag_retriever.get_stats()
    }

    with open(extraction_file, 'w') as f:
        json.dump(extraction_data, f, indent=2, default=str)
    logger.info(f"RAG extraction data saved to: {extraction_file}")

    return request


if __name__ == "__main__":
    print("\nProposal Generator - Starting...")
    exit_code = asyncio.run(main(
        pdf_path="RFP_test.pdf",
        client_name="Govt of India",
        project_name="",
        project_type=""
    ))
    exit(exit_code)
