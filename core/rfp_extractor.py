"""
Simple RFP Information Extractor using OpenAI
"""

import json
from typing import Dict, Optional
from openai import OpenAI
from core.document_processor import DocumentProcessor
from core.simple_cost_tracker import SimpleCostTracker
import logging

logger = logging.getLogger(__name__)


class RFPExtractor:
    """Extract structured information from RFP documents using OpenAI"""

    def __init__(self, config: Dict, cost_tracker: Optional[SimpleCostTracker] = None):
        self.config = config
        self.openai_client = OpenAI()
        self.document_processor = DocumentProcessor()
        self.cost_tracker = cost_tracker

    def extract_rfp_info(self, pdf_path: str) -> Dict[str, str]:
        """Extract client information from RFP PDF"""
        logger.info(f"Extracting RFP information from {pdf_path}")

        # Process PDF and get text content
        chunks = self.document_processor.process_pdf(pdf_path)
        if not chunks:
            logger.warning("No content extracted from PDF")
            return self._get_default_info()

        # Combine more chunks for better extraction (use more content)
        combined_text = "\n".join([chunk.text for chunk in chunks[:5]])[:10000]

        # Extract using OpenAI
        try:
            model_used = self.config.get('openai', {}).get('model', 'gpt-4o')
            logger.info(f"Calling OpenAI API with model: {model_used}")
            logger.info(f"Content length to extract: {len(combined_text)} characters")

            response = self.openai_client.chat.completions.create(
                model=model_used,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert RFP information extractor fluent in Arabic and English. Extract specific information from this RFP document. "
                            "CRITICAL: This document may be in Arabic - read carefully and extract actual names and titles. "
                            "DO NOT use generic terms like 'Unknown Client' or 'RFP Project'. "
                            "Return ONLY valid JSON with these keys:\n"
                            "- client_name: Extract the actual organization name (look for company names like 'NHC', 'الشركة الوطنية للإسكان', 'National Housing Company', etc.)\n"
                            "- project_name: Extract the specific project title (look for project descriptions like 'تتبع وتحليل سلوك المستخدمين', 'User Behavior Tracking', etc.)\n"
                            "- project_description: Brief summary in English of what the project is about\n"
                            "- project_type: Choose from web_development, mobile_app, data_analytics, healthcare, ai_solution, consulting, infrastructure, general\n"
                            "- timeline: Extract specific timeline (look for '12 شهر', '6 أسابيع', months, weeks, etc.)\n"
                            "- budget_range: Extract any budget information mentioned\n"
                            "- key_requirements: Array of main technical requirements in English\n"
                            "- location: Extract location info (look for 'Riyadh', 'الرياض', 'KSA', etc.)\n"
                            "- language: 'ar' for Arabic documents, 'en' for English\n"
                            "- mau_info: Extract MAU (Monthly Active Users) numbers if mentioned\n"
                            "EXAMPLES of what to look for:\n"
                            "- Company: 'NHC', 'الشركة الوطنية للإسكان' → client_name: 'National Housing Company (NHC)'\n"
                            "- Project: 'مشروع تتبع وتحليل سلوك المستخدمين' → project_name: 'User Behavior Tracking and Analysis Platform'\n"
                            "- Timeline: '12 شهر' → timeline: '12 months implementation duration'\n"
                            "Search thoroughly in headers, company logos, contact information, and project titles."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Extract information from this RFP:\n\n{combined_text}"
                    }
                ],
                temperature=0.1,
                max_completion_tokens=500,
                response_format={"type": "json_object"}
            )

            # Track cost if tracker available
            if self.cost_tracker:
                self.cost_tracker.track_completion(response, model=model_used)

            raw_response = response.choices[0].message.content
            logger.info(f"Raw OpenAI response: {raw_response}")

            extracted_info = json.loads(raw_response)
            logger.info(f"Parsed extracted info: {extracted_info}")

            # Validate the extraction - check if we got actual names
            client_name = extracted_info.get('client_name', '').strip()
            project_name = extracted_info.get('project_name', '').strip()

            if not client_name or not project_name:
                logger.warning(f"Extraction returned empty names - client: '{client_name}', project: '{project_name}'")
                logger.warning("This indicates the extraction failed to find proper names in the document")
            else:
                logger.info(f"✅ Successfully extracted - Client: '{client_name}', Project: '{project_name}'")

            return extracted_info

        except Exception as e:
            logger.error(f"Extraction failed with error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_default_info()

    def _get_default_info(self) -> Dict[str, str]:
        """Default values when extraction fails"""
        return {
            "client_name": "",
            "project_name": "",
            "project_type": "general",
            "timeline": "As per RFP requirements",
            "budget_range": "As per RFP specifications",
            "evaluation_criteria": [],
            "required_documents": [],
            "team_requirements": [],
            "language": "en",
            "data_residency": "",
            "sla_requirements": "",
            "financial_proposal_required": False
        }