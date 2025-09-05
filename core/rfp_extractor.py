"""
Simple RFP Information Extractor using OpenAI
"""

import json
from typing import Dict
from openai import OpenAI
from core.document_processor import DocumentProcessor
import logging

logger = logging.getLogger(__name__)


class RFPExtractor:
    """Extract structured information from RFP documents using OpenAI"""

    def __init__(self, config: Dict):
        self.config = config
        self.openai_client = OpenAI()
        self.document_processor = DocumentProcessor()

    def extract_rfp_info(self, pdf_path: str) -> Dict[str, str]:
        """Extract client information from RFP PDF"""
        logger.info(f"Extracting RFP information from {pdf_path}")

        # Process PDF and get text content
        chunks = self.document_processor.process_pdf(pdf_path)
        if not chunks:
            logger.warning("No content extracted from PDF")
            return self._get_default_info()

        # Combine first few chunks (limit tokens)
        combined_text = "\n".join([chunk.text for chunk in chunks[:3]])[:6000]

        # Extract using OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an RFP information extractor. Return ONLY valid JSON. "
                            "Keys (all optional but include if present): "
                            "client_name, project_name, project_type, timeline, budget_range, "
                            "evaluation_criteria (array of {name, weight_percent}), "
                            "required_documents (array of strings), team_requirements (array of strings), "
                            "language (one of: en, ar, hi, fr, es), data_residency, sla_requirements, "
                            "financial_proposal_required (boolean). "
                            "For project_type choose from: web_development, mobile_app, data_analytics, healthcare, ai_solution, consulting, infrastructure, general. "
                            "Use empty string or sensible defaults if not found."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Extract information from this RFP:\n\n{combined_text}"
                    }
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            extracted_info = json.loads(response.choices[0].message.content)
            logger.info(f"Extracted: {extracted_info}")
            return extracted_info

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
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