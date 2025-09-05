"""
RFP PDF Parser - Extracts requirements from RFP documents
"""

import re
import json
from typing import Dict, List, Any, Optional
import logging
import PyPDF2
import pdfplumber
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RFPParser:
    """Parse RFP PDF documents and extract requirements"""
    
    def __init__(self):
        self.requirement_patterns = {
            'timeline': [
                r'timeline[:\s]+([^\.]+)',
                r'duration[:\s]+([^\.]+)',
                r'completion[:\s]+([^\.]+)',
                r'deadline[:\s]+([^\.]+)',
                r'(?:must|should|shall) be completed? (?:by|within) ([^\.]+)',
                r'(\d+)\s*(?:weeks?|months?|days?)',
            ],
            'budget': [
                r'budget[:\s]+\$?([0-9,]+)',
                r'cost[:\s]+\$?([0-9,]+)',
                r'not to exceed[:\s]+\$?([0-9,]+)',
                r'maximum[:\s]+\$?([0-9,]+)',
                r'\$([0-9,]+)\s*(?:to|-)\s*\$([0-9,]+)',
            ],
            'deliverables': [
                r'deliverables?[:\s]+([^\.]+)',
                r'outputs?[:\s]+([^\.]+)',
                r'(?:must|shall|will) (?:deliver|provide)[:\s]+([^\.]+)',
            ],
            'scope': [
                r'scope[:\s]+([^\.]+)',
                r'objectives?[:\s]+([^\.]+)',
                r'requirements?[:\s]+([^\.]+)',
            ],
            'technology': [
                r'(?:technology|technical) stack[:\s]+([^\.]+)',
                r'(?:must use|requires?|using)[:\s]+([\w\s,]+)',
                r'platform[:\s]+([^\.]+)',
            ]
        }
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse RFP PDF and extract structured requirements
        
        Args:
            pdf_path: Path to the RFP PDF file
            
        Returns:
            Dictionary with extracted requirements
        """
        try:
            logger.info(f"Parsing RFP PDF: {pdf_path}")
            
            # Extract text using multiple methods for better accuracy
            text = self._extract_text_pdfplumber(pdf_path)
            if not text:
                text = self._extract_text_pypdf2(pdf_path)
            
            if not text:
                raise ValueError("Could not extract text from PDF")
            
            # Parse the extracted text
            requirements = self._parse_requirements(text)
            
            # Extract specific sections
            sections = self._extract_sections(text)
            
            # Combine all extracted information
            result = {
                "source_file": pdf_path,
                "extracted_at": datetime.now().isoformat(),
                "requirements": requirements,
                "sections": sections,
                "raw_text": text[:5000],  # First 5000 chars for reference
                "metadata": self._extract_metadata(text)
            }
            
            logger.info(f"Successfully parsed RFP with {len(requirements)} requirement categories")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing RFP PDF: {str(e)}")
            raise
    
    def _extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for tables)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Also extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
            
            return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    def _extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
            return ""
    
    def _parse_requirements(self, text: str) -> Dict[str, Any]:
        """Parse requirements from text using patterns"""
        
        requirements = {}
        text_lower = text.lower()
        
        # Extract timeline
        timeline_info = self._extract_pattern_matches(text_lower, self.requirement_patterns['timeline'])
        if timeline_info:
            requirements['timeline'] = self._normalize_timeline(timeline_info[0])
        
        # Extract budget
        budget_info = self._extract_pattern_matches(text, self.requirement_patterns['budget'])
        if budget_info:
            requirements['budget'] = self._normalize_budget(budget_info[0])
        
        # Extract deliverables
        deliverables_info = self._extract_pattern_matches(text, self.requirement_patterns['deliverables'])
        if deliverables_info:
            requirements['deliverables'] = self._parse_list_items(deliverables_info[0])
        
        # Extract scope
        scope_info = self._extract_pattern_matches(text, self.requirement_patterns['scope'])
        if scope_info:
            requirements['scope'] = scope_info[0].strip()
        
        # Extract technology requirements
        tech_info = self._extract_pattern_matches(text, self.requirement_patterns['technology'])
        if tech_info:
            requirements['technology'] = self._parse_technologies(tech_info[0])
        
        # Extract non-functional requirements
        requirements['non_functional'] = self._extract_non_functional_requirements(text)
        
        # Extract evaluation criteria
        requirements['evaluation_criteria'] = self._extract_evaluation_criteria(text)
        
        # Extract submission requirements
        requirements['submission'] = self._extract_submission_requirements(text)
        
        return requirements
    
    def _extract_pattern_matches(self, text: str, patterns: List[str]) -> List[str]:
        """Extract matches for a list of patterns"""
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            matches.extend(found)
        return matches
    
    def _normalize_timeline(self, timeline_str: str) -> Dict[str, Any]:
        """Normalize timeline information"""
        timeline = {
            "raw": timeline_str,
            "duration": None,
            "unit": None,
            "phases": []
        }
        
        # Extract duration
        duration_match = re.search(r'(\d+)\s*(weeks?|months?|days?)', timeline_str, re.IGNORECASE)
        if duration_match:
            timeline['duration'] = int(duration_match.group(1))
            timeline['unit'] = duration_match.group(2).rstrip('s')
        
        # Look for phases
        phase_pattern = r'(?:phase|stage)\s*(\d+)[:\s]+([^,\.]+)'
        phases = re.findall(phase_pattern, timeline_str, re.IGNORECASE)
        if phases:
            timeline['phases'] = [{"number": int(p[0]), "description": p[1].strip()} for p in phases]
        
        return timeline
    
    def _normalize_budget(self, budget_str: str) -> Dict[str, Any]:
        """Normalize budget information"""
        budget = {
            "raw": budget_str,
            "min": None,
            "max": None,
            "currency": "USD"
        }
        
        # Extract numbers
        numbers = re.findall(r'\$?([0-9,]+)', budget_str)
        if numbers:
            amounts = [int(n.replace(',', '')) for n in numbers]
            if len(amounts) == 1:
                budget['max'] = amounts[0]
            elif len(amounts) >= 2:
                budget['min'] = min(amounts)
                budget['max'] = max(amounts)
        
        return budget
    
    def _parse_list_items(self, text: str) -> List[str]:
        """Parse list items from text"""
        items = []
        
        # Split by common delimiters
        for delimiter in [',', ';', '\n', '•', '-', '*']:
            if delimiter in text:
                parts = text.split(delimiter)
                items.extend([p.strip() for p in parts if p.strip()])
                break
        
        if not items:
            items = [text.strip()]
        
        return items
    
    def _parse_technologies(self, tech_str: str) -> List[str]:
        """Parse technology requirements"""
        technologies = []
        
        # Common technology keywords to look for
        tech_keywords = [
            'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes',
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'ci/cd', 'devops', 'agile', 'scrum', 'microservices'
        ]
        
        tech_str_lower = tech_str.lower()
        for keyword in tech_keywords:
            if keyword in tech_str_lower:
                technologies.append(keyword.upper() if len(keyword) <= 4 else keyword.capitalize())
        
        return technologies
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract major sections from the RFP"""
        sections = {}
        
        # Common RFP section headers
        section_headers = [
            'background', 'introduction', 'overview',
            'scope of work', 'scope', 'objectives',
            'requirements', 'technical requirements', 'functional requirements',
            'deliverables', 'timeline', 'schedule',
            'budget', 'cost', 'pricing',
            'evaluation criteria', 'selection criteria',
            'submission', 'proposal submission', 'instructions'
        ]
        
        text_lines = text.split('\n')
        
        for i, line in enumerate(text_lines):
            line_lower = line.lower().strip()
            for header in section_headers:
                if header in line_lower and len(line) < 100:  # Likely a header
                    # Extract content until next section
                    section_content = []
                    for j in range(i+1, min(i+50, len(text_lines))):  # Look ahead up to 50 lines
                        next_line = text_lines[j]
                        # Check if we hit another section
                        if any(h in next_line.lower() for h in section_headers):
                            break
                        section_content.append(next_line)
                    
                    sections[header] = '\n'.join(section_content[:20])  # Store first 20 lines
                    break
        
        return sections
    
    def _extract_non_functional_requirements(self, text: str) -> Dict[str, Any]:
        """Extract non-functional requirements"""
        nfr = {}
        
        # Availability
        availability_match = re.search(r'availability[:\s]+([0-9\.]+)%?', text, re.IGNORECASE)
        if availability_match:
            nfr['availability'] = f"{availability_match.group(1)}%"
        
        # Performance
        performance_match = re.search(r'(?:response time|latency)[:\s]+([^\.]+)', text, re.IGNORECASE)
        if performance_match:
            nfr['performance'] = performance_match.group(1).strip()
        
        # Security
        security_keywords = ['security', 'encryption', 'compliance', 'gdpr', 'iso', 'soc']
        security_items = []
        for keyword in security_keywords:
            if keyword in text.lower():
                security_items.append(keyword.upper())
        if security_items:
            nfr['security'] = security_items
        
        # Scalability
        if 'scalab' in text.lower():
            nfr['scalability'] = "Required"
        
        # Region/Location
        region_match = re.search(r'region[:\s]+([a-z\-0-9]+)', text, re.IGNORECASE)
        if region_match:
            nfr['region'] = region_match.group(1)
        
        return nfr
    
    def _extract_evaluation_criteria(self, text: str) -> List[Dict[str, Any]]:
        """Extract evaluation criteria"""
        criteria = []
        
        # Look for percentage-based criteria
        percentage_pattern = r'([^\.]+?)\s*[\(:\-]\s*([0-9]+)%'
        matches = re.findall(percentage_pattern, text)
        
        for match in matches:
            criteria.append({
                "criterion": match[0].strip(),
                "weight": f"{match[1]}%"
            })
        
        # Also look for ordered criteria
        if not criteria:
            criteria_keywords = ['technical', 'cost', 'experience', 'timeline', 'quality']
            for keyword in criteria_keywords:
                if keyword in text.lower():
                    criteria.append({"criterion": keyword.capitalize(), "weight": "TBD"})
        
        return criteria
    
    def _extract_submission_requirements(self, text: str) -> Dict[str, Any]:
        """Extract submission requirements"""
        submission = {}
        
        # Deadline
        deadline_pattern = r'(?:deadline|due date|submission date)[:\s]+([^\.]+)'
        deadline_match = re.search(deadline_pattern, text, re.IGNORECASE)
        if deadline_match:
            submission['deadline'] = deadline_match.group(1).strip()
        
        # Format requirements
        if 'pdf' in text.lower():
            submission['format'] = 'PDF'
        elif 'word' in text.lower() or 'doc' in text.lower():
            submission['format'] = 'Word'
        
        # Page limit
        page_limit_match = re.search(r'(\d+)\s*pages?', text, re.IGNORECASE)
        if page_limit_match:
            submission['page_limit'] = int(page_limit_match.group(1))
        
        return submission
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from the RFP"""
        metadata = {}
        
        # RFP number/reference
        rfp_pattern = r'(?:rfp|request for proposal|reference)[#:\s]+([a-zA-Z0-9\-]+)'
        rfp_match = re.search(rfp_pattern, text, re.IGNORECASE)
        if rfp_match:
            metadata['rfp_reference'] = rfp_match.group(1)
        
        # Organization name
        org_pattern = r'(?:issued by|organization|company)[:\s]+([^\.]+)'
        org_match = re.search(org_pattern, text, re.IGNORECASE)
        if org_match:
            metadata['organization'] = org_match.group(1).strip()
        
        # Contact information
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        if emails:
            metadata['contact_emails'] = emails[:3]  # First 3 emails
        
        return metadata

    def generate_requirements_summary(self, parsed_data: Dict) -> Dict[str, Any]:
        """
        Generate a structured requirements summary for proposal generation
        
        Args:
            parsed_data: Output from parse_pdf
            
        Returns:
            Structured requirements for ProposalRequest
        """
        requirements = parsed_data.get('requirements', {})
        
        summary = {
            "project_type": self._determine_project_type(parsed_data),
            "timeline": self._format_timeline(requirements.get('timeline', {})),
            "budget_range": self._format_budget(requirements.get('budget', {})),
            "requirements": {
                "scope": requirements.get('scope', ''),
                "deliverables": requirements.get('deliverables', []),
                "technology": requirements.get('technology', []),
                "non_functional": requirements.get('non_functional', {}),
            },
            "evaluation_criteria": requirements.get('evaluation_criteria', []),
            "submission": requirements.get('submission', {}),
            "metadata": parsed_data.get('metadata', {})
        }
        
        return summary
    
    def _determine_project_type(self, parsed_data: Dict) -> str:
        """Determine project type from parsed data"""
        text = str(parsed_data).lower()
        
        if 'cloud' in text and 'migration' in text:
            return 'cloud_migration'
        elif 'web' in text or 'website' in text:
            return 'web_development'
        elif 'mobile' in text or 'app' in text:
            return 'mobile_app'
        elif 'data' in text or 'analytics' in text:
            return 'data_analytics'
        elif 'devops' in text or 'ci/cd' in text:
            return 'devops'
        else:
            return 'custom_development'
    
    def _format_timeline(self, timeline: Dict) -> str:
        """Format timeline for display"""
        if not timeline:
            return "TBD"
        
        if timeline.get('duration') and timeline.get('unit'):
            return f"{timeline['duration']} {timeline['unit']}"
        
        return timeline.get('raw', 'TBD')
    
    def _format_budget(self, budget: Dict) -> str:
        """Format budget for display"""
        if not budget:
            return "TBD"
        
        if budget.get('min') and budget.get('max'):
            return f"${budget['min']:,} - ${budget['max']:,}"
        elif budget.get('max'):
            return f"Up to ${budget['max']:,}"
        
        return budget.get('raw', 'TBD')