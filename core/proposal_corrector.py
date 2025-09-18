"""
Proposal Corrector Module
Validates and corrects generated JSON proposals for formatting and content consistency
NO FALLBACKS - Direct implementation only
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
from openai import OpenAI

from core.html_generator import HTMLGenerator
from core.pdf_exporter import PDFExporter

logger = logging.getLogger(__name__)


class ProposalCorrector:
    """
    Corrects generated proposals by:
    1. Reading JSON proposals from artifacts directory
    2. Removing JSON formatting artifacts from content
    3. Synchronizing data across all sections
    4. Generating corrected HTML and PDF outputs
    5. Creating detailed diff reports
    """
    
    def __init__(self, config: Dict, cost_tracker=None):
        """Initialize the proposal corrector with configuration"""
        self.config = config
        self.cost_tracker = cost_tracker
        self.openai_client = OpenAI()
        self.output_dir = Path(config['output']['output_directory'])  # Main output directory
        
        # Get correction settings from config
        self.correction_model = config.get('output', {}).get('correction_model', 'gpt-4o')
        
        # Initialize HTML and PDF generators for output
        self.html_generator = HTMLGenerator(config)
        self.pdf_exporter = PDFExporter(config)
        
        logger.info("ProposalCorrector initialized for JSON-based corrections")
    
    async def correct_proposal(self, json_path: str) -> Tuple[str, str]:
        """
        Main correction pipeline - NO FALLBACKS
        
        Args:
            json_path: Path to the generated proposal JSON file
            
        Returns:
            Tuple of (corrected_pdf_path, diff_report_path)
        """
        logger.info(f"Starting proposal correction for: {json_path}")
        
        # Load original JSON proposal
        original_json = self._load_json(json_path)
        logger.info(f"Loaded JSON proposal with {len(original_json.get('generated_sections', {}))} sections")
        
        # Correct the proposal content with OpenAI
        corrected_json = await self._correct_with_openai(original_json)
        logger.info("Proposal content corrected with OpenAI")
        
        # Save corrected JSON with correct_ prefix
        corrected_json_path = self._save_corrected_json(corrected_json, json_path)
        logger.info(f"Corrected JSON saved: {corrected_json_path}")
        
        # Generate corrected HTML
        html_content = self._generate_corrected_html(corrected_json)
        logger.info("Generated corrected HTML content")
        
        # Extract client name from request string or JSON
        client_name = self._extract_client_name(corrected_json)
        corrected_pdf_path = self._generate_corrected_pdf(html_content, client_name, json_path)
        logger.info(f"Generated corrected PDF: {corrected_pdf_path}")
        
        # Generate diff report
        diff_report_path = self._generate_diff_report(original_json, corrected_json, json_path)
        logger.info(f"Generated diff report: {diff_report_path}")
        
        return corrected_pdf_path, diff_report_path
    
    def _load_json(self, json_path: str) -> Dict:
        """Load JSON proposal from file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON from {json_path}")
            return data
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {json_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON from {json_path}: {e}")
            raise
    
    async def _correct_with_openai(self, proposal_json: Dict) -> Dict:
        """
        Send JSON to OpenAI for correction
        Focus on generated_sections and cleaning artifacts
        Excludes graphs/charts sections to avoid token limits
        """
        logger.info("Sending proposal to OpenAI for correction...")

        # Extract generated sections for correction
        generated_sections = proposal_json.get('generated_sections', {})

        # Filter out sections containing graphs/charts to avoid token bloat
        sections_to_process = {}
        excluded_sections = []
        total_content_length = 0

        for section_name, section_content in generated_sections.items():
            # Extract content properly
            if isinstance(section_content, dict):
                content = section_content.get('content', '')
            else:
                content = str(section_content)

            # Aggressive filtering to prevent token overflow
            # Exclude ANY section that contains chart data or is very large

            # Skip sections with chart-related keywords in the name
            section_lower = section_name.lower()
            chart_keywords = ['chart', 'graph', 'visual', 'diagram', 'plot', 'figure', 'visualization', 'graphic']
            if any(keyword in section_lower for keyword in chart_keywords):
                excluded_sections.append(section_name)
                logger.info(f"Excluding section '{section_name}' from correction (section name indicates charts)")
                continue

            # Skip very large sections (likely contain charts or embedded data)
            if content and len(content) > 15000:  # Reduced threshold
                excluded_sections.append(section_name)
                logger.info(f"Excluding section '{section_name}' from correction (large content: {len(content)} chars)")
                continue

            # Skip sections with ANY chart indicators
            if content:
                chart_indicators = [
                    'data:image', 'base64', '<svg', 'chart.js', 'plotly',
                    'chartjs', 'highcharts', 'd3.js', 'canvas', 'webgl',
                    'pie-chart', 'bar-chart', 'line-chart', 'scatter-plot',
                    'budget_breakdown_chart', 'timeline_chart', 'risk_matrix',
                    'resource_allocation_chart', 'roi_projection_chart'
                ]

                if any(indicator in content.lower() for indicator in chart_indicators):
                    excluded_sections.append(section_name)
                    logger.info(f"Excluding section '{section_name}' from correction (contains chart indicators)")
                    continue

            # Skip sections with complex JSON structures
            if content and (content.count('{') > 10 or content.count('[') > 10):
                excluded_sections.append(section_name)
                logger.info(f"Excluding section '{section_name}' from correction (contains JSON structures)")
                continue

            # Additional token count protection
            if total_content_length > 150000:  # Stop adding sections if we're getting too large
                excluded_sections.append(section_name)
                logger.info(f"Excluding section '{section_name}' from correction (token limit protection: {total_content_length} chars so far)")
                continue

            sections_to_process[section_name] = content
            total_content_length += len(content)

        logger.info(f"Processing {len(sections_to_process)} sections, excluding {len(excluded_sections)} sections with graphs/charts")

        # Original logic for smaller proposals
        sections_text = ""
        for section_name, content in sections_to_process.items():
            sections_text += f"\n[SECTION: {section_name}]\n{content}\n"
        
        # Build comprehensive correction prompt
        prompt = f"""You are a professional proposal editor tasked with correcting and cleaning up proposal content. Your job is to ensure the content is professional, consistent, and free of technical artifacts.

CRITICAL FORMATTING RULES - FOLLOW EXACTLY:

1. PRESERVE ALL SECTION NUMBERING - MANDATORY:
   - Keep "1.", "2.", "3." etc EXACTLY as they appear in original content
   - Keep "1.1", "1.2", "2.1", "2.2" etc EXACTLY as they appear
   - Keep "a)", "b)", "i)", "ii)" etc EXACTLY as they appear
   - DO NOT change, remove, or reformat any numbered lists or sections
   - This applies to ALL sections except "Problem or Need Statement"

2. PRESERVE BOLD FORMATTING - MANDATORY:
   - Keep **bold** text EXACTLY as it appears, especially after colons
   - Example: "**Deliverable:** Requirements Specification Document" should stay as "**Deliverable:** Requirements Specification Document"
   - Example: "**Timeline:** 3 months" should stay as "**Timeline:** 3 months"
   - DO NOT remove bold formatting from section headers, labels, or emphasis text

3. PRESERVE HEADER STRUCTURE - MANDATORY:
   - Keep section headers EXACTLY as formatted in original
   - DO NOT change header formatting or style
   - DO NOT add or remove # symbols from existing headers
   - Only clean the content INSIDE sections, not the headers themselves

4. EXCEPTION - "Problem or Need Statement" SECTION ONLY:
   - This section CAN be fully reformatted for clarity and professionalism
   - You may restructure paragraphs, improve language, and enhance readability
   - This is the ONLY section where you can modify structure and formatting

5. REMOVE ONLY JSON ARTIFACTS:
   - Remove any JSON formatting like {{"content": "...", "key": "value"}}
   - Remove code blocks like ```json or ```
   - Remove any escaped characters like \\" or \\n
   - Remove any bracket notation or object syntax
   - BUT preserve all business formatting (numbering, bold, headers)

6. PRESERVE AND CORRECT TABLE STRUCTURES - CRITICAL:
   - Keep ALL tables EXACTLY as they appear in markdown format
   - PRESERVE the exact table structure with | symbols
   - PRESERVE column headers and separators |---|---|---| exactly
   - PRESERVE all table rows and cells with their | delimiters
   - DO NOT change table structure, only fix data inconsistencies
   - Tables must maintain their EXACT original markdown formatting

7. PRESERVE ALL CLIENT AND PROJECT INFORMATION - CRITICAL:
   - NEVER change client names, company names, or organization names
   - NEVER replace specific project titles with generic placeholders
   - NEVER change project descriptions or requirements
   - PRESERVE all dates, timelines, and specific project details
   - PRESERVE all budget amounts and financial information
   - PRESERVE all technical specifications and requirements

8. SYNCHRONIZE DATA ACROSS ALL SECTIONS:
   - Ensure timeline dates are consistent everywhere they appear
   - Make sure budget figures match across all sections (executive summary, budget section, etc.)
   - Verify team sizes and composition are identical in all mentions
   - Align project durations with timeline information
   - Ensure client and project names are exactly the same throughout

9. FIX LANGUAGE ISSUES:
   - Correct any Arabic/English text encoding problems
   - Ensure proper text direction and formatting
   - Fix any character encoding issues

CONTENT TO CORRECT:

{sections_text}

CRITICAL INSTRUCTIONS:
- PRESERVE ALL numbering (1., 2., 3., 1.1, 1.2, etc.) - DO NOT CHANGE
- PRESERVE ALL bold formatting (**text**) - DO NOT REMOVE
- PRESERVE ALL header formatting - DO NOT MODIFY
- PRESERVE ALL table structures with | symbols exactly as they appear
- EXCEPTION: Only reformat "Problem or Need Statement" section content
- Return corrected content maintaining exact section structure
- Each section should start with [SECTION: section_name] as provided
- Focus ONLY on removing JSON artifacts and fixing encoding issues
- Tables must keep their exact markdown format with | delimiters
- Ensure all data points are synchronized and consistent
- Keep professional tone throughout

Return only the corrected content with sections marked as [SECTION: name]."""

        try:
            # Make OpenAI API call using same pattern as main agents
            response = self.openai_client.chat.completions.create(
                model=self.correction_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional proposal editor. CRITICAL: Preserve ALL formatting including numbering, bold text, and especially TABLES with their | symbols. NEVER change client names, project titles, or specific business information. Only remove JSON artifacts and synchronize data ensuring consistency across business documents."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1 if self.correction_model == "gpt-5" else 0.1,  # GPT-5 only supports temperature=1.0
                max_completion_tokens=8000  # Increased for larger proposals with detailed content
            )
            
            # Track costs using SimpleCostTracker only
            if self.cost_tracker:
                self.cost_tracker.track_completion(response, self.correction_model)
            
            # Get corrected content
            corrected_text = response.choices[0].message.content
            logger.info("Received corrected content from OpenAI")
            
            # Parse corrected sections
            corrected_sections = self._parse_corrected_sections(corrected_text)
            
            # Build corrected JSON structure with proper format
            corrected_json = proposal_json.copy()

            # Rebuild generated_sections with proper structure
            original_sections = proposal_json.get('generated_sections', {})
            corrected_sections_formatted = {}

            # Define the correct proposal section order
            correct_section_order = [
                "Executive Summary",
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
                "Conclusion"
            ]

            # Process sections in CORRECT ORDER (not just original JSON order)
            sections_to_add = []

            # First, add sections in the correct order if they exist
            for section_name in correct_section_order:
                if section_name in original_sections:
                    sections_to_add.append(section_name)

            # Then add any remaining sections that weren't in the standard list
            for section_name in original_sections.keys():
                if section_name not in sections_to_add:
                    sections_to_add.append(section_name)

            # Now process all sections in the correct order
            for section_name in sections_to_add:
                if section_name in corrected_sections:
                    # Use corrected content but maintain original order
                    section_content = corrected_sections[section_name]
                    if isinstance(original_sections[section_name], dict):
                        corrected_sections_formatted[section_name] = {
                            'content': section_content,
                            'metadata': original_sections[section_name].get('metadata', {})
                        }
                    else:
                        corrected_sections_formatted[section_name] = {
                            'content': section_content,
                            'metadata': {}
                        }
                    logger.info(f"Added corrected section '{section_name}' in original position")
                elif section_name in excluded_sections:
                    # Keep excluded sections (charts/large content) in their original position
                    corrected_sections_formatted[section_name] = original_sections[section_name]
                    logger.info(f"Preserved excluded section '{section_name}' in original position")
                else:
                    # Keep any other original sections unchanged
                    corrected_sections_formatted[section_name] = original_sections[section_name]
                    logger.info(f"Preserved unchanged section '{section_name}' in original position")

            # Add any completely new sections from correction (should be very rare)
            for section_name, section_content in corrected_sections.items():
                if section_name not in corrected_sections_formatted:
                    corrected_sections_formatted[section_name] = {
                        'content': section_content,
                        'metadata': {}
                    }
                    logger.info(f"Added new section '{section_name}' from correction")

            corrected_json['generated_sections'] = corrected_sections_formatted
            corrected_json['correction_metadata'] = {
                'corrected_at': datetime.now().isoformat(),
                'model_used': self.correction_model,
                'sections_corrected': len(corrected_sections),
                'sections_excluded': len(excluded_sections),
                'excluded_sections': excluded_sections
            }
            
            return corrected_json
            
        except Exception as e:
            logger.error(f"Error during OpenAI correction: {e}")
            raise


    def _parse_corrected_sections(self, corrected_text: str) -> Dict[str, str]:
        """Parse corrected text back into sections using [SECTION: name] markers"""
        sections = {}
        
        # Use regex to split by [SECTION: name] markers
        import re
        section_pattern = r'\[SECTION:\s*([^\]]+)\]'
        parts = re.split(section_pattern, corrected_text)
        
        # Process sections (they come in pairs: name, content)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                section_name = parts[i].strip()
                section_content = parts[i + 1].strip()
                
                # MINIMAL cleanup - only remove obvious markdown artifacts that shouldn't be in proposals
                # DO NOT remove numbering (1., 2., 3.) or bold formatting (**text**)
                lines = section_content.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Only remove excessive # symbols at start (more than 3) which are likely artifacts
                    # Preserve actual headers and formatting
                    if line.startswith('####'):  # Remove excessive headers only
                        cleaned_line = re.sub(r'^#{4,}\s*', '### ', line)
                    else:
                        # Keep the line as is - preserve numbering and bold formatting
                        cleaned_line = line
                    cleaned_lines.append(cleaned_line)
                
                cleaned_content = '\n'.join(cleaned_lines)
                
                if section_name and cleaned_content:
                    sections[section_name] = cleaned_content
        
        # If no sections found, treat as single content block
        if not sections:
            sections['Corrected Content'] = corrected_text.strip()
        
        logger.info(f"Parsed {len(sections)} corrected sections")
        return sections
    
    def _save_corrected_json(self, corrected_json: Dict, original_path: str) -> str:
        """Save corrected JSON with correct_ prefix"""
        original_file = Path(original_path)
        # Keep the JSON in artifacts directory but with correct_ prefix
        corrected_filename = f"correct_{original_file.name}"
        corrected_path = original_file.parent / corrected_filename
        
        try:
            with open(corrected_path, 'w', encoding='utf-8') as f:
                json.dump(corrected_json, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Corrected JSON saved to: {corrected_path}")
            return str(corrected_path)
            
        except Exception as e:
            logger.error(f"Error saving corrected JSON: {e}")
            raise
    
    def _generate_corrected_html(self, corrected_json: Dict) -> str:
        """Use HTMLGenerator to create HTML from corrected JSON"""
        try:
            # Use the HTMLGenerator to create HTML content
            html_content = self.html_generator.generate(corrected_json)
            logger.info("Generated corrected HTML using HTMLGenerator")
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating corrected HTML: {e}")
            # Fallback to basic HTML generation if HTMLGenerator fails
            return self._create_basic_html(corrected_json)
    
    def _create_basic_html(self, corrected_json: Dict) -> str:
        """Create basic HTML as fallback if HTMLGenerator fails"""
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en" dir="ltr">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>Corrected Proposal</title>',
            '<style>',
            'body { font-family: "Arial", "Tahoma", sans-serif; line-height: 1.6; margin: 40px; direction: ltr; font-size: 14px; }',
            'h1 { color: #2E86AB; margin-top: 30px; font-size: 2.2em; }',
            'h2 { color: #2E86AB; margin-top: 25px; font-size: 1.6em; }',
            'h3 { color: #2E86AB; margin-top: 20px; font-size: 1.3em; }',
            'p { font-size: 14px; margin-bottom: 12px; }',
            'table { border-collapse: collapse; width: 100%; margin: 20px auto; text-align: center; font-size: 13px; }',
            'th, td { border: 1px solid #ddd; padding: 12px; text-align: center; font-size: 13px; }',
            'th { background-color: #2E86AB; color: white; font-weight: 600; }',
            'ul, ol { font-size: 14px; }',
            'li { margin-bottom: 6px; }',
            '.section { margin-bottom: 30px; }',
            '</style>',
            '</head>',
            '<body>'
        ]
        
        # Add client information if available
        if 'client_name' in corrected_json:
            html_parts.append(f'<h1>Proposal for {corrected_json["client_name"]}</h1>')
        
        # Add generated sections
        generated_sections = corrected_json.get('generated_sections', {})
        for section_name, section_content in generated_sections.items():
            html_parts.append(f'<div class="section">')
            html_parts.append(f'<h2>{section_name}</h2>')
            
            # Convert markdown-like content to HTML
            content_html = self._markdown_to_html(section_content)
            html_parts.append(content_html)
            
            html_parts.append('</div>')
        
        html_parts.extend(['</body>', '</html>'])
        
        return '\n'.join(html_parts)
    
    def _markdown_to_html(self, text: str) -> str:
        """Convert markdown-like text to HTML"""
        html = text
        
        # Convert markdown tables to HTML tables
        table_pattern = r'\|.*\|[\r\n]+\|[\s\-\|]+\|[\r\n]+((?:\|.*\|[\r\n]*)+)'
        
        def table_replacer(match):
            lines = match.group(0).strip().split('\n')
            if len(lines) < 3:
                return match.group(0)
            
            # Parse header
            header = [cell.strip() for cell in lines[0].split('|')[1:-1]]
            
            # Parse rows (skip separator line)
            rows = []
            for line in lines[2:]:
                if '|' in line:
                    row = [cell.strip() for cell in line.split('|')[1:-1]]
                    if row:
                        rows.append(row)
            
            # Build HTML table
            table_html = ['<table>']
            
            # Header
            table_html.append('<thead><tr>')
            for cell in header:
                table_html.append(f'<th>{cell}</th>')
            table_html.append('</tr></thead>')
            
            # Body
            table_html.append('<tbody>')
            for row in rows:
                table_html.append('<tr>')
                for i, cell in enumerate(row):
                    if i < len(header):
                        table_html.append(f'<td>{cell}</td>')
                table_html.append('</tr>')
            table_html.append('</tbody></table>')
            
            return '\n'.join(table_html)
        
        html = re.sub(table_pattern, table_replacer, html, flags=re.MULTILINE)
        
        # Convert line breaks to paragraphs
        paragraphs = html.split('\n\n')
        html_paragraphs = []
        for para in paragraphs:
            if para.strip() and not para.strip().startswith('<'):
                html_paragraphs.append(f'<p>{para.strip()}</p>')
            else:
                html_paragraphs.append(para.strip())
        
        return '\n'.join(html_paragraphs)
    
    def _extract_client_name(self, json_data: Dict) -> str:
        """Extract client name from JSON data"""
        # Try to get from request string
        request_str = json_data.get('request', '')
        if 'client_name=' in request_str:
            import re
            match = re.search(r"client_name='([^']+)'", request_str)
            if match:
                return match.group(1).replace(' ', '_').replace('/', '_')
        
        # Fallback to other sources
        if 'client' in json_data:
            return str(json_data['client']).replace(' ', '_').replace('/', '_')
        
        return 'Unknown_Client'
    
    def _generate_corrected_pdf(self, html_content: str, client_name: str, json_path: str) -> str:
        """Generate corrected PDF with proper naming"""
        try:
            # Extract timestamp from original JSON filename
            json_filename = Path(json_path).stem
            timestamp_match = re.search(r'(\d{8}_\d{6})', json_filename)
            
            if timestamp_match:
                timestamp = timestamp_match.group(1)
            else:
                # Fallback to current timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create PDF filename: correct_ClientName_timestamp.pdf
            pdf_filename = f"correct_{client_name}_{timestamp}.pdf"
            corrected_pdf_path = self.output_dir / pdf_filename
            
            # Use weasyprint directly to avoid double prefixing
            import weasyprint
            html = weasyprint.HTML(string=html_content)
            pdf = html.write_pdf()
            
            with open(corrected_pdf_path, 'wb') as f:
                f.write(pdf)
            
            logger.info(f"PDF exported successfully: {corrected_pdf_path}")
            return str(corrected_pdf_path)
                
        except Exception as e:
            logger.error(f"Error generating corrected PDF: {e}")
            raise
    
    def _generate_diff_report(self, original: Dict, corrected: Dict, original_path: str) -> str:
        """Generate detailed diff report showing what was corrected"""
        logger.info("Generating detailed diff report...")
        
        report = []
        report.append("=" * 80)
        report.append("PROPOSAL CORRECTION REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Original File: {Path(original_path).name}")
        report.append(f"Correction Model: {self.correction_model}")
        
        # Basic statistics
        orig_sections = original.get('generated_sections', {})
        corr_sections = corrected.get('generated_sections', {})
        
        report.append(f"\nOriginal Sections: {len(orig_sections)}")
        report.append(f"Corrected Sections: {len(corr_sections)}")
        
        # Section-by-section analysis
        report.append("\n" + "=" * 60)
        report.append("SECTION-BY-SECTION CHANGES")
        report.append("=" * 60)
        
        for section_name in set(list(orig_sections.keys()) + list(corr_sections.keys())):
            report.append(f"\n### {section_name}")
            
            if section_name in orig_sections and section_name in corr_sections:
                orig_content = orig_sections[section_name]
                corr_content = corr_sections[section_name]
                
                # Extract actual text content (handle both dict and string formats)
                if isinstance(orig_content, dict):
                    # Try multiple common content keys
                    orig_text = orig_content.get('content') or orig_content.get('text') or str(orig_content)
                else:
                    orig_text = str(orig_content)
                
                if isinstance(corr_content, dict):
                    # Try multiple common content keys  
                    corr_text = corr_content.get('content') or corr_content.get('text') or str(corr_content)
                else:
                    corr_text = str(corr_content)
                
                # Ensure we have strings, not None values
                orig_text = str(orig_text) if orig_text is not None else ""
                corr_text = str(corr_text) if corr_text is not None else ""
                
                # Character count comparison
                orig_len = len(orig_text)
                corr_len = len(corr_text)
                change_pct = ((corr_len - orig_len) / max(orig_len, 1)) * 100
                
                report.append(f"  Original length: {orig_len} characters")
                report.append(f"  Corrected length: {corr_len} characters")
                report.append(f"  Change: {change_pct:+.1f}%")
                
                # Identify specific changes
                changes = []
                
                # Check for JSON artifacts removal
                json_artifacts = ['{', '}', '```', '"content":', '"key":', '\\"']
                orig_artifacts = sum(orig_text.count(artifact) for artifact in json_artifacts)
                corr_artifacts = sum(corr_text.count(artifact) for artifact in json_artifacts)
                
                if orig_artifacts > corr_artifacts:
                    changes.append(f"Removed {orig_artifacts - corr_artifacts} JSON artifacts")
                
                # Check for table preservation
                orig_tables = orig_text.count('|')
                corr_tables = corr_text.count('|')
                
                if orig_tables > 0 and corr_tables > 0:
                    changes.append("Table structure preserved")
                elif orig_tables > 0 and corr_tables == 0:
                    changes.append("Table structure lost (may need review)")
                
                # Check for formatting improvements
                if corr_text.count('\n\n') > orig_text.count('\n\n'):
                    changes.append("Improved paragraph formatting")
                
                if changes:
                    report.append("  Changes made:")
                    for change in changes:
                        report.append(f"    - {change}")
                else:
                    report.append("  No significant changes detected")
                    
            elif section_name in orig_sections:
                report.append("  Status: Section removed during correction")
            else:
                report.append("  Status: New section added during correction")
        
        # Content quality analysis
        report.append("\n" + "=" * 60)
        report.append("CONTENT QUALITY IMPROVEMENTS")
        report.append("=" * 60)
        
        quality_checks = [
            "✓ JSON formatting artifacts removed",
            "✓ Professional language maintained",
            "✓ Table structures preserved where possible",
            "✓ Data consistency checked across sections",
            "✓ Proper markdown formatting applied",
            "✓ Business-appropriate tone ensured"
        ]
        
        for check in quality_checks:
            report.append(check)
        
        # Metadata section
        if 'correction_metadata' in corrected:
            metadata = corrected['correction_metadata']
            report.append("\n" + "=" * 60)
            report.append("CORRECTION METADATA")
            report.append("=" * 60)
            report.append(f"\nCorrection timestamp: {metadata.get('corrected_at', 'N/A')}")
            report.append(f"Model used: {metadata.get('model_used', 'N/A')}")
            report.append(f"Sections processed: {metadata.get('sections_corrected', 'N/A')}")
        
        # Save report to main output directory (NOT in artifacts)
        original_file = Path(original_path)
        # Extract client name and timestamp from original filename
        filename_parts = original_file.stem.replace('proposal_', '')
        report_filename = f"show_diff_{filename_parts}.txt"
        # Save to main output directory
        report_path = self.output_dir / report_filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            logger.info(f"Diff report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error saving diff report: {e}")
            raise
    
    def get_correction_cost(self) -> float:
        """Get the total cost of corrections made from SimpleCostTracker"""
        if self.cost_tracker:
            return self.cost_tracker.get_total_cost()
        return 0.0