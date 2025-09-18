"""
HTML Generator for proposal output
"""

import json
from typing import Dict, Any  # Added proper imports
from jinja2 import Template
from datetime import datetime
import base64

class HTMLGenerator:
    """Generate HTML output for proposals"""
    
    def __init__(self, config: Dict, output_format: str = 'interactive'):
        self.config = config
        self.output_format = output_format  # 'interactive' for HTML, 'static' for PDF
        self.template = self._load_template()
    
    def _load_template(self) -> Template:
        """Load HTML template"""
        template_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - Proposal</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .header {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .metadata {
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        
        .metadata-item {
            margin: 10px 0;
        }
        
        .metadata-item label {
            font-weight: bold;
            color: #666;
            margin-right: 10px;
        }
        
        .toc {
            padding: 30px 40px;
            background: #fafafa;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .toc h2 {
            color: #2E86AB;
            margin-bottom: 20px;
        }
        
        .toc ul {
            list-style: none;
            padding-left: 0;
        }
        
        .toc li {
            padding: 8px 0;
            border-bottom: 1px dotted #ddd;
        }
        
        .toc a {
            color: #333;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .toc a:hover {
            color: #2E86AB;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 50px;
            page-break-inside: avoid;
        }
        
        .section h2 {
            color: #2E86AB;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #2E86AB;
        }
        
        .section h3 {
            color: #444;
            font-size: 1.3em;
            margin: 20px 0 10px 0;
        }
        
        .section p {
            margin-bottom: 15px;
            text-align: justify;
        }
        
        .section ul, .section ol {
            margin: 15px 0 15px 30px;
        }
        
        .section li {
            margin-bottom: 8px;
        }
        
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }
        
        th {
            background: #2E86AB;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
            border-left: 4px solid #2E86AB;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        
        .footer {
            background: #2c3e50;
            color: white;
            padding: 30px 40px;
            text-align: center;
        }
        
        .footer .company {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        
        .footer .contact {
            opacity: 0.8;
        }
        
        @media print {
            .toc {
                page-break-after: always;
            }
            
            .section {
                page-break-inside: avoid;
            }
            
            .header {
                background: #2E86AB !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
        }
        
        .executive-summary {
            background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        /* Table Styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border-radius: 8px;
            overflow: hidden;
        }
        
        table thead {
            background: linear-gradient(135deg, #2E86AB 0%, #4A90A4 100%);
            color: white;
        }
        
        table th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        
        table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }
        
        table tbody tr:last-child td {
            border-bottom: none;
        }
        
        table tbody tr:hover {
            background: #f8f9fa;
            transition: background 0.3s ease;
        }
        
        table tbody tr:nth-child(even) {
            background: #fafafa;
        }
        
        /* Responsive tables */
        @media (max-width: 768px) {
            table {
                font-size: 0.9em;
            }
            
            table th, table td {
                padding: 10px;
            }
        }
        
        .budget-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .budget-table th {
            background: linear-gradient(135deg, #2E86AB 0%, #4A90A4 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        .budget-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .budget-table tbody tr:hover {
            background: #f8f9fa;
        }
        
        .budget-table tbody tr:nth-child(even) {
            background: #fafafa;
        }
        
        .page-info {
            text-align: center;
            color: #666;
            font-style: italic;
            margin: 10px 0;
            padding: 5px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, #2E86AB, #A23B72);
            margin: 30px 0;
            border-radius: 1px;
        }
        
        .timeline-graphic {
            width: 100%;
            height: 400px;
            margin: 20px 0;
        }
    </style>
    
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{{ title }}</h1>
            <div class="subtitle">Proposal for {{ client }}</div>
        </div>
        
        <!-- Metadata -->
        <div class="metadata">
            <div class="metadata-item">
                <label>Date:</label>
                <span>{{ date }}</span>
            </div>
            <div class="metadata-item">
                <label>Project:</label>
                <span>{{ project }}</span>
            </div>
            <div class="metadata-item">
                <label>Timeline:</label>
                <span>{{ timeline }}</span>
            </div>
            <div class="metadata-item">
                <label>Proposal ID:</label>
                <span>{{ proposal_id }}</span>
            </div>
        </div>
        
        <!-- Table of Contents -->
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                {% for section_name in sections.keys() %}
                <li><a href="#{{ section_name | replace(' ', '-') | lower }}">{{ loop.index }}. {{ section_name }}</a></li>
                {% endfor %}
            </ul>
        </div>
        
        <!-- Content Sections -->
        <div class="content">
            {% for section_name, section_content in sections.items() %}
            <div class="section" id="{{ section_name | replace(' ', '-') | lower }}">
                <h2>{{ loop.index }}. {{ section_name }}</h2>
                
                {% if section_name == "Executive Summary" %}
                <div class="executive-summary">
                    {{ section_content.content | safe }}
                </div>
                {% else %}
                    {{ section_content.content | safe }}
                {% endif %}
                
            </div>
            {% endfor %}
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <div class="company">{{ company_name }}</div>
            <div class="contact">
                {{ company_email }} | {{ company_website }}
            </div>
            <div style="margin-top: 20px; opacity: 0.6;">
                © {{ year }} {{ company_name }}. All rights reserved.
            </div>
        </div>
    </div>
    
    <script>
        // Add smooth scrolling to TOC links
        document.querySelectorAll('.toc a').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
</body>
</html>
        '''
        return Template(template_content)
    
    def generate(self, proposal: Dict) -> str:
        """Generate HTML from proposal data"""
        
        # Format sections content (handle both 'sections' and 'generated_sections')
        formatted_sections = {}
        sections_data = proposal.get('generated_sections', proposal.get('sections', {}))
        for section_name, section_data in sections_data.items():
            if isinstance(section_data, dict):
                content = section_data.get('content', '')
                if isinstance(content, dict):
                    content = self._format_content_dict(content)
                # Convert markdown to HTML and strip duplicate headings that repeat section title
                rendered_html = self._markdown_to_html(content)
                rendered_html = self._strip_duplicate_heading(section_name, rendered_html)
                # Remove tables for specific sections if requested
                if section_name in ["Technical Approach and Methodology Visualization", "Technical Approach and Methodology"]:
                    rendered_html = self._remove_tables(rendered_html)
                formatted_sections[section_name] = {
                    'content': rendered_html
                }
            else:
                rendered_html = self._markdown_to_html(str(section_data))
                rendered_html = self._strip_duplicate_heading(section_name, rendered_html)
                if section_name in ["Technical Approach and Methodology Visualization", "Technical Approach and Methodology"]:
                    rendered_html = self._remove_tables(rendered_html)
                formatted_sections[section_name] = {
                    'content': rendered_html
                }
    

    

        # Prepare template context
        context = {
            'title': f"{proposal.get('project', 'Project')} Proposal",
            'client': proposal.get('client', 'Client'),
            'project': proposal.get('project', 'Project'),
            'date': datetime.now().strftime('%B %d, %Y'),
            'timeline': proposal.get('timeline', 'TBD'),
            'proposal_id': f"AzmX-{datetime.now().strftime('%Y%m%d')}-001",
            'sections': formatted_sections,            
            'company_name': self.config.get('company', {}).get('name', 'AzmX Technologies'),
            'company_email': self.config.get('company', {}).get('email', 'contact@azmx.com'),
            'company_website': self.config.get('company', {}).get('website', 'www.azmx.com'),
            'year': datetime.now().year
        }
        
        # Render template
        return self.template.render(**context)

    def _clean_json_artifacts(self, content: str) -> str:
        """Clean JSON artifacts from content"""
        if not content:
            return ""

        import re
        import json

        # Step 1: Remove code blocks and extract content
        content = re.sub(r'```json\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'```\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)

        # Step 2: Remove agent response artifacts
        content = re.sub(r'RunResult:\s*\n-\s*Last agent:.*?Agent\(name=.*?\).*?\n', '', content, flags=re.DOTALL)
        content = re.sub(r'Final output \(str\):\s*\n', '', content, flags=re.DOTALL)
        content = re.sub(r'- \d+ new item\(s\)\s*\n', '', content, flags=re.DOTALL)
        content = re.sub(r'- \d+ raw response\(s\)\s*\n', '', content, flags=re.DOTALL)
        content = re.sub(r'- \d+ input guardrail result\(s\)\s*\n', '', content, flags=re.DOTALL)
        content = re.sub(r'- \d+ output guardrail result\(s\)\s*\n', '', content, flags=re.DOTALL)
        content = re.sub(r'\(See `RunResult` for more details\)', '', content, flags=re.DOTALL)

        # Step 3: Try to parse and extract readable content from JSON structures
        json_block_pattern = r'\{\s*\n\s*"(?:topic|section_title|content)"[^}]*\}'
        
        def extract_readable_json(match):
            json_text = match.group(0)
            try:
                parsed = json.loads(json_text)
                # Extract content from common fields
                for field in ['content', 'summary', 'text', 'body', 'section_content']:
                    if field in parsed and isinstance(parsed[field], str):
                        return parsed[field]
                
                # Handle research agent format
                if 'topic' in parsed and 'summary' in parsed:
                    parts = []
                    if parsed.get('summary'):
                        parts.append(parsed['summary'])
                    if 'findings' in parsed and isinstance(parsed['findings'], list):
                        findings = [f['insight'] for f in parsed['findings'] if isinstance(f, dict) and 'insight' in f]
                        if findings:
                            parts.extend(findings)
                    return '\n\n'.join(parts)
                
                # If we can't extract specific fields, return original
                return json_text
                
            except json.JSONDecodeError:
                # If parsing fails, remove the JSON block entirely
                return ""

        content = re.sub(json_block_pattern, extract_readable_json, content, flags=re.DOTALL)

        # Step 4: Remove remaining JSON artifacts
        json_patterns = [
            r'\{\s*\n\s*"[^"]+"\s*:\s*"[^"]*"[^}]*\}',  # Simple JSON objects
            r'"key_points"\s*:\s*\[[^\]]*\]',  # Key points arrays
            r'"suggested_visuals"\s*:\s*\[[^\]]*\]',  # Suggested visuals arrays
            r'"confidence_score"\s*:\s*\d+\.\d+',  # Confidence scores
            r'"word_count"\s*:\s*\d+',  # Word counts
            r'"metadata"\s*:\s*\{[^}]*\}',  # Metadata objects
        ]

        for pattern in json_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL)

        # Step 5: Clean up extra whitespace and newlines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Reduce multiple newlines
        content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)  # Remove leading whitespace
        content = content.strip()

        return content

    def _markdown_to_html(self, content: str) -> str:
        """Enhanced markdown to HTML conversion with table support"""
        if not content:
            return ""

        import re

        # Remove JSON artifacts and code blocks first
        content = self._clean_json_artifacts(content)

        html = content

        # Process tables first (before other markdown)
        html = self._process_markdown_tables(html)

        # Headers - fix the order and add h4 support
        html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)

        # Bold and italics
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)

        # Process lists and paragraphs with better handling
        html = self._process_lists_and_paragraphs(html)

        return html

    def _process_markdown_tables(self, text: str) -> str:
        """Convert markdown tables to HTML tables"""
        import re

        # Find table blocks (lines with | separators)
        table_pattern = r'(\|[^\n]*\|\n(?:\|[^\n]*\|\n)*)'

        def convert_table(match):
            table_text = match.group(1).strip()
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]

            if len(lines) < 2:
                return table_text  # Not a valid table

            # Check if second line is separator (contains -, |, :)
            separator_line = lines[1]
            if not re.match(r'^\|[\s\-\:\|]*\|$', separator_line):
                return table_text  # Not a markdown table

            # Build HTML table
            html_parts = ['<table class="budget-table">']

            # Header row
            header_cells = [cell.strip() for cell in lines[0].split('|')[1:-1]]
            html_parts.append('<thead><tr>')
            for cell in header_cells:
                html_parts.append(f'<th>{cell}</th>')
            html_parts.append('</tr></thead>')

            # Data rows
            html_parts.append('<tbody>')
            for line in lines[2:]:  # Skip header and separator
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                html_parts.append('<tr>')
                for cell in cells:
                    html_parts.append(f'<td>{cell}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody>')

            html_parts.append('</table>')
            return '\n'.join(html_parts)

        return re.sub(table_pattern, convert_table, text, flags=re.MULTILINE)

    def _process_lists_and_paragraphs(self, html: str) -> str:
        """Process lists and paragraphs with better formatting"""
        lines = html.split('\n')
        new_lines = []
        in_list = False
        in_ordered_list = False
        in_table = False

        for line in lines:
            line = line.strip()

            # Skip empty lines in tables
            if '<table' in line:
                in_table = True
            elif '</table>' in line:
                in_table = False

            if in_table:
                new_lines.append(line)
                continue

            # Handle unordered lists
            if line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                if in_ordered_list:
                    new_lines.append('</ol>')
                    in_ordered_list = False
                if not in_list:
                    new_lines.append('<ul>')
                    in_list = True
                new_lines.append(f"<li>{line[2:].strip()}</li>")
            # Handle ordered lists (simple 1-9 prefix)
            elif line[:3].isdigit() and line[1:3] == '. ' or line.startswith(('1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. ')):
                if in_list:
                    new_lines.append('</ul>')
                    in_list = False
                if not in_ordered_list:
                    new_lines.append('<ol>')
                    in_ordered_list = True
                # remove leading number and dot
                content_part = line.split('. ', 1)[1] if '. ' in line else line
                new_lines.append(f"<li>{content_part.strip()}</li>")
            else:
                if in_list:
                    new_lines.append('</ul>')
                    in_list = False
                if in_ordered_list:
                    new_lines.append('</ol>')
                    in_ordered_list = False

                # Skip lines that are already HTML tags
                if line and not line.startswith('<') and not line.endswith('>'):
                    # Handle special budget formatting
                    if '---' in line:
                        new_lines.append('<hr>')
                    elif line.startswith('Page ') and 'of' in line:
                        new_lines.append(f'<div class="page-info">{line}</div>')
                    else:
                        new_lines.append(f"<p>{line}</p>")
                elif line:
                    new_lines.append(line)

        if in_list:
            new_lines.append('</ul>')
        if in_ordered_list:
            new_lines.append('</ol>')

        return '\n'.join(new_lines)

    def _strip_duplicate_heading(self, section_name: str, html_content: str) -> str:
        """Remove a leading heading that duplicates the section title.

        Handles cases like "<h2>1. Section Title</h2>" or "<h3>Section Title</h3>".
        Comparison is case-insensitive and ignores leading numbering and punctuation.
        """
        import re

        try:
            # Normalize function to handle separators and punctuation variations
            def _normalize_title(text: str) -> str:
                t = text.strip().lower()
                # Handle various separators
                t = t.replace('&amp;', 'and')
                t = t.replace('&', 'and')
                t = t.replace('/', ' ')
                t = t.replace('-', ' ')
                # Remove all punctuation and normalize spaces
                t = re.sub(r'[^a-z0-9\s]', '', t)
                t = re.sub(r'\s+', ' ', t).strip()
                return t

            normalized_title = _normalize_title(section_name)

            # Pattern to capture the very first heading tag h2-h4
            # Also check for pattern with newlines/spaces after the opening tag
            pattern = r'^(\s*<(h[2-4])>[\s\n]*(.*?)[\s\n]*</\2>\s*)'
            match = re.match(pattern, html_content, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                return html_content

            heading_inner = match.group(3)

            # Remove any leading numbering like "1.", "1)" or "1 -"
            cleaned_heading = re.sub(r'^\s*\d+[\.|\)|-]?\s*', '', heading_inner)
            cleaned_heading = _normalize_title(cleaned_heading)

            # Also check variations of the section name
            section_variations = [
                normalized_title,
                _normalize_title(section_name.replace('/', ' & ')),
                _normalize_title(section_name.replace('/', ' and ')),
                _normalize_title(section_name.replace('Team/', 'Team and ')),
                _normalize_title(section_name.replace('Team/', 'Team & '))
            ]

            if cleaned_heading in section_variations:
                # Strip the duplicate heading
                return html_content[len(match.group(1)):]  # remove the matched heading block

            return html_content
        except Exception:
            return html_content

    def _remove_tables(self, html_content: str) -> str:
        """Remove all HTML <table> blocks from content."""
        import re
        try:
            # Remove complete table blocks
            return re.sub(r'<table[\s\S]*?</table>', '', html_content, flags=re.IGNORECASE)
        except Exception:
            return html_content

    
    def _is_valid_html_structure(self, html_content: str) -> bool:
        """Basic validation of HTML structure"""
        try:
            # Check for balanced tags (basic validation)
            import re
            
            # Count opening and closing div tags
            open_divs = len(re.findall(r'<div[^>]*>', html_content))
            close_divs = len(re.findall(r'</div>', html_content))
            
            # Allow some flexibility in tag balance
            divs_balanced = abs(open_divs - close_divs) <= 2
            
            # Check for basic HTML structure
            has_content = len(html_content.strip()) > 0
            no_malicious_content = not any(dangerous in html_content.lower() for dangerous in [
                'javascript:', 'data:text/html', '<iframe', '<object', '<embed', 'onload=', 'onerror='
            ])
            
            return divs_balanced and has_content and no_malicious_content
            
        except Exception:
            return False
    
    def _format_content_dict(self, content_dict: Dict) -> str:
        """Format a content dictionary into HTML string"""
        parts = []
        
        for key, value in content_dict.items():
            if key == 'main':
                parts.append(str(value))
            elif key == 'subsections':
                for subsection in value:
                    parts.append(f"### {subsection.get('title', '')}")
                    parts.append(subsection.get('content', ''))
            elif isinstance(value, list):
                parts.append(f"### {key.replace('_', ' ').title()}")
                for item in value:
                    parts.append(f"- {item}")
            else:
                parts.append(str(value))
        
        return '\n\n'.join(parts)