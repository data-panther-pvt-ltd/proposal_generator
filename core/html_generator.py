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
        
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: center;
        }
        
        .chart-title {
            font-weight: bold;
            margin-bottom: 15px;
            color: #666;
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
    {% if include_plotly %}
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <style>
        /* Plotly chart specific styles */
        .plotly-graph-div {
            width: 100% !important;
            height: auto !important;
            min-height: 400px;
            margin: 20px 0;
        }
        
        .chart-container .plotly-graph-div {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .chart-error {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
        }
        
        .chart-wrapper {
            overflow-x: auto;
            overflow-y: visible;
        }
    </style>
    {% endif %}
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
                
                {% if section_name in charts and charts[section_name] %}
                <div class="chart-container" id="chart-container-{{ loop.index }}">
                    <div class="chart-title">{{ section_name }} Visualization</div>
                    <div class="chart-wrapper">
                        {{ charts[section_name] | safe }}
                    </div>
                </div>
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
        
        # Format sections content
        formatted_sections = {}
        for section_name, section_data in proposal.get('sections', {}).items():
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
        
        # Process charts first - convert base64 or other formats to HTML (supports lists)
        formatted_charts = self._process_charts(proposal.get('charts', {}))

        # Suppress charts for specific sections entirely (per product requirement)
        suppressed_chart_sections = {
            "Project Scope Visualization",
            "Implementation Strategy Visualization",
            "Technical Approach and Methodology Visualization",
            "Technical Approach and Methodology"
        }
        for suppressed in list(formatted_charts.keys()):
            if suppressed in suppressed_chart_sections:
                del formatted_charts[suppressed]

        # Check if we need Plotly library (only for interactive charts)
        def _has_any_chart(charts_dict):
            if not charts_dict:
                return False
            for v in charts_dict.values():
                if isinstance(v, list) and any(bool(item) for item in v):
                    return True
                if isinstance(v, str) and v.strip():
                    return True
                if isinstance(v, dict) and v:
                    return True
            return False

        include_plotly = self.output_format == 'interactive' and _has_any_chart(formatted_charts)

        # Inject charts inline into section content, except preserved sections that should use template placement
        preserved_template_sections = {"Proposed Solution Visualization"}
        charts_for_template = {}

        for section_name in list(formatted_sections.keys()):
            charts_for_section = formatted_charts.get(section_name)
            if not charts_for_section:
                continue

            if section_name in preserved_template_sections:
                # Keep original behavior: let the template render the chart below the content
                # Ensure a single HTML string is passed to the template (use first if list)
                if isinstance(charts_for_section, list) and len(charts_for_section) > 0:
                    charts_for_template[section_name] = charts_for_section[0]
                else:
                    charts_for_template[section_name] = charts_for_section
            else:
                # Default: inject inline and do not pass to template to avoid duplicates
                injected = self._inject_charts_into_content(section_name, formatted_sections[section_name]['content'], charts_for_section)
                formatted_sections[section_name]['content'] = injected

        # Prepare template context
        context = {
            'title': f"{proposal.get('project', 'Project')} Proposal",
            'client': proposal.get('client', 'Client'),
            'project': proposal.get('project', 'Project'),
            'date': datetime.now().strftime('%B %d, %Y'),
            'timeline': proposal.get('timeline', 'TBD'),
            'proposal_id': f"AzmX-{datetime.now().strftime('%Y%m%d')}-001",
            'sections': formatted_sections,
            'charts': charts_for_template,
            'include_plotly': include_plotly,
            'company_name': self.config['company']['name'],
            'company_email': self.config['company']['email'],
            'company_website': self.config['company']['website'],
            'year': datetime.now().year
        }
        
        # Render template
        return self.template.render(**context)

    def _process_charts(self, charts: Dict) -> Dict:
        """Enhanced chart processing with better error handling and list support"""
        formatted_charts = {}

        for section_name, chart_info in charts.items():
            try:
                # Support multiple charts per section
                if isinstance(chart_info, list):
                    section_charts = []
                    for item in chart_info:
                        if isinstance(item, dict):
                            chart_data = item.get('data', '')
                            chart_type = item.get('type', 'chart')
                        elif isinstance(item, str):
                            chart_data = item
                            chart_type = 'chart'
                        else:
                            section_charts.append(self._create_chart_placeholder('Invalid chart format'))
                            continue

                        if self._is_plotly_html(chart_data):
                            if self.output_format == 'interactive':
                                section_charts.append(self._clean_plotly_html(chart_data))
                            else:
                                section_charts.append(self._create_chart_placeholder(
                                    f'Interactive {chart_type} chart - view in HTML format'
                                ))
                        elif self._is_base64_image(chart_data):
                            section_charts.append(self._process_base64_image(chart_data, chart_type))
                        elif self._contains_html(chart_data):
                            section_charts.append(self._validate_and_clean_chart_html(chart_data) or self._create_chart_placeholder('Invalid chart HTML'))
                        else:
                            section_charts.append(self._create_chart_placeholder(f'{chart_type.title()} visualization placeholder'))

                    formatted_charts[section_name] = section_charts
                    continue

                if isinstance(chart_info, dict):
                    chart_data = chart_info.get('data', '')
                    chart_type = chart_info.get('type', 'chart')
                elif isinstance(chart_info, str):
                    chart_data = chart_info
                    chart_type = 'chart'
                else:
                    formatted_charts[section_name] = self._create_chart_placeholder('Invalid chart format')
                    continue

                # Process chart based on content type
                if self._is_plotly_html(chart_data):
                    if self.output_format == 'interactive':
                        formatted_charts[section_name] = self._clean_plotly_html(chart_data)
                    else:
                        formatted_charts[section_name] = self._create_chart_placeholder(
                            f'Interactive {chart_type} chart - view in HTML format'
                        )
                elif self._is_base64_image(chart_data):
                    formatted_charts[section_name] = self._process_base64_image(chart_data, chart_type)
                elif self._contains_html(chart_data):
                    formatted_charts[section_name] = self._validate_and_clean_chart_html(chart_data)
                else:
                    formatted_charts[section_name] = self._create_chart_placeholder(
                        f'{chart_type.title()} visualization placeholder'
                    )


            except Exception as e:
                print(f"[HTMLGenerator] Error processing chart for {section_name}: {str(e)}")
                formatted_charts[section_name] = self._create_chart_placeholder(
                    f'Chart processing error for {section_name}'
                )

        return formatted_charts

    def _inject_charts_into_content(self, section_name: str, html_content: str, charts_for_section: Any) -> str:
        """Place charts inline based on placeholders or append at the end if none.

        Supported placeholders inside content HTML (case-insensitive):
        - [[CHART]] or <!--CHART--> inserts the next chart in order
        - [[CHART:N]] inserts the Nth chart (1-based index)
        If charts_for_section is a single string, it's treated as one chart.
        """
        import re

        if not charts_for_section:
            return html_content

        charts_list = charts_for_section if isinstance(charts_for_section, list) else [charts_for_section]
        charts_list = [c for c in charts_list if isinstance(c, str) and c.strip()]
        if not charts_list:
            return html_content

        def replace_indexed(match):
            idx_group = match.group(1)
            if idx_group:
                try:
                    idx = int(idx_group) - 1
                except ValueError:
                    idx = -1
                if 0 <= idx < len(charts_list):
                    return f'<div class="chart-container"><div class="chart-title">{section_name} Visualization</div><div class="chart-wrapper">{charts_list[idx]}</div></div>'
                return ''
            # unindexed, pop from the front if available
            if charts_list:
                chart_html = charts_list.pop(0)
                return f'<div class="chart-container"><div class="chart-title">{section_name} Visualization</div><div class="chart-wrapper">{chart_html}</div></div>'
            return ''

        # Insert at placeholders
        pattern = re.compile(r"\[\[CHART(?::(\d+))?\]\]|<!--\s*CHART(?:\s*:\s*(\d+))?\s*-->", re.IGNORECASE)

        def repl(m):
            # combine either group 1 or 2 for index
            index_str = m.group(1) or m.group(2)
            return replace_indexed(type('obj', (), {'group': lambda _i: index_str}))

        new_html = pattern.sub(lambda m: replace_indexed(type('obj', (), {
            'group': lambda i: (m.group(1) if i == 1 else (m.group(2) if i == 2 else None))
        })), html_content)

        # If there are any remaining charts and no placeholders consumed them, append them after the content
        if new_html == html_content or any(token in html_content for token in ['[[CHART', '<!--CHART']):
            # Append any charts not yet placed
            if charts_list:
                appended = '\n'.join([
                    f'<div class="chart-container"><div class="chart-title">{section_name} Visualization</div><div class="chart-wrapper">{c}</div></div>'
                    for c in charts_list
                ])
                new_html = f"{new_html}\n{appended}"

        return new_html

    def _contains_html(self, content: str) -> bool:
        """Check if content contains HTML tags"""
        return '<' in content and '>' in content


    def _process_single_chart(self, chart_data: str, chart_type: str, chart_generator, chart_title: str = None) -> str:
        """Process a single chart based on its format and output type with enhanced Plotly support"""
        
        if not chart_data:
            return self._create_chart_placeholder('No chart data available')
        
        # Check if it's already HTML from Plotly (interactive or static)
        if self._is_plotly_html(chart_data):
            if self.output_format == 'interactive':
                # For interactive mode, return Plotly HTML as-is
                return self._clean_plotly_html(chart_data)
            else:
                # For static mode, if we get interactive HTML, try to extract any embedded images
                # or create a placeholder since we can't render interactive content in PDF
                return self._extract_static_from_plotly_html(chart_data, chart_type)
        
        # Check if it's a base64 image (from static chart generation)
        elif self._is_base64_image(chart_data):
            return self._process_base64_image(chart_data, chart_type)
        
        # Check if it's markdown with embedded image
        elif '![' in chart_data and '](' in chart_data:
            return self._process_markdown_image(chart_data, chart_type)
        
        # Check if it's raw HTML that looks like a div or complete chart
        elif '<' in chart_data and '>' in chart_data:
            # Clean and validate the HTML
            cleaned_html = self._validate_and_clean_chart_html(chart_data)
            return cleaned_html if cleaned_html else chart_data
        
        # Try to process as JSON data and regenerate chart
        else:
            return self._regenerate_chart_from_data(chart_data, chart_type, chart_generator)
    
    def _is_plotly_html(self, content: str) -> bool:
        """Check if content is Plotly generated HTML"""
        plotly_indicators = [
            'plotly-graph-div',
            'Plotly.newPlot',
            'data-plotly-domain',
            'plotly.js',
            'plotly-latest.min.js'
        ]
        return any(indicator in content for indicator in plotly_indicators)
    
    def _clean_plotly_html(self, html_content: str, chart_title: str = None) -> str:
        """Clean and prepare Plotly HTML for embedding with enhanced processing"""
        import re
        import uuid
        
        try:
            print(f"[HTMLGenerator] Cleaning Plotly HTML for {chart_title or 'chart'}")
            
            # Remove any external Plotly script tags since we include CDN in head
            external_script_patterns = [
                r'<script[^>]*src[^>]*plotly[^>]*></script>',
                r'<script[^>]*src[^>]*plot\.ly[^>]*></script>',
                r'<script[^>]*src[^>]*cdn\.plot\.ly[^>]*></script>'
            ]
            
            for pattern in external_script_patterns:
                html_content = re.sub(pattern, '', html_content, flags=re.IGNORECASE)
            
            # Generate unique ID to prevent conflicts
            unique_id = f"plotly-chart-{str(uuid.uuid4())[:8]}"
            
            # Find and replace div IDs
            div_patterns = [
                r'<div[^>]*id="([^"]+)"([^>]*)>',
                r'<div[^>]*id=\'([^\']+)\'([^>]*)>'
            ]
            
            old_ids = []
            for pattern in div_patterns:
                matches = re.finditer(pattern, html_content)
                for match in matches:
                    old_id = match.group(1)
                    old_ids.append(old_id)
                    # Replace in div tag
                    html_content = html_content.replace(
                        f'id="{old_id}"', f'id="{unique_id}"'
                    ).replace(
                        f"id='{old_id}'", f"id='{unique_id}'"
                    )
            
            # Replace IDs in script content as well
            for old_id in old_ids:
                html_content = html_content.replace(
                    f'"{old_id}"', f'"{unique_id}"'
                ).replace(
                    f"'{old_id}'", f"'{unique_id}'"
                )
            
            # Ensure div has proper CSS class for styling
            if 'class="plotly-graph-div"' not in html_content:
                html_content = re.sub(
                    r'<div([^>]*id="' + unique_id + r'"[^>]*)>',
                    r'<div\1 class="plotly-graph-div">',
                    html_content
                )
            
            # Wrap scripts in DOM ready function for better compatibility
            script_pattern = r'<script[^>]*>(.*?)</script>'
            scripts = re.findall(script_pattern, html_content, re.DOTALL)
            
            if scripts:
                # Remove old script tags
                html_content = re.sub(script_pattern, '', html_content, flags=re.DOTALL)
                
                # Create new wrapped script
                wrapped_script = '\n'.join(scripts)
                if 'Plotly.' in wrapped_script:
                    # Ensure script runs after DOM is ready
                    new_script = f'''
                    <script type="text/javascript">
                    document.addEventListener('DOMContentLoaded', function() {{
                        try {{
                            {wrapped_script}
                        }} catch (error) {{
                            console.error('Plotly chart error:', error);
                            document.getElementById('{unique_id}').innerHTML = '<div class="chart-error">Chart rendering failed: ' + error.message + '</div>';
                        }}
                    }});
                    </script>
                    '''
                    html_content += new_script
            
            print(f"[HTMLGenerator] Successfully cleaned Plotly HTML, unique ID: {unique_id}")
            return html_content
            
        except Exception as e:
            print(f"[HTMLGenerator] Error cleaning Plotly HTML: {str(e)}")
            # Return original content if cleaning fails
            return html_content
    
    def _validate_plotly_html_structure(self, html_content: str) -> bool:
        """Validate that Plotly HTML has proper structure"""
        import re
        
        try:
            # Check for essential Plotly components
            has_div = bool(re.search(r'<div[^>]*class="plotly-graph-div"[^>]*>', html_content) or 
                          re.search(r'<div[^>]*id="plotly-chart-[^"]*"[^>]*>', html_content))
            has_script = '<script' in html_content and 'Plotly.' in html_content
            
            # Check for basic HTML structure
            has_proper_tags = '<div' in html_content and '</div>' in html_content
            
            # Validate no unclosed tags (basic check)
            open_divs = len(re.findall(r'<div[^>]*>', html_content))
            close_divs = len(re.findall(r'</div>', html_content))
            balanced_divs = abs(open_divs - close_divs) <= 1  # Allow some flexibility
            
            print(f"[HTMLGenerator] Plotly validation - div: {has_div}, script: {has_script}, tags: {has_proper_tags}, balanced: {balanced_divs}")
            
            return has_div and has_script and has_proper_tags and balanced_divs
            
        except Exception as e:
            print(f"[HTMLGenerator] Error validating Plotly HTML: {str(e)}")
            return False
    
    def _extract_static_from_plotly_html(self, html_content: str, chart_type: str) -> str:
        """Extract static content from Plotly HTML or create placeholder with enhanced detection"""
        import re
        
        print(f"[HTMLGenerator] Extracting static content from Plotly HTML for {chart_type}")
        
        # Look for any embedded base64 images in the HTML (multiple patterns)
        img_patterns = [
            r'src=["\'](data:image/[^"\']*)["\']',
            r'"(data:image/png;base64,[A-Za-z0-9+/=]+)"',
            r'"(data:image/jpeg;base64,[A-Za-z0-9+/=]+)"',
            r'"(data:image/svg\+xml;base64,[A-Za-z0-9+/=]+)"'
        ]
        
        for pattern in img_patterns:
            img_match = re.search(pattern, html_content, re.IGNORECASE)
            if img_match:
                img_data = img_match.group(1)
                print(f"[HTMLGenerator] Found embedded image data in Plotly HTML")
                return f'''
                <div style="text-align: center; margin: 20px 0;">
                    <img src="{img_data}" alt="{chart_type}" 
                         style="max-width: 100%; height: auto; border: 1px solid #e0e0e0; 
                                border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
                </div>
                '''
        
        # Look for canvas or svg elements that might contain static content
        if '<canvas' in html_content or '<svg' in html_content:
            print(f"[HTMLGenerator] Found canvas/svg content in Plotly HTML")
            # Extract canvas or svg element
            canvas_match = re.search(r'(<canvas[^>]*>.*?</canvas>)', html_content, re.DOTALL)
            svg_match = re.search(r'(<svg[^>]*>.*?</svg>)', html_content, re.DOTALL)
            
            if canvas_match:
                return f'''
                <div style="text-align: center; margin: 20px 0;">
                    {canvas_match.group(1)}
                </div>
                '''
            elif svg_match:
                return f'''
                <div style="text-align: center; margin: 20px 0;">
                    {svg_match.group(1)}
                </div>
                '''
        
        # Check if HTML contains viewport meta tag or other indicators of full HTML document
        if '<!DOCTYPE' in html_content or '<html' in html_content:
            # Try to extract just the chart div and relevant scripts
            div_match = re.search(r'(<div[^>]*plotly[^>]*>.*?</div>)', html_content, re.DOTALL | re.IGNORECASE)
            if div_match:
                return self._create_chart_placeholder(f'Static rendering not supported for full HTML document ({chart_type})')
        
        # If no static content found, create an informative placeholder
        print(f"[HTMLGenerator] No static content found in Plotly HTML, creating placeholder")
        return self._create_chart_placeholder(f'Interactive {chart_type} chart - please view in HTML format for full functionality')
    
    def _is_base64_image(self, content: str) -> bool:
        """Check if content contains base64 image data"""
        return 'data:image/' in content and 'base64,' in content
    
    def _process_base64_image(self, content: str, chart_type: str) -> str:
        """Process base64 image content"""
        import re
        
        # Extract base64 data URL if it's wrapped in markdown or other format
        data_url_match = re.search(r'(data:image/[^;]*;base64,[^)"\s]*)', content)
        if data_url_match:
            data_url = data_url_match.group(1)
            return f'''
            <div style="text-align: center; margin: 20px 0;">
                <img src="{data_url}" alt="{chart_type}" 
                     style="max-width: 100%; height: auto; border: 1px solid #e0e0e0; 
                            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
            </div>
            '''
        
        # If content starts with data:image/, it's already a data URL
        if content.startswith('data:image/'):
            return f'''
            <div style="text-align: center; margin: 20px 0;">
                <img src="{content}" alt="{chart_type}" 
                     style="max-width: 100%; height: auto; border: 1px solid #e0e0e0; 
                            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
            </div>
            '''
        
        return self._create_chart_placeholder('Invalid image data')
    
    def _process_markdown_image(self, content: str, chart_type: str) -> str:
        """Process markdown image syntax"""
        import re
        
        # Extract image URL from markdown syntax ![alt](url)
        img_match = re.search(r'!\[.*?\]\(([^)]+)\)', content)
        if img_match:
            img_url = img_match.group(1)
            if img_url.startswith('data:image/'):
                return self._process_base64_image(img_url, chart_type)
            else:
                return f'<img src="{img_url}" alt="{chart_type}" style="max-width: 100%; height: auto;">'
        
        return self._create_chart_placeholder('Invalid markdown image')
    
    def _regenerate_chart_from_data(self, data: str, chart_type: str, chart_generator) -> str:
        """Try to regenerate chart from raw data"""
        import json
        
        try:
            # Try to parse as JSON
            data_dict = json.loads(data)
            
            # Attempt to regenerate based on chart type
            if chart_type.lower() in ['gantt', 'timeline']:
                return chart_generator.generate_gantt_chart(data_dict)
            elif chart_type.lower() in ['budget', 'pie', 'cost']:
                return chart_generator.create_budget_chart(data_dict)
            elif chart_type.lower() in ['risk', 'matrix']:
                return chart_generator.build_risk_matrix(data_dict.get('risks', []))
            elif chart_type.lower() in ['resource', 'team']:
                return chart_generator.generate_resource_chart(data_dict)
            else:
                # Try generic chart generation with available data
                if 'categories' in data_dict and 'values' in data_dict:
                    return chart_generator.create_budget_chart(data_dict)
                else:
                    return self._create_chart_placeholder(f'Unsupported chart type: {chart_type}')
                
        except json.JSONDecodeError:
            # If not JSON, treat as plain text but check if it might be HTML
            if '<' in data and '>' in data:
                return self._validate_and_clean_chart_html(data)
            else:
                return self._create_chart_placeholder(f'Chart data: {data[:100]}...' if len(data) > 100 else data)
    
    def _set_chart_output_format(self, format_type):
        """Set the output format for chart generation tools in current thread"""
        import threading
        threading.current_thread()._chart_output_format = format_type

    def _create_chart_placeholder(self, message: str) -> str:
        """Create a placeholder for charts that can't be processed with improved styling"""
        # Remove emoji for professional appearance as per project guidelines
        return f'''
        <div class="chart-placeholder" style="
            padding: 30px 20px; 
            text-align: center; 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px dashed #dee2e6; 
            border-radius: 12px; 
            color: #6c757d; 
            margin: 20px 0;
            font-style: italic;
            font-size: 14px;
            line-height: 1.5;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; color: #495057;">Chart Placeholder</div>
            <div>{message}</div>
        </div>
        '''

    def _markdown_to_html(self, content: str) -> str:
        """Enhanced markdown to HTML conversion with table support"""
        if not content:
            return ""

        import re

        # Remove JSON code blocks first
        content = re.sub(r'```json\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'```\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)

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

    def _validate_and_clean_chart_html(self, html_content: str) -> str:
        """Validate and clean chart HTML content for proper embedding with enhanced validation"""
        import re
        import uuid
        
        try:
            print(f"[HTMLGenerator] Validating and cleaning chart HTML, length: {len(html_content)}")
            
            # Remove potentially dangerous script content for security
            if self.output_format == 'interactive':
                # Remove external Plotly script tags since we include CDN
                external_patterns = [
                    r'<script[^>]*src[^>]*plotly[^>]*></script>',
                    r'<script[^>]*src[^>]*plot\.ly[^>]*></script>',
                    r'<script[^>]*src[^>]*cdn\.plot\.ly[^>]*></script>'
                ]
                for pattern in external_patterns:
                    html_content = re.sub(pattern, '', html_content, flags=re.IGNORECASE)
            
            # Validate HTML structure
            if not self._is_valid_html_structure(html_content):
                print(f"[HTMLGenerator] Invalid HTML structure detected")
                return None
            
            # Generate unique ID to prevent conflicts
            unique_id = f"chart-{str(uuid.uuid4())[:8]}"
            
            # Replace all IDs to prevent conflicts
            id_patterns = [
                r'id="([^"]+)"',
                r"id='([^']+)'"
            ]
            
            old_ids = []
            for pattern in id_patterns:
                matches = re.finditer(pattern, html_content)
                for match in matches:
                    old_id = match.group(1)
                    if old_id not in old_ids:
                        old_ids.append(old_id)
            
            # Replace all occurrences of old IDs with unique ID
            for old_id in old_ids:
                html_content = html_content.replace(f'id="{old_id}"', f'id="{unique_id}"')
                html_content = html_content.replace(f"id='{old_id}'", f"id='{unique_id}'")
                # Also replace in JavaScript references
                html_content = html_content.replace(f'"{old_id}"', f'"{unique_id}"')
                html_content = html_content.replace(f"'{old_id}'", f"'{unique_id}'")
            
            # Ensure proper CSS classes for styling
            if '<div' in html_content and 'class=' not in html_content:
                html_content = re.sub(
                    r'<div([^>]*id="' + unique_id + r'"[^>]*)>',
                    r'<div\1 class="chart-content">',
                    html_content
                )
            
            # Wrap standalone content in a container div if needed
            if not html_content.strip().startswith('<div'):
                html_content = f'<div class="chart-wrapper" id="{unique_id}">{html_content}</div>'
            
            # Add error boundaries for JavaScript execution
            if '<script' in html_content and 'try' not in html_content:
                script_pattern = r'<script[^>]*>(.*?)</script>'
                scripts = re.findall(script_pattern, html_content, re.DOTALL)
                if scripts:
                    html_content = re.sub(script_pattern, '', html_content, flags=re.DOTALL)
                    for script_content in scripts:
                        if script_content.strip():
                            wrapped_script = f'''
                            <script type="text/javascript">
                            try {{
                                {script_content}
                            }} catch (error) {{
                                console.error('Chart script error:', error);
                                var element = document.getElementById('{unique_id}');
                                if (element) {{
                                    element.innerHTML = '<div class="chart-error">Chart rendering failed: ' + error.message + '</div>';
                                }}
                            }}
                            </script>
                            '''
                            html_content += wrapped_script
            
            print(f"[HTMLGenerator] Successfully validated and cleaned chart HTML")
            return html_content
            
        except Exception as e:
            print(f"[HTMLGenerator] Error validating chart HTML: {str(e)}")
            return f'<div class="chart-error">Chart validation failed: {str(e)}</div>'
    
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