"""
PDF Exporter using WeasyPrint
"""

import os
import re
import base64
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import weasyprint
import logging

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = None
    BrowserContext = None
    Page = None

try:
    from utils.agent_logger import agent_logger
    logger = agent_logger.get_agent_logger('pdf_exporter')
except ImportError:
    logger = logging.getLogger(__name__)

class PDFExporter:

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config['output']['output_directory']
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    
    def export(self, html_content: str, client_name: str) -> Optional[str]:
        """
        Args:
            html_content: HTML string to convert
            client_name: Client name for filename
            
        Returns:
            Path to generated PDF or None if failed
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{client_name.replace(' ', '_')}_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            
            # Add print-specific CSS
            html_with_print_css = self._add_print_css(html_content)
            
            # Create PDF
            html = weasyprint.HTML(string=html_with_print_css)
            pdf = html.write_pdf()
            
            # Save PDF
            with open(filepath, 'wb') as f:
                f.write(pdf)

            
            logger.info(f"PDF exported successfully: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"PDF export failed: {str(e)}")
            return None
    
    def _add_print_css(self, html_content: str) -> str:
        """Add print-specific CSS to HTML"""
        
        print_css = '''
        <style>
            @page {
                size: A4;
                margin: 1.5cm;
                @bottom-right {
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 10pt;
                    color: #666;
                }
            }
            
            @media print {
                body {
                    font-size: 11pt;
                }
                
                .header {
                    page-break-after: avoid;
                }
                
                .toc {
                    page-break-after: always;
                }
                
                .section {
                    page-break-inside: avoid;
                }
                
                h2 {
                    page-break-after: avoid;
                }
                
                table {
                    page-break-inside: avoid;
                }
                
                
                .no-print {
                    display: none;
                }
            }
        </style>
        '''
        
        # Insert print CSS before closing head tag
        return html_content.replace('</head>', f'{print_css}</head>')
    
