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
# import weasyprint
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
    """Export HTML to PDF with Playwright screenshot support for charts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config['output']['output_directory']
        self.screenshot_for_pdf = config.get('charts', {}).get('screenshot_for_pdf', False)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Screenshot settings
        self.screenshot_temp_dir = os.path.join(self.output_dir, 'temp_screenshots')
        if self.screenshot_for_pdf and PLAYWRIGHT_AVAILABLE:
            os.makedirs(self.screenshot_temp_dir, exist_ok=True)
        
        # Chart settings
        self.chart_render_timeout = 10000  # 10 seconds
        self.chart_wait_timeout = 2000     # 2 seconds after render
        
        if self.screenshot_for_pdf and not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available. Chart screenshots disabled. Install with: pip install playwright && playwright install chromium")
    
    def export(self, html_content: str, client_name: str) -> Optional[str]:
        """
        Export HTML content to PDF with optional chart screenshots
        
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
            
            # Process charts with screenshots if enabled
            if self.screenshot_for_pdf and PLAYWRIGHT_AVAILABLE:
                html_content = self._generate_pdf_with_charts(html_content)
            
            # Add print-specific CSS
            html_with_print_css = self._add_print_css(html_content)
            
            # Create PDF
            # html = weasyprint.HTML(string=html_with_print_css)
            html = "Hello"
            pdf = html.write_pdf()
            
            # Save PDF
            with open(filepath, 'wb') as f:
                f.write(pdf)
            
            # Cleanup temporary screenshots
            if self.screenshot_for_pdf and PLAYWRIGHT_AVAILABLE:
                self._cleanup_temp_screenshots()
            
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
                
                .chart-container {
                    page-break-inside: avoid;
                    max-height: 600px;
                }
                
                .no-print {
                    display: none;
                }
            }
        </style>
        '''
        
        # Insert print CSS before closing head tag
        return html_content.replace('</head>', f'{print_css}</head>')
    
    def _generate_pdf_with_charts(self, html_content: str) -> str:
        """Generate PDF with embedded chart screenshots"""
        try:
            # Save HTML to temporary file for Playwright
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
                temp_file.write(html_content)
                temp_html_path = temp_file.name
            
            # Capture chart screenshots - handle both sync and async contexts
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use create_task
                    import nest_asyncio
                    nest_asyncio.apply()
                    chart_screenshots = asyncio.run(self._capture_chart_screenshots(temp_html_path))
                else:
                    # No running loop, use asyncio.run
                    chart_screenshots = asyncio.run(self._capture_chart_screenshots(temp_html_path))
            except RuntimeError:
                # No event loop, create one
                chart_screenshots = asyncio.run(self._capture_chart_screenshots(temp_html_path))
            except ImportError:
                # nest_asyncio not installed, skip screenshot capture
                logger.warning("Chart screenshots disabled - install nest_asyncio for chart capture in async context")
                os.unlink(temp_html_path)
                return html_content
            
            # Replace chart divs with image tags
            modified_html = self._embed_chart_screenshots(html_content, chart_screenshots)
            
            # Cleanup temporary HTML file
            os.unlink(temp_html_path)
            
            return modified_html
            
        except Exception as e:
            logger.error(f"Chart screenshot generation failed: {str(e)}")
            # Cleanup on error
            if 'temp_html_path' in locals():
                try:
                    os.unlink(temp_html_path)
                except:
                    pass
            return html_content  # Fallback to original HTML
    
    async def _capture_chart_screenshots(self, html_file_path: str) -> Dict[str, str]:
        """Capture screenshots of Plotly charts from HTML file"""
        chart_screenshots = {}
        
        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
                
                try:
                    context = await browser.new_context(
                        viewport={'width': 1200, 'height': 800}
                    )
                    page = await context.new_page()
                    
                    # Load HTML file
                    file_url = f"file://{os.path.abspath(html_file_path)}"
                    await page.goto(file_url, wait_until='networkidle')
                    
                    # Wait for Plotly to render
                    await self._wait_for_plotly_render(page)
                    
                    # Find all chart containers
                    chart_containers = await page.query_selector_all('.plotly-graph-div')
                    
                    logger.info(f"Found {len(chart_containers)} chart containers")
                    
                    for i, container in enumerate(chart_containers):
                        try:
                            # Get chart ID or generate one
                            chart_id = await container.get_attribute('id')
                            if not chart_id:
                                chart_id = f"chart_{i}"
                            
                            # Take screenshot of the chart
                            screenshot_bytes = await container.screenshot(
                                type='png',
                                quality=95
                            )
                            
                            # Save as base64
                            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                            chart_screenshots[chart_id] = screenshot_b64
                            
                            logger.debug(f"Captured screenshot for chart: {chart_id}")
                            
                        except Exception as e:
                            logger.error(f"Failed to capture screenshot for chart {i}: {str(e)}")
                    
                    await context.close()
                    
                finally:
                    await browser.close()
        
        except Exception as e:
            logger.error(f"Browser automation failed: {str(e)}")
        
        return chart_screenshots
    
    async def _wait_for_plotly_render(self, page: Page) -> None:
        """Wait for Plotly charts to fully render"""
        try:
            # Wait for Plotly to be available
            await page.wait_for_function(
                "typeof Plotly !== 'undefined'",
                timeout=self.chart_render_timeout
            )
            
            # Wait for all plots to be rendered
            await page.wait_for_function(
                """() => {
                    const plots = document.querySelectorAll('.plotly-graph-div');
                    if (plots.length === 0) return true;
                    
                    for (let plot of plots) {
                        if (!plot._fullLayout || !plot._fullData) {
                            return false;
                        }
                    }
                    return true;
                }""",
                timeout=self.chart_render_timeout
            )
            
            # Additional wait for animations to complete
            await page.wait_for_timeout(self.chart_wait_timeout)
            
            logger.debug("Plotly charts rendered successfully")
            
        except Exception as e:
            logger.warning(f"Plotly render wait failed: {str(e)}. Proceeding anyway.")
    
    def _embed_chart_screenshots(self, html_content: str, chart_screenshots: Dict[str, str]) -> str:
        """Replace chart divs with image tags containing screenshots"""
        modified_html = html_content
        
        for chart_id, screenshot_b64 in chart_screenshots.items():
            try:
                # Create image tag with base64 data
                img_tag = f'<img src="data:image/png;base64,{screenshot_b64}" ' \
                         f'alt="Chart: {chart_id}" ' \
                         f'style="max-width: 100%; height: auto; display: block; margin: 10px auto;" />'
                
                # Replace chart div with image tag
                # Try to find the specific chart by ID first
                chart_div_pattern = f'<div[^>]*id="{chart_id}"[^>]*class="[^"]*plotly-graph-div[^"]*"[^>]*>.*?</div>'
                if re.search(chart_div_pattern, modified_html, re.DOTALL):
                    modified_html = re.sub(chart_div_pattern, img_tag, modified_html, flags=re.DOTALL)
                else:
                    # Fallback: replace first available plotly div
                    general_pattern = r'<div[^>]*class="[^"]*plotly-graph-div[^"]*"[^>]*>.*?</div>'
                    modified_html = re.sub(general_pattern, img_tag, modified_html, count=1, flags=re.DOTALL)
                
                logger.debug(f"Embedded screenshot for chart: {chart_id}")
                
            except Exception as e:
                logger.error(f"Failed to embed screenshot for chart {chart_id}: {str(e)}")
        
        return modified_html
    
    def _cleanup_temp_screenshots(self) -> None:
        """Clean up temporary screenshot files"""
        try:
            if os.path.exists(self.screenshot_temp_dir):
                import shutil
                shutil.rmtree(self.screenshot_temp_dir)
                logger.debug("Cleaned up temporary screenshot files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp screenshots: {str(e)}")