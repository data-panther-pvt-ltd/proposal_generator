import pypandoc
import os
from datetime import datetime

class DOCXExporter:
    """Export proposal to DOCX using Pandoc via pypandoc"""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get('output', {}).get('output_directory', 'generated_proposals')
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            # Check for pypandoc. If it's not installed, this will fail.
            import pypandoc

            # Check if pandoc is available. If not, download it.
            try:
                pypandoc.get_pandoc_version()
                print(f"[DOCXExporter] pypandoc version: {pypandoc.get_pandoc_version()}")
            except RuntimeError:
                print("[DOCXExporter] Pandoc not found. Attempting to download...")
                try:
                    pypandoc.download_pandoc()
                    print("[DOCXExporter] Pandoc downloaded successfully.")
                    print(f"[DOCXExporter] pypandoc version: {pypandoc.get_pandoc_version()}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download Pandoc: {str(e)}")

        except ImportError:
            raise ImportError("pypandoc library is not installed. Please run 'pip install pypandoc'.")
        except Exception as e:
            raise RuntimeError(f"pypandoc not properly installed or configured: {str(e)}")

    # In docx_exporter.py - Replace the existing export method
    def export(self, html_content: str, client_name: str) -> str:
        """Convert HTML content to DOCX file and save"""

        print(f"[DOCXExporter] Starting DOCX generation...")

        # Validate inputs
        if not html_content or not html_content.strip():
            raise ValueError("HTML content is empty or None")

        if not client_name or not client_name.strip():
            client_name = "Unknown_Client"
            print(f"[DOCXExporter] Client name was empty, using: {client_name}")

        # Clean client name for filename
        clean_client_name = client_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename_pattern = self.config.get("output", {}).get("filename_pattern", "proposal_{client}_{date}.{format}")
        filename = filename_pattern.format(
            client=clean_client_name,
            date=timestamp,
            format="docx"
        )

        docx_path = os.path.join(self.output_dir, filename)

        print(f"[DOCXExporter] Output directory: {self.output_dir}")
        print(f"[DOCXExporter] Filename: {filename}")
        print(f"[DOCXExporter] Full path: {docx_path}")
        print(f"[DOCXExporter] HTML content length: {len(html_content)}")

        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[DOCXExporter] Output directory created/verified: {self.output_dir}")

            # Check if pypandoc is working
            try:
                import pypandoc
                version = pypandoc.get_pandoc_version()
                print(f"[DOCXExporter] Using pandoc version: {version}")
            except Exception as e:
                raise RuntimeError(f"Pandoc not available: {str(e)}")

            # Clean HTML content for better conversion
            cleaned_html = self._clean_html_for_docx(html_content)
            print(f"[DOCXExporter] HTML cleaned, new length: {len(cleaned_html)}")

            # Convert using pypandoc
            extra_args = ['--standalone']
            
            # Add reference document if it exists
            if os.path.exists('templates/reference.docx'):
                extra_args.append('--reference-doc=templates/reference.docx')
            
            pypandoc.convert_text(
                cleaned_html,
                'docx',
                format='html',
                outputfile=docx_path,
                extra_args=extra_args
            )

            # Verify file was created
            if os.path.exists(docx_path):
                file_size = os.path.getsize(docx_path)
                print(f"[DOCXExporter] ✓ DOCX file created successfully - Size: {file_size} bytes")
                return docx_path
            else:
                raise RuntimeError("DOCX file was not created by pypandoc")

        except Exception as e:
            print(f"[DOCXExporter] ✗ Error details: {str(e)}")
            print(f"[DOCXExporter] ✗ Error type: {type(e).__name__}")
            raise RuntimeError(f"Failed to export DOCX: {str(e)}")

    def _clean_html_for_docx(self, html_content: str) -> str:
        """Clean HTML content for better DOCX conversion"""
        import re

        # Remove problematic elements that don't convert well to DOCX
        cleaned = html_content

        # Remove script tags
        cleaned = re.sub(r'<script[^>]*>.*?</script>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove style tags (keep inline styles)
        cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove problematic CSS classes and IDs
        cleaned = re.sub(r'\s*class="[^"]*"', '', cleaned)
        cleaned = re.sub(r'\s*id="[^"]*"', '', cleaned)

        # Convert div to p for better paragraph handling
        cleaned = re.sub(r'<div([^>]*)>', r'<p\1>', cleaned)
        cleaned = re.sub(r'</div>', '</p>', cleaned)

        # Remove empty paragraphs
        cleaned = re.sub(r'<p[^>]*>\s*</p>', '', cleaned)

        return cleaned