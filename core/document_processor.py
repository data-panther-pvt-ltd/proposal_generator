"""
Smart Document Processor with Semantic Chunking
Implements intelligent text splitting with overlap and metadata preservation
Simple fallback PDF processing without external dependencies
"""

import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import logging
from dataclasses import dataclass
import json

if TYPE_CHECKING:
    from core.simple_cost_tracker import SimpleCostTracker

logger = logging.getLogger(__name__)

# Try PyPDF2 for simple PDF extraction
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    text: str
    chunk_id: str
    page_numbers: List[int]
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "page_numbers": self.page_numbers,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "metadata": self.metadata
        }

class DocumentProcessor:
    """Process documents with semantic chunking"""
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,  # 15% of 800
        encoding_model: str = "cl100k_base",  # GPT-4 encoding
        min_chunk_size: int = 100,
        config: Optional[Dict] = None,
        cost_tracker: Optional['SimpleCostTracker'] = None
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            encoding_model: Tiktoken encoding model
            min_chunk_size: Minimum tokens per chunk
            config: Optional configuration dictionary
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.encoding = tiktoken.get_encoding(encoding_model)  # Keep for chunking operations
        self.config = config or {}
        self.cost_tracker = cost_tracker
        
        # Semantic boundaries for intelligent splitting
        self.section_markers = [
            r'\n#{1,6}\s+',  # Markdown headers
            r'\n\d+\.\s+',    # Numbered sections
            r'\n[A-Z][^.!?]*:\s*\n',  # Section titles
            r'\n\n',          # Paragraph breaks
            r'\.\s+',         # Sentence endings
        ]
    
    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Process PDF using RFPParser first, then fallback to PyPDF2
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document chunks with metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Try RFPParser first for better extraction
        try:
            chunks = self.process_pdf_with_rfp_parser(pdf_path)
            if chunks:
                return chunks
        except Exception as e:
            logger.debug(f"RFPParser not available or failed: {e}")
        
        # Fallback to PyPDF2
        if not PYPDF2_AVAILABLE:
            logger.warning("PyPDF2 not available, using basic text extraction")
            return self._basic_pdf_processing(pdf_path)
        
        try:
            pages_text = []
            
            # Extract text using PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        # Clean the text
                        text = self._clean_text(text)
                        pages_text.append((page_num, text))
                        logger.debug(f"Extracted {len(text)} characters from page {page_num}")
            
            if not pages_text:
                logger.warning(f"No text extracted from {pdf_path}")
                return []
            
            logger.info(f"Extracted text from {len(pages_text)} pages")
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(pages_text, pdf_path)
            
            logger.info(f"Created {len(chunks)} chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return self._basic_pdf_processing(pdf_path)
    
    def _basic_pdf_processing(self, pdf_path: str) -> List[DocumentChunk]:
        """Simple fallback PDF processing using OpenAI if available"""
        try:
            import os
            from openai import OpenAI
            
            # Check if OpenAI API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OpenAI API key not found for fallback processing")
                return []
            
            client = OpenAI(api_key=api_key)
            
            # Read PDF file
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            
            # Use OpenAI to extract text (this is a placeholder - actual implementation would need proper PDF handling)
            logger.info("Using OpenAI for PDF text extraction")
            
            # For now, return empty chunks as OpenAI doesn't directly process PDFs
            # You would need to convert PDF to images or text first
            # When API calls are implemented here, use self.cost_tracker.track_completion(response, model)
            return []
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return []
    
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camelCase
        
        # Preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    
    def _create_semantic_chunks(
        self,
        pages_text: List[Tuple[int, str]],
        source_path: str
    ) -> List[DocumentChunk]:
        """
        Create semantic chunks from page text
        
        Args:
            pages_text: List of (page_number, text) tuples
            source_path: Source file path
            
        Returns:
            List of document chunks
        """
        chunks = []
        chunk_id = 0
        
        # Combine all text with page tracking
        full_text = ""
        char_to_page = {}
        current_pos = 0
        
        for page_num, text in pages_text:
            for char in text:
                char_to_page[current_pos] = page_num
                current_pos += 1
                full_text += char
        
        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(full_text)
        
        # Create chunks respecting boundaries
        current_chunk = ""
        current_tokens = 0
        chunk_start = 0
        chunk_pages = set()
        
        for i, boundary in enumerate(boundaries):
            segment = full_text[boundaries[i-1] if i > 0 else 0:boundary]
            segment_tokens = len(self.encoding.encode(segment))
            
            # Check if adding segment would exceed chunk size
            if current_tokens + segment_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_end = boundaries[i-1] if i > 0 else boundary
                
                # Get page numbers for chunk
                for pos in range(chunk_start, chunk_end):
                    if pos in char_to_page:
                        chunk_pages.add(char_to_page[pos])
                
                chunks.append(DocumentChunk(
                    text=current_chunk.strip(),
                    chunk_id=f"chunk_{chunk_id:04d}",
                    page_numbers=sorted(list(chunk_pages)),
                    start_char=chunk_start,
                    end_char=chunk_end,
                    token_count=current_tokens,
                    metadata={
                        "source": source_path,
                        "chunk_method": "semantic",
                        "has_overlap": True
                    }
                ))
                
                chunk_id += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and chunks:
                    # Get overlap text from end of current chunk
                    overlap_tokens = min(self.chunk_overlap, current_tokens)
                    overlap_text = self._get_token_overlap(current_chunk, overlap_tokens)
                    current_chunk = overlap_text + segment
                    current_tokens = len(self.encoding.encode(current_chunk))
                else:
                    current_chunk = segment
                    current_tokens = segment_tokens
                
                chunk_start = boundaries[i-1] if i > 0 else 0
                chunk_pages = set()
            else:
                # Add segment to current chunk
                current_chunk += segment
                current_tokens += segment_tokens
        
        # Add final chunk if exists
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_end = len(full_text)
            
            for pos in range(chunk_start, chunk_end):
                if pos in char_to_page:
                    chunk_pages.add(char_to_page[pos])
            
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id:04d}",
                page_numbers=sorted(list(chunk_pages)),
                start_char=chunk_start,
                end_char=chunk_end,
                token_count=current_tokens,
                metadata={
                    "source": source_path,
                    "chunk_method": "semantic",
                    "has_overlap": False
                }
            ))
        
        return chunks
    
    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """
        Find semantic boundaries in text
        
        Args:
            text: Input text
            
        Returns:
            List of boundary positions
        """
        boundaries = []
        
        # Find all potential boundaries using patterns
        for pattern in self.section_markers:
            for match in re.finditer(pattern, text):
                boundaries.append(match.end())
        
        # Add start and end
        boundaries.append(0)
        boundaries.append(len(text))
        
        # Sort and deduplicate
        boundaries = sorted(list(set(boundaries)))
        
        # Merge boundaries that are too close
        min_distance = 200  # Increased minimum characters between boundaries for better chunking
        merged = [boundaries[0]]
        
        for boundary in boundaries[1:]:
            if boundary - merged[-1] >= min_distance:
                merged.append(boundary)
        
        # If we have very few boundaries, create artificial ones based on chunk size
        if len(merged) < 5:  # Require at least 5 boundaries
            # Create boundaries every ~1000 characters for better chunking
            step_size = 1000
            for i in range(step_size, len(text), step_size):
                # Find nearest sentence end or paragraph break
                search_start = max(0, i - 100)
                search_end = min(len(text), i + 100)
                
                # Try to find paragraph break first
                para_break = text.find('\n\n', search_start, search_end)
                if para_break != -1:
                    merged.append(para_break + 2)
                else:
                    # Try sentence end
                    sentence_end = text.find('. ', search_start, search_end)
                    if sentence_end != -1:
                        merged.append(sentence_end + 2)
                    else:
                        # Force boundary at position
                        merged.append(i)
            
            merged = sorted(list(set(merged)))
        
        logger.debug(f"Found {len(merged)} boundaries in text of length {len(text)}")
        return merged
    
    def _get_token_overlap(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens from text for overlap
        
        Args:
            text: Source text
            overlap_tokens: Number of tokens to extract
            
        Returns:
            Overlap text
        """
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= overlap_tokens:
            return text
        
        # Get last N tokens
        overlap_token_ids = tokens[-overlap_tokens:]
        
        # Decode back to text
        overlap_text = self.encoding.decode(overlap_token_ids)
        
        # Try to start at a word boundary
        first_space = overlap_text.find(' ')
        if first_space > 0 and first_space < 50:
            overlap_text = overlap_text[first_space:].lstrip()
        
        return overlap_text
    
    def process_text(self, text: str, source: str = "text") -> List[DocumentChunk]:
        """
        Process raw text into chunks
        
        Args:
            text: Input text
            source: Source identifier
            
        Returns:
            List of document chunks
        """
        # Clean text
        text = self._clean_text(text)
        
        # Find boundaries
        boundaries = self._find_semantic_boundaries(text)
        
        chunks = []
        chunk_id = 0
        
        for i in range(len(boundaries) - 1):
            chunk_text = text[boundaries[i]:boundaries[i+1]].strip()
            
            if not chunk_text:
                continue
            
            token_count = len(self.encoding.encode(chunk_text))
            
            if token_count < self.min_chunk_size:
                continue
            
            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_id=f"chunk_{chunk_id:04d}",
                page_numbers=[],
                start_char=boundaries[i],
                end_char=boundaries[i+1],
                token_count=token_count,
                metadata={
                    "source": source,
                    "chunk_method": "semantic",
                    "has_overlap": False
                }
            ))
            
            chunk_id += 1
        
        return chunks
    
    def process_pdf_with_rfp_parser(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Process PDF using RFPParser for better extraction
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            from utils.rfp_parser import RFPParser
            parser = RFPParser()
            
            logger.info(f"Using RFPParser for PDF: {pdf_path}")
            parsed_data = parser.parse_pdf(pdf_path)
            
            # Extract text from sections
            pages_text = []
            
            # Add raw text
            if 'raw_text' in parsed_data:
                pages_text.append((1, parsed_data['raw_text']))
            
            # Add sections as separate pages
            if 'sections' in parsed_data:
                for i, (section_name, section_content) in enumerate(parsed_data['sections'].items(), 2):
                    if section_content:
                        pages_text.append((i, f"## {section_name}\n{section_content}"))
            
            if pages_text:
                return self._create_semantic_chunks(pages_text, pdf_path)
            else:
                # Fallback to basic processing
                return self.process_pdf(pdf_path)
                
        except Exception as e:
            logger.warning(f"RFPParser failed: {e}, using basic processing")
            return self.process_pdf(pdf_path)
    
    def get_stats(self, chunks: List[DocumentChunk]) -> Dict:
        """Get statistics about chunks"""
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "min_tokens": 0,
                "max_completion_tokens": 0,
                "total_pages": 0
            }
        
        token_counts = [c.token_count for c in chunks]
        all_pages = set()
        
        for chunk in chunks:
            all_pages.update(chunk.page_numbers)
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_completion_tokens": max(token_counts),
            "total_pages": len(all_pages),
            "pages_covered": sorted(list(all_pages))
        }