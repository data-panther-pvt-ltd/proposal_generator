"""
Document Cache System for PDF Fingerprinting and Vector DB Management
Handles caching of vector databases for each PDF file using SHA-256 hashing
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DocumentCache:
    """Manages document fingerprinting and vector DB caching"""
    
    def __init__(self, cache_dir: str = "vector_db"):
        """
        Initialize document cache
        
        Args:
            cache_dir: Base directory for storing vector databases
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = "metadata.json"
        
    def get_document_hash(self, pdf_path: str) -> str:
        """
        Calculate SHA-256 hash of PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            SHA-256 hash string
        """
        sha256_hash = hashlib.sha256()
        
        with open(pdf_path, "rb") as f:
            # Read file in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()
    
    def get_vector_db_path(self, pdf_path: str) -> Tuple[Path, str]:
        """
        Get the vector database path for a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (vector_db_directory, document_hash)
        """
        doc_hash = self.get_document_hash(pdf_path)
        vector_db_dir = self.cache_dir / doc_hash
        
        return vector_db_dir, doc_hash
    
    def is_cached(self, pdf_path: str) -> bool:
        """
        Check if vector database exists for PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if cached, False otherwise
        """
        vector_db_dir, _ = self.get_vector_db_path(pdf_path)
        
        # Check if directory exists and contains index file
        if not vector_db_dir.exists():
            return False
            
        index_path = vector_db_dir / "index.faiss"
        metadata_path = vector_db_dir / self.metadata_file
        
        return index_path.exists() and metadata_path.exists()
    
    def get_cache_metadata(self, pdf_path: str) -> Optional[Dict]:
        """
        Get metadata for cached vector database
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Metadata dictionary or None if not cached
        """
        vector_db_dir, _ = self.get_vector_db_path(pdf_path)
        metadata_path = vector_db_dir / self.metadata_file
        
        if not metadata_path.exists():
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
            return None
    
    def save_cache_metadata(
        self, 
        pdf_path: str,
        chunk_count: int,
        embedding_model: str,
        embedding_dim: int,
        chunk_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        additional_metadata: Optional[Dict] = None
    ) -> None:
        """
        Save metadata for vector database cache
        
        Args:
            pdf_path: Path to PDF file
            chunk_count: Number of chunks created
            embedding_model: Model used for embeddings
            embedding_dim: Dimension of embeddings
            chunk_strategy: Chunking strategy used
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks
            additional_metadata: Additional metadata to store
        """
        vector_db_dir, doc_hash = self.get_vector_db_path(pdf_path)
        vector_db_dir.mkdir(exist_ok=True)
        
        metadata = {
            "pdf_path": str(pdf_path),
            "pdf_hash": doc_hash,
            "processed_date": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dim,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "file_size": os.path.getsize(pdf_path),
            "cache_version": "1.0.0"
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        metadata_path = vector_db_dir / self.metadata_file
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved cache metadata for {pdf_path}")
    
    def validate_cache(self, pdf_path: str, embedding_model: str) -> bool:
        """
        Validate if cached vector DB is still valid
        
        Args:
            pdf_path: Path to PDF file
            embedding_model: Expected embedding model
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not self.is_cached(pdf_path):
            return False
        
        metadata = self.get_cache_metadata(pdf_path)
        if not metadata:
            return False
        
        # Check if PDF hash matches (file unchanged)
        current_hash = self.get_document_hash(pdf_path)
        if metadata.get("pdf_hash") != current_hash:
            logger.info(f"PDF has changed, cache invalid for {pdf_path}")
            return False
        
        # Check if embedding model matches
        if metadata.get("embedding_model") != embedding_model:
            logger.info(f"Embedding model mismatch, cache invalid for {pdf_path}")
            return False
        
        # Check cache version compatibility
        if metadata.get("cache_version") != "1.0.0":
            logger.info(f"Cache version incompatible for {pdf_path}")
            return False
        
        return True
    
    def clear_cache(self, pdf_path: str) -> None:
        """
        Clear cached vector database for a PDF
        
        Args:
            pdf_path: Path to PDF file
        """
        vector_db_dir, _ = self.get_vector_db_path(pdf_path)
        
        if vector_db_dir.exists():
            import shutil
            shutil.rmtree(vector_db_dir)
            logger.info(f"Cleared cache for {pdf_path}")
    
    def get_all_cached_documents(self) -> Dict[str, Dict]:
        """
        Get information about all cached documents
        
        Returns:
            Dictionary mapping document hashes to metadata
        """
        cached_docs = {}
        
        for dir_path in self.cache_dir.iterdir():
            if dir_path.is_dir():
                metadata_path = dir_path / self.metadata_file
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            cached_docs[dir_path.name] = metadata
                    except Exception as e:
                        logger.error(f"Error reading metadata from {dir_path}: {e}")
        
        return cached_docs
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache
        
        Returns:
            Dictionary with cache statistics
        """
        cached_docs = self.get_all_cached_documents()
        
        total_size = 0
        total_chunks = 0
        
        for doc_hash, metadata in cached_docs.items():
            vector_db_dir = self.cache_dir / doc_hash
            
            # Calculate directory size
            for path in vector_db_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
            
            total_chunks += metadata.get("chunk_count", 0)
        
        return {
            "total_documents": len(cached_docs),
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_directory": str(self.cache_dir)
        }