"""
Vector Store Manager using FAISS for local vector database storage
Implements efficient similarity search with IVF and PQ optimization
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(
        self,
        dimension: int = 1536,  # OpenAI text-embedding-ada-002 dimension
        index_type: str = "auto",  # Auto-select based on dataset size
        use_gpu: bool = False
    ):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ("auto" for automatic selection based on size)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu and self._check_gpu_availability()
        
        # Initialize index
        self.index = None
        self.id_to_metadata = {}
        self.next_id = 0
        
        # Store original index type for dynamic selection
        self.original_index_type = index_type
        self.current_index_type = None
        
        # Don't create index immediately for auto mode
        if index_type != "auto":
            self._create_index()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for FAISS"""
        try:
            import faiss.contrib.torch_utils
            return faiss.get_num_gpus() > 0
        except:
            return False
    
    def _select_index_type(self, n_vectors: int) -> str:
        """
        Select appropriate index type based on dataset size
        
        Args:
            n_vectors: Number of vectors in the dataset
            
        Returns:
            Appropriate index type string
        """
        if n_vectors <= 10:
            # Very small dataset - force flat index to avoid segfaults
            logger.warning(f"Using Flat index for very small dataset ({n_vectors} vectors) to avoid segmentation faults")
            return "Flat"
        elif n_vectors < 100:
            # Small dataset - use flat index (no training needed)
            return "Flat"
        elif n_vectors < 1000:
            # Small dataset - use simple IVF with few clusters
            n_clusters = min(n_vectors // 4, 100)  # At least 4 vectors per cluster
            return f"IVF{n_clusters},Flat"
        elif n_vectors < 10000:
            # Medium dataset - use IVF with PQ compression
            n_clusters = min(n_vectors // 4, 1024)
            return f"IVF{n_clusters},PQ32"
        else:
            # Large dataset - use more clusters and stronger compression
            n_clusters = min(n_vectors // 4, 4096)
            return f"IVF{n_clusters},PQ64"
    
    def _create_index(self, n_vectors: Optional[int] = None) -> None:
        """
        Create FAISS index based on configuration
        
        Args:
            n_vectors: Optional number of vectors for auto index selection
        """
        
        # Determine index type
        if self.original_index_type == "auto":
            if n_vectors is None:
                # Default to flat index for unknown size
                self.current_index_type = "Flat"
            else:
                self.current_index_type = self._select_index_type(n_vectors)
            logger.info(f"Auto-selected index type: {self.current_index_type} for {n_vectors} vectors")
        else:
            self.current_index_type = self.index_type
        
        if "IVF" in self.current_index_type:
            # Create IVF index with PQ compression
            # IVF4096,PQ64 means 4096 clusters with 64-byte PQ encoding
            
            # Start with a flat quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            
            # Parse index configuration
            parts = self.current_index_type.split(",")
            n_clusters = int(parts[0].replace("IVF", ""))
            
            if len(parts) > 1 and "PQ" in parts[1]:
                # Product Quantization for compression
                m = int(parts[1].replace("PQ", ""))  # Number of subquantizers
                n_bits = 8  # Bits per subquantizer
                
                self.index = faiss.IndexIVFPQ(
                    quantizer,
                    self.dimension,
                    n_clusters,
                    m,
                    n_bits
                )
            elif len(parts) > 1 and "Flat" in parts[1]:
                # Standard IVF without compression
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.dimension,
                    n_clusters
                )
            else:
                # Default IVF flat
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.dimension,
                    n_clusters
                )
            
            # Set search parameters based on number of clusters
            self.index.nprobe = min(64, max(1, n_clusters // 16))  # Adaptive nprobe
            
        else:
            # Default to flat L2 index with ID support
            base_index = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
        
        # Move to GPU if available and requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU acceleration for FAISS")
            except Exception as e:
                logger.warning(f"Failed to use GPU: {e}, falling back to CPU")
                self.use_gpu = False
        
        logger.info(f"Created FAISS index: {self.current_index_type}")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add embeddings to the index
        
        Args:
            embeddings: Numpy array of embeddings (n_samples, dimension)
            metadata: List of metadata dictionaries for each embedding
            
        Returns:
            List of assigned IDs
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Create index if not exists (for auto mode)
        if self.index is None:
            n_vectors = len(embeddings)
            logger.info(f"Creating index for {n_vectors} vectors")
            self._create_index(n_vectors)
        
        # Train index if needed (for IVF indices)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
            logger.info("FAISS index training complete")
        
        # Generate IDs
        ids = list(range(self.next_id, self.next_id + len(embeddings)))
        
        # Store metadata
        for i, meta in enumerate(metadata):
            self.id_to_metadata[ids[i]] = meta
        
        # Add to index
        if hasattr(self.index, 'add_with_ids'):
            self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
        else:
            self.index.add(embeddings)
        
        self.next_id += len(embeddings)
        
        logger.info(f"Added {len(embeddings)} embeddings to index (total: {self.index.ntotal})")
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_func: Optional function to filter results
            
        Returns:
            List of (metadata, distance) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Safety check for very small indices
        if self.index.ntotal <= 2:
            logger.warning(f"Very small index ({self.index.ntotal} vectors), using brute force search")
            # For very small indices, return all vectors
            k = min(k, self.index.ntotal)
        
        # Ensure query is the right shape and type
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search with safety for small indices
        search_k = min(k * 2, self.index.ntotal)
        try:
            distances, indices = self.index.search(query_embedding, search_k)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            # Fallback: return empty results
            return []
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
                
            metadata = self.id_to_metadata.get(int(idx))
            if metadata is None:
                continue
            
            # Apply filter if provided
            if filter_func and not filter_func(metadata):
                continue
            
            results.append((metadata, float(dist)))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> List[List[Tuple[Dict[str, Any], float]]]:
        """
        Search for multiple queries in batch
        
        Args:
            query_embeddings: Multiple query embeddings (n_queries, dimension)
            k: Number of results per query
            
        Returns:
            List of result lists
        """
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Batch search
        distances, indices = self.index.search(query_embeddings, min(k, self.index.ntotal))
        
        # Collect results for each query
        all_results = []
        for query_dists, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_dists, query_indices):
                if idx == -1:
                    continue
                    
                metadata = self.id_to_metadata.get(int(idx))
                if metadata:
                    results.append((metadata, float(dist)))
            
            all_results.append(results)
        
        return all_results
    
    def save(self, directory: Path) -> None:
        """
        Save index and metadata to disk
        
        Args:
            directory: Directory to save index files
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        index_path = directory / "index.faiss"
        
        # Move to CPU for saving if on GPU
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = directory / "id_to_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.id_to_metadata, f)
        
        # Save configuration
        config_path = directory / "config.json"
        config = {
            "dimension": self.dimension,
            "index_type": self.original_index_type,
            "current_index_type": self.current_index_type,
            "next_id": self.next_id,
            "total_vectors": self.index.ntotal if self.index else 0
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved vector store to {directory}")
    
    def load(self, directory: Path) -> None:
        """
        Load index and metadata from disk
        
        Args:
            directory: Directory containing index files
        """
        directory = Path(directory)
        
        # Load configuration
        config_path = directory / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.dimension = config["dimension"]
        self.original_index_type = config.get("index_type", "auto")
        self.current_index_type = config.get("current_index_type", config.get("index_type", "Flat"))
        self.next_id = config["next_id"]
        
        # Load FAISS index
        index_path = directory / "index.faiss"
        self.index = faiss.read_index(str(index_path))
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                logger.warning(f"Failed to use GPU: {e}")
                self.use_gpu = False
        
        # Load metadata
        metadata_path = directory / "id_to_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            self.id_to_metadata = pickle.load(f)
        
        logger.info(f"Loaded vector store from {directory} ({self.index.ntotal} vectors)")
    
    def clear(self) -> None:
        """Clear all embeddings from the index"""
        # Reset to initial state
        self.index = None
        self.id_to_metadata = {}
        self.next_id = 0
        self.current_index_type = None
        
        # Only create index if not in auto mode
        if self.original_index_type != "auto":
            self._create_index()
        
        logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": self.current_index_type or self.original_index_type,
            "original_index_type": self.original_index_type,
            "using_gpu": self.use_gpu,
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if not self.index:
            return 0.0
        
        # Rough estimation based on index type and number of vectors
        n_vectors = self.index.ntotal
        index_type = self.current_index_type or self.original_index_type
        
        if "PQ" in index_type:
            # PQ compressed index
            pq_bytes = int(index_type.split("PQ")[1]) if "PQ" in index_type else 64
            bytes_per_vector = pq_bytes + 8  # PQ code + ID
        elif "IVF" in index_type:
            # IVF flat index
            bytes_per_vector = self.dimension * 4 + 8  # float32 + ID
        else:
            # Flat index
            bytes_per_vector = self.dimension * 4
        
        # Add metadata overhead (rough estimate)
        metadata_bytes = len(str(self.id_to_metadata)) if self.id_to_metadata else 0
        
        total_bytes = n_vectors * bytes_per_vector + metadata_bytes
        return round(total_bytes / (1024 * 1024), 2)