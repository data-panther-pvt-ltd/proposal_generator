"""
OpenAI Embedding Service - Direct API implementation without LangChain
Uses OpenAI's text-embedding-3-large model for high-quality embeddings
"""

import numpy as np
from openai import OpenAI
from typing import List, Dict, Any, Optional
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.simple_cost_tracker import SimpleCostTracker

logger = logging.getLogger(__name__)

class EmbeddingService:
    """OpenAI embedding service with batching and retry logic"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = 1536,  # Dimensions for text-embedding-ada-002
        batch_size: int = 100,
        max_retries: int = 3,
        cost_tracker: Optional['SimpleCostTracker'] = None
    ):
        """
        Initialize embedding service
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            dimensions: Output dimensions (3072 for large, can be reduced)
            batch_size: Number of texts to embed in one API call
            max_retries: Maximum retry attempts
        """
        # Make OpenAI API key mandatory - no fallbacks
        if not api_key:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenAI API key is required. Please set OPENAI_API_KEY environment variable. "
                    "No fallback methods are allowed."
                )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cost_tracker = cost_tracker
        
        # Token counting for the model
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Model-specific limits - support both old and new models
        self.model_limits = {
            "text-embedding-ada-002": {
                "max_completion_tokens": 8191,
                "dimensions": 1536
            },
            "text-embedding-3-small": {
                "max_completion_tokens": 8191,
                "dimensions": 1536
            },
            "text-embedding-3-large": {
                "max_completion_tokens": 8191,
                "dimensions": 3072
            }
        }
        
        # These will be updated from actual API responses, not tiktoken estimates
        self.total_tokens_used = 0
        self.total_api_calls = 0
        
        logger.info(f"Initialized embedding service with model: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call OpenAI embedding API with retry logic
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Create embeddings
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            # Track usage from actual API response
            self.total_api_calls += 1
            actual_tokens_used = 0
            
            # Get actual token usage from API response
            if hasattr(response, 'usage') and response.usage:
                actual_tokens_used = response.usage.total_tokens
                self.total_tokens_used += actual_tokens_used
                
                # Track cost if cost_tracker is available  
                if self.cost_tracker:
                    self.cost_tracker.track_completion(response, self.model)
                    
                logger.debug(f"Embedded {len(texts)} texts using {actual_tokens_used} actual tokens from API")
            else:
                # Fallback to tiktoken estimate only if API doesn't provide usage
                estimated_tokens = sum(len(self.encoding.encode(text)) for text in texts)
                self.total_tokens_used += estimated_tokens
                logger.warning(f"API response missing usage data, using tiktoken estimate: {estimated_tokens} tokens")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings (n_texts, dimensions)
        """
        if not texts:
            return np.array([])
        
        # Validate and truncate texts if needed
        processed_texts = []
        max_completion_tokens = self.model_limits.get(self.model, {}).get("max_completion_tokens", 8191)
        
        for text in texts:
            # Count tokens
            tokens = self.encoding.encode(text)
            
            if len(tokens) > max_completion_tokens:
                # Truncate to fit within limits
                truncated_tokens = tokens[:max_completion_tokens]
                truncated_text = self.encoding.decode(truncated_tokens)
                processed_texts.append(truncated_text)
                logger.warning(f"Truncated text from {len(tokens)} to {max_completion_tokens} tokens")
            else:
                processed_texts.append(text)
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(processed_texts), self.batch_size):
            batch = processed_texts[i:i + self.batch_size]
            
            # Add delay to avoid rate limits
            if i > 0:
                time.sleep(0.1)
            
            embeddings = self._call_embedding_api(batch)
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def embed_with_metadata(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Embed texts with associated metadata
        
        Args:
            texts: List of texts to embed
            metadata: List of metadata dictionaries
            
        Returns:
            List of dictionaries with embeddings and metadata
        """
        if len(texts) != len(metadata):
            raise ValueError("Number of texts must match number of metadata items")
        
        embeddings = self.embed_texts(texts)
        
        results = []
        for i, (text, meta, embedding) in enumerate(zip(texts, metadata, embeddings)):
            result = {
                "text": text,
                "embedding": embedding.tolist(),
                "metadata": meta,
                "embedding_model": self.model,
                "dimensions": len(embedding)
            }
            results.append(result)
        
        return results
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """
        Create embedding for a search query
        
        Args:
            query: Search query
            
        Returns:
            Query embedding
        """
        # For queries, we might want to add a prefix for better retrieval
        # This is a best practice for some embedding models
        prefixed_query = f"Search query: {query}"
        
        return self.embed_text(prefixed_query)
    
    def create_hyde_embedding(self, query: str, hypothetical_answer: str) -> np.ndarray:
        """
        Create HyDE (Hypothetical Document Embedding) for improved retrieval
        
        Args:
            query: Original query
            hypothetical_answer: Generated hypothetical answer
            
        Returns:
            HyDE embedding
        """
        # Combine query context with hypothetical answer
        hyde_text = f"Question: {query}\n\nAnswer: {hypothetical_answer}"
        
        return self.embed_text(hyde_text)
    
    def batch_embed_with_progress(
        self,
        texts: List[str],
        progress_callback: Optional[callable] = None
    ) -> np.ndarray:
        """
        Embed texts with progress tracking
        
        Args:
            texts: List of texts to embed
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Numpy array of embeddings
        """
        total = len(texts)
        all_embeddings = []
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Process batch
            embeddings = self._call_embedding_api(batch)
            all_embeddings.extend(embeddings)
            
            # Update progress
            processed = min(i + self.batch_size, total)
            progress = processed / total * 100
            
            if progress_callback:
                progress_callback(processed, total, progress)
            
            logger.info(f"Embedding progress: {processed}/{total} ({progress:.1f}%)")
            
            # Rate limiting
            if i + self.batch_size < total:
                time.sleep(0.1)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def estimate_tokens(self, texts: List[str]) -> Dict[str, int]:
        """
        Estimate token usage for texts
        
        Args:
            texts: List of texts to estimate
            
        Returns:
            Token estimation dictionary
        """
        total_tokens = sum(len(self.encoding.encode(text)) for text in texts)
        
        model_info = self.model_limits.get(self.model, {})
        
        return {
            "total_texts": len(texts),
            "total_tokens": total_tokens,
            "model": self.model,
            "max_completion_tokens_per_text": model_info.get("max_completion_tokens", 8191)
        }
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        
        return {
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "model": self.model,
            "dimensions": self.dimensions
        }
    
    def reduce_dimensions(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Reduce embedding dimensions using PCA
        
        Args:
            embeddings: Original embeddings
            target_dim: Target dimension size
            
        Returns:
            Reduced embeddings
        """
        from sklearn.decomposition import PCA
        
        if embeddings.shape[1] <= target_dim:
            return embeddings
        
        pca = PCA(n_components=target_dim)
        reduced = pca.fit_transform(embeddings)
        
        logger.info(f"Reduced embeddings from {embeddings.shape[1]} to {target_dim} dimensions")
        
        return reduced.astype(np.float32)