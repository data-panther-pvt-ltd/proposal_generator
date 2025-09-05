"""
RAG Retriever with HyDE (Hypothetical Document Embeddings) and Reranking
Implements advanced retrieval strategies for better context extraction
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from openai import OpenAI

from core.document_processor import DocumentProcessor, DocumentChunk
from core.embedding_service import EmbeddingService
from core.vector_store import VectorStore
from core.document_cache import DocumentCache

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Advanced RAG retriever with HyDE and reranking capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG retriever with configuration
        
        Args:
            config: Configuration dictionary from settings.yml
        """
        self.config = config
        self.vector_config = config.get('vector_db', {})
        self.openai_config = config.get('openai', {})
        
        # Track loaded PDFs and their vector stores
        self.current_pdf_path = None
        self.loaded_vector_stores = {}  # Cache multiple vector stores in memory
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=self.vector_config.get('chunk_size', 800),
            chunk_overlap=self.vector_config.get('chunk_overlap', 120),
            min_chunk_size=self.vector_config.get('min_chunk_size', 100),
            config=config  # Pass full config for PDF processing settings
        )
        
        self.embedding_service = EmbeddingService(
            model=self.vector_config.get('embedding_model', 'text-embedding-ada-002'),
            dimensions=self.vector_config.get('embedding_dimensions', 1536),
            batch_size=self.vector_config.get('batch_size', 100)
        )
        
        # Initialize with None - will create on demand for each PDF
        self.vector_store = None
        
        self.document_cache = DocumentCache(
            cache_dir=self.vector_config.get('cache_directory', 'vector_db')
        )
        
        # OpenAI client for HyDE generation
        import os
        api_key = os.getenv(self.openai_config.get('api_key_env', 'OPENAI_API_KEY'))
        self.openai_client = OpenAI(api_key=api_key)
        
        # Model configuration from settings
        self.llm_model = self.openai_config.get('model', 'gpt-4o')
        self.llm_temperature = 0.7
        self.llm_max_tokens = 300
        
        # Settings from configuration file
        self.enable_hyde = self.vector_config.get('enable_hyde', False)
        self.enable_multi_query = self.vector_config.get('enable_multi_query', False)
        self.search_k = self.vector_config.get('search_k', 10)
        self.rerank_top_k = self.vector_config.get('rerank_top_k', 5)
        
        logger.info(f"RAG Retriever initialized with HyDE={self.enable_hyde}, MultiQuery={self.enable_multi_query}")
    
    def _get_or_create_vector_store(self, pdf_path: str) -> VectorStore:
        """Get existing or create new vector store for a PDF"""
        # Check if we have it in memory cache
        if pdf_path in self.loaded_vector_stores:
            logger.debug(f"Using cached vector store for {pdf_path}")
            return self.loaded_vector_stores[pdf_path]
        
        # Create new vector store for this PDF
        logger.info(f"Creating new vector store for {pdf_path}")
        vector_store = VectorStore(
            dimension=self.vector_config.get('embedding_dimensions', 1536),
            index_type=self.vector_config.get('index_type', 'auto'),
            use_gpu=self.vector_config.get('use_gpu', False)
        )
        
        # Cache it
        self.loaded_vector_stores[pdf_path] = vector_store
        return vector_store
    
    def process_and_index_pdf(self, pdf_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Process PDF and create/load vector database
        
        Args:
            pdf_path: Path to PDF file
            force_reindex: Force reindexing even if cache exists
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Check if already loaded in memory
        if not force_reindex and pdf_path in self.loaded_vector_stores:
            logger.debug(f"PDF already loaded in memory: {pdf_path}")
            self.current_pdf_path = pdf_path
            self.vector_store = self.loaded_vector_stores[pdf_path]
            stats = self.vector_store.get_stats()
            return {
                "status": "already_loaded", 
                "pdf_path": pdf_path,
                "num_chunks": stats.get('total_vectors', 0)
            }
        
        # Check disk cache
        if not force_reindex and self.document_cache.is_cached(pdf_path):
            if self.document_cache.validate_cache(pdf_path, self.vector_config.get('embedding_model', 'text-embedding-ada-002')):
                logger.info("Loading existing vector database from cache")
                result = self._load_from_cache(pdf_path)
                self.current_pdf_path = pdf_path
                return result
            else:
                logger.info("Cache invalid, will reindex")
        
        # Process PDF
        logger.info("Creating new vector database")
        try:
            chunks = self.document_processor.process_pdf(pdf_path)
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            # Return error with num_chunks for consistency
            return {
                'status': 'error',
                'message': f'PDF processing failed: {str(e)}',
                'pdf_path': pdf_path,
                'num_chunks': 0
            }
        
        if not chunks:
            logger.warning(f"No chunks extracted from {pdf_path}")
            return {
                'status': 'error',
                'message': 'No content extracted from PDF',
                'pdf_path': pdf_path,
                'num_chunks': 0
            }
        
        logger.info(f"Generated {len(chunks)} chunks from PDF")
        
        # Create embeddings
        logger.info("Creating embeddings for chunks")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        
        # Get or create vector store for this PDF
        vector_store = self._get_or_create_vector_store(pdf_path)
        
        # Clear any existing data in this vector store
        vector_store.clear()
        
        # Store in vector database
        logger.info("Storing embeddings in vector database")
        vector_db_dir, doc_hash = self.document_cache.get_vector_db_path(pdf_path)
        vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Add embeddings to vector store
        metadata_list = [
            {
                'chunk_id': chunk.chunk_id,
                'page_numbers': chunk.page_numbers,
                'text': chunk.text,
                'token_count': chunk.token_count,
                'metadata': chunk.metadata
            }
            for chunk in chunks
        ]
        
        chunk_ids = vector_store.add_embeddings(embeddings, metadata_list)
        
        # Save vector store to disk
        index_path = vector_db_dir / "index.faiss"
        vector_store.save(str(vector_db_dir))
        
        # Save metadata
        additional_metadata = {
            'doc_hash': doc_hash,
            'chunk_ids': chunk_ids,
            'index_type': self.vector_config.get('index_type', 'auto'),
            'vector_store_stats': vector_store.get_stats()
        }
        
        self.document_cache.save_cache_metadata(
            pdf_path=pdf_path,
            chunk_count=len(chunks),
            embedding_model=self.vector_config.get('embedding_model', 'text-embedding-ada-002'),
            embedding_dim=self.vector_config.get('embedding_dimensions', 1536),
            chunk_strategy=self.vector_config.get('chunk_strategy', 'semantic'),
            chunk_size=self.vector_config.get('chunk_size', 800),
            chunk_overlap=self.vector_config.get('chunk_overlap', 120),
            additional_metadata=additional_metadata
        )
        
        # Set as current
        self.current_pdf_path = pdf_path
        self.vector_store = vector_store
        
        logger.info(f"Vector database saved to {vector_db_dir}")
        
        return {
            'status': 'success',
            'doc_hash': doc_hash,
            'num_chunks': len(chunks),
            'vector_db_path': str(vector_db_dir),
            'metadata': additional_metadata
        }
    
    def _load_from_cache(self, pdf_path: str) -> Dict[str, Any]:
        """Load vector database from cache"""
        vector_db_dir, doc_hash = self.document_cache.get_vector_db_path(pdf_path)
        index_path = vector_db_dir / "index.faiss"
        
        # Get or create vector store for this PDF
        vector_store = self._get_or_create_vector_store(pdf_path)
        
        # Load from disk
        vector_store.load(str(vector_db_dir))
        
        # Set as current
        self.vector_store = vector_store
        self.current_pdf_path = pdf_path
        
        # Load metadata
        metadata_path = vector_db_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded vector database from {vector_db_dir}")
        
        return {
            'status': 'cached',
            'doc_hash': doc_hash,
            'num_chunks': metadata.get('chunk_count', 0),  # Default to 0 if not found
            'vector_db_path': str(vector_db_dir),
            'metadata': metadata
        }
    
    def retrieve(self, query: str, pdf_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for query using HyDE and reranking
        
        Args:
            query: Search query
            pdf_path: Optional PDF to search in (loads its vector DB)
            
        Returns:
            List of relevant chunks with scores
        """
        logger.debug(f"Retrieving chunks for query: {query[:100]}...")
        
        # Handle PDF switching
        if pdf_path:
            # Check if we need to switch PDFs
            if pdf_path != self.current_pdf_path:
                logger.info(f"Switching from {self.current_pdf_path} to {pdf_path}")
                
                # Check if already loaded in memory
                if pdf_path in self.loaded_vector_stores:
                    self.vector_store = self.loaded_vector_stores[pdf_path]
                    self.current_pdf_path = pdf_path
                    logger.debug(f"Switched to cached vector store for {pdf_path}")
                elif self.document_cache.is_cached(pdf_path):
                    # Load from disk cache
                    self._load_from_cache(pdf_path)
                else:
                    # Process the PDF first
                    logger.info(f"PDF {pdf_path} not loaded, processing it first")
                    self.process_and_index_pdf(pdf_path)
        
        # Check if we have a vector store
        if not self.vector_store or self.vector_store.get_stats()['total_vectors'] == 0:
            logger.warning("No vector store loaded or empty index")
            return []
        
        # Generate query variations if enabled
        queries = [query]
        if self.enable_multi_query:
            queries.extend(self._generate_query_variations(query))
            logger.info(f"Generated {len(queries)} query variations")
        
        # Apply HyDE if enabled
        if self.enable_hyde:
            hyde_docs = self._generate_hyde_documents(query)
            queries.extend(hyde_docs)
            logger.info(f"Generated {len(hyde_docs)} HyDE documents")
        
        # Retrieve chunks for all queries
        all_results = []
        for q in queries:
            query_embedding = self.embedding_service.create_query_embedding(q)
            results = self.vector_store.search(query_embedding, k=self.search_k)
            all_results.extend(results)
        
        # Deduplicate and rerank
        unique_results = self._deduplicate_results(all_results)
        reranked_results = self._rerank_results(unique_results, query)
        
        # Return top k after reranking
        final_results = reranked_results[:self.rerank_top_k]
        
        logger.info(f"Retrieved {len(final_results)} chunks after reranking")
        
        return final_results
    
    def clear_pdf_cache(self, pdf_path: str):
        """Clear both memory and disk cache for a specific PDF"""
        logger.info(f"Clearing cache for {pdf_path}")
        
        # Clear from memory
        if pdf_path in self.loaded_vector_stores:
            del self.loaded_vector_stores[pdf_path]
            logger.debug(f"Removed {pdf_path} from memory cache")
        
        # Clear from disk
        self.document_cache.clear_cache(pdf_path)
        
        # Reset current if it was the active one
        if self.current_pdf_path == pdf_path:
            self.current_pdf_path = None
            self.vector_store = None
    
    def clear_all_caches(self):
        """Clear all cached vector stores from memory"""
        logger.info("Clearing all cached vector stores from memory")
        self.loaded_vector_stores.clear()
        self.current_pdf_path = None
        self.vector_store = None
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better retrieval"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate 2-3 alternative phrasings of the given query to improve search results. Return only the variations, one per line."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens
            )
            
            variations = response.choices[0].message.content.strip().split('\n')
            return [v.strip() for v in variations if v.strip()][:3]
            
        except Exception as e:
            logger.warning(f"Failed to generate query variations: {e}")
            return []
    
    def _generate_hyde_documents(self, query: str) -> List[str]:
        """Generate hypothetical documents that would answer the query"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a hypothetical document excerpt that would perfectly answer the given query. Be specific and detailed. Maximum 200 words."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0.5,
                max_tokens=self.llm_max_tokens
            )
            
            hyde_doc = response.choices[0].message.content.strip()
            return [hyde_doc] if hyde_doc else []
            
        except Exception as e:
            logger.warning(f"Failed to generate HyDE document: {e}")
            return []
    
    def _deduplicate_results(self, results: List[Tuple]) -> List[Tuple]:
        """Remove duplicate results based on chunk ID"""
        seen = {}
        for metadata, score in results:
            chunk_id = metadata.get('chunk_id')
            if chunk_id not in seen or score < seen[chunk_id][1]:
                seen[chunk_id] = (metadata, score)
        
        return list(seen.values())
    
    def _rerank_results(self, results: List[Tuple], query: str) -> List[Dict[str, Any]]:
        """
        Rerank results using semantic similarity and relevance scoring
        """
        if not results:
            return []
        
        # Extract texts for reranking
        texts = [metadata.get('text', '') for metadata, score in results]
        
        # Calculate semantic similarity scores
        query_embedding = self.embedding_service.create_query_embedding(query)
        text_embeddings = self.embedding_service.embed_texts(texts)
        
        # Calculate cosine similarities
        similarities = np.dot(text_embeddings, query_embedding.T).flatten()
        
        # Combine with original scores (weighted average)
        reranked = []
        for i, (metadata, score) in enumerate(results):
            # Lower score is better in FAISS, so invert it
            faiss_score = 1.0 / (1.0 + score)
            semantic_score = float(similarities[i])
            
            # Weighted combination (70% semantic, 30% FAISS)
            combined_score = 0.7 * semantic_score + 0.3 * faiss_score
            
            reranked.append({
                'chunk_id': metadata.get('chunk_id'),
                'text': metadata.get('text', ''),
                'page_numbers': metadata.get('page_numbers', []),
                'score': combined_score,
                'faiss_score': score,
                'semantic_score': semantic_score,
                'metadata': metadata.get('metadata', {})
            })
        
        # Sort by combined score (higher is better)
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        return reranked
    
    def get_context_for_agent(self, query: str, pdf_path: str, max_tokens: int = 2000) -> str:
        """
        Get formatted context for agent consumption
        
        Args:
            query: The query to search for
            pdf_path: Path to the PDF to search in
            max_tokens: Maximum tokens to return
            
        Returns:
            Formatted context string
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(query, pdf_path)
        
        if not chunks:
            return "No relevant context found in the document."
        
        # Format chunks into context
        context_parts = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_tokens = chunk['metadata'].get('token_count', len(chunk_text.split()) * 1.3)
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            pages = chunk.get('page_numbers', [])
            page_str = f"Pages {min(pages)}-{max(pages)}" if pages else "Unknown pages"
            
            context_parts.append(f"[{page_str}]\n{chunk_text}")
            current_tokens += chunk_tokens
        
        context = "\n\n---\n\n".join(context_parts)
        
        return f"Relevant context from the RFP document:\n\n{context}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        stats = {
            'current_pdf': self.current_pdf_path,
            'loaded_pdfs': list(self.loaded_vector_stores.keys()),
            'num_loaded_pdfs': len(self.loaded_vector_stores),
            'cache_stats': self.document_cache.get_cache_stats(),
            'embedding_stats': self.embedding_service.get_usage_stats(),
        }
        
        # Add current vector store stats if available
        if self.vector_store:
            stats['current_vector_store_stats'] = self.vector_store.get_stats()
        
        return stats