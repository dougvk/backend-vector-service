"""
Indexing module for the backend vector service.
Handles storage and retrieval of embeddings and associated metadata.
"""

import os
import logging
import sys
import json
from typing import List, Dict, Any, Tuple, Optional
import time

from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.embeddings import resolve_embed_model

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.embedding_module import get_embedding, get_local_embedding_model

# Configure logging
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config.LOG_DIR), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config.LOG_DIR, 'indexing_module.log'),
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptIndex:
    """
    Class to manage the vector index for podcast transcripts.
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize the transcript index.
        
        Args:
            storage_dir: Directory to store the index (default from config)
        """
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config.INDEX_STORAGE_DIR
        )
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            logger.info(f"Created storage directory: {self.storage_dir}")
        
        # Set up local embedding model for LlamaIndex if using local embeddings
        if config.USE_LOCAL_EMBEDDINGS_FOR_TESTS:
            from sentence_transformers import SentenceTransformer
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            # Create a HuggingFaceEmbedding instance with our model
            embed_model = HuggingFaceEmbedding(model_name=config.LOCAL_EMBEDDING_MODEL)
            logger.info(f"Using local embedding model for LlamaIndex: {config.LOCAL_EMBEDDING_MODEL}")
        else:
            # Use OpenAI embedding model
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = OpenAIEmbedding(api_key=config.OPENAI_API_KEY, model=config.EMBEDDING_MODEL)
            logger.info(f"Using OpenAI embedding model for LlamaIndex: {config.EMBEDDING_MODEL}")
        
        # Try to load existing index or create a new one
        try:
            self.storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.index = load_index_from_storage(self.storage_context, embed_model=embed_model)
            logger.info(f"Loaded existing index from {self.storage_dir}")
        except Exception as e:
            logger.info(f"No existing index found or error loading index: {str(e)}")
            logger.info("Creating new index")
            self.index = VectorStoreIndex([], embed_model=embed_model)
            self.storage_context = None
    
    def insert_transcript_chunks(self, podcast_title: str, chunks: List[Tuple[str, str]]) -> None:
        """
        Insert transcript chunks into the index.
        
        Args:
            podcast_title: Title of the podcast
            chunks: List of (chunk_id, chunk_text) tuples
        """
        nodes = []
        
        for chunk_id, chunk_text in chunks:
            # Create a node with metadata
            node = TextNode(
                text=chunk_text,
                metadata={
                    "podcast_title": podcast_title,
                    "chunk_id": chunk_id
                }
            )
            nodes.append(node)
        
        # Insert nodes into the index
        self.index.insert_nodes(nodes)
        
        # Persist the index
        self.index.storage_context.persist(persist_dir=self.storage_dir)
        
        logger.info(f"Inserted {len(chunks)} chunks for podcast '{podcast_title}' into the index")
    
    def similarity_search(self, query_text: str, top_k: int = 10, podcast_filter: str = None, use_local: bool = None) -> List[Dict[str, Any]]:
        """
        Perform a similarity search on the index.
        
        Args:
            query_text: Query text to search for
            top_k: Number of results to return
            podcast_filter: Optional filter for specific podcast title
            use_local: Whether to use a local model for embedding (default from config.USE_LOCAL_EMBEDDINGS_FOR_TESTS)
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        try:
            # Determine whether to use local embeddings
            if use_local is None:
                use_local = config.USE_LOCAL_EMBEDDINGS_FOR_TESTS
            
            # Set up metadata filter if podcast_filter is provided
            metadata_filter = None
            if podcast_filter:
                metadata_filter = MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="podcast_title",
                            value=podcast_filter,
                            operator=FilterOperator.EQ
                        )
                    ]
                )
            
            # Perform similarity search
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=metadata_filter
            )
            
            # If we need to use a different embedding model than what was used to build the index
            if use_local != config.USE_LOCAL_EMBEDDINGS_FOR_TESTS:
                # Generate embedding for the query using the specified model
                query_embedding = get_embedding(query_text, use_local=use_local)
                logger.info(f"Generated query embedding with dimension: {len(query_embedding)}")
                
                # Create a query bundle with the pre-computed embedding
                query_bundle = QueryBundle(
                    query_str=query_text,
                    embedding=query_embedding
                )
                
                # Retrieve nodes using the query bundle with pre-computed embedding
                nodes = retriever.retrieve(query_bundle)
            else:
                # Use the default embedding model of the index
                nodes = retriever.retrieve(query_text)
            
            # Format results
            results = []
            for node in nodes:
                results.append({
                    "podcast_title": node.metadata.get("podcast_title", "Unknown"),
                    "chunk_id": node.metadata.get("chunk_id", "Unknown"),
                    "text": node.text,
                    "score": node.score if hasattr(node, "score") else None
                })
            
            logger.info(f"Performed similarity search for query: '{query_text}', found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise

if __name__ == "__main__":
    # Test the module
    index = TranscriptIndex()
    
    # Test inserting some dummy data
    podcast_title = "Test Podcast"
    chunks = [
        ("test_chunk_1", "This is a test chunk about artificial intelligence and machine learning."),
        ("test_chunk_2", "Podcasts are a great way to learn about new topics while commuting."),
        ("test_chunk_3", "Vector embeddings allow for semantic search capabilities in applications.")
    ]
    
    index.insert_transcript_chunks(podcast_title, chunks)
    
    # Test similarity search
    query = "Tell me about AI"
    results = index.similarity_search(query)
    
    print(f"Search results for query: '{query}'")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Podcast: {result['podcast_title']}")
        print(f"  Chunk ID: {result['chunk_id']}")
        print(f"  Text: {result['text']}")
        print(f"  Score: {result['score']}")
