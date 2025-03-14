"""
Script to query the existing vector index without rebuilding it.
"""

import os
import sys
import logging
import time
import argparse
from typing import List, Dict, Any

# Add the parent directory to the path to import config and modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.embedding_module import get_embedding

# Configure logging
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), config.LOG_DIR), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'query_index.log')),
        logging.StreamHandler()  # This will output to console
    ]
)
logger = logging.getLogger(__name__)

def query_index(query_text, top_k=5, podcast_filter=None, use_openai=True):
    """Query the index with the given text."""
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
    from llama_index.core.settings import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    print("\n" + "=" * 80)
    print(f"QUERYING INDEX: '{query_text}'")
    if podcast_filter:
        print(f"Filtering by podcast: '{podcast_filter}'")
    print("=" * 80)
    
    # Set up a local embedding model for LlamaIndex
    embed_model = HuggingFaceEmbedding(model_name=config.LOCAL_EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    
    # Set the embedding model for query
    if use_openai:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = False
        print(f"Using OpenAI embeddings for query: {config.EMBEDDING_MODEL}")
    else:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = True
        print(f"Using local embeddings for query: {config.LOCAL_EMBEDDING_MODEL}")
    
    # Generate query embedding
    query_embedding = get_embedding(query_text, use_local=not use_openai)
    print(f"Query embedding dimensions: {len(query_embedding)}")
    
    # Load the index
    storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.INDEX_STORAGE_DIR)
    
    if not os.path.exists(storage_dir):
        print(f"Error: Index directory '{storage_dir}' does not exist.")
        print("Please run rebuild_index.py first to create the index.")
        return []
    
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context, embed_model=None)  # No embedding needed
    
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
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=metadata_filter
    )
    
    # Create a query bundle with the pre-computed embedding
    from llama_index.core.schema import QueryBundle
    query_bundle = QueryBundle(
        query_str=query_text,
        embedding=query_embedding
    )
    
    # Retrieve nodes
    start_time = time.time()
    nodes = retriever.retrieve(query_bundle)
    search_time = time.time() - start_time
    
    # Format results
    results = []
    for node in nodes:
        results.append({
            "podcast_title": node.metadata.get("podcast_title", "Unknown"),
            "chunk_id": node.metadata.get("chunk_id", "Unknown"),
            "text": node.text,
            "score": node.score if hasattr(node, "score") else None
        })
    
    print(f"Found {len(results)} results in {search_time:.2f} seconds")
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Podcast: {result['podcast_title']}")
        print(f"  Chunk ID: {result['chunk_id']}")
        print(f"  Relevance Score: {result['score']:.4f}")
        
        # Truncate text if it's too long for display
        text = result['text']
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"  Text: {text}")
    
    print("\n" + "=" * 80)
    print("QUERY COMPLETED")
    print("=" * 80)
    
    return results

def main():
    """Run the main program."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query the vector index")
    parser.add_argument("--query", type=str, help="Query to search for")
    parser.add_argument("--podcast", type=str, help="Filter by podcast title")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--local", action="store_true", help="Use local embeddings instead of OpenAI")
    
    args = parser.parse_args()
    
    # If no query is provided, prompt the user
    query_text = args.query
    if not query_text:
        query_text = input("Enter your query: ")
    
    # Query the index
    query_index(query_text, args.top_k, args.podcast, not args.local)

if __name__ == "__main__":
    main()
