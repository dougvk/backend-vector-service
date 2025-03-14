"""
Script to build and query a vector index from transcript files.
This script processes all transcripts, generates embeddings, and creates a searchable index.
"""

import os
import sys
import logging
import time
from typing import List, Dict, Tuple
import argparse

# Add the parent directory to the path to import config and modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.input_module import load_transcripts, process_new_transcripts
from modules.embedding_module import get_embedding, batch_get_embeddings
from modules.indexing_module import TranscriptIndex

# Configure logging to output to both file and console
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), config.LOG_DIR), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'index_build.log')),
        logging.StreamHandler()  # This will output to console
    ]
)
logger = logging.getLogger(__name__)

def build_index(use_openai: bool = False):
    """
    Build a vector index from all transcript files.
    
    Args:
        use_openai: Whether to use OpenAI embeddings instead of local embeddings
    """
    print("\n" + "=" * 80)
    print("BUILDING VECTOR INDEX FROM TRANSCRIPTS")
    print("=" * 80)
    
    # Override the local embeddings setting if specified
    if use_openai:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = False
        print(f"Using OpenAI embeddings: {config.EMBEDDING_MODEL}")
    else:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = True
        print(f"Using local embeddings: {config.LOCAL_EMBEDDING_MODEL}")
    
    # Define the transcript directory
    transcript_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.TRANSCRIPT_DIR)
    print(f"Loading transcripts from: {transcript_dir}")
    
    # Process transcripts to get chunks
    start_time = time.time()
    processed_transcripts = process_new_transcripts(transcript_dir)
    processing_time = time.time() - start_time
    
    # Count total chunks and transcripts
    total_transcripts = len(processed_transcripts)
    total_chunks = sum(len(chunks) for chunks in processed_transcripts.values())
    
    print(f"Processed {total_transcripts} transcripts into {total_chunks} chunks in {processing_time:.2f} seconds")
    
    # Initialize the index
    index = TranscriptIndex()
    
    # Insert all transcript chunks into the index
    start_time = time.time()
    for podcast_title, chunks in processed_transcripts.items():
        print(f"Indexing: {podcast_title} ({len(chunks)} chunks)")
        index.insert_transcript_chunks(podcast_title, chunks)
    
    indexing_time = time.time() - start_time
    print(f"Indexed {total_chunks} chunks in {indexing_time:.2f} seconds")
    
    print("\n" + "=" * 80)
    print("INDEX BUILDING COMPLETED")
    print(f"Total transcripts: {total_transcripts}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total time: {processing_time + indexing_time:.2f} seconds")
    print("=" * 80)
    
    return index

def query_index(index: TranscriptIndex, query: str, top_k: int = 5, podcast_filter: str = None, use_openai: bool = False):
    """
    Query the vector index with a natural language query.
    
    Args:
        index: The TranscriptIndex instance
        query: The natural language query
        top_k: Number of results to return
        podcast_filter: Optional filter for specific podcast title
        use_openai: Whether to use OpenAI embeddings for the query
    """
    print("\n" + "=" * 80)
    print(f"QUERYING INDEX: '{query}'")
    if podcast_filter:
        print(f"Filtering by podcast: '{podcast_filter}'")
    
    # Set the embedding model to match what was used for indexing
    if use_openai:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = False
        print(f"Using OpenAI embeddings for query: {config.EMBEDDING_MODEL}")
    else:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = True
        print(f"Using local embeddings for query: {config.LOCAL_EMBEDDING_MODEL}")
    
    print("=" * 80)
    
    # Perform the search
    start_time = time.time()
    results = index.similarity_search(query, top_k=top_k, podcast_filter=podcast_filter)
    search_time = time.time() - start_time
    
    # Display results
    print(f"Found {len(results)} results in {search_time:.2f} seconds:")
    
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

def interactive_query(index: TranscriptIndex, use_openai: bool = False):
    """
    Start an interactive query session.
    
    Args:
        index: The TranscriptIndex instance
        use_openai: Whether to use OpenAI embeddings for queries
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE QUERY SESSION")
    print("Type 'exit' to quit")
    print("=" * 80)
    
    # Set the embedding model to match what was used for indexing
    if use_openai:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = False
        print(f"Using OpenAI embeddings for queries: {config.EMBEDDING_MODEL}")
    else:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = True
        print(f"Using local embeddings for queries: {config.LOCAL_EMBEDDING_MODEL}")
    
    while True:
        # Get query from user
        query = input("\nEnter your query: ")
        
        # Exit if requested
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        # Get optional podcast filter
        podcast_filter = input("Filter by podcast title (leave empty for no filter): ")
        if podcast_filter.strip() == "":
            podcast_filter = None
        
        # Get number of results
        try:
            top_k = int(input("Number of results to show (default 5): ") or "5")
        except ValueError:
            top_k = 5
        
        # Perform the query
        query_index(index, query, top_k, podcast_filter, use_openai)
    
    print("\n" + "=" * 80)
    print("INTERACTIVE SESSION ENDED")
    print("=" * 80)

def main():
    """Run the main program."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build and query a vector index from transcript files")
    parser.add_argument("--build", action="store_true", help="Build the index")
    parser.add_argument("--query", type=str, help="Query to search for")
    parser.add_argument("--interactive", action="store_true", help="Start interactive query session")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI embeddings instead of local embeddings")
    parser.add_argument("--podcast", type=str, help="Filter by podcast title")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    # Default to building the index if no arguments are provided
    if not (args.build or args.query or args.interactive):
        args.build = True
        args.interactive = True
    
    # Build the index if requested
    index = None
    if args.build:
        index = build_index(use_openai=args.openai)
    else:
        # Load existing index
        index = TranscriptIndex()
    
    # Query the index if requested
    if args.query:
        query_index(index, args.query, args.top_k, args.podcast, use_openai=args.openai)
    
    # Start interactive session if requested
    if args.interactive:
        interactive_query(index, use_openai=args.openai)

if __name__ == "__main__":
    main()
