"""
Script to rebuild and query a vector index from transcript files.
This script ensures consistent embedding dimensions throughout the process.
"""

import os
import sys
import logging
import time
import shutil
from typing import List, Dict, Tuple, Any

# Add the parent directory to the path to import config and modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.input_module import process_new_transcripts
from modules.embedding_module import get_embedding, batch_get_embeddings

# Configure logging
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), config.LOG_DIR), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'rebuild_index.log')),
        logging.StreamHandler()  # This will output to console
    ]
)
logger = logging.getLogger(__name__)

def clean_index_directory():
    """Remove existing index directory to start fresh."""
    index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.INDEX_STORAGE_DIR)
    if os.path.exists(index_dir):
        print(f"Removing existing index directory: {index_dir}")
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    print(f"Created fresh index directory: {index_dir}")

def process_transcripts():
    """Process all transcripts and return chunks."""
    transcript_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.TRANSCRIPT_DIR)
    print(f"Loading transcripts from: {transcript_dir}")
    
    start_time = time.time()
    processed_transcripts = process_new_transcripts(transcript_dir)
    processing_time = time.time() - start_time
    
    total_transcripts = len(processed_transcripts)
    total_chunks = sum(len(chunks) for chunks in processed_transcripts.values())
    
    print(f"Processed {total_transcripts} transcripts into {total_chunks} chunks in {processing_time:.2f} seconds")
    return processed_transcripts, total_transcripts, total_chunks

def embed_chunks(processed_transcripts, use_openai=True):
    """Generate embeddings for all chunks."""
    # Set the embedding model
    if use_openai:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = False
        print(f"Using OpenAI embeddings: {config.EMBEDDING_MODEL}")
    else:
        config.USE_LOCAL_EMBEDDINGS_FOR_TESTS = True
        print(f"Using local embeddings: {config.LOCAL_EMBEDDING_MODEL}")
    
    all_embeddings = {}
    all_texts = {}
    all_metadata = {}
    
    start_time = time.time()
    for podcast_title, chunks in processed_transcripts.items():
        print(f"Embedding: {podcast_title} ({len(chunks)} chunks)")
        
        # Extract texts and create metadata
        texts = [chunk_text for _, chunk_text in chunks]
        chunk_ids = [chunk_id for chunk_id, _ in chunks]
        
        # Generate embeddings
        embeddings = batch_get_embeddings(texts, use_local=not use_openai)
        
        # Store embeddings and metadata
        for i, chunk_id in enumerate(chunk_ids):
            key = f"{podcast_title}_{chunk_id}"
            all_embeddings[key] = embeddings[i]
            all_texts[key] = texts[i]
            all_metadata[key] = {
                "podcast_title": podcast_title,
                "chunk_id": chunk_id
            }
    
    embedding_time = time.time() - start_time
    print(f"Generated embeddings for {len(all_embeddings)} chunks in {embedding_time:.2f} seconds")
    
    # Check embedding dimensions
    sample_key = next(iter(all_embeddings))
    sample_embedding = all_embeddings[sample_key]
    print(f"Embedding dimensions: {len(sample_embedding)}")
    
    return all_embeddings, all_texts, all_metadata

def build_index(all_embeddings, all_texts, all_metadata):
    """Build a vector index using the generated embeddings."""
    from llama_index.core import VectorStoreIndex, Document, StorageContext
    from llama_index.core.schema import TextNode
    from llama_index.core.settings import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    # Set up a local embedding model for LlamaIndex
    # This is just for index creation, we'll use our pre-computed embeddings
    embed_model = HuggingFaceEmbedding(model_name=config.LOCAL_EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    
    # Create nodes
    nodes = []
    for key in all_embeddings:
        node = TextNode(
            text=all_texts[key],
            metadata=all_metadata[key],
            embedding=all_embeddings[key]
        )
        nodes.append(node)
    
    print(f"Created {len(nodes)} nodes for indexing")
    
    # Build the index
    start_time = time.time()
    index = VectorStoreIndex(nodes, embed_model=None)  # No embedding needed since nodes already have embeddings
    
    # Persist the index
    storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.INDEX_STORAGE_DIR)
    index.storage_context.persist(persist_dir=storage_dir)
    
    indexing_time = time.time() - start_time
    print(f"Built and persisted index in {indexing_time:.2f} seconds")
    
    return index

def query_index(query_text, top_k=5, podcast_filter=None, use_openai=True):
    """Query the index with the given text."""
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
    from llama_index.core.settings import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
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
    
    return results

def main():
    """Run the main program."""
    print("\n" + "=" * 80)
    print("REBUILDING VECTOR INDEX FROM TRANSCRIPTS")
    print("=" * 80)
    
    # Use OpenAI embeddings
    use_openai = True
    
    # Clean index directory
    clean_index_directory()
    
    # Process transcripts
    processed_transcripts, total_transcripts, total_chunks = process_transcripts()
    
    # Generate embeddings
    all_embeddings, all_texts, all_metadata = embed_chunks(processed_transcripts, use_openai)
    
    # Build index
    index = build_index(all_embeddings, all_texts, all_metadata)
    
    print("\n" + "=" * 80)
    print("INDEX BUILDING COMPLETED")
    print(f"Total transcripts: {total_transcripts}")
    print(f"Total chunks: {total_chunks}")
    print("=" * 80)
    
    # Test query
    print("\n" + "=" * 80)
    print("TESTING QUERY")
    print("=" * 80)
    
    query_text = "What happened during the execution of King Louis XVI?"
    query_index(query_text, top_k=3, use_openai=use_openai)
    
    print("\n" + "=" * 80)
    print("QUERY TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
