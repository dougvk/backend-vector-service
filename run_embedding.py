"""
Script to test the embedding module functionality with transcript chunks.
"""

import os
import sys
import logging
import time
from typing import List, Dict, Tuple

# Add the parent directory to the path to import config and modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.input_module import load_transcripts, process_new_transcripts
from modules.embedding_module import get_embedding, batch_get_embeddings

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'embedding_test.log')),
        logging.StreamHandler()  # This will output to console
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the embedding test on transcript chunks."""
    print("\n" + "=" * 80)
    print("TESTING EMBEDDING MODULE WITH TRANSCRIPT CHUNKS")
    print("=" * 80)
    
    # Define the transcript directory
    transcript_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.TRANSCRIPT_DIR)
    print(f"Loading transcripts from: {transcript_dir}")
    
    # Process transcripts to get chunks
    processed_transcripts = process_new_transcripts(transcript_dir)
    print(f"Processed {len(processed_transcripts)} transcripts")
    
    # Count total chunks
    total_chunks = sum(len(chunks) for chunks in processed_transcripts.values())
    print(f"Total chunks to embed: {total_chunks}")
    
    # Generate embeddings for each transcript's chunks
    for title, chunks in processed_transcripts.items():
        print(f"\nProcessing transcript: {title}")
        print(f"Number of chunks: {len(chunks)}")
        
        # Extract just the text from the chunks
        chunk_texts = [chunk_text for _, chunk_text in chunks]
        
        # Time the embedding generation
        start_time = time.time()
        
        # First try with OpenAI API
        print(f"Using OpenAI model: {config.EMBEDDING_MODEL}")
        
        # Generate embeddings in batch
        try:
            # Explicitly set use_local to False to use OpenAI
            embeddings = batch_get_embeddings(chunk_texts, use_local=False)
            
            # Print embedding information
            elapsed_time = time.time() - start_time
            print(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f} seconds")
            print(f"Embedding dimensions: {len(embeddings[0])}")
            
            # Print a sample of the first embedding
            print(f"First embedding sample (first 5 values): {embeddings[0][:5]}")
            
            # Calculate average embedding magnitude
            avg_magnitude = sum(sum(e**2 for e in emb)**0.5 for emb in embeddings) / len(embeddings)
            print(f"Average embedding magnitude: {avg_magnitude:.4f}")
            
        except Exception as e:
            print(f"Error generating embeddings with OpenAI: {str(e)}")
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            
            # Fall back to local model if OpenAI fails
            print(f"\nFalling back to local model: {config.LOCAL_EMBEDDING_MODEL}")
            try:
                start_time = time.time()
                embeddings = batch_get_embeddings(chunk_texts, use_local=True)
                
                # Print embedding information
                elapsed_time = time.time() - start_time
                print(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f} seconds")
                print(f"Embedding dimensions: {len(embeddings[0])}")
                print(f"First embedding sample (first 5 values): {embeddings[0][:5]}")
                
                # Calculate average embedding magnitude
                avg_magnitude = sum(sum(e**2 for e in emb)**0.5 for emb in embeddings) / len(embeddings)
                print(f"Average embedding magnitude: {avg_magnitude:.4f}")
                
            except Exception as e2:
                print(f"Error generating embeddings with local model: {str(e2)}")
                logger.error(f"Error generating embeddings with local model: {str(e2)}")
    
    print("\n" + "=" * 80)
    print("EMBEDDING TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
