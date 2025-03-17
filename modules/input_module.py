"""
Input module for the backend vector service.
Handles loading and processing of transcript files.
"""

import os
import logging
import sys
from typing import List, Dict, Tuple

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    filename=os.path.join(config.LOG_DIR, 'input_module.log'),
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_transcripts(directory_path: str) -> Dict[str, str]:
    """
    Load all .txt files from the specified directory.
    
    Args:
        directory_path: Path to directory containing transcript files
        
    Returns:
        Dictionary mapping podcast titles to transcript text
    """
    transcripts = {}
    
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
        
        # Get all .txt files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                # Extract podcast title from filename
                podcast_title = os.path.basename(filename).split('.')[0]
                
                # Read the transcript file
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    transcript_text = file.read()
                
                transcripts[podcast_title] = transcript_text
                logger.info(f"Loaded transcript: {podcast_title}")
    
    except Exception as e:
        logger.error(f"Error loading transcripts: {str(e)}")
        raise
    
    return transcripts

def split_transcript(text: str, chunk_size: int = config.CHUNK_SIZE, overlap_percent: float = 0.1) -> List[str]:
    """
    Split transcript text into chunks of specified size with overlap.
    
    Args:
        text: Transcript text to split
        chunk_size: Size of each chunk in words (default from config)
        overlap_percent: Percentage of overlap between chunks (default 10%)
        
    Returns:
        List of text chunks
    """
    # Split text into words
    words = text.split()
    
    # Calculate overlap size in words
    overlap_size = int(chunk_size * overlap_percent)
    effective_chunk_size = chunk_size - overlap_size
    
    # Calculate number of chunks
    if effective_chunk_size <= 0:
        logger.warning(f"Overlap too large for chunk size. Using no overlap.")
        effective_chunk_size = chunk_size
        overlap_size = 0
    
    # Calculate number of chunks needed
    if len(words) <= chunk_size:
        num_chunks = 1
    else:
        num_chunks = 1 + (len(words) - chunk_size) // effective_chunk_size
        # Add one more chunk if there are remaining words
        if (len(words) - chunk_size) % effective_chunk_size > 0:
            num_chunks += 1
    
    # Split words into chunks with overlap
    chunks = []
    for i in range(num_chunks):
        start_idx = i * effective_chunk_size
        end_idx = min(start_idx + chunk_size, len(words))
        chunk = ' '.join(words[start_idx:end_idx])
        chunks.append(chunk)
    
    logger.info(f"Split transcript into {len(chunks)} chunks of approximately {chunk_size} words each with {overlap_percent*100}% overlap")
    return chunks

def process_new_transcripts(directory_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Process new transcript files in the specified directory.
    
    Args:
        directory_path: Path to directory containing transcript files
        
    Returns:
        Dictionary mapping podcast titles to lists of (chunk_id, chunk_text) tuples
    """
    # Load all transcripts
    transcripts = load_transcripts(directory_path)
    
    # Process each transcript
    processed_transcripts = {}
    for podcast_title, transcript_text in transcripts.items():
        # Split transcript into chunks
        chunks = split_transcript(transcript_text)
        
        # Create list of (chunk_id, chunk_text) tuples
        chunk_tuples = [(f"{podcast_title}_chunk_{i}", chunk) for i, chunk in enumerate(chunks)]
        
        processed_transcripts[podcast_title] = chunk_tuples
        logger.info(f"Processed transcript: {podcast_title} into {len(chunks)} chunks")
    
    return processed_transcripts

if __name__ == "__main__":
    # Test the module
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config.TRANSCRIPT_DIR)
    transcripts = load_transcripts(test_dir)
    print(f"Loaded {len(transcripts)} transcripts")
    
    # Process a sample transcript if available
    if transcripts:
        podcast_title = next(iter(transcripts))
        transcript_text = transcripts[podcast_title]
        chunks = split_transcript(transcript_text)
        print(f"Split transcript '{podcast_title}' into {len(chunks)} chunks")
