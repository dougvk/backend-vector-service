"""
Script to test the input module functionality.
"""

import os
import sys
import logging

# Add the parent directory to the path to import config and modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.input_module import load_transcripts, split_transcript, process_new_transcripts

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'input_module_test.log')),
        logging.StreamHandler()  # This will output to console
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the input module test."""
    print("\n" + "=" * 80)
    print("TESTING INPUT MODULE")
    print("=" * 80)
    
    # Define the transcript directory
    transcript_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.TRANSCRIPT_DIR)
    print(f"Loading transcripts from: {transcript_dir}")
    
    # Load transcripts
    transcripts = load_transcripts(transcript_dir)
    print(f"Loaded {len(transcripts)} transcripts:")
    for title in transcripts.keys():
        print(f"  - {title}")
    
    # Process transcripts
    processed_transcripts = process_new_transcripts(transcript_dir)
    print(f"\nProcessed {len(processed_transcripts)} transcripts:")
    for title, chunks in processed_transcripts.items():
        print(f"  - {title}: {len(chunks)} chunks")
        # Print a sample of the first chunk
        if chunks:
            chunk_id, chunk_text = chunks[0]
            preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            print(f"    First chunk preview: {preview}")
    
    print("\n" + "=" * 80)
    print("INPUT MODULE TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
