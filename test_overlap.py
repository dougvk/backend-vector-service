"""
Test script to demonstrate the chunking with overlap functionality.
"""

import os
import sys
from modules.input_module import split_transcript

# Sample text for testing
sample_text = """
This is a sample transcript text that will be used to test the chunking functionality with overlap.
We need to make sure that the chunks are created correctly with the specified overlap percentage.
The overlap should be exactly 10% of the chunk size, meaning that each chunk (except the first one)
will start with the last 10% of words from the previous chunk. This helps maintain context between
chunks and improves the quality of vector search results by ensuring that related content isn't
arbitrarily split between chunks. Let's add some more text to make sure we have enough content
for multiple chunks. The quick brown fox jumps over the lazy dog. A stitch in time saves nine.
All that glitters is not gold. Actions speak louder than words. Better late than never.
Don't judge a book by its cover. Easy come, easy go. Fortune favors the bold.
"""

# Test with different chunk sizes
chunk_sizes = [20, 50]

for chunk_size in chunk_sizes:
    print(f"\nTesting with chunk size: {chunk_size}")
    chunks = split_transcript(sample_text, chunk_size=chunk_size)
    
    print(f"Number of chunks: {len(chunks)}")
    
    # Print each chunk with word count
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        print(f"\nChunk {i} ({word_count} words):")
        print(f"{chunk}")
        
        # If not the first chunk, check for overlap with previous chunk
        if i > 0:
            prev_chunk_words = chunks[i-1].split()
            curr_chunk_words = chunk.split()
            
            # Expected overlap size
            expected_overlap = int(chunk_size * 0.1)
            
            # Check if the first words of current chunk match the last words of previous chunk
            overlap_words = []
            for j in range(min(expected_overlap, len(prev_chunk_words), len(curr_chunk_words))):
                if j < len(prev_chunk_words) and j < len(curr_chunk_words):
                    if prev_chunk_words[-expected_overlap+j] == curr_chunk_words[j]:
                        overlap_words.append(curr_chunk_words[j])
            
            print(f"Overlap with previous chunk: {len(overlap_words)} words")
            if overlap_words:
                print(f"Overlap words: {' '.join(overlap_words)}")

if __name__ == "__main__":
    print("Testing chunking with overlap functionality")
