"""
Test script for the backend vector service.
This script tests the basic functionality of the service.
"""

import os
import sys
import logging
import shutil

# Add the parent directory to the path to import config and modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.input_module import load_transcripts, split_transcript, process_new_transcripts
from modules.embedding_module import get_embedding, get_local_embedding, get_local_embedding_model
from modules.indexing_module import TranscriptIndex

# Define test-specific directories
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_artifacts")
TEST_TRANSCRIPT_DIR = os.path.join(TEST_DIR, "transcripts")
TEST_INDEX_STORAGE_DIR = os.path.join(TEST_DIR, "index_storage")
TEST_LOG_DIR = os.path.join(TEST_DIR, "logs")

# Define main directories (that should be kept clean during tests)
MAIN_TRANSCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.TRANSCRIPT_DIR)
MAIN_INDEX_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.INDEX_STORAGE_DIR)

# Create necessary test directories
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(TEST_TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(TEST_INDEX_STORAGE_DIR, exist_ok=True)
os.makedirs(TEST_LOG_DIR, exist_ok=True)

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(TEST_LOG_DIR, 'test_service.log')),
        logging.StreamHandler()  # This will output to console
    ]
)
logger = logging.getLogger(__name__)

# Also configure root logger to output to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)
logging.getLogger().setLevel(logging.INFO)

def cleanup_test_artifacts(clean_all=False):
    """
    Clean up test artifacts after testing.
    
    Args:
        clean_all: If True, removes the entire test directory. 
                   If False, only cleans contents but keeps the directory structure.
    """
    if clean_all:
        # Remove the entire test directory
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
            logger.info(f"Removed test directory: {TEST_DIR}")
    else:
        # Clean up transcript files
        for file in os.listdir(TEST_TRANSCRIPT_DIR):
            file_path = os.path.join(TEST_TRANSCRIPT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clean up index storage
        for item in os.listdir(TEST_INDEX_STORAGE_DIR):
            item_path = os.path.join(TEST_INDEX_STORAGE_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        logger.info("Cleaned up test artifacts but kept directory structure")

def cleanup_main_directories():
    """
    Clean up any test artifacts that might have been created in the main directories.
    This ensures that tests don't interfere with the actual application data.
    """
    # Clean up main transcript directory
    if os.path.exists(MAIN_TRANSCRIPT_DIR):
        for file in os.listdir(MAIN_TRANSCRIPT_DIR):
            if file.startswith("Sample Podcast") or file.startswith("Test Podcast"):
                file_path = os.path.join(MAIN_TRANSCRIPT_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed test file from main directory: {file_path}")
    
    # We don't clean up the main index storage as it might contain real data
    # Instead, we ensure our tests use the test-specific directory
    
    logger.info("Cleaned up test artifacts from main directories")

def create_sample_transcript():
    """Create a sample transcript file for testing."""
    os.makedirs(TEST_TRANSCRIPT_DIR, exist_ok=True)
    
    sample_transcript = """
    This is a sample podcast transcript about artificial intelligence and machine learning.
    Vector embeddings are a powerful way to represent text data for semantic search.
    Podcast transcripts can be processed and indexed to enable natural language queries.
    This backend service uses OpenAI's embedding API to generate vector representations.
    The llama-index library provides efficient storage and retrieval capabilities.
    Flask is used to create a simple REST API for querying the vector index.
    """
    
    sample_file_path = os.path.join(TEST_TRANSCRIPT_DIR, "Sample Podcast.txt")
    with open(sample_file_path, 'w', encoding='utf-8') as f:
        f.write(sample_transcript)
    
    logger.info(f"Created sample transcript file: {sample_file_path}")
    return sample_file_path

def test_input_module():
    """Test the input module functionality."""
    print("\n" + "=" * 30 + " TESTING INPUT MODULE " + "=" * 30)
    logger.info("Testing input module...")
    
    # Create sample transcript
    create_sample_transcript()
    
    # Load transcripts
    transcripts = load_transcripts(TEST_TRANSCRIPT_DIR)
    
    logger.info(f"Loaded {len(transcripts)} transcripts")
    
    # Test splitting
    if transcripts:
        podcast_title = next(iter(transcripts))
        transcript_text = transcripts[podcast_title]
        chunks = split_transcript(transcript_text)
        logger.info(f"Split transcript '{podcast_title}' into {len(chunks)} chunks")
    
    # Test processing
    processed_transcripts = process_new_transcripts(TEST_TRANSCRIPT_DIR)
    logger.info(f"Processed {len(processed_transcripts)} transcripts")
    
    return processed_transcripts

def test_embedding_module():
    """Test the embedding module functionality."""
    print("\n" + "=" * 30 + " TESTING EMBEDDING MODULE " + "=" * 30)
    logger.info("Testing embedding module...")
    
    # Use local embedding model for tests
    if config.USE_LOCAL_EMBEDDINGS_FOR_TESTS:
        print(f"Using local embedding model for tests: {config.LOCAL_EMBEDDING_MODEL}")
        logger.info(f"Using local embedding model for tests: {config.LOCAL_EMBEDDING_MODEL}")
        test_text = "This is a test text for generating embeddings."
        
        # Load the model and log its properties
        model = get_local_embedding_model()
        print(f"Local model loaded successfully: {model}")
        logger.info(f"Local model loaded successfully: {model}")
        
        # Generate embedding and log its properties
        embedding = get_local_embedding(test_text)
        print(f"Generated local embedding with dimension: {len(embedding)}")
        print(f"First 5 values of local embedding: {embedding[:5]}")
        logger.info(f"Generated local embedding with dimension: {len(embedding)}")
        logger.info(f"First 5 values of local embedding: {embedding[:5]}")
        return embedding
    
    # Fall back to OpenAI if local embeddings are disabled
    if config.OPENAI_API_KEY == "your-openai-api-key":
        print("Skipping OpenAI embedding test as no valid API key is provided")
        print("For testing purposes, we'll use a mock embedding")
        logger.warning("Skipping OpenAI embedding test as no valid API key is provided")
        logger.info("For testing purposes, we'll use a mock embedding")
        return [0.1] * 1536  # Mock embedding vector
    
    # Test embedding generation with OpenAI
    test_text = "This is a test text for generating embeddings."
    embedding = get_embedding(test_text, use_local=False)
    
    print(f"Generated OpenAI embedding with dimension: {len(embedding)}")
    logger.info(f"Generated OpenAI embedding with dimension: {len(embedding)}")
    return embedding

def test_indexing_module(processed_transcripts=None):
    """Test the indexing module functionality."""
    print("\n" + "=" * 30 + " TESTING INDEXING MODULE " + "=" * 30)
    logger.info("Testing indexing module...")
    
    # Initialize index with test storage directory
    index = TranscriptIndex(storage_dir=TEST_INDEX_STORAGE_DIR)
    
    # Insert test data if no processed transcripts
    if not processed_transcripts:
        podcast_title = "Test Podcast"
        chunks = [
            ("test_chunk_1", "This is a test chunk about artificial intelligence and machine learning."),
            ("test_chunk_2", "Podcasts are a great way to learn about new topics while commuting."),
            ("test_chunk_3", "Vector embeddings allow for semantic search capabilities in applications.")
        ]
        
        index.insert_transcript_chunks(podcast_title, chunks)
        print(f"Inserted test chunks for '{podcast_title}'")
        logger.info(f"Inserted test chunks for '{podcast_title}'")
    else:
        # Insert processed transcripts
        for podcast_title, chunks in processed_transcripts.items():
            index.insert_transcript_chunks(podcast_title, chunks)
            print(f"Inserted {len(chunks)} chunks for '{podcast_title}'")
            logger.info(f"Inserted {len(chunks)} chunks for '{podcast_title}'")
    
    # Test similarity search
    query = "Tell me about AI"
    print(f"Testing similarity search with query: '{query}'")
    logger.info(f"Testing similarity search with query: '{query}'")
    
    # Use local embeddings for search if configured
    if config.USE_LOCAL_EMBEDDINGS_FOR_TESTS:
        print(f"Using local embedding model for similarity search: {config.LOCAL_EMBEDDING_MODEL}")
        logger.info(f"Using local embedding model for similarity search: {config.LOCAL_EMBEDDING_MODEL}")
    else:
        print(f"Using OpenAI embedding model for similarity search: {config.EMBEDDING_MODEL}")
        logger.info(f"Using OpenAI embedding model for similarity search: {config.EMBEDDING_MODEL}")
    
    # Perform search
    results = index.similarity_search(query)
    
    print(f"Search results for query: '{query}'")
    logger.info(f"Search results for query: '{query}'")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Podcast: {result['podcast_title']}")
        print(f"  Chunk ID: {result['chunk_id']}")
        print(f"  Text: {result['text'][:100]}...")  # Show first 100 chars
        print(f"  Score: {result['score']}")
        logger.info(f"Result {i+1}:")
        logger.info(f"  Podcast: {result['podcast_title']}")
        logger.info(f"  Chunk ID: {result['chunk_id']}")
        logger.info(f"  Text: {result['text'][:100]}...")  # Show first 100 chars
        logger.info(f"  Score: {result['score']}")

def main():
    """Run all tests."""
    # Clean up any existing test artifacts before starting
    cleanup_test_artifacts()
    
    # Clean up any test artifacts in main directories
    cleanup_main_directories()
    
    print("\n" + "=" * 80)
    print("STARTING BACKEND VECTOR SERVICE TESTS")
    print("=" * 80)
    logger.info("=" * 50)
    logger.info("STARTING BACKEND VECTOR SERVICE TESTS")
    logger.info("=" * 50)
    
    print(f"Using local embeddings for tests: {config.USE_LOCAL_EMBEDDINGS_FOR_TESTS}")
    logger.info(f"Using local embeddings for tests: {config.USE_LOCAL_EMBEDDINGS_FOR_TESTS}")
    
    if config.USE_LOCAL_EMBEDDINGS_FOR_TESTS:
        print(f"Local embedding model: {config.LOCAL_EMBEDDING_MODEL}")
        logger.info(f"Local embedding model: {config.LOCAL_EMBEDDING_MODEL}")
    
    # Test input module
    processed_transcripts = test_input_module()
    
    # Test embedding module
    test_embedding_module()
    
    # Test indexing module
    test_indexing_module(processed_transcripts)
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print(f"Local embedding tests {'were used' if config.USE_LOCAL_EMBEDDINGS_FOR_TESTS else 'were not used'}")
    print("=" * 80)
    logger.info("=" * 50)
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
    logger.info(f"Local embedding tests {'were used' if config.USE_LOCAL_EMBEDDINGS_FOR_TESTS else 'were not used'}")
    logger.info("=" * 50)
    
    # Clean up test artifacts after testing
    cleanup_test_artifacts(clean_all=False)
    
    # Instructions for running the API
    print("\nTo run the API server:")
    print("1. Activate the virtual environment:")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("2. Run the Flask app:")
    print("   python app/main.py")
    print("3. Access the API at http://localhost:8080")
    print("4. Example query: http://localhost:8080/query?search=artificial%20intelligence")
    logger.info("\nTo run the API server:")
    logger.info("1. Activate the virtual environment:")
    logger.info("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    logger.info("2. Run the Flask app:")
    logger.info("   python app/main.py")
    logger.info("3. Access the API at http://localhost:8080")
    logger.info("4. Example query: http://localhost:8080/query?search=artificial%20intelligence")

if __name__ == "__main__":
    main()
