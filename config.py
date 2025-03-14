"""
Configuration file for the backend vector service.
Contains all configurable parameters for the application.
"""

# OpenAI API configuration
OPENAI_API_KEY = ""  # Set this via environment variables
EMBEDDING_MODEL = "text-embedding-3-small"

# Local embedding model configuration
USE_LOCAL_EMBEDDINGS_FOR_TESTS = True  # Set to True to use local models for tests
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight model good for testing

# Transcript processing configuration
TRANSCRIPT_DIR = "transcripts"  # Directory to store transcript files
CHUNK_SIZE = 2000  # Default chunk size for splitting transcripts (in words)

# Vector index configuration
INDEX_STORAGE_DIR = "index_storage"  # Directory to store the vector index

# API configuration
API_HOST = "0.0.0.0"  # Host to bind the API server to
API_PORT = 8080  # Port to bind the API server to
API_DEBUG = True  # Whether to run the API server in debug mode

# Logging configuration
LOG_DIR = "logs"  # Directory to store log files
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
