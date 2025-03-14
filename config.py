"""
Configuration file for the backend vector service.
Contains all configurable parameters for the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Get API key from environment variable
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Local embedding model configuration
USE_LOCAL_EMBEDDINGS_FOR_TESTS = os.getenv("USE_LOCAL_EMBEDDINGS_FOR_TESTS", "True").lower() == "true"
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Transcript processing configuration
TRANSCRIPT_DIR = "transcripts"  # Directory to store transcript files
CHUNK_SIZE = 500  # Default chunk size for splitting transcripts (in words)

# Vector index configuration
INDEX_STORAGE_DIR = "index_storage"  # Directory to store the vector index

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")  # Host to bind the API server to
API_PORT = int(os.getenv("API_PORT", 8080))  # Port to bind the API server to
API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"  # Whether to run the API server in debug mode

# Logging configuration
LOG_DIR = "logs"  # Directory to store log files
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
