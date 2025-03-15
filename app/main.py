"""
Main API entry point for the backend vector service.
Provides a GET endpoint for querying the vector index.
"""

import os
import sys
import logging
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the parent directory to the path to import config and modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.embedding_module import get_embedding
from modules.indexing_module import TranscriptIndex
from modules.input_module import process_new_transcripts

# Configure logging
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config.LOG_DIR), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config.LOG_DIR, 'api.log'),
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Default to config setting, but allow override via app.config
app.config['USE_LOCAL_EMBEDDINGS'] = config.USE_LOCAL_EMBEDDINGS_FOR_TESTS

# Create necessary directories
transcript_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    config.TRANSCRIPT_DIR
)
os.makedirs(transcript_dir, exist_ok=True)

index_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    config.INDEX_STORAGE_DIR
)
os.makedirs(index_dir, exist_ok=True)

# Initialize the transcript index at the module level
index = TranscriptIndex()
logger.info("Initialized TranscriptIndex")

@app.route('/query', methods=['GET'])
def query():
    """
    GET endpoint for querying the vector index.
    
    Query parameters:
    - search: Query string to search for
    - top_k: Number of results to return (default: 10)
    - podcast: Optional filter for specific podcast title
    
    Returns:
        JSON response with search results
    """
    try:
        # Get query parameters
        search_query = request.args.get('search', '')
        top_k = int(request.args.get('top_k', 10))
        podcast_filter = request.args.get('podcast', None)
        
        if not search_query:
            return jsonify({
                'error': 'Missing required parameter: search'
            }), 400
        
        # Ensure index is initialized
        global index
        if index is None:
            logger.error("TranscriptIndex is not initialized")
            index = TranscriptIndex()
            logger.info("Re-initialized TranscriptIndex")
        
        # Perform similarity search with the app config embedding setting
        results = index.similarity_search(
            search_query, 
            top_k, 
            podcast_filter, 
            use_local=app.config['USE_LOCAL_EMBEDDINGS']
        )
        
        # Return results as JSON
        return jsonify({
            'query': search_query,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            'error': f"Error processing query: {str(e)}"
        }), 500

# Rest of your routes...

# This function can be called to configure the app for Gunicorn
def configure_app(use_openai=False):
    """Configure the application for production use."""
    global index
    
    if use_openai:
        app.config['USE_LOCAL_EMBEDDINGS'] = False
        logger.info("Using OpenAI embeddings for API queries")
    else:
        logger.info(f"Using local embeddings for API queries: {config.LOCAL_EMBEDDING_MODEL}")
    
    # Ensure index is initialized
    if index is None:
        index = TranscriptIndex()
        logger.info("Initialized TranscriptIndex in configure_app")
    
    return app

# When running directly, this will be executed
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backend Vector Service API')
    parser.add_argument('--use-openai', action='store_true', 
                        help='Use OpenAI embeddings instead of local embeddings')
    args = parser.parse_args()
    
    # Configure the app
    configure_app(use_openai=args.use_openai)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )