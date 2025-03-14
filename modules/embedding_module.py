"""
Embedding module for the backend vector service.
Handles generation of embeddings using OpenAI's API or local models.
"""

import os
import logging
import sys
import time
from typing import List, Dict, Any, Union

from openai import OpenAI
from openai import RateLimitError, APIError

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config.LOG_DIR), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config.LOG_DIR, 'embedding_module.log'),
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy-load sentence_transformers only when needed
_sentence_transformer_model = None
# Lazy-load OpenAI client
_openai_client = None

def get_openai_client(api_key: str = config.OPENAI_API_KEY):
    """
    Lazy-load and return the OpenAI client.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        The OpenAI client instance
    """
    global _openai_client
    if _openai_client is None:
        try:
            _openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    return _openai_client

def get_local_embedding_model():
    """
    Lazy-load and return the sentence transformer model.
    
    Returns:
        The sentence transformer model instance
    """
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {config.LOCAL_EMBEDDING_MODEL}")
            _sentence_transformer_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
            logger.info(f"Local embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {str(e)}")
            raise
    return _sentence_transformer_model

def get_local_embedding(text: str, model_name: str = config.LOCAL_EMBEDDING_MODEL) -> List[float]:
    """
    Generate an embedding vector for the given text using a local model.
    
    Args:
        text: Text to generate embedding for
        model_name: Name of the local model to use
        
    Returns:
        Embedding vector as a list of floats
    """
    try:
        # Get the model
        model = get_local_embedding_model()
        
        # Generate embedding
        embedding = model.encode(text).tolist()
        
        logger.info(f"Generated local embedding with model {model_name}, vector dimension: {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating local embedding: {str(e)}")
        raise

def get_embedding(text: str, api_key: str = config.OPENAI_API_KEY, model: str = config.EMBEDDING_MODEL, use_local: bool = None) -> List[float]:
    """
    Generate an embedding vector for the given text using OpenAI's API or a local model.
    
    Args:
        text: Text to generate embedding for
        api_key: OpenAI API key (default from config)
        model: OpenAI embedding model to use (default from config)
        use_local: Whether to use a local model (default from config.USE_LOCAL_EMBEDDINGS_FOR_TESTS)
        
    Returns:
        Embedding vector as a list of floats
    """
    # Determine whether to use local embeddings
    if use_local is None:
        use_local = config.USE_LOCAL_EMBEDDINGS_FOR_TESTS
    
    # Use local embedding if specified
    if use_local:
        return get_local_embedding(text)
    
    # Otherwise use OpenAI API
    try:
        # Get the OpenAI client
        client = get_openai_client(api_key)
        
        # Handle rate limiting with retries
        max_retries = 5
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get the embedding from OpenAI using the new API
                response = client.embeddings.create(
                    input=text,
                    model=model
                )
                
                # Extract the embedding vector
                embedding = response.data[0].embedding
                
                logger.info(f"Generated OpenAI embedding with model {model}, vector dimension: {len(embedding)}")
                return embedding
                
            except RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Rate limit exceeded after maximum retries")
                    raise
            except Exception as e:
                logger.error(f"Error generating OpenAI embedding: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise

def batch_get_embeddings(texts: List[str], api_key: str = config.OPENAI_API_KEY, model: str = config.EMBEDDING_MODEL, use_local: bool = None) -> List[List[float]]:
    """
    Generate embedding vectors for a batch of texts.
    
    Args:
        texts: List of texts to generate embeddings for
        api_key: OpenAI API key (default from config)
        model: OpenAI embedding model to use (default from config)
        use_local: Whether to use a local model (default from config.USE_LOCAL_EMBEDDINGS_FOR_TESTS)
        
    Returns:
        List of embedding vectors
    """
    # Determine whether to use local embeddings
    if use_local is None:
        use_local = config.USE_LOCAL_EMBEDDINGS_FOR_TESTS
    
    # Use local embeddings if specified
    if use_local:
        try:
            # Get the model
            model = get_local_embedding_model()
            
            # Generate embeddings in batch
            from tqdm.auto import tqdm
            
            # Process in batches of 32 to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
                batch = texts[i:i+batch_size]
                batch_embeddings = model.encode(batch).tolist()
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} local embeddings with model {config.LOCAL_EMBEDDING_MODEL}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating local embeddings: {str(e)}")
            raise
    
    # Otherwise use OpenAI API
    try:
        # Get the OpenAI client
        client = get_openai_client(api_key)
        
        # Handle rate limiting with retries
        max_retries = 5
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get embeddings from OpenAI using the new API
                response = client.embeddings.create(
                    input=texts,
                    model=model
                )
                
                # Extract the embedding vectors
                embeddings = [item.embedding for item in response.data]
                
                logger.info(f"Generated {len(embeddings)} OpenAI embeddings with model {model}")
                return embeddings
                
            except RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Rate limit exceeded after maximum retries")
                    raise
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the module
    test_text = "This is a test text for generating embeddings."
    
    # Test local embedding
    if config.USE_LOCAL_EMBEDDINGS_FOR_TESTS:
        print(f"Testing local embedding with model: {config.LOCAL_EMBEDDING_MODEL}")
        local_embedding = get_local_embedding(test_text)
        print(f"Generated local embedding with dimension: {len(local_embedding)}")
        print(f"First 5 values: {local_embedding[:5]}")
    
    # Test OpenAI embedding
    if config.OPENAI_API_KEY != "your-openai-api-key":
        print(f"Testing OpenAI embedding with model: {config.EMBEDDING_MODEL}")
        try:
            openai_embedding = get_embedding(test_text, use_local=False)
            print(f"Generated OpenAI embedding with dimension: {len(openai_embedding)}")
            print(f"First 5 values: {openai_embedding[:5]}")
        except Exception as e:
            print(f"Error testing OpenAI embedding: {str(e)}")
    else:
        print("Skipping OpenAI embedding test as no valid API key is provided")
