"""
Test script for the backend vector service API.
This script tests the API endpoints.
"""

import os
import sys
import json
import unittest
import requests
import argparse
from threading import Thread
from time import sleep

# Add the parent directory to the path to import config and modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from app.main import app

# Parse command line arguments before unittest takes over
parser = argparse.ArgumentParser(description='API Test Script')
parser.add_argument('--use-openai', action='store_true', 
                    help='Use OpenAI embeddings instead of local embeddings')
# Parse only known args to avoid conflicts with unittest
args, remaining_args = parser.parse_known_args()

# Update sys.argv to only include the args that unittest should see
sys.argv = [sys.argv[0]] + remaining_args

class TestVectorServiceAPI(unittest.TestCase):
    """Test cases for the Vector Service API."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class by starting the Flask server in a separate thread."""
        # Set up the Flask app with the appropriate embedding setting
        if args.use_openai:
            print("Using OpenAI embeddings for API tests")
            app.config['USE_LOCAL_EMBEDDINGS'] = False
        else:
            print(f"Using local embeddings for API tests: {config.LOCAL_EMBEDDING_MODEL}")
            app.config['USE_LOCAL_EMBEDDINGS'] = True
        
        # Start the Flask app in a separate thread
        cls.server_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=8080, debug=False))
        cls.server_thread.daemon = True
        cls.server_thread.start()
        
        # Give the server a moment to start
        sleep(1)
        
        # Base URL for API requests
        cls.base_url = "http://localhost:8080"
    
    def test_query_endpoint(self):
        """Test the /query endpoint."""
        # Test with a valid query
        query = "artificial intelligence"
        response = requests.get(f"{self.base_url}/query?search={query}")
        
        # Check response status code
        self.assertEqual(response.status_code, 200)
        
        # Parse JSON response
        data = response.json()
        
        # Check response structure
        self.assertIn('query', data)
        self.assertIn('results', data)
        self.assertEqual(data['query'], query)
        self.assertIsInstance(data['results'], list)
        
        # Print results for inspection
        print(f"\nQuery: {query}")
        print(f"Found {len(data['results'])} results")
        
        # Check at least one result if index is populated
        if len(data['results']) > 0:
            result = data['results'][0]
            self.assertIn('podcast_title', result)
            self.assertIn('chunk_id', result)
            self.assertIn('text', result)
            self.assertIn('score', result)
            
            # Print the first result
            print(f"Top result: {result['podcast_title']} - {result['chunk_id']}")
            print(f"Score: {result['score']}")
            print(f"Text: {result['text'][:100]}...")
    
    def test_query_endpoint_with_filter(self):
        """Test the /query endpoint with podcast filter."""
        # Test with a valid query and podcast filter
        query = "artificial intelligence"
        podcast_filter = "1_The French Revolution The Execution of the King Part 4"  # Use an actual podcast title
        
        response = requests.get(f"{self.base_url}/query?search={query}&podcast={podcast_filter}")
        
        # Check response status code
        self.assertEqual(response.status_code, 200)
        
        # Parse JSON response
        data = response.json()
        
        # Check response structure
        self.assertIn('query', data)
        self.assertIn('results', data)
        
        # Print results for inspection
        print(f"\nQuery with filter: {query}, Podcast: {podcast_filter}")
        print(f"Found {len(data['results'])} results")
        
        # If there are results, check they match the filter
        for result in data['results']:
            self.assertEqual(result['podcast_title'], podcast_filter)
    
    def test_query_endpoint_missing_search(self):
        """Test the /query endpoint with missing search parameter."""
        response = requests.get(f"{self.base_url}/query")
        
        # Check response status code (should be 400 Bad Request)
        self.assertEqual(response.status_code, 400)
        
        # Parse JSON response
        data = response.json()
        
        # Check error message
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Missing required parameter: search')

if __name__ == "__main__":
    unittest.main()
