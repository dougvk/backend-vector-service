# Backend Vector Service

A backend service for processing podcast transcripts, generating embeddings, and enabling natural language queries via a GET API endpoint.

## Project Overview

This service allows you to:
1. Import podcast transcript text files
2. Process and split transcripts into manageable chunks
3. Generate embeddings for each chunk using either OpenAI's API or local embedding models
4. Store embeddings and metadata in a vector index
5. Query the index using natural language to find relevant podcast segments

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (optional if using local embeddings)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend-vector-service
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
   - Copy the `.env.example` file to create a new `.env` file:
   ```bash
   cp .env.example .env
   ```
   - Edit the `.env` file and add your OpenAI API key:
   ```bash
   # Open the .env file in your favorite editor
   nano .env
   
   # Update the OPENAI_API_KEY value with your actual API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   - Adjust other configuration parameters in the `.env` file as needed

### Directory Structure

```
backend-vector-service/
├── app/
│   └── main.py            # Main API entry point
├── modules/
│   ├── input_module.py    # Transcript processing
│   ├── embedding_module.py # Embedding generation
│   └── indexing_module.py # Vector indexing and search
├── logs/                  # Log files
├── transcripts/           # Transcript files
├── index_storage/         # Vector index storage
├── config.py              # Configuration file
├── test_service.py        # Main test script
├── test_api.py            # API test script
├── .env.example           # Template for environment variables
├── .env                   # Environment variables (not in version control)
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Usage

### Adding Transcript Files

1. Place your podcast transcript files (in `.txt` format) in the `transcripts` directory.
2. The filename should represent the podcast title (e.g., `My Podcast Episode.txt`).

### Running the API Server

1. Start the Flask server:
```bash
cd backend-vector-service
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run with local embeddings (default)
python app/main.py

# Run with OpenAI embeddings
python app/main.py --use-openai
```

2. The API will be available at `http://localhost:8080`.

### Command-line Arguments

The application supports the following command-line arguments:

- `--use-openai`: Use OpenAI embeddings instead of local embeddings for API queries
  - Example: `python app/main.py --use-openai`

### API Endpoints

#### 1. Query Endpoint
- **URL**: `/query`
- **Method**: GET
- **Parameters**:
  - `search`: Query string to search for (required)
  - `top_k`: Number of results to return (optional, default: 10)
  - `podcast`: Filter by podcast title (optional)
- **Example**: `http://localhost:8080/query?search=artificial intelligence&top_k=5`

#### 2. Update Endpoint
- **URL**: `/update`
- **Method**: POST
- **Description**: Processes new transcript files in the `transcripts` directory and updates the index.
- **Example**: `curl -X POST http://localhost:8080/update`

### Testing

The application includes comprehensive test scripts:

1. Run the main service tests:
```bash
# Test with local embeddings (default)
python test_service.py

# Test with OpenAI embeddings
python test_service.py --use-openai
```

2. Run the API tests:
```bash
# Test with local embeddings (default)
python test_api.py

# Test with OpenAI embeddings
python test_api.py --use-openai
```

## Environment Variables

The application uses environment variables for configuration. These can be set in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | (required for OpenAI embeddings) |
| `EMBEDDING_MODEL` | OpenAI embedding model to use | text-embedding-3-small |
| `USE_LOCAL_EMBEDDINGS_FOR_TESTS` | Whether to use local embeddings for tests | True |
| `LOCAL_EMBEDDING_MODEL` | Local embedding model to use | all-MiniLM-L6-v2 |
| `API_HOST` | Host to bind the API server to | 0.0.0.0 |
| `API_PORT` | Port to bind the API server to | 8080 |
| `API_DEBUG` | Whether to run the API in debug mode | True |

## Embedding Models

The application supports two types of embedding models:

1. **Local Embeddings**: Uses the Sentence Transformers library with the `all-MiniLM-L6-v2` model
   - Faster and doesn't require API credits
   - Suitable for testing and development
   - Set `USE_LOCAL_EMBEDDINGS_FOR_TESTS = True` in `.env` to use local embeddings by default

2. **OpenAI Embeddings**: Uses OpenAI's embedding API
   - Higher quality embeddings for production use
   - Requires an API key and consumes API credits
   - Use the `--use-openai` flag to override the default setting

## Deployment

### DigitalOcean Droplet Deployment

1. Create a DigitalOcean droplet with Ubuntu.
2. SSH into your droplet.
3. Install Python and Git:
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git
```

4. Clone the repository and set up the application:
```bash
git clone <repository-url>
cd backend-vector-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn  # Production WSGI server
```

5. Configure the application:
```bash
cp .env.example .env
nano .env  # Update with your OpenAI API key and production settings
```

6. Set up a systemd service for automatic startup:
```bash
sudo nano /etc/systemd/system/vector-service.service
```

7. Add the following content to the service file:
```
[Unit]
Description=Backend Vector Service
After=network.target

[Service]
User=<your-username>
WorkingDirectory=/path/to/backend-vector-service
ExecStart=/path/to/backend-vector-service/venv/bin/gunicorn --workers=1 --threads=4 -b 0.0.0.0:8080 app.main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

8. Enable and start the service:
```bash
sudo systemctl enable vector-service
sudo systemctl start vector-service
```

### Memory Requirements

The vector service requires approximately 1GB of memory when running with OpenAI embeddings for ~500 transcripts. For optimal performance on a DigitalOcean droplet:

- **Recommended VPS size**: 2GB RAM Basic Droplet ($8/month)
- **Worker configuration**: 1 worker with 4 threads (as configured above)

This configuration balances memory usage and performance for a moderate number of transcripts. If you need to scale beyond 1000 transcripts, consider upgrading to a 4GB droplet.

## Maintenance

### Updating the Index

To update the index with new transcript files:
1. Add new transcript files to the `transcripts` directory.
2. Make a POST request to the `/update` endpoint.

### Monitoring

- Check the log files in the `logs` directory for errors and information.
- Monitor API rate limits from OpenAI to avoid exceeding quotas.

## Troubleshooting

- **API Key Issues**: Ensure your OpenAI API key is correctly set in the `.env` file if using OpenAI embeddings.
- **File Format Issues**: Transcript files must be plain text (`.txt`) files.
- **Index Errors**: If the index becomes corrupted, delete the `index_storage` directory and restart the application to rebuild the index.
- **Port Conflicts**: If port 8080 is already in use, modify the port number in your `.env` file.
- **Environment Variables**: If changes to the `.env` file don't take effect, restart the application.

## License

[Specify your license here]
