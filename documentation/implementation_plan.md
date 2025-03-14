# Implementation plan

## Phase 1: Environment Setup

1.  Create a new project folder named `backend-vector-service` on your local machine. (Project Goal)
2.  Open a terminal in the project folder and create a Python virtual environment with the command: `python3 -m venv venv`. (Tech Stack: Python)
3.  Activate the virtual environment by running `source venv/bin/activate` on Unix/macOS or `venv\Scripts\activate` on Windows. (Tech Stack: Python)
4.  Install required Python packages: run `pip install flask openai llamaindex` to install Flask for our GET API endpoint, OpenAI for embeddings, and llamaindex for vector indexing. (Tech Stack, Core Features)
5.  **Validation**: Run `python -c "import flask, openai, llamaindex"` to check that the packages are installed correctly. Read the current llamaindex docs to make sure implementation conforms to current documentation and recommendations (<https://docs.llamaindex.ai/>).

## Phase 2: Backend Project Structure

1.  Create a directory structure as follows:

    *   `/app` for the main application
    *   `/modules` for modular components
    *   `/logs` for log files
    *   Create a configuration file at the root named `config.py` (Project Goal, App Structure)

2.  **Validation**: List the directory structure using `tree -L 2` to ensure all folders are created.

## Phase 3: Input Module Development

1.  Create the file `/modules/input_module.py`. (Core Feature: Transcript Import and Preprocessing)
2.  In `/modules/input_module.py`, implement a function `load_transcripts(directory_path)` to read all `.txt` files from a specified directory. (Core Features: Import .txt Files)
3.  Implement functionality within the same file to extract the podcast title from the filename (e.g., using `os.path.basename(filename).split('.')[0]`). (Core Features: Metadata Extraction)
4.  Write a function `split_transcript(text, chunk_size=2000)` to split transcript text into chunks of 2000 words (default configurable). (Core Features: Transcript Splitting)
5.  **Validation**: Create a temporary test file (e.g., `/modules/test_input_module.py`) that imports `load_transcripts` and `split_transcript`, process a sample `.txt` file, and prints the number of chunks. Run this file to confirm correct splitting.

## Phase 4: Embedding Module Development

1.  Create the file `/modules/embedding_module.py`. (Core Feature: Embedding Generation)
2.  In `/modules/embedding_module.py`, implement a function `get_embedding(text, api_key, model='text-embedding-ada-002')` that calls OpenAI's Embedding API to generate an embedding vector for the given text. (Core Features: Using OpenAI's API)
3.  Make the API key and other details configurable by reading them from `config.py`. (Core Features: Configurable API connection)
4.  **Validation**: Create a test script (e.g., `/modules/test_embedding_module.py`) that uses a sample text and prints out the embedding vector (or its length) to ensure the call is functional.

## Phase 5: Indexing and Storage Module Development

1.  Create the file `/modules/indexing_module.py`. (Core Feature: Indexing and Storage)
2.  In `/modules/indexing_module.py`, implement a class (e.g., `TranscriptIndex`) that initializes a llamaindex instance to store embeddings and associated metadata (podcast title and transcript chunk).
3.  Add methods to insert new transcript chunks and metadata into the index and to perform a similarity search. (Core Features: Similarity Search)
4.  **Validation**: Write a test function within the module or in a separate test file (e.g., `/modules/test_indexing_module.py`) to insert dummy data and perform a similarity search ensuring results are returned.

## Phase 6: Query Interface (API) Development

1.  Create the file `/app/main.py` to serve as the main API entry point. (Core Feature: Query Interface)
2.  In `/app/main.py`, import Flask and initialize a Flask application. (Core Features: GET Request Interface)
3.  Define a GET endpoint `/query` that accepts a query string parameter (e.g., `?search=...`). (Core Features: GET Endpoint)
4.  In the `/query` endpoint, import the function `get_embedding` from `/modules/embedding_module.py` and convert the user's query into an embedding vector. (Core Features: Query Embedding)
5.  Call the similarity search method from `/modules/indexing_module.py` using the query embedding to retrieve the top 10 most relevant transcript segments along with their podcast titles. (Core Feature: Vector Search)
6.  Return the results as a JSON response. (Project Goal)
7.  **Validation**: Run the Flask server locally using `python /app/main.py`, then use a tool like `curl` to send a GET request (e.g., `curl "http://localhost:5000/query?search=sample"`) and check that a JSON response with results is returned.

## Phase 7: Updates and Automatic Processing

1.  In `/modules/input_module.py`, add a function `process_new_transcripts(directory_path)` that scans for new `.txt` files and processes them to update the index automatically. (Core Features: Updates and Maintenance)
2.  In `/modules/indexing_module.py`, integrate a logging mechanism to log each indexing operation (e.g., use Python’s built-in `logging` module and output logs to the `/logs` directory). (Core Features: Logging)
3.  **Validation**: Simulate the addition of a new transcript file, run the update function manually, and confirm via logs that the new transcript was processed and indexed.

## Phase 8: Integration

1.  Ensure that `/app/main.py` imports and uses functions from `/modules/input_module.py`, `/modules/embedding_module.py`, and `/modules/indexing_module.py` to provide end-to-end functionality. (Project Goal, App Structure)
2.  Incorporate configuration settings from `config.py` (e.g., API keys, chunk_size, file paths) across all modules for consistency and easy updates. (Core Features: Configuration)
3.  **Validation**: Run the complete application (Flask server), then execute an end-to-end scenario: add a transcript file to the designated folder, process it, and perform a query via the GET endpoint to verify integrated functionality.

## Phase 9: Deployment

1.  Prepare a `requirements.txt` file in the project root by running `pip freeze > requirements.txt`. (Tech Stack: Python)
2.  Write a simple deployment script or instructions to run the app on a DigitalOcean droplet. (Deployment: DigitalOcean Droplet)
3.  On the DigitalOcean droplet, install Python and clone the project repository. (Deployment: DigitalOcean Droplet)
4.  Set up a virtual environment on the droplet, install dependencies from `requirements.txt`, and configure environment variables as needed (e.g., OpenAI API key). (Deployment: DigitalOcean Droplet)
5.  Configure a process manager (such as systemd or supervisord) to run the Flask app with a production server like Gunicorn. (Deployment: DigitalOcean Droplet)
6.  **Validation**: On the droplet, start the application and use `curl` or a browser to make a GET request to the `/query` endpoint ensuring production functionality.

## Phase 10: Final Testing and Documentation

1.  Write comprehensive documentation (e.g., a `README.md`) that explains how to set up, run, and maintain the system, including configuration options and update procedures. (Project Goal, Updates and Maintenance)
2.  Create unit tests for each module within a `/tests` directory and run them to ensure code quality. (Core Features: Validation)
3.  **Validation**: Run all tests using a test runner such as `pytest` to confirm that every part of the system works as expected.
4.  Ensure that logging is functioning and monitor resource usage on the DigitalOcean droplet, adjusting configurations if necessary. (Known Issues & Potential Pitfalls)

# Note

*   All configuration parameters (e.g., OpenAI API key, chunk size, directories) should be stored in and read from `config.py` for ease of updates.
*   Monitor API rate limits from OpenAI and adjust error handling/retry logic in the embedding module if needed. (Known Issues & Potential Pitfalls)

By following these steps, you will create a backend service capable of processing podcast transcripts, generating and indexing embeddings, and enabling natural language queries via a GET API endpoint.
