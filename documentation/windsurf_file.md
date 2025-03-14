# .windsurfrules

## Project Overview

*   **Type:** windsurf_file
*   **Description:** This project builds a backend service that leverages a vector database to perform natural language queries over podcast transcripts. It reads local .txt transcript files, preprocesses them by splitting into configurable chunks (default 2000 words), generates vector embeddings using OpenAI's embedding API via the llamaindex library, and indexes these embeddings to support efficient similarity searches.
*   **Primary Goal:** Build a robust backend that returns the top 10 most relevant transcript segments (along with the associated podcast title) in response to natural language GET requests.

## Project Structure

### Framework-Specific Routing

*   **Directory Rules:**

    *   [python_3.x]: Use a modular, file-based architecture with a dedicated "src/" directory for code organization.
    *   Example 1: "main.py" initializes the application, integrating all modules.
    *   Example 2: "src/input.py" handles file scanning, transcript import, metadata extraction, and chunking logic.
    *   Example 3: "src/query.py" sets up the GET endpoint for handling query requests via similarity search.

### Core Directories

*   **Versioned Structure:**

    *   [src/]: Houses the complete backend service modules, each aligned with specific tasks:

        *   Input Module: Reads transcript files, extracts podcast titles from file names, and splits text into chunks.
        *   Embedding Module: Interfaces with OpenAI's embedding API to generate vector representations of each chunk.
        *   Indexing and Storage Module: Uses the llamaindex library to store and index embeddings for rapid query responses.
        *   Query Module: Exposes a simple GET endpoint to process user queries and retrieve the top 10 matching transcript segments along with metadata.

### Key Files

*   **Stack-Versioned Patterns:**

    *   [main.py]: Orchestrates initialization and integration of all service components.
    *   [src/input.py]: Implements file scanning, metadata extraction, and splitting of transcripts into configurable chunks.
    *   [src/embedding.py]: Manages embedding generation using OpenAI's embedding API.
    *   [src/indexing.py]: Handles storing and indexing of embeddings via the llamaindex library.
    *   [src/query.py]: Provides the GET endpoint for executing similarity searches and returning query results.

## Tech Stack Rules

*   **Version Enforcement:**

    *   [<python@3.x>]: Develop the application with Python 3.x, using best practices for modular code and configuration via environment variables.
    *   [llamaindex@latest]: Employ the latest version of the llamaindex library to facilitate vector database operations.
    *   [openai_embedding_api@default]: Integrate with OpenAI's embedding API using default settings (with configurable parameters as needed).

## PRD Compliance

*   **Non-Negotiable:**

    *   "The system must return the top 10 most relevant transcript segments along with the associated podcast title." : This requirement enforces processing local .txt files, stringent logging during indexing and querying, and an efficient backend pipeline.

## App Flow Integration

*   **Stack-Aligned Flow:**

    *   Python Backend Pipeline → The application begins by reading transcript files in src/input.py, processes them to generate vector embeddings via src/embedding.py, indexes these embeddings using src/indexing.py, and finally handles GET requests in src/query.py that perform similarity searches to return the top 10 transcript segments with metadata.
