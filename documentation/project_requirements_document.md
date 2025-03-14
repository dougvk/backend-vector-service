# Project Requirements Document

## 1. Project Overview

This project is about building a backend system that uses a vector database to search podcast transcripts through natural language queries. The system reads local .txt files containing podcast transcripts, splits them into manageable chunks (defaulting to 2000 words), generates vector embeddings for each chunk using OpenAI’s embedding API, and stores these embeddings using the llamaindex Python library. When a user sends a GET request with a query, the system searches through the vector database and returns the top 10 most relevant transcript segments along with the podcast title extracted from the file name.

The purpose of this project is to provide a simple yet effective way to search through large amounts of podcast transcript data by retrieving contextually similar segments based on natural language queries. The core objectives include efficient transcript processing, meaningful embedding generation, quick and accurate query responses, and low-cost deployment on a DigitalOcean droplet. Success is measured by the system’s accuracy in returning relevant transcript sections and its smooth operation on minimal infrastructure.

## 2. In-Scope vs. Out-of-Scope

**In-Scope:**

*   Reading and importing .txt podcast transcript files stored locally in the project repository.
*   Extracting metadata (e.g., podcast title) from the file names.
*   Splitting transcripts into chunks with a configurable default size of 2000 words.
*   Generating vector embeddings for each chunk using OpenAI's embedding API.
*   Storing and indexing the generated embeddings using the llamaindex library.
*   Providing a simple backend GET endpoint to accept user queries.
*   Returning the top 10 most relevant transcript segments along with the associated podcast title.
*   Logging operations for debugging during transcript processing, embedding generation, indexing, and querying.
*   Deploying the entire application on a basic DigitalOcean droplet.

**Out-of-Scope:**

*   Developing a full-fledged user interface or web front-end; the service will only respond to GET requests.
*   Implementing any authentication or access control mechanisms.
*   Adding support for file formats other than .txt.
*   Supporting multiple user roles or advanced access management.
*   Integrating other NLP features or media types beyond transcript text search.
*   Extensive scalability features beyond the immediate requirements for personal use.

## 3. User Flow

A typical user interaction begins when the system is deployed and set up to scan the local repository for .txt transcript files. Once the system has processed these files—splitting them into chunks, generating embeddings, and indexing the results—a backend GET endpoint is ready to handle incoming requests. When a user sends a GET request with a natural language query, the system converts the query into a vector using OpenAI's API and performs a similarity search on the pre-indexed transcript chunks.

After the similarity search, the system retrieves the top 10 most relevant transcript segments. Each result includes the transcript snippet and the podcast title (extracted from the file’s name), ensuring that users have clear context for the returned segments. The response is then sent back to the user in a straightforward list format, making the process simple to operate and debug without a complex front-end interface.

## 4. Core Features

*   **Transcript Import and Preprocessing:**

    *   Read local .txt files from the project repository.
    *   Extract metadata (e.g., podcast title) from the file names.
    *   Split transcripts into chunks (default 2000 words, configurable).

*   **Embedding Generation:**

    *   Use OpenAI’s embedding API to convert text chunks into vector embeddings.
    *   Allow configuration of API settings (currently using OpenAI’s default embedding API).

*   **Indexing and Storage:**

    *   Store and index the generated embeddings using the llamaindex Python library.
    *   Maintain the association of each transcript chunk with its related metadata.

*   **Query Interface:**

    *   Accept GET requests containing natural language queries.
    *   Convert user query into an embedding vector.
    *   Perform similarity search against the indexed vector database.
    *   Return the top 10 most relevant transcript segments, along with the associated podcast title.

*   **Logging and Debugging:**

    *   Implement logging for all major operations (file import, embedding generation, indexing, querying) to aid in troubleshooting.

*   **Deployment on DigitalOcean:**

    *   Ensure the entire backend service is hosted effectively on an affordable DigitalOcean droplet.

## 5. Tech Stack & Tools

*   **Programming Language:** Python

*   **Backend Libraries:**

    *   llamaindex (for vector database operations, indexing, storage)
    *   Standard Python libraries for file I/O and logging

*   **Embedding API:** OpenAI's embedding API (for generating vector embeddings from transcript chunks)

*   **Deployment:** DigitalOcean Droplet (basic VPS for hosting backend service)

*   **Development Tools:**

    *   Windsurf (modern IDE with integrated AI coding assistance)

The integration of these tools will enable efficient processing of transcript files, generation and indexing of embeddings, and rapid response to query requests.

## 6. Non-Functional Requirements

*   **Performance:**

    *   The system should return query responses with minimal latency typical of a single-user personal service.
    *   Embedding generation and indexing should be efficient yet are scheduled as part of the transcript import process, not during a user query.

*   **Security:**

    *   As this is a backend service running on a secure VPS for personal use, standard Python security best practices should be followed.
    *   Sensitive API keys (for OpenAI’s embedding API) must be stored securely, e.g., using environment variables.

*   **Usability:**

    *   The application is designed to be easy to deploy and debug with clear logging.
    *   The configuration (such as transcript chunk size and API details) should be easily adjustable via configuration files or environment variables.

*   **Scalability:**

    *   While it is currently targeted for personal use, the design should allow added transcripts (and occasional new episodes) without extensive rework.

*   **Reliability:**

    *   Logging and error handling should catch and document any issues during file processing, API calls, or indexing, ensuring easier maintenance.

## 7. Constraints & Assumptions

*   The application assumes that all podcast transcript files are in .txt format and are stored locally within the project repository.
*   It is assumed that OpenAI’s embedding API and llamaindex library are available and can be accessed without rate limiting issues; however, considerable care should be taken in managing API keys and quota.
*   The chunk size for transcript processing starts at 2000 words by default, but it must be configurable.
*   The deployment environment is a basic DigitalOcean droplet, meaning there are limited resources available (networking, memory, storage) tailored for personal use.
*   The backend service is designed to operate without a front-end user interface or multi-user authentication systems.

## 8. Known Issues & Potential Pitfalls

*   API Rate Limits: OpenAI’s embedding API might have rate limits. Ensure that the system has proper error handling and retry logic if rate limits are exceeded.
*   File I/O: As the system processes around 700 .txt files, any I/O bottleneck or file corruption could slow down preprocessing. Monitor file integrity and processing performance.
*   Embedding Accuracy: The quality of embeddings is crucial; inconsistencies in transcript formatting might influence accuracy. Preprocessing should include normalization of text.
*   Configuration Errors: Misconfigured parameters (such as an incorrect chunk size or API key) can lead to failures during embedding generation or indexing. Use clear configuration validation.
*   Resource Limits: Running on a basic DigitalOcean droplet might impose limits. Be mindful of memory and storage usage, especially during batch processing. Consider logging resource usage to identify potential bottlenecks.
*   Debugging: Minimal logging may not capture all edge cases. It is recommended to gradually add more detailed logging if issues arise.

This document serves as the comprehensive and unambiguous reference for the entire project. It lays out the project’s purpose, the complete list of features, technology choices, and potential risks to ensure that downstream technical documents can be generated without any ambiguity.
