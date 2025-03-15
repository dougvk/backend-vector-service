# Query API Documentation

## Endpoint Details

- **URL**: `/query`
- **Method**: GET
- **Purpose**: Searches for relevant podcast transcript chunks based on a semantic query

## Query Parameters

1. **`search`** (required)
   - The search query text you want to find in the podcast transcripts
   - Example: `?search=artificial intelligence in healthcare`

2. **`top_k`** (optional)
   - Number of results to return
   - Default: 10
   - Example: `?search=AI&top_k=5`

3. **`podcast`** (optional)
   - Filter results to a specific podcast title
   - Example: `?search=climate change&podcast=Science Weekly`

## Response Format

The API returns a JSON response with the following structure:

```json
{
  "query": "your search query",
  "results": [
    {
      "podcast_title": "Podcast Name",
      "chunk_id": "unique_chunk_identifier",
      "text": "The actual transcript text that matched your query...",
      "score": 0.87654 // Similarity score (higher is better)
    },
    // More results...
  ]
}
```

## Error Responses

- **400 Bad Request**: If the required `search` parameter is missing
  ```json
  {
    "error": "Missing required parameter: search"
  }
  ```

- **500 Internal Server Error**: If there's a server-side error processing the query
  ```json
  {
    "error": "Error processing query: [error details]"
  }
  ```

## Implementation Notes

1. The API uses vector embeddings to perform semantic search, meaning it can find conceptually related content even if the exact words don't match.

2. Results are sorted by relevance (similarity score).

3. The backend uses either OpenAI embeddings or local embeddings depending on server configuration.

4. The service is designed to handle hundreds of podcast transcripts efficiently.

## Example Usage

For your frontend, you might implement a search function like this:

```javascript
async function searchTranscripts(query, topK = 10, podcastFilter = null) {
  let url = `/query?search=${encodeURIComponent(query)}&top_k=${topK}`;
  
  if (podcastFilter) {
    url += `&podcast=${encodeURIComponent(podcastFilter)}`;
  }
  
  try {
    const response = await fetch(url);
    const data = await response.json();
    
    if (response.ok) {
      return data.results;
    } else {
      throw new Error(data.error || 'Unknown error occurred');
    }
  } catch (error) {
    console.error('Search failed:', error);
    throw error;
  }
}
```
