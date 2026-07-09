# Agent Integration

Zoom Search can be used as an evidence-retrieval tool inside an agent loop. The agent calls Zoom Search when it needs grounded web evidence, then consumes a structured response containing the answer draft, source list, warnings, and runtime metrics.

This keeps the core search workflow independent while making it easy to plug into LangGraph, LangChain, or other tool-calling agents.

## LangGraph / LangChain Tool

See [`examples/langgraph_tool.py`](../examples/langgraph_tool.py) for a minimal async tool wrapper.

```python
from langchain.tools import tool

from zoom_search import search


@tool
async def zoom_search_evidence(query: str, include_raw_diagnostics: bool = False) -> dict:
    """Search for grounded web evidence and return answer, sources, warnings, and metrics."""

    response = await search(
        question=query,
        output_mode="answer_with_sources",
        include_raw_diagnostics=include_raw_diagnostics,
    )
    payload = {
        "request_id": response.request_id,
        "answer": response.answer,
        "sources": response.results,
        "metrics": response.metrics,
        "warnings": response.warnings,
    }
    if include_raw_diagnostics:
        payload["raw_diagnostics"] = response.raw_diagnostics
    return payload
```

In production, pass real providers and API keys the same way as the standard Zoom Search API:

```python
response = await search(
    question=query,
    llm_engine="gemini",
    llm_model="gemini-2.5-flash",
    llm_api_key="YOUR_GEMINI_API_KEY",
    search_engine="tavily",
    search_api_key="YOUR_TAVILY_API_KEY",
    output_mode="answer_with_sources",
)
```

## Tool Output Shape

The recommended agent-facing output is JSON-like and reviewable:

```python
{
    "request_id": "...",
    "answer": "...",
    "sources": [
        {
            "title": "...",
            "snippet": "...",
            "url": "...",
            "source_domain": "...",
        }
    ],
    "metrics": {
        "elapsed_ms": 1234,
        "search_requests": {...},
        "llm_usage": {...},
    },
    "warnings": [],
}
```

Agents can use this output to:

- answer with citations
- inspect source provenance before final response generation
- decide whether to run follow-up searches
- track quality, latency, token use, and search request cost
- expose warnings or low-confidence retrieval paths for human review

## Why Use Zoom Search As An Agent Tool

- Query refinement improves source discovery before the agent writes an answer.
- Source-domain zoom-in gives the agent deeper evidence from high-value domains.
- Duplicate provenance and source domains make outputs easier to inspect.
- Metrics make agent behavior easier to evaluate and regress-test.

## MCP Server Adapter

Zoom Search also ships an optional stdio MCP server that exposes a `zoom_search` tool.

Install MCP support:

```bash
pip install "zoom-search[mcp]"
```

Run the server:

```bash
zoom-search-mcp
```

For real providers, pass credentials through environment variables in your MCP client configuration:

```json
{
  "mcpServers": {
    "zoom-search": {
      "command": "zoom-search-mcp",
      "env": {
        "ZOOM_SEARCH_LLM_ENGINE": "gemini",
        "ZOOM_SEARCH_LLM_MODEL": "gemini-2.5-flash",
        "ZOOM_SEARCH_LLM_API_KEY": "YOUR_GEMINI_API_KEY",
        "ZOOM_SEARCH_SEARCH_ENGINE": "tavily",
        "ZOOM_SEARCH_SEARCH_API_KEY": "YOUR_TAVILY_API_KEY"
      }
    }
  }
}
```

The MCP tool accepts:

```json
{
  "question": "Which vector databases support hybrid search and metadata filtering for Python apps?",
  "previous_conversation": [],
  "output_mode": "answer_with_sources",
  "demo_mode": false,
  "zoomout_num_results": 5,
  "zoomin_num_results": 5,
  "top_k_domains_per_query": 1,
  "include_raw_diagnostics": false
}
```

For local no-key demos, call the tool with `demo_mode: true`.
For reliability testing, set `include_raw_diagnostics: true` to include raw LLM response diagnostics for query rewriting and answer synthesis. Keep it disabled for normal agent calls.

The server reads these environment variables:

| Variable | Purpose |
|---|---|
| `ZOOM_SEARCH_LLM_ENGINE` | Built-in LLM provider engine, e.g. `gemini`, `deepseek`, or `openai` |
| `ZOOM_SEARCH_LLM_MODEL` | Provider model name |
| `ZOOM_SEARCH_LLM_API_KEY` | LLM provider API key |
| `ZOOM_SEARCH_LLM_BASE_URL` | Optional LLM base URL override |
| `ZOOM_SEARCH_SEARCH_ENGINE` | Built-in search engine, e.g. `tavily`, `serpapi`, or `brave` |
| `ZOOM_SEARCH_SEARCH_API_KEY` | Search provider API key |
| `ZOOM_SEARCH_SEARCH_BASE_URL` | Optional search base URL override |
| `ZOOM_SEARCH_DEMO_MODE` | Set to `true` for demo-mode default |
