# Advanced Configuration

This document keeps the detailed workflow, provider, parameter, output, streaming, and error reference for Zoom Search.

## Requirements

- Python `>=3.10`
- `httpx>=0.27.0`
- `uv` is recommended for development workflows

## Workflow

1. Normalize a `SearchRequest`, request dictionary, or flat keyword parameters.
2. Resolve LLM and search providers from capability declarations.
3. Rewrite the question into structured search groups and query variants.
4. Run broad zoom-out searches.
5. Select high-value source domains.
6. Run targeted zoom-in searches on those domains.
7. Deduplicate results and preserve traceability.
8. Format evidence and optionally synthesize an answer.

<p align="center">
  <img src="./assets/zoom-search-workflow.svg" alt="Zoom Search workflow" />
</p>

<p align="center">
  <img src="./assets/direct-vs-zoom-search.svg" alt="Direct Search vs Zoom Search comparison" />
</p>

## Example Queries

- What hotels in Shenzhen have rooms with exercise bikes?
- Which vector databases support hybrid search and metadata filtering for Python apps?
- What are the latest SQLite performance improvements and what versions introduced them?
- Which AI coding tools support self-hosted deployment for enterprise teams?
- What are the differences between Tavily, Brave Search API, and Serper for AI search apps?

## Output Modes

| Mode | Answer synthesis | Returned fields |
|---|---:|---|
| `answer` | yes | `request_id`, `metrics`, `answer`, `warnings` |
| `answer_with_sources` | yes | `request_id`, `metrics`, `answer`, `results`, `search_context`, `warnings` |
| `results_simple` | no | `request_id`, `metrics`, `results`, `warnings` |
| `results_detailed` | no | `request_id`, `metrics`, `results`, `warnings` |

Detailed results include `source_domain`, `traceability`, and `duplicate_traceabilities`.

## Streaming Events

Answer modes emit `search_started`, `search_completed`, `answer_started`, `answer_delta`, `answer_completed`, and `completed`.

| Event | When it appears |
|---|---|
| `search_started` | The workflow has accepted and normalized the request. |
| `search_completed` | Search, zoom-in, deduplication, and evidence formatting are complete. |
| `answer_started` | LLM answer synthesis is about to start. |
| `answer_delta` | A streamed answer text chunk is available. |
| `answer_completed` | Answer synthesis has finished. |
| `completed` | The final `SearchResponse` is available on the event. |

## Conversation History

Use `previous_conversation` when the latest question depends on recent context. Zoom Search keeps the latest two non-empty entries and uses them during query rewriting and answer synthesis.

```python
response = await search(
    question="What about hotels with in-room fitness equipment?",
    previous_conversation=[
        "I am planning a business trip to Shenzhen.",
        "I prefer hotels with wellness facilities.",
    ],
    demo_mode=True,
    output_mode="answer_with_sources",
)
```

## Request Parameters

Top-level parameters:

| Parameter | Description |
|---|---|
| `question` | User question to search and answer. |
| `previous_conversation` | Recent context strings; only the latest two non-empty entries are kept. |
| `output_mode` | `answer`, `answer_with_sources`, `results_simple`, or `results_detailed`. |
| `demo_mode` | Use deterministic local demo providers without API keys. |
| `seed` | Reproducibility hint for demo mode and supported LLM providers. |
| `http_proxy` | Global provider proxy URL. |

LLM parameters:

| Parameter | Description |
|---|---|
| `llm_engine` | Built-in engine name, `openai-compatible`, or `custom`. |
| `llm_model` | Model name passed to the selected LLM provider. |
| `llm_api_key` | API key for the LLM provider. |
| `llm_base_url` | Custom or OpenAI-compatible LLM endpoint base URL. |
| `llm_headers` | Additional request headers for the LLM provider. |
| `llm_http_proxy` | LLM-specific proxy URL. |
| `llm_extra` | Provider-specific mapping or adapter options. |
| `llm_request_options` | Optional feature flags for `temperature`, `response_format`, `seed`, `stream`, and `reasoning`. |

Search parameters:

| Parameter | Description |
|---|---|
| `search_engine` | Built-in search engine name or `custom`. |
| `search_api_key` | API key for the search provider. |
| `search_base_url` | Custom search endpoint URL. |
| `search_headers` | Additional request headers for the search provider. |
| `search_http_proxy` | Search-specific proxy URL. |
| `search_extra` | Provider-specific search options. |
| `search_result_collection_path` | Dot path to the result list for custom search responses. |
| `search_title_fields` | Candidate title fields for custom search result mapping. |
| `search_snippet_fields` | Candidate snippet fields for custom search result mapping. |
| `search_url_fields` | Candidate URL fields for custom search result mapping. |

Search limits:

| Parameter | Default | Description |
|---|---:|---|
| `zoomout_num_results` | `5` | Broad search result count per query. |
| `zoomin_num_results` | `5` | Domain-focused search result count. |
| `top_k_domains_per_query` | `1` | Number of source domains selected for zoom-in search. |

## Provider Architecture

Zoom Search has three provider kinds:

- `demo`: built-in deterministic LLM and search providers.
- `builtin`: providers declared in `zoom_search.providers.capabilities`.
- `custom`: caller-supplied OpenAI-compatible or native HTTP providers.

Most built-in LLM providers use the shared OpenAI-compatible adapter. `claude` and `replicate` use dedicated native adapters. Custom LLMs can use `llm_engine="openai-compatible"` for Chat Completions-like endpoints or `llm_engine="custom"` for native HTTP endpoints with explicit mappings.

Search adapters normalize provider requests and responses through `UnifiedSearchRequest` and `UnifiedSearchResponse`. Capability declarations control query paths, result paths, site restriction behavior, provider-side result counts, and local filtering.

## Built-In Engines

Built-in `llm_engine` options:

`openai`, `gemini`, `doubao-global`, `doubao-china`, `qwen-global`, `qwen-china`, `glm-china`, `glm-global`, `baichuan`, `spark`, `huggingface`, `claude`, `replicate`, `minimax-global`, `minimax-china`, `deepseek`, `kimi-china`, `kimi-global`, `yi`, `hunyuan`, `stepfun`, `siliconflow`, `together`, `fireworks`, `groq`, `cerebras`, `perplexity`, `grok`, `mistral`, `cohere`, `openrouter`, `mimo`, `deepinfra`, `novita`, `hyperbolic`, `lepton`, `ollama`, `openai-compatible`, `custom`.

Built-in `search_engine` options:

`tavily`, `serper`, `brave`, `you`, `360search`, `firecrawl`, `baidu`, `linkup`, `perplexity`, `glm`, `volcengine`, `exa`, `bocha`, `querit`, `serpapi`, `metasota`, `searxng`, `tiangong`, `custom`.

## Custom Providers

OpenAI-compatible LLM endpoint:

```python
response = await search(
    question="Summarize the latest SQLite performance news.",
    llm_engine="openai-compatible",
    llm_model="custom-model",
    llm_base_url="https://llm.example.com/v1",
    llm_api_key="YOUR_CUSTOM_LLM_API_KEY",
    search_engine="tavily",
    search_api_key="YOUR_TAVILY_API_KEY",
)
```

Custom native LLM endpoint:

```python
response = await search(
    question="How should I store bananas on the counter?",
    llm_engine="custom",
    llm_model="native-model",
    llm_base_url="https://native-llm.example.com",
    llm_headers={"x-api-key": "YOUR_NATIVE_LLM_API_KEY"},
    llm_extra={
        "endpoint_path": "/v1/generate",
        "request_mapping": {
            "model_path": "payload.modelName",
            "messages_path": "payload.chatMessages",
            "response_format_path": "payload.responseFormat",
        },
        "response_mapping": {
            "content_path": "data.output.text",
            "json_payload_path": "data.output.json",
            "usage_total_tokens_path": "usage.total",
            "finish_reason_path": "state.finish",
        },
    },
    search_engine="tavily",
    search_api_key="YOUR_TAVILY_API_KEY",
)
```

Custom search endpoint:

```python
response = await search(
    question="Latest SQLite performance news.",
    llm_engine="gemini",
    llm_model="gemini-2.5-flash",
    llm_api_key="YOUR_GEMINI_API_KEY",
    search_engine="custom",
    search_base_url="https://search.example.com/search",
    search_api_key="YOUR_SEARCH_API_KEY",
    search_result_collection_path="data.items",
    search_title_fields=["name", "title"],
    search_snippet_fields=["summary", "snippet"],
    search_url_fields=["link", "url"],
)
```

## Typed Requests

```python
from zoom_search.models import LLMConfig
from zoom_search.models import SearchConfig
from zoom_search.models import SearchRequest

request = SearchRequest(
    question="Summarize the latest SQLite performance news.",
    llm=LLMConfig(
        engine="openai-compatible",
        model="custom-model",
        base_url="https://llm.example.com/v1",
        api_key="YOUR_CUSTOM_LLM_API_KEY",
    ),
    search=SearchConfig(
        engine="custom",
        base_url="https://search.example.com/search",
        api_key="YOUR_SEARCH_API_KEY",
    ),
)
```

## Errors and Warnings

Catch `ZoomSearchError` and use `error.details.error_type` for stable application logic.

```python
from zoom_search import ZoomSearchError

try:
    response = await search(...)
except ZoomSearchError as error:
    print(error.details.error_type)
    print(error.details.reason_code)
    print(error.details.invalid_fields)
```

Stable `error_type` values include `configuration_error`, `authentication_error`, `invalid_request_error`, `rate_limit_error`, `quota_exceeded_error`, `content_filtered_error`, `network_error`, `provider_error`, and `empty_result_error`.

Warnings are returned on `SearchResponse.warnings` and stream events. Common warning codes include `low_confidence_site_restriction`, `zoom_in_domain_filtered_results`, `zoom_in_filtered_to_empty`, `result_collection_path_not_list`, and `search_call_failure`.

## Development Checks

```bash
# Install development dependencies.
uv sync

# Run tests.
uv run pytest

# Build package artifacts.
uv run python -m build

# Check package metadata.
uv run twine check dist/*
```

See [`docs/development.md`](./development.md) for evaluation assets and additional checks.
