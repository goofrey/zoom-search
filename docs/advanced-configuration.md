# Advanced Configuration

This document keeps the longer provider, parameter, output, and error reference for Zoom Search.

## Output Modes

| Mode | Answer synthesis | Returned fields |
|---|---:|---|
| `answer` | yes | `request_id`, `metrics`, `answer`, `warnings` |
| `answer_with_sources` | yes | `request_id`, `metrics`, `answer`, `results`, `search_context`, `warnings` |
| `results_simple` | no | `request_id`, `metrics`, `results`, `warnings` |
| `results_detailed` | no | `request_id`, `metrics`, `results`, `warnings` |

Detailed results include `source_domain`, `traceability`, and `duplicate_traceabilities`.

## Request Parameters

Top-level parameters:

- `question`: user question.
- `previous_conversation`: recent context strings; only the latest two non-empty entries are kept.
- `output_mode`: one of `answer`, `answer_with_sources`, `results_simple`, or `results_detailed`.
- `demo_mode`: use deterministic demo providers.
- `seed`: reproducibility hint for demo mode and supported LLM providers.
- `http_proxy`: global provider proxy URL.

LLM parameters:

- `llm_engine`, `llm_model`, `llm_api_key`, `llm_base_url`
- `llm_headers`, `llm_http_proxy`, `llm_extra`
- `llm_request_options` with optional booleans for `temperature`, `response_format`, `seed`, `stream`, and `reasoning`

Search parameters:

- `search_engine`, `search_api_key`, `search_base_url`
- `search_headers`, `search_http_proxy`, `search_extra`
- `search_result_collection_path`, `search_title_fields`, `search_snippet_fields`, `search_url_fields` for custom search mapping

Search limits:

- `zoomout_num_results`: default `5`
- `zoomin_num_results`: default `5`
- `top_k_domains_per_query`: default `1`

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
    llm_headers={"x-api-key": "secret"},
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
    search_api_key="secret",
)
```

Custom search endpoint:

```python
response = await search(
    question="Latest SQLite performance news.",
    llm_engine="gemini",
    llm_model="gemini-2.5-flash",
    llm_api_key="secret",
    search_engine="custom",
    search_base_url="https://search.example.com/search",
    search_api_key="secret",
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
        api_key="secret",
    ),
    search=SearchConfig(
        engine="custom",
        base_url="https://search.example.com/search",
        api_key="secret",
    ),
)
```

## Errors And Warnings

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
