import pytest

from zoom_search.models import SearchRequest
from zoom_search.orchestration.runtime import build_runtime_context
from zoom_search.providers.resolver import resolve_providers
from zoom_search.transport import create_transport_context
from zoom_search.validation import normalize_and_validate_request


def test_request_default_normalization() -> None:
    normalized = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "demo_mode": True,
            "previous_conversation": ["first", "second", "third"],
        }
    )

    assert normalized.output_mode == "answer"
    assert normalized.previous_conversation == ["second", "third"]
    assert normalized.search_limits.zoomout_num_results == 5
    assert normalized.search_limits.zoomin_num_results == 5
    assert normalized.search_limits.top_k_domains_per_query == 1


def test_string_previous_conversation_is_ignored() -> None:
    normalized = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "demo_mode": True,
            "previous_conversation": "abc",
        }
    )

    assert normalized.previous_conversation == []


@pytest.mark.parametrize(
    ("field", "value", "category", "invalid_fields"),
    [
        ("search_limits", "bad", "search_configuration_error", ["search_limits"]),
        ("llm", "bad", "llm_configuration_error", ["llm"]),
        ("search", "bad", "search_configuration_error", ["search"]),
        ("proxy", "bad", "proxy_configuration_error", ["proxy"]),
    ],
)
def test_invalid_nested_config_types_raise_configuration_error(
    field: str,
    value: object,
    category: str,
    invalid_fields: list[str],
) -> None:
    with pytest.raises(Exception) as exc_info:
        normalize_and_validate_request(
            {
                "question": "What is Zoom Search?",
                "demo_mode": True,
                field: value,
            }
        )

    error = exc_info.value
    assert error.category == category
    assert error.details.error_type == "configuration_error"
    assert error.details.invalid_fields == invalid_fields


def test_non_integer_search_limits_raise_configuration_error() -> None:
    with pytest.raises(Exception) as exc_info:
        normalize_and_validate_request(
            {
                "question": "What is Zoom Search?",
                "demo_mode": True,
                "zoomout_num_results": "x",
            }
        )

    error = exc_info.value
    assert error.category == "search_configuration_error"
    assert error.details.error_type == "configuration_error"
    assert error.details.invalid_fields == ["search_limits.zoomout_num_results"]


def test_flat_parameters_are_normalized_to_internal_models() -> None:
    normalized = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm_engine": "gemini",
            "llm_model": "gemini-2.5-flash",
            "llm_api_key": "secret",
            "llm_http_proxy": "https://llm-proxy.example.com",
            "llm_content_path": "data.output.text",
            "llm_json_payload_path": "data.output.json",
            "llm_usage_total_tokens_path": "stats.total",
            "llm_finish_reason_path": "state.finish",
            "search_engine": "tavily",
            "search_api_key": "search-secret",
            "search_http_proxy": "https://search-proxy.example.com",
            "search_result_collection_path": "data.items",
            "search_title_fields": ["name", "title"],
            "search_snippet_fields": "summary,snippet",
            "search_url_fields": ["link"],
            "http_proxy": "https://global-proxy.example.com",
            "zoomout_num_results": 3,
            "zoomin_num_results": 4,
            "top_k_domains_per_query": 2,
        }
    )

    assert normalized.llm is not None
    assert normalized.llm.engine == "gemini"
    assert normalized.llm.model == "gemini-2.5-flash"
    assert normalized.llm.api_key == "secret"
    assert normalized.llm.http_proxy == "https://llm-proxy.example.com"
    assert normalized.llm.extra["response_mapping"] == {
        "content_path": "data.output.text",
        "json_payload_path": "data.output.json",
        "usage_total_tokens_path": "stats.total",
        "finish_reason_path": "state.finish",
    }
    assert normalized.search is not None
    assert normalized.search.engine == "tavily"
    assert normalized.search.api_key == "search-secret"
    assert normalized.search.http_proxy == "https://search-proxy.example.com"
    assert normalized.search.extra["result_collection_path"] == "data.items"
    assert normalized.search.extra["title_fields"] == ["name", "title"]
    assert normalized.search.extra["snippet_fields"] == "summary,snippet"
    assert normalized.search.extra["url_fields"] == ["link"]
    assert normalized.proxy.http_proxy == "https://global-proxy.example.com"
    assert normalized.search_limits.zoomout_num_results == 3
    assert normalized.search_limits.zoomin_num_results == 4
    assert normalized.search_limits.top_k_domains_per_query == 2


def test_llm_request_options_are_normalized_and_resolved() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {
                "engine": "deepseek",
                "model": "deepseek-chat",
                "api_key": "secret",
                "request_options": {
                    "temperature": False,
                    "response_format": True,
                    "seed": "false",
                    "stream": "true",
                    "reasoning": "false",
                },
            },
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    assert request.llm is not None
    assert request.llm.request_options.temperature is False
    assert request.llm.request_options.response_format is True
    assert request.llm.request_options.seed is False
    assert request.llm.request_options.stream is True
    assert request.llm.request_options.reasoning is False

    providers = resolve_providers(request)
    assert providers.llm.request_options.temperature is False
    assert providers.llm.request_options.response_format is True
    assert providers.llm.request_options.seed is False
    assert providers.llm.request_options.stream is True
    assert providers.llm.request_options.reasoning is False


def test_prefixed_llm_request_options_are_normalized_and_resolved() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm_engine": "deepseek",
            "llm_model": "deepseek-v4-flash",
            "llm_api_key": "secret",
            "llm_request_options": {
                "temperature": False,
                "response_format": True,
                "seed": False,
                "stream": True,
                "reasoning": False,
            },
            "search_engine": "tavily",
            "search_api_key": "search-secret",
        }
    )

    assert request.llm is not None
    assert request.llm.request_options.temperature is False
    assert request.llm.request_options.response_format is True
    assert request.llm.request_options.seed is False
    assert request.llm.request_options.stream is True
    assert request.llm.request_options.reasoning is False

    providers = resolve_providers(request)
    assert providers.llm.request_options.temperature is False
    assert providers.llm.request_options.response_format is True
    assert providers.llm.request_options.seed is False
    assert providers.llm.request_options.stream is True
    assert providers.llm.request_options.reasoning is False


def test_invalid_engine_configuration_error() -> None:
    with pytest.raises(Exception) as exc_info:
        request = normalize_and_validate_request(
            {
                "question": "What is Zoom Search?",
                "llm": {
                    "engine": "invalid-llm",
                    "model": "gpt-test",
                    "api_key": "secret",
                },
                "search": {
                    "engine": "tavily",
                    "api_key": "secret",
                },
            }
        )
        resolve_providers(request)

    error = exc_info.value
    assert error.category == "llm_configuration_error"
    assert error.details.error_type == "configuration_error"
    assert error.details.invalid_fields == ["llm.engine"]


def test_invalid_llm_base_url_is_configuration_error() -> None:
    with pytest.raises(Exception) as exc_info:
        normalize_and_validate_request(
            {
                "question": "What is Zoom Search?",
                "llm": {"engine": "openai", "model": "gpt-test", "api_key": "secret", "base_url": "not-a-url"},
                "search": {"engine": "tavily", "api_key": "secret"},
            }
        )

    error = exc_info.value
    assert error.category == "llm_configuration_error"
    assert error.details.error_type == "configuration_error"
    assert error.details.invalid_fields == ["llm.base_url"]


def test_invalid_search_base_url_is_configuration_error() -> None:
    with pytest.raises(Exception) as exc_info:
        normalize_and_validate_request(
            {
                "question": "What is Zoom Search?",
                "llm": {"engine": "openai", "model": "gpt-test", "api_key": "secret"},
                "search": {"engine": "tavily", "api_key": "secret", "base_url": "ftp://search.example.com"},
            }
        )

    error = exc_info.value
    assert error.category == "search_configuration_error"
    assert error.details.error_type == "configuration_error"
    assert error.details.invalid_fields == ["search.base_url"]


def test_new_builtin_llm_provider_resolution() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {"engine": "deepseek", "model": "deepseek-chat", "api_key": "secret"},
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    providers = resolve_providers(request)

    assert providers.llm.engine == "deepseek"
    assert providers.llm.protocol == "openai_compatible"
    assert providers.llm.base_url == "https://api.deepseek.com/v1"


def test_huggingface_requires_org_model_format() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {"engine": "huggingface", "model": "llama-3", "api_key": "secret"},
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    with pytest.raises(Exception) as exc_info:
        resolve_providers(request)

    error = exc_info.value
    assert error.category == "llm_configuration_error"
    assert error.details.invalid_fields == ["llm.model"]


def test_doubao_global_requires_explicit_region_base_url() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {"engine": "doubao-global", "model": "seed-1-6", "api_key": "secret"},
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    with pytest.raises(Exception) as exc_info:
        resolve_providers(request)

    error = exc_info.value
    assert error.category == "llm_configuration_error"
    assert error.details.invalid_fields == ["llm.base_url"]


def test_lepton_requires_explicit_model_endpoint_base_url() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {"engine": "lepton", "model": "mistral-7b", "api_key": "secret"},
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    with pytest.raises(Exception) as exc_info:
        resolve_providers(request)

    error = exc_info.value
    assert error.category == "llm_configuration_error"
    assert error.details.invalid_fields == ["llm.base_url"]


def test_custom_openai_compatible_llm_streaming_is_opt_in() -> None:
    default_request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {
                "engine": "openai-compatible",
                "model": "custom-model",
                "base_url": "https://llm.example.com/v1",
            },
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )
    streaming_request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {
                "engine": "openai-compatible",
                "model": "custom-model",
                "base_url": "https://llm.example.com/v1",
                "extra": {"supports_streaming": True},
            },
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    default_provider = resolve_providers(default_request).llm
    streaming_provider = resolve_providers(streaming_request).llm

    assert default_provider.engine == "openai-compatible"
    assert default_provider.protocol == "openai_compatible"
    assert default_provider.capability.supports_streaming is False
    assert default_provider.capability.streaming_level == "unsupported"
    assert streaming_provider.capability.supports_streaming is True
    assert streaming_provider.capability.streaming_level == "sse_full_fidelity"
    assert streaming_provider.capability.execution_model == "sync_with_stream"


def test_custom_native_llm_defaults_to_native_protocol() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {
                "engine": "custom",
                "model": "native-model",
                "base_url": "https://native.example.com",
                "extra": {"response_mapping": {"content_path": "data.answer"}},
            },
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    provider = resolve_providers(request).llm

    assert provider.engine == "custom"
    assert provider.protocol == "native"
    assert provider.capability.adapter_class == "custom_native"
    assert provider.capability.supports_seed == "supported"


def test_legacy_custom_openai_compatible_protocol_is_still_supported() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {
                "engine": "custom",
                "protocol": "openai_compatible",
                "model": "custom-model",
                "base_url": "https://llm.example.com/v1",
            },
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    provider = resolve_providers(request).llm

    assert provider.engine == "custom"
    assert provider.protocol == "openai_compatible"
    assert provider.capability.adapter_class == "openai_compatible"


def test_ollama_builtin_llm_allows_missing_api_key() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {"engine": "ollama", "model": "llama3.2"},
            "search": {"engine": "tavily", "api_key": "search-secret"},
        }
    )

    providers = resolve_providers(request)

    assert providers.llm.engine == "ollama"
    assert providers.llm.api_key is None
    assert providers.llm.base_url == "http://localhost:11434/v1"


def test_request_id_and_runtime_context_behavior() -> None:
    request = normalize_and_validate_request(
        SearchRequest(
            question="What is Zoom Search?",
            demo_mode=True,
        )
    )

    providers = resolve_providers(request)

    context = build_runtime_context(
        request=request,
        llm_provider=providers.llm,
        search_provider=providers.search,
    )

    assert context.request_id.startswith("zs_")
    assert context.llm_provider.provider_kind == "demo"
    assert context.search_provider.provider_kind == "demo"
    assert context.retry_budget == {"llm": 1, "search": 1}


def test_provider_proxy_overrides_global_proxy() -> None:
    request = normalize_and_validate_request(
        {
            "question": "What is Zoom Search?",
            "llm": {
                "engine": "openai",
                "model": "gpt-test",
                "api_key": "secret",
                "http_proxy": "https://llm-proxy.example.com",
            },
            "search": {
                "engine": "tavily",
                "api_key": "secret",
                "http_proxy": "https://search-proxy.example.com",
            },
            "proxy": {"http_proxy": "https://global-proxy.example.com"},
        }
    )
    providers = resolve_providers(request)
    context = build_runtime_context(
        request=request,
        llm_provider=providers.llm,
        search_provider=providers.search,
    )

    transport_context = create_transport_context(context=context)

    assert transport_context.llm.proxy == "https://llm-proxy.example.com"
    assert transport_context.search.proxy == "https://search-proxy.example.com"


def test_invalid_provider_proxy_is_rejected() -> None:
    with pytest.raises(Exception) as exc_info:
        normalize_and_validate_request(
            {
                "question": "What is Zoom Search?",
                "llm": {
                    "engine": "openai",
                    "model": "gpt-test",
                    "api_key": "secret",
                    "http_proxy": "ftp://bad-proxy",
                },
                "search": {"engine": "tavily", "api_key": "secret"},
            }
        )

    error = exc_info.value
    assert error.category == "proxy_configuration_error"
    assert error.details.invalid_fields == ["llm.http_proxy"]
