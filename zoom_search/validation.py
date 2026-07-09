"""Request normalization and validation."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlparse

from zoom_search.errors import configuration_error
from zoom_search.models import LLMConfig
from zoom_search.models import LLMRequestOptions
from zoom_search.models import OutputMode
from zoom_search.models import ProxyConfig
from zoom_search.models import SearchConfig
from zoom_search.models import SearchLimits
from zoom_search.models import SearchRequest

ALLOWED_OUTPUT_MODES: set[OutputMode] = {
    "answer",
    "answer_with_sources",
    "results_simple",
    "results_detailed",
}


def normalize_and_validate_request(request: SearchRequest | dict) -> SearchRequest:
    normalized = _coerce_request(request)
    _validate_request(normalized)
    return normalized


def _coerce_request(request: SearchRequest | dict) -> SearchRequest:
    if isinstance(request, SearchRequest):
        data = {
            "question": request.question,
            "previous_conversation": list(request.previous_conversation or []),
            "output_mode": request.output_mode,
            "search_limits": request.search_limits,
            "llm": request.llm,
            "search": request.search,
            "demo_mode": request.demo_mode,
            "proxy": request.proxy,
            "seed": request.seed,
            "include_raw_diagnostics": request.include_raw_diagnostics,
        }
    elif isinstance(request, dict):
        data = dict(request)
    else:
        raise configuration_error(
            category="llm_configuration_error",
            component="validation",
            message="Unsupported request type.",
            user_message="Request must be a SearchRequest object or dict.",
            invalid_fields=["request"],
        )

    data = _expand_flat_parameters(data)

    previous_conversation = data.get("previous_conversation") or []
    if isinstance(previous_conversation, tuple):
        previous_conversation = list(previous_conversation)
    output_mode = data.get("output_mode") or "answer"

    search_limits_data = data.get("search_limits") or {}
    if isinstance(search_limits_data, SearchLimits):
        search_limits = SearchLimits(
            zoomout_num_results=search_limits_data.zoomout_num_results,
            zoomin_num_results=search_limits_data.zoomin_num_results,
            top_k_domains_per_query=search_limits_data.top_k_domains_per_query,
        )
    elif isinstance(search_limits_data, Mapping):
        search_limits_mapping = dict(search_limits_data)
        search_limits = SearchLimits(
            zoomout_num_results=search_limits_mapping.get("zoomout_num_results", 5),
            zoomin_num_results=search_limits_mapping.get("zoomin_num_results", 5),
            top_k_domains_per_query=search_limits_mapping.get("top_k_domains_per_query", 1),
        )
    else:
        raise configuration_error(
            category="search_configuration_error",
            component="validation",
            message="search_limits must be a mapping or SearchLimits object.",
            user_message="search_limits must be an object with positive integer values.",
            invalid_fields=["search_limits"],
        )

    proxy_data = data.get("proxy") or {}
    if isinstance(proxy_data, ProxyConfig):
        proxy = ProxyConfig(http_proxy=proxy_data.http_proxy)
    elif isinstance(proxy_data, Mapping):
        proxy = ProxyConfig(http_proxy=proxy_data.get("http_proxy"))
    else:
        raise configuration_error(
            category="proxy_configuration_error",
            component="validation",
            message="proxy must be a mapping or ProxyConfig object.",
            user_message="proxy must be an object with proxy settings.",
            invalid_fields=["proxy"],
        )

    llm_data = data.get("llm")
    llm = _coerce_llm_config(llm_data)

    search_data = data.get("search")
    search = _coerce_search_config(search_data)

    return SearchRequest(
        question=(data.get("question") or "").strip(),
        previous_conversation=_normalize_previous_conversation(previous_conversation),
        output_mode=output_mode,
        search_limits=search_limits,
        llm=llm,
        search=search,
        demo_mode=bool(data.get("demo_mode", False)),
        proxy=proxy,
        seed=data.get("seed"),
        include_raw_diagnostics=bool(data.get("include_raw_diagnostics", False)),
    )


def _expand_flat_parameters(data: dict) -> dict:
    expanded = dict(data)

    search_limits_value = expanded.get("search_limits")
    if isinstance(search_limits_value, SearchLimits):
        search_limits = {
            "zoomout_num_results": search_limits_value.zoomout_num_results,
            "zoomin_num_results": search_limits_value.zoomin_num_results,
            "top_k_domains_per_query": search_limits_value.top_k_domains_per_query,
        }
    elif search_limits_value is None:
        search_limits = {}
    elif isinstance(search_limits_value, Mapping):
        search_limits = dict(search_limits_value)
    else:
        raise configuration_error(
            category="search_configuration_error",
            component="validation",
            message="search_limits must be a mapping or SearchLimits object.",
            user_message="search_limits must be an object with positive integer values.",
            invalid_fields=["search_limits"],
        )
    for field_name in ("zoomout_num_results", "zoomin_num_results", "top_k_domains_per_query"):
        if field_name in expanded:
            search_limits[field_name] = expanded.pop(field_name)
    if search_limits:
        expanded["search_limits"] = search_limits

    proxy_value = expanded.get("proxy")
    if isinstance(proxy_value, ProxyConfig):
        proxy = {"http_proxy": proxy_value.http_proxy}
    elif proxy_value is None:
        proxy = {}
    elif isinstance(proxy_value, Mapping):
        proxy = dict(proxy_value)
    else:
        raise configuration_error(
            category="proxy_configuration_error",
            component="validation",
            message="proxy must be a mapping or ProxyConfig object.",
            user_message="proxy must be an object with proxy settings.",
            invalid_fields=["proxy"],
        )
    if "http_proxy" in expanded:
        proxy["http_proxy"] = expanded.pop("http_proxy")
    if proxy:
        expanded["proxy"] = proxy

    llm = _merge_custom_llm_mapping(expanded, expanded.get("llm"))
    llm = _merge_prefixed_config(llm, expanded, prefix="llm_")
    if llm is not None:
        expanded["llm"] = llm

    search_config = _merge_custom_search_mapping(expanded, expanded.get("search"))
    search_config = _merge_prefixed_config(search_config, expanded, prefix="search_")
    if search_config is not None:
        expanded["search"] = search_config

    return expanded


def _merge_custom_search_mapping(data: dict, value: LLMConfig | SearchConfig | dict | None) -> LLMConfig | SearchConfig | dict | None:
    mapping_keys = {
        "result_collection_path": "result_collection_path",
        "title_fields": "title_fields",
        "snippet_fields": "snippet_fields",
        "url_fields": "url_fields",
    }
    mapping = {target: data.pop(f"search_{source}") for source, target in mapping_keys.items() if f"search_{source}" in data}
    if not mapping:
        return value
    if isinstance(value, SearchConfig):
        base = {
            "engine": value.engine,
            "api_key": value.api_key,
            "base_url": value.base_url,
            "http_proxy": value.http_proxy,
            "headers": dict(value.headers),
            "extra": dict(value.extra),
        }
    elif value is None or isinstance(value, Mapping):
        base = dict(value or {})
    else:
        raise configuration_error(
            category="search_configuration_error",
            component="validation",
            message="search must be a mapping or SearchConfig object.",
            user_message="search must be an object with search provider settings.",
            invalid_fields=["search"],
        )
    extra = dict(base.get("extra") or {})
    extra.update(mapping)
    base["extra"] = extra
    return base


def _merge_custom_llm_mapping(data: dict, value: LLMConfig | SearchConfig | dict | None) -> LLMConfig | SearchConfig | dict | None:
    mapping_keys = {
        "content_path": "content_path",
        "text_path": "text_path",
        "json_payload_path": "json_payload_path",
        "usage_path": "usage_path",
        "usage_input_tokens_path": "usage_input_tokens_path",
        "usage_output_tokens_path": "usage_output_tokens_path",
        "usage_total_tokens_path": "usage_total_tokens_path",
        "usage_reasoning_tokens_path": "usage_reasoning_tokens_path",
        "usage_cached_input_tokens_path": "usage_cached_input_tokens_path",
        "finish_reason_path": "finish_reason_path",
    }
    mapping = {target: data.pop(f"llm_{source}") for source, target in mapping_keys.items() if f"llm_{source}" in data}
    if not mapping:
        return value
    if isinstance(value, LLMConfig):
        base = {
            "engine": value.engine,
            "model": value.model,
            "api_key": value.api_key,
            "base_url": value.base_url,
            "protocol": value.protocol,
            "http_proxy": value.http_proxy,
            "headers": dict(value.headers),
            "request_options": {
                "temperature": value.request_options.temperature,
                "response_format": value.request_options.response_format,
                "seed": value.request_options.seed,
                "stream": value.request_options.stream,
                "reasoning": value.request_options.reasoning,
            },
            "extra": dict(value.extra),
        }
    elif value is None or isinstance(value, Mapping):
        base = dict(value or {})
    else:
        raise configuration_error(
            category="llm_configuration_error",
            component="validation",
            message="llm must be a mapping or LLMConfig object.",
            user_message="llm must be an object with LLM settings.",
            invalid_fields=["llm"],
        )
    extra = dict(base.get("extra") or {})
    response_mapping = dict(extra.get("response_mapping") or {})
    response_mapping.update(mapping)
    extra["response_mapping"] = response_mapping
    base["extra"] = extra
    return base


def _merge_prefixed_config(value: LLMConfig | SearchConfig | dict | None, data: dict, *, prefix: str) -> LLMConfig | SearchConfig | dict | None:
    prefixed = {
        key.removeprefix(prefix): data.pop(key)
        for key in list(data)
        if key.startswith(prefix) and key not in {"search_limits"}
    }
    if not prefixed:
        return value
    if isinstance(value, LLMConfig):
        base = {
            "engine": value.engine,
            "model": value.model,
            "api_key": value.api_key,
            "base_url": value.base_url,
            "protocol": value.protocol,
            "http_proxy": value.http_proxy,
            "headers": dict(value.headers),
            "request_options": {
                "temperature": value.request_options.temperature,
                "response_format": value.request_options.response_format,
                "seed": value.request_options.seed,
                "stream": value.request_options.stream,
                "reasoning": value.request_options.reasoning,
            },
            "extra": dict(value.extra),
        }
    elif isinstance(value, SearchConfig):
        base = {
            "engine": value.engine,
            "api_key": value.api_key,
            "base_url": value.base_url,
            "http_proxy": value.http_proxy,
            "headers": dict(value.headers),
            "extra": dict(value.extra),
        }
    elif value is None or isinstance(value, Mapping):
        base = dict(value or {})
    else:
        category = "llm_configuration_error" if prefix == "llm_" else "search_configuration_error"
        field = prefix.removesuffix("_")
        raise configuration_error(
            category=category,
            component="validation",
            message=f"{field} must be a mapping or config object.",
            user_message=f"{field} must be an object with provider settings.",
            invalid_fields=[field],
        )
    base.update(prefixed)
    return base


def _coerce_llm_config(value: LLMConfig | dict | None) -> LLMConfig | None:
    if value is None:
        return None
    if isinstance(value, LLMConfig):
        return LLMConfig(
            engine=value.engine,
            model=value.model,
            api_key=value.api_key,
            base_url=value.base_url,
            protocol=value.protocol,
            http_proxy=value.http_proxy,
            headers=dict(value.headers),
            request_options=LLMRequestOptions(
                temperature=value.request_options.temperature,
                response_format=value.request_options.response_format,
                seed=value.request_options.seed,
                stream=value.request_options.stream,
                reasoning=value.request_options.reasoning,
            ),
            extra=dict(value.extra),
        )
    if not isinstance(value, Mapping):
        raise configuration_error(
            category="llm_configuration_error",
            component="validation",
            message="llm must be a mapping or LLMConfig object.",
            user_message="llm must be an object with LLM settings.",
            invalid_fields=["llm"],
        )
    return LLMConfig(
        engine=value.get("engine"),
        model=value.get("model"),
        api_key=value.get("api_key"),
        base_url=value.get("base_url"),
        protocol=value.get("protocol"),
        http_proxy=value.get("http_proxy"),
        headers=dict(value.get("headers") or {}),
        request_options=_coerce_llm_request_options(value.get("request_options")),
        extra=dict(value.get("extra") or {}),
    )


def _coerce_search_config(value: SearchConfig | dict | None) -> SearchConfig | None:
    if value is None:
        return None
    if isinstance(value, SearchConfig):
        return SearchConfig(
            engine=value.engine,
            api_key=value.api_key,
            base_url=value.base_url,
            http_proxy=value.http_proxy,
            headers=dict(value.headers),
            extra=dict(value.extra),
        )
    if not isinstance(value, Mapping):
        raise configuration_error(
            category="search_configuration_error",
            component="validation",
            message="search must be a mapping or SearchConfig object.",
            user_message="search must be an object with search provider settings.",
            invalid_fields=["search"],
        )
    return SearchConfig(
        engine=value.get("engine"),
        api_key=value.get("api_key"),
        base_url=value.get("base_url"),
        http_proxy=value.get("http_proxy"),
        headers=dict(value.get("headers") or {}),
        extra=dict(value.get("extra") or {}),
    )


def _coerce_llm_request_options(value: LLMRequestOptions | dict | None) -> LLMRequestOptions:
    if isinstance(value, LLMRequestOptions):
        return LLMRequestOptions(
            temperature=_coerce_optional_bool(value.temperature),
            response_format=_coerce_optional_bool(value.response_format),
            seed=_coerce_optional_bool(value.seed),
            stream=_coerce_optional_bool(value.stream),
            reasoning=_coerce_optional_bool(value.reasoning),
        )
    mapping = dict(value or {}) if isinstance(value, dict) else {}
    return LLMRequestOptions(
        temperature=_coerce_optional_bool(mapping.get("temperature")),
        response_format=_coerce_optional_bool(mapping.get("response_format")),
        seed=_coerce_optional_bool(mapping.get("seed")),
        stream=_coerce_optional_bool(mapping.get("stream")),
        reasoning=_coerce_optional_bool(mapping.get("reasoning")),
    )


def _normalize_previous_conversation(previous_conversation: object) -> list[str]:
    if isinstance(previous_conversation, (str, bytes)) or not isinstance(previous_conversation, list):
        return []
    values = [str(item).strip() for item in previous_conversation if str(item).strip()]
    return values[-2:]


def _validate_request(request: SearchRequest) -> None:
    if not request.question:
        raise configuration_error(
            category="llm_configuration_error",
            component="validation",
            message="Question is required.",
            user_message="question is required.",
            invalid_fields=["question"],
        )

    if request.output_mode not in ALLOWED_OUTPUT_MODES:
        raise configuration_error(
            category="llm_configuration_error",
            component="validation",
            message="Invalid output_mode.",
            user_message="output_mode must be one of answer, answer_with_sources, results_simple, results_detailed.",
            invalid_fields=["output_mode"],
        )

    _validate_limits(request.search_limits)
    _validate_proxy(request.proxy.http_proxy, field_name="proxy.http_proxy")

    if request.llm is not None:
        _validate_proxy(request.llm.http_proxy, field_name="llm.http_proxy")
        _validate_base_url(request.llm.base_url, field_name="llm.base_url", category="llm_configuration_error")
    if request.search is not None:
        _validate_proxy(request.search.http_proxy, field_name="search.http_proxy")
        _validate_base_url(request.search.base_url, field_name="search.base_url", category="search_configuration_error")

    if request.demo_mode:
        return

    if request.llm is None or not request.llm.engine:
        raise configuration_error(
            category="llm_configuration_error",
            component="validation",
            message="llm.engine is required outside demo mode.",
            user_message="llm.engine is required when demo_mode is false.",
            invalid_fields=["llm.engine"],
        )
    if request.search is None or not request.search.engine:
        raise configuration_error(
            category="search_configuration_error",
            component="validation",
            message="search.engine is required outside demo mode.",
            user_message="search.engine is required when demo_mode is false.",
            invalid_fields=["search.engine"],
        )


def _validate_limits(search_limits: SearchLimits) -> None:
    invalid_fields: list[str] = []
    if not _is_positive_int(search_limits.zoomout_num_results):
        invalid_fields.append("search_limits.zoomout_num_results")
    if not _is_positive_int(search_limits.zoomin_num_results):
        invalid_fields.append("search_limits.zoomin_num_results")
    if not _is_positive_int(search_limits.top_k_domains_per_query):
        invalid_fields.append("search_limits.top_k_domains_per_query")
    if invalid_fields:
        raise configuration_error(
            category="search_configuration_error",
            component="validation",
            message="Search limit values must be positive integers.",
            user_message="search_limits values must be greater than or equal to 1.",
            invalid_fields=invalid_fields,
        )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _validate_proxy(proxy_value: str | None, *, field_name: str) -> None:
    if not proxy_value:
        return
    parsed = urlparse(proxy_value)
    if parsed.scheme not in {"http", "https", "socks5"} or not parsed.netloc:
        raise configuration_error(
            category="proxy_configuration_error",
            component="proxy",
            message="Invalid proxy configuration.",
            user_message=f"{field_name} must be a valid http, https, or socks5 URL.",
            invalid_fields=[field_name],
        )


def _validate_base_url(base_url: str | None, *, field_name: str, category: str) -> None:
    if not base_url:
        return
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise configuration_error(
            category=category,
            component="validation",
            message="Invalid provider base_url configuration.",
            user_message=f"{field_name} must be a valid http or https URL.",
            invalid_fields=[field_name],
        )


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None
