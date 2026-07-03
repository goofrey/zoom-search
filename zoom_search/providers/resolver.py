"""Provider resolver and descriptor normalization."""

from __future__ import annotations

from zoom_search.errors import configuration_error
from zoom_search.models import LLMConfig
from zoom_search.models import ProviderCapability
from zoom_search.models import ResolvedProvider
from zoom_search.models import ResolvedProviders
from zoom_search.models import SearchConfig
from zoom_search.models import SearchRequest
from zoom_search.providers.capabilities import BUILTIN_LLM_CAPABILITIES
from zoom_search.providers.capabilities import BUILTIN_SEARCH_CAPABILITIES
from zoom_search.providers.capabilities import DEMO_LLM_CAPABILITY
from zoom_search.providers.capabilities import DEMO_SEARCH_CAPABILITY


def resolve_providers(request: SearchRequest) -> ResolvedProviders:
    return ResolvedProviders(
        llm=resolve_llm_provider(request),
        search=resolve_search_provider(request),
    )


def resolve_llm_provider(request: SearchRequest) -> ResolvedProvider:
    if request.demo_mode:
        return ResolvedProvider(
            engine=DEMO_LLM_CAPABILITY.engine,
            provider_kind="demo",
            component="llm",
            capability=DEMO_LLM_CAPABILITY,
            protocol=DEMO_LLM_CAPABILITY.protocol,
        )

    assert request.llm is not None
    config = request.llm
    if config.engine in BUILTIN_LLM_CAPABILITIES:
        _validate_builtin_llm(config)
        capability = BUILTIN_LLM_CAPABILITIES[config.engine]
        return _resolved_provider_from_llm(config=config, capability=capability, provider_kind="builtin")

    if config.engine in {"custom", "openai-compatible"}:
        _validate_custom_llm(config)
        is_openai_compatible = config.engine == "openai-compatible" or config.protocol == "openai_compatible"
        supports_streaming = is_openai_compatible and _coerce_bool(config.extra.get("supports_streaming"))
        capability = ProviderCapability(
            engine="openai-compatible" if is_openai_compatible else "custom",
            provider_kind="custom",
            protocol="openai_compatible" if is_openai_compatible else "native",
            adapter_class="openai_compatible" if is_openai_compatible else "custom_native",
            structured_output_level="native_json_object_only",
            supports_structured_output=is_openai_compatible,
            supports_json_mode=is_openai_compatible,
            supports_streaming=supports_streaming,
            supports_seed="supported",
            streaming_level="sse_full_fidelity" if supports_streaming else "unsupported",
            execution_model="sync_with_stream" if supports_streaming else "sync_request_response",
        )
        return _resolved_provider_from_llm(config=config, capability=capability, provider_kind="custom")

    raise configuration_error(
        category="llm_configuration_error",
        component="llm",
        message=f"Unsupported llm.engine: {config.engine}",
        user_message=f"llm.engine must be one of {_format_supported_engines(BUILTIN_LLM_CAPABILITIES)}, custom, or openai-compatible.",
        invalid_fields=["llm.engine"],
        provider_engine=config.engine,
    )


def resolve_search_provider(request: SearchRequest) -> ResolvedProvider:
    if request.demo_mode:
        return ResolvedProvider(
            engine=DEMO_SEARCH_CAPABILITY.engine,
            provider_kind="demo",
            component="search",
            capability=DEMO_SEARCH_CAPABILITY,
        )

    assert request.search is not None
    config = request.search
    if config.engine in BUILTIN_SEARCH_CAPABILITIES:
        _validate_builtin_search(config)
        capability = BUILTIN_SEARCH_CAPABILITIES[config.engine]
        return _resolved_provider_from_search(config=config, capability=capability, provider_kind="builtin")

    if config.engine == "custom":
        _validate_custom_search(config)
        capability = ProviderCapability(
            engine="custom",
            provider_kind="custom",
            supports_site_restriction=False,
            site_restriction_mode="query_side",
            supports_query_site_operator="unknown",
            recommended_zoom_in_strategy="query_side",
            query_param_path="query",
            supports_provider_side_num_results=False,
            num_results_mode="unsupported",
            result_collection_path=_coerce_optional_text(config.extra.get("result_collection_path")),
            field_candidates={
                "title": _coerce_field_candidates(config.extra.get("title_fields"), ["title"]),
                "snippet": _coerce_field_candidates(
                    config.extra.get("snippet_fields"),
                    ["snippet", "summary", "summary_ai", "description", "content", "highlights", "text"],
                ),
                "url": _coerce_field_candidates(config.extra.get("url_fields"), ["url", "link"]),
            },
        )
        return _resolved_provider_from_search(config=config, capability=capability, provider_kind="custom")

    raise configuration_error(
        category="search_configuration_error",
        component="search",
        message=f"Unsupported search.engine: {config.engine}",
        user_message="search.engine must be one of tavily, serper, brave, you, 360search, firecrawl, baidu, linkup, perplexity, glm, volcengine, exa, bocha, querit, serpapi, searxng, metasota, tiangong, or custom.",
        invalid_fields=["search.engine"],
        provider_engine=config.engine,
    )


def _resolved_provider_from_llm(
    *,
    config: LLMConfig,
    capability: ProviderCapability,
    provider_kind: str,
) -> ResolvedProvider:
    return ResolvedProvider(
        engine=config.engine or capability.engine,
        provider_kind=provider_kind,
        component="llm",
        capability=capability,
        model=config.model,
        protocol=config.protocol or capability.protocol,
        base_url=config.base_url or capability.default_base_url,
        api_key=config.api_key,
        headers=dict(config.headers),
        http_proxy=config.http_proxy,
        request_options=config.request_options,
        extra=dict(config.extra),
    )


def _resolved_provider_from_search(
    *,
    config: SearchConfig,
    capability: ProviderCapability,
    provider_kind: str,
) -> ResolvedProvider:
    return ResolvedProvider(
        engine=config.engine or capability.engine,
        provider_kind=provider_kind,
        component="search",
        capability=capability,
        base_url=config.base_url or capability.default_base_url,
        api_key=config.api_key,
        headers=dict(config.headers),
        http_proxy=config.http_proxy,
        extra=dict(config.extra),
    )


def _validate_builtin_llm(config: LLMConfig) -> None:
    invalid_fields: list[str] = []
    if not config.model:
        invalid_fields.append("llm.model")
    if config.engine != "ollama" and not config.api_key:
        invalid_fields.append("llm.api_key")
    if config.engine == "huggingface" and config.model and "/" not in config.model:
        invalid_fields.append("llm.model")
    if config.engine == "doubao-global" and not config.base_url:
        invalid_fields.append("llm.base_url")
    if config.engine == "lepton" and not config.base_url:
        invalid_fields.append("llm.base_url")
    if invalid_fields:
        user_message = "Built-in llm providers require llm.model and llm.api_key."
        if config.engine == "huggingface" and "llm.model" in invalid_fields:
            user_message = "HuggingFace requires llm.model in org/model format and llm.api_key."
        if config.engine == "doubao-global" and "llm.base_url" in invalid_fields:
            user_message = "doubao-global requires an explicit llm.base_url for the target region, plus llm.model and llm.api_key."
        if config.engine == "lepton" and "llm.base_url" in invalid_fields:
            user_message = "lepton requires an explicit llm.base_url for the target model endpoint, plus llm.model and llm.api_key."
        raise configuration_error(
            category="llm_configuration_error",
            component="llm",
            message="Built-in llm provider config is incomplete.",
            user_message=user_message,
            invalid_fields=invalid_fields,
            provider_engine=config.engine,
        )


def _format_supported_engines(capabilities: dict[str, ProviderCapability]) -> str:
    return ", ".join(sorted(capabilities))


def _validate_custom_llm(config: LLMConfig) -> None:
    invalid_fields: list[str] = []
    if config.protocol is not None and config.protocol not in {"openai_compatible", "native"}:
        invalid_fields.append("llm.protocol")
    if not config.base_url:
        invalid_fields.append("llm.base_url")
    if invalid_fields:
        raise configuration_error(
            category="llm_configuration_error",
            component="llm",
            message="Custom llm provider config is incomplete.",
            user_message="Custom llm providers require llm.base_url. Use llm.engine=custom for native providers or llm.engine=openai-compatible for OpenAI-compatible endpoints.",
            invalid_fields=invalid_fields,
            provider_engine=config.engine,
        )


def _validate_custom_search(config: SearchConfig) -> None:
    if not config.base_url:
        raise configuration_error(
            category="search_configuration_error",
            component="search",
            message="Custom search provider config is incomplete.",
            user_message="Custom search providers require search.base_url.",
            invalid_fields=["search.base_url"],
            provider_engine=config.engine,
        )


def _coerce_optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _coerce_field_candidates(value: object, default: list[str]) -> list[str]:
    if isinstance(value, str):
        candidates = [item.strip() for item in value.split(",") if item.strip()]
        return candidates or list(default)
    if isinstance(value, (list, tuple)):
        candidates = [str(item).strip() for item in value if str(item).strip()]
        return candidates or list(default)
    return list(default)


def _validate_builtin_search(config: SearchConfig) -> None:
    if config.engine == "volcengine":
        has_api_key = bool(config.api_key)
        has_aksk = bool(config.extra.get("access_key")) and bool(config.extra.get("secret_key"))
        if has_api_key or has_aksk:
            return
        invalid_fields = ["search.api_key", "search.extra.access_key", "search.extra.secret_key"]
    elif config.engine == "searxng":
        if config.base_url:
            return
        invalid_fields = ["search.base_url"]
    elif config.engine == "tiangong":
        if config.api_key and config.extra.get("app_secret"):
            return
        invalid_fields = ["search.api_key", "search.extra.app_secret"]
    elif config.api_key:
        return
    else:
        invalid_fields = ["search.api_key"]
    raise configuration_error(
        category="search_configuration_error",
        component="search",
        message="Built-in search provider config is incomplete.",
        user_message="Built-in search providers require the provider-specific credentials. Most engines require search.api_key. Volcengine also supports search.extra.access_key plus search.extra.secret_key. Searxng requires search.base_url. Tiangong requires search.api_key plus search.extra.app_secret.",
        invalid_fields=invalid_fields,
        provider_engine=config.engine,
    )
