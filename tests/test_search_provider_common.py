from __future__ import annotations

import hashlib
import hmac
import json

from zoom_search.errors import call_failure
from zoom_search.models import ProviderCapability
from zoom_search.models import UnifiedSearchRequest
from zoom_search.providers.capabilities import BUILTIN_SEARCH_CAPABILITIES
from zoom_search.providers.resolver import resolve_search_provider
from zoom_search.models import SearchConfig
from zoom_search.models import SearchRequest
from zoom_search.providers.search import BaseSearchAdapter
from zoom_search.providers.search import SearchCapabilityRegistry
from zoom_search.providers.search import build_site_query
from zoom_search.providers.search import clean_html
from zoom_search.providers.search import clean_site_restriction_domain
from zoom_search.providers.search import encode_site_restriction_value
from zoom_search.providers.search import create_builtin_search_adapter
from zoom_search.providers.search import extract_provider_error
from zoom_search.providers.search import normalize_search_exception
from zoom_search.providers.search import normalize_search_error
from zoom_search.providers.search import read_result_collection
from zoom_search.providers.search import inspect_result_collection
from zoom_search.providers.search import read_field_path
from zoom_search.providers.search import write_field_path
from zoom_search.search.domain import normalize_url


class _FakeTiangongProvider:
    def __init__(self) -> None:
        self.capability = BUILTIN_SEARCH_CAPABILITIES["tiangong"]
        self.base_url = None
        self.api_key = "app-key"
        self.extra = {"app_secret": "app-secret"}


class PostAdapter(BaseSearchAdapter):
    def build_base_request(self, request: UnifiedSearchRequest) -> dict:
        return {"fixed": True}


class GetAdapter(BaseSearchAdapter):
    method = "GET"


def test_capability_registry_reads_engine_capabilities() -> None:
    registry = SearchCapabilityRegistry(BUILTIN_SEARCH_CAPABILITIES)

    assert registry.supports_provider_side_site_restriction("tavily") is True
    assert registry.recommended_zoom_in_strategy("searxng") == "best_effort"
    assert registry.get("brave").num_results_param == "count"
    assert registry.get("you").num_results_mode == "per_collection_limit"
    assert registry.get("360search").query_param_path == "q"
    assert registry.get("serpapi").num_results_mode == "unsupported"


def test_normalize_url_rejects_invalid_port_without_raising() -> None:
    assert normalize_url("https://example.com:bad/path") == ""


def test_normalize_url_rejects_non_web_scheme() -> None:
    assert normalize_url("ftp://example.com/file") == ""
    assert normalize_url("mailto:user@example.com") == ""


def test_zoomout_and_zoom_in_use_same_unified_interface() -> None:
    capability = BUILTIN_SEARCH_CAPABILITIES["tavily"]
    adapter = PostAdapter(capability=capability, endpoint="https://example.com/search")

    zoomout = adapter.build_request(UnifiedSearchRequest(task="zoomout_search", query="hotels", num_results=5))
    zoom_in = adapter.build_request(
        UnifiedSearchRequest(
            task="zoomin_search",
            query="What is the hotel room setup?",
            num_results=3,
            site_restriction_mode="provider_side",
            site_restriction_domain="https://www.trip.com/path",
        )
    )

    assert zoomout.json_body == {"fixed": True, "query": "hotels", "max_results": 5}
    assert zoom_in.json_body == {
        "fixed": True,
        "query": "What is the hotel room setup?",
        "include_domains": ["www.trip.com"],
        "max_results": 3,
    }


def test_get_and_post_request_construction() -> None:
    brave = GetAdapter(capability=BUILTIN_SEARCH_CAPABILITIES["brave"], endpoint="https://brave.example")
    baidu = PostAdapter(capability=BUILTIN_SEARCH_CAPABILITIES["baidu"], endpoint="https://baidu.example")

    get_request = brave.build_request(UnifiedSearchRequest(task="zoomout_search", query="shenzhen hotel", num_results=4))
    post_request = baidu.build_request(
        UnifiedSearchRequest(
            task="zoomin_search",
            query="trip room",
            num_results=2,
            site_restriction_mode="provider_side",
            site_restriction_domain="trip.com",
        )
    )

    assert get_request.method == "GET"
    assert get_request.params == {"q": "shenzhen hotel", "count": 4}
    assert post_request.method == "POST"
    assert post_request.json_body == {
        "fixed": True,
        "messages": [{"content": "trip room"}],
        "resource_type_filter": [{"top_k": 2}],
        "search_filter": {"match": {"site": ["trip.com"]}},
    }


def test_group_a_provider_capabilities_are_declared() -> None:
    expected = {
        "tavily",
        "linkup",
        "perplexity",
        "glm",
        "baidu",
        "volcengine",
        "exa",
        "firecrawl",
        "bocha",
        "querit",
        "serpapi",
    }

    assert expected.issubset(BUILTIN_SEARCH_CAPABILITIES.keys())


def test_group_b_provider_capabilities_are_declared() -> None:
    expected = {"serper", "brave", "you", "360search"}

    assert expected.issubset(BUILTIN_SEARCH_CAPABILITIES.keys())
    assert BUILTIN_SEARCH_CAPABILITIES["serper"].recommended_zoom_in_strategy == "query_side"
    assert BUILTIN_SEARCH_CAPABILITIES["brave"].supports_query_site_operator == "true"
    assert BUILTIN_SEARCH_CAPABILITIES["you"].num_results_param == "count"
    assert BUILTIN_SEARCH_CAPABILITIES["you"].default_base_url == "https://ydc-index.io/v1/search"
    assert BUILTIN_SEARCH_CAPABILITIES["360search"].field_candidates["snippet"][0] == "summary_ai"


def test_site_restriction_domain_uses_host_value() -> None:
    assert clean_site_restriction_domain("https://monkeycode-ai.com/console") == "monkeycode-ai.com"
    assert build_site_query(domain="https://www.trip.com/path", split_question="hotel room") == "site:www.trip.com hotel room"


def test_high_uncertainty_provider_capabilities_are_declared() -> None:
    expected = {"searxng", "metasota", "tiangong"}

    assert expected.issubset(BUILTIN_SEARCH_CAPABILITIES.keys())
    assert BUILTIN_SEARCH_CAPABILITIES["searxng"].supports_query_site_operator == "unknown"
    assert BUILTIN_SEARCH_CAPABILITIES["metasota"].recommended_zoom_in_strategy == "best_effort"
    assert BUILTIN_SEARCH_CAPABILITIES["tiangong"].supports_provider_side_num_results is False


def test_priority_provider_site_restriction_and_num_results_mapping() -> None:
    expectations = {
        "tavily": ("include_domains", ["www.trip.com"], "max_results", 20),
        "linkup": ("includeDomains", ["www.trip.com"], "maxResults", 7),
        "perplexity": ("search_domain_filter", ["www.trip.com"], "max_results", 20),
        "glm": ("search_domain_filter", "www.trip.com", "count", 50),
    }

    for engine, (site_path, site_value, num_path, num_value) in expectations.items():
        adapter = create_builtin_search_adapter(
            type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES[engine], "base_url": None, "api_key": "key", "extra": {}})()
        )
        built = adapter.build_request(
            UnifiedSearchRequest(
                task="zoomin_search",
                query="hotel room",
                num_results=99 if engine in {"tavily", "perplexity", "glm"} else 7,
                site_restriction_mode="provider_side",
                site_restriction_domain="https://www.trip.com/path",
            )
        )

        payload = built.params if built.method == "GET" else built.json_body
        assert read_field_path(payload, site_path) == site_value
        assert read_field_path(payload, num_path) == num_value


def test_glm_query_is_truncated_to_provider_limit() -> None:
    glm = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["glm"], "base_url": None, "api_key": "glm-key", "extra": {}})()
    )

    built = glm.build_request(
        UnifiedSearchRequest(task="zoomout_search", query="x" * 80, num_results=2)
    )

    assert built.json_body["search_query"] == "x" * 70
    assert built.headers["Authorization"] == "Bearer glm-key"


def test_special_group_a_provider_patches() -> None:
    linkup = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["linkup"], "base_url": None, "api_key": "linkup-key", "extra": {}})()
    )
    tavily = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["tavily"], "base_url": None, "api_key": "tvly-key", "extra": {}})()
    )
    baidu = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["baidu"], "base_url": None, "api_key": "token", "extra": {}})()
    )
    volcengine = create_builtin_search_adapter(
        type(
            "Provider",
            (),
            {
                "capability": BUILTIN_SEARCH_CAPABILITIES["volcengine"],
                "base_url": None,
                "api_key": None,
                "extra": {"access_key": "ak", "secret_key": "sk"},
            },
        )()
    )
    serpapi = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["serpapi"], "base_url": None, "api_key": "secret", "extra": {}})()
    )
    firecrawl = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["firecrawl"], "base_url": None, "api_key": "k", "extra": {}})()
    )

    linkup_request = linkup.build_request(
        UnifiedSearchRequest(task="zoomin_search", query="trip room", num_results=7, site_restriction_mode="provider_side", site_restriction_domain="trip.com")
    )
    tavily_request = tavily.build_request(
        UnifiedSearchRequest(task="zoomin_search", query="trip room", num_results=99, site_restriction_mode="provider_side", site_restriction_domain="trip.com")
    )
    baidu_request = baidu.build_request(
        UnifiedSearchRequest(task="zoomin_search", query="x" * 80, num_results=3, site_restriction_mode="provider_side", site_restriction_domain="trip.com")
    )
    volcengine_request = volcengine.build_request(
        UnifiedSearchRequest(task="zoomin_search", query="trip room", num_results=4, site_restriction_mode="provider_side", site_restriction_domain="trip.com")
    )
    serpapi_request = serpapi.build_request(
        UnifiedSearchRequest(task="zoomin_search", query="trip room", num_results=2, site_restriction_mode="provider_side", site_restriction_domain="trip.com")
    )
    firecrawl_request = firecrawl.build_request(
        UnifiedSearchRequest(task="zoomin_search", query="trip room", num_results=2, site_restriction_mode="provider_side", site_restriction_domain="https://trip.com/a")
    )

    assert linkup_request.json_body["outputType"] == "searchResults"
    assert linkup_request.json_body["depth"] == "standard"
    assert linkup_request.json_body["includeDomains"] == ["trip.com"]
    assert linkup_request.json_body["maxResults"] == 7
    assert linkup_request.headers["Authorization"] == "Bearer linkup-key"
    assert linkup_request.headers["Content-Type"] == "application/json"
    assert tavily_request.json_body["include_domains"] == ["trip.com"]
    assert tavily_request.json_body["max_results"] == 20
    assert tavily_request.headers["Authorization"] == "Bearer tvly-key"
    assert tavily_request.headers["Content-Type"] == "application/json"
    assert len(baidu_request.json_body["messages"][0]["content"]) == 72
    assert baidu_request.json_body["messages"][0]["role"] == "user"
    assert baidu_request.headers["Authorization"] == "Bearer token"
    assert baidu_request.headers["X-Appbuilder-Authorization"] == "Bearer token"
    assert volcengine_request.endpoint.startswith("https://mercury.volcengineapi.com")
    assert volcengine_request.json_body["Filter"]["Sites"] == "trip.com"
    assert volcengine_request.json_body["SearchType"] == "web"
    assert "X-Signature" in volcengine_request.headers
    assert serpapi_request.method == "GET"
    assert serpapi_request.params["api_key"] == "secret"
    assert serpapi_request.params["as_sitesearch"] == "trip.com"
    assert serpapi_request.params["as_dt"] == "i"
    assert serpapi_request.params["engine"] == "google"
    assert firecrawl_request.json_body["sources"] == ["web"]
    assert firecrawl_request.json_body["includeDomains"] == ["trip.com"]


def test_group_a_search_auth_headers_are_sent() -> None:
    exa = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["exa"], "base_url": None, "api_key": "exa-key", "extra": {}})()
    )
    perplexity = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["perplexity"], "base_url": None, "api_key": "pplx-key", "extra": {}})()
    )
    glm = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["glm"], "base_url": None, "api_key": "glm-key", "extra": {}})()
    )
    bocha = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["bocha"], "base_url": None, "api_key": "bocha-key", "extra": {}})()
    )
    exa_request = exa.build_request(
        UnifiedSearchRequest(task="zoomout_search", query="trip room", num_results=2)
    )
    perplexity_request = perplexity.build_request(
        UnifiedSearchRequest(task="zoomout_search", query="trip room", num_results=2)
    )
    glm_request = glm.build_request(
        UnifiedSearchRequest(task="zoomout_search", query="trip room", num_results=2)
    )
    bocha_request = bocha.build_request(
        UnifiedSearchRequest(task="zoomout_search", query="trip room", num_results=2)
    )

    assert exa_request.headers["x-api-key"] == "exa-key"
    assert exa_request.headers["Content-Type"] == "application/json"
    assert perplexity_request.headers["Authorization"] == "Bearer pplx-key"
    assert perplexity_request.headers["Content-Type"] == "application/json"
    assert glm_request.headers["Authorization"] == "Bearer glm-key"
    assert glm_request.headers["Content-Type"] == "application/json"
    assert bocha_request.headers["Authorization"] == "Bearer bocha-key"
    assert bocha_request.headers["Content-Type"] == "application/json"


def test_search_api_key_backed_engines_send_provider_auth_headers() -> None:
    serper = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["serper"], "base_url": None, "api_key": "serper-key", "extra": {}})()
    )
    brave = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["brave"], "base_url": None, "api_key": "brave-key", "extra": {}})()
    )
    you = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["you"], "base_url": None, "api_key": "you-key", "extra": {}})()
    )
    search360 = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["360search"], "base_url": None, "api_key": "360-key", "extra": {}})()
    )
    firecrawl = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["firecrawl"], "base_url": None, "api_key": "firecrawl-key", "extra": {}})()
    )

    serper_request = serper.build_request(UnifiedSearchRequest(task="zoomout_search", query="hotel room", num_results=3))
    brave_request = brave.build_request(UnifiedSearchRequest(task="zoomout_search", query="hotel room", num_results=3))
    you_request = you.build_request(UnifiedSearchRequest(task="zoomout_search", query="hotel room", num_results=3))
    search360_request = search360.build_request(UnifiedSearchRequest(task="zoomout_search", query="hotel room", num_results=3))
    firecrawl_request = firecrawl.build_request(UnifiedSearchRequest(task="zoomout_search", query="hotel room", num_results=3))

    assert serper_request.headers["X-API-KEY"] == "serper-key"
    assert serper_request.headers["Content-Type"] == "application/json"
    assert brave_request.headers["X-Subscription-Token"] == "brave-key"
    assert brave_request.headers["Accept"] == "application/json"
    assert you_request.headers["X-API-Key"] == "you-key"
    assert you_request.headers["Accept"] == "application/json"
    assert search360_request.headers["Authorization"] == "Bearer 360-key"
    assert search360_request.headers["Content-Type"] == "application/json"
    assert firecrawl_request.headers["Authorization"] == "Bearer firecrawl-key"
    assert firecrawl_request.headers["Content-Type"] == "application/json"


def test_group_a_result_normalization_and_debug_output() -> None:
    exa = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["exa"], "base_url": None, "api_key": "k", "extra": {}})()
    )
    baidu = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["baidu"], "base_url": None, "api_key": "k", "extra": {}})()
    )
    firecrawl = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["firecrawl"], "base_url": None, "api_key": "k", "extra": {}})()
    )

    exa_response = exa.normalize_response(
        payload={"results": [{"title": "A", "url": "https://exa.ai/a", "highlights": ["first", "second"]}]},
        request=UnifiedSearchRequest(task="zoomout_search", query="q", num_results=3),
    )
    baidu_response = baidu.normalize_response(
        payload={"references": [{"title": "B", "url": "https://trip.com/b", "content": "snippet text"}]},
        request=UnifiedSearchRequest(task="zoomout_search", query="q", num_results=3),
    )
    firecrawl_response = firecrawl.normalize_response(
        payload={"data": {"web": [{"title": "C", "url": "https://firecrawl.dev/c", "description": "web desc"}], "news": [{"title": "ignore", "url": "https://n.com", "snippet": "n"}]}},
        request=UnifiedSearchRequest(task="zoomin_search", query="q", num_results=3, site_restriction_mode="provider_side", site_restriction_domain="firecrawl.dev"),
    )

    assert exa_response.results[0].snippet == "first second"
    assert baidu_response.results[0].snippet == "snippet text"
    assert firecrawl_response.results[0].snippet == "web desc"
    assert firecrawl_response.debug_events[0]["result_collection_path"] == "data.web"


def test_provider_specific_error_normalization() -> None:
    bocha = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["bocha"], "base_url": None, "api_key": "k", "extra": {}})()
    )
    querit = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["querit"], "base_url": None, "api_key": "k", "extra": {}})()
    )

    bocha_error = normalize_search_error(error=RuntimeError("You do not have enough money"), provider="bocha", request_id="req-1")
    querit_error = normalize_search_error(error=RuntimeError("rate limited"), provider="querit", request_id="req-2")

    assert bocha_error.category == "search_call_failure"
    assert bocha_error.details.reason_code == "provider_call_failed"
    assert querit_error.category == "search_call_failure"


def test_querit_real_payload_shape_and_success_code_are_supported() -> None:
    adapter = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["querit"], "base_url": None, "api_key": "k", "extra": {}})()
    )
    payload = {
        "took": "5ms",
        "error_code": 200,
        "error_msg": "",
        "results": {
            "result": [
                {
                    "title": "Querit.ai | Search Smarter, Build Faster",
                    "snippet": "Pricing and API details",
                    "url": "https://www.querit.ai/",
                }
            ]
        },
    }

    assert extract_provider_error(payload=payload, provider="querit", request_id="req-ok") is None
    response = adapter.normalize_response(
        payload=payload,
        request=UnifiedSearchRequest(
            task="zoomin_search",
            query="search api pricing",
            num_results=5,
            site_restriction_mode="provider_side",
            site_restriction_domain="querit.ai",
        ),
    )

    assert [item.url for item in response.results] == ["https://www.querit.ai/"]
    assert response.debug_events[0]["result_collection_path"] == "results.result"


def test_volcengine_real_payload_shape_is_supported() -> None:
    adapter = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["volcengine"], "base_url": None, "api_key": "k", "extra": {}})()
    )
    payload = {
        "ResponseMetadata": {"RequestId": "req-1"},
        "Result": {
            "ResultCount": 2,
            "WebResults": [
                {
                    "SortId": 2,
                    "Title": "Lower score",
                    "Url": "https://example.com/lower",
                    "Snippet": "lower snippet",
                    "RankScore": 0.4,
                },
                {
                    "SortId": 1,
                    "Title": "Higher score",
                    "Url": "https://example.com/higher",
                    "Snippet": "higher snippet",
                    "RankScore": 0.9,
                },
            ],
        },
    }

    response = adapter.normalize_response(
        payload=payload,
        request=UnifiedSearchRequest(task="zoomout_search", query="q", num_results=2),
    )

    assert [item.url for item in response.results] == ["https://example.com/higher", "https://example.com/lower"]
    assert response.results[0].title == "Higher score"
    assert response.results[0].snippet == "higher snippet"
    assert response.results[0].raw_item["RankScore"] == 0.9
    assert response.debug_events[0]["result_collection_path"] == "Result.WebResults"


def test_volcengine_aksk_signature_uses_final_payload() -> None:
    adapter = create_builtin_search_adapter(
        type(
            "Provider",
            (),
            {
                "capability": BUILTIN_SEARCH_CAPABILITIES["volcengine"],
                "base_url": None,
                "api_key": None,
                "extra": {"access_key": "ak", "secret_key": "sk"},
            },
        )()
    )

    built = adapter.build_request(
        UnifiedSearchRequest(
            task="zoomin_search",
            query="hotel room",
            num_results=4,
            site_restriction_mode="provider_side",
            site_restriction_domain="trip.com",
        )
    )
    signing_payload = json.dumps(built.json_body, ensure_ascii=False, separators=(",", ":"))
    expected = hmac.new(b"sk", signing_payload.encode("utf-8"), hashlib.sha256).hexdigest()

    assert built.headers["X-Signature"] == expected


def test_result_collection_path_reading() -> None:
    payload = {"data": {"web": [{"title": "A"}]}, "results": {"web": [{"title": "B"}]}}

    assert read_result_collection(payload, "data.web") == [{"title": "A"}]
    assert read_result_collection(payload, "results.web") == [{"title": "B"}]
    assert read_result_collection(payload, "missing.path") == []
    assert inspect_result_collection(payload, "results") == "non_list"
    assert inspect_result_collection(payload, "results.web") == "list"


def test_non_list_result_collection_path_emits_warning() -> None:
    capability = ProviderCapability(
        engine="test",
        provider_kind="builtin",
        result_collection_path="results",
        field_candidates={"title": ["title"], "snippet": ["snippet"], "url": ["url"]},
    )
    adapter = PostAdapter(capability=capability)

    response = adapter.normalize_response(
        payload={"results": {"items": [{"title": "A", "snippet": "B", "url": "https://example.com"}]}},
        request=UnifiedSearchRequest(task="zoomout_search", query="q", num_results=1),
    )

    assert response.results == []
    assert any(warning.code == "result_collection_path_not_list" for warning in response.normalized_warnings)


def test_custom_search_capability_uses_configured_result_mapping() -> None:
    provider = resolve_search_provider(
        SearchRequest(
            question="q",
            search=SearchConfig(
                engine="custom",
                base_url="https://search.example.com/api",
                extra={
                    "result_collection_path": "data.items",
                    "title_fields": ["name", "title"],
                    "snippet_fields": "summary,snippet",
                    "url_fields": ["link", "url"],
                },
            ),
        )
    )

    assert provider.capability.result_collection_path == "data.items"
    assert provider.capability.field_candidates["title"] == ["name", "title"]
    assert provider.capability.field_candidates["snippet"] == ["summary", "snippet"]
    assert provider.capability.field_candidates["url"] == ["link", "url"]


def test_title_snippet_url_fallback_and_html_cleanup() -> None:
    capability = ProviderCapability(
        engine="test",
        provider_kind="builtin",
        result_collection_path="results",
        field_candidates={
            "title": ["name", "title"],
            "snippet": ["snippets", "description", "content"],
            "url": ["link", "url"],
        },
    )
    adapter = PostAdapter(capability=capability)
    response = adapter.normalize_response(
        payload={
            "results": [
                {
                    "name": "Example",
                    "snippets": ["<b>Hello</b>", "world"],
                    "link": "https://example.com/a",
                }
            ]
        },
        request=UnifiedSearchRequest(task="zoomout_search", query="q", num_results=1),
    )

    assert response.results[0].title == "Example"
    assert response.results[0].snippet == "Hello world"
    assert response.results[0].url == "https://example.com/a"
    assert clean_html("<p>A<br>B</p>") == " A B "


def test_missing_fields_are_discarded_with_debug_count() -> None:
    capability = BUILTIN_SEARCH_CAPABILITIES["firecrawl"]
    adapter = PostAdapter(capability=capability)
    response = adapter.normalize_response(
        payload={
            "data": {
                "web": [
                    {"title": "ok", "description": "desc", "url": "https://a.com"},
                    {"title": "missing url", "description": "desc"},
                    {"description": "missing title", "url": "https://b.com"},
                ]
            }
        },
        request=UnifiedSearchRequest(task="zoomout_search", query="q", num_results=3),
    )

    assert len(response.results) == 1
    assert response.debug_events[0]["discarded_result_count"] == 2


def test_results_are_ranked_by_position_then_score_then_original_order() -> None:
    capability = ProviderCapability(
        engine="test",
        provider_kind="builtin",
        result_collection_path="results",
        field_candidates={"title": ["title"], "snippet": ["snippet"], "url": ["url"]},
    )
    adapter = PostAdapter(capability=capability)

    response = adapter.normalize_response(
        payload={
            "results": [
                {"title": "score low", "snippet": "a", "url": "https://example.com/a", "score": 0.2},
                {"title": "position two", "snippet": "b", "url": "https://example.com/b", "position": 2},
                {"title": "plain", "snippet": "c", "url": "https://example.com/c"},
                {"title": "score high", "snippet": "d", "url": "https://example.com/d", "score": "0.9"},
                {"title": "rank score", "snippet": "f", "url": "https://example.com/f", "RankScore": 0.8},
                {"title": "position one", "snippet": "e", "url": "https://example.com/e", "position": "1"},
            ]
        },
        request=UnifiedSearchRequest(task="zoomout_search", query="q", num_results=5),
    )

    assert [item.title for item in response.results] == ["position one", "position two", "score high", "rank score", "score low", "plain"]
    assert [item.provider_result_index for item in response.results] == [5, 1, 3, 4, 0, 2]


def test_best_effort_query_side_filters_non_matching_domains_and_emits_warning() -> None:
    adapter = GetAdapter(capability=BUILTIN_SEARCH_CAPABILITIES["searxng"])
    response = adapter.normalize_response(
        payload={
            "results": [
                {"title": "keep", "content": "desc", "url": "https://sub.trip.com/a"},
                {"title": "drop", "content": "desc", "url": "https://booking.com/b"},
            ]
        },
        request=UnifiedSearchRequest(
            task="zoomin_search",
            query="site:trip.com hotel",
            num_results=2,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        ),
    )

    assert [item.url for item in response.results] == ["https://sub.trip.com/a"]
    assert response.normalized_warnings[0].code == "low_confidence_site_restriction"
    assert response.debug_events[0]["discarded_result_count"] == 1
    assert response.debug_events[0]["raw_result_count"] == 2
    assert response.debug_events[0]["normalized_result_count"] == 1


def test_error_normalization_preserves_zoom_search_error() -> None:
    original = call_failure(
        category="search_call_failure",
        component="search",
        message="bad request",
        user_message="bad request",
        request_id="req-1",
        provider_engine="brave",
        reason_code="invalid_query_or_request",
    )

    assert normalize_search_error(error=original, provider="brave", request_id="req-1") is original
    normalized = normalize_search_error(error=ValueError("boom"), provider="brave", request_id="req-2")
    assert normalized.category == "search_call_failure"
    assert normalized.details.error_type == "provider_error"
    assert normalized.details.reason_code == "provider_call_failed"


def test_search_http_status_error_preserves_status_and_provider_fields() -> None:
    import httpx

    request = httpx.Request("GET", "https://search.example")
    response = httpx.Response(
        429,
        json={"error": {"message": "Too many requests", "type": "rate_limit_error", "code": "rate_limit"}},
        request=request,
    )
    error = httpx.HTTPStatusError("too many", request=request, response=response)
    context = type(
        "Context",
        (),
        {
            "request_id": "req-429",
            "search_provider": type("Provider", (), {"engine": "brave"})(),
        },
    )()

    normalized = normalize_search_exception(error=error, context=context)

    assert normalized.category == "search_call_failure"
    assert normalized.details.error_type == "rate_limit_error"
    assert normalized.details.reason_code == "rate_limited"
    assert normalized.details.http_status == 429
    assert normalized.details.provider_error_code == "rate_limit"
    assert normalized.details.provider_error_type == "rate_limit_error"
    assert normalized.retryable is True
    assert normalized.raw_diagnostics is not None
    assert normalized.raw_diagnostics.http_status == 429


def test_search_provider_business_error_preserves_request_id_and_body() -> None:
    payload = {"error_code": "BadRequest", "error_msg": "invalid query", "type": "invalid_request_error"}

    error = extract_provider_error(payload=payload, provider="baidu", request_id="req-business")

    assert error is not None
    assert error.request_id == "req-business"
    assert error.category == "search_call_failure"
    assert error.details.error_type == "provider_error"
    assert error.details.reason_code == "provider_call_failed"
    assert error.details.provider_error_code == "BadRequest"
    assert error.details.provider_error_type == "invalid_request_error"
    assert error.raw_diagnostics is not None
    assert error.raw_diagnostics.provider_error_body == payload


def test_helper_builders_cover_site_encoding_and_nested_path_writer() -> None:
    payload: dict = {}
    write_field_path(payload, "messages[].content", "hello")
    write_field_path(payload, "resource_type_filter[].top_k", 5)

    assert payload == {"messages": [{"content": "hello"}], "resource_type_filter": [{"top_k": 5}]}
    assert encode_site_restriction_value(["https://trip.com/path"], value_type="string") == "trip.com"
    assert encode_site_restriction_value(["trip.com", "booking.com"], value_type="string_array") == ["trip.com", "booking.com"]
    assert encode_site_restriction_value(["trip.com", "booking.com"], value_type="delimiter_string") == "trip.com|booking.com"
    assert build_site_query(domain="https://www.trip.com/path", split_question="  hotel room  ") == "site:www.trip.com hotel room"


def test_group_b_provider_parameter_mapping_and_query_side_zoom_in() -> None:
    providers = {
        "serper": ("POST", "q", "num", 8),
        "brave": ("GET", "q", "count", 8),
        "you": ("GET", "query", "count", 8),
        "360search": ("GET", "q", "count", 8),
    }

    for engine, (method, query_path, num_path, expected_num) in providers.items():
        adapter = create_builtin_search_adapter(
            type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES[engine], "base_url": None, "api_key": "key", "extra": {}})()
        )
        zoomout = adapter.build_request(UnifiedSearchRequest(task="zoomout_search", query="hotel room", num_results=8))
        zoom_in = adapter.build_request(
            UnifiedSearchRequest(
                task="zoomin_search",
                query="site:trip.com What is the room setup?",
                num_results=8,
                site_restriction_mode="query_side",
            )
        )

        zoomout_payload = zoomout.params if zoomout.method == "GET" else zoomout.json_body
        zoom_in_payload = zoom_in.params if zoom_in.method == "GET" else zoom_in.json_body

        assert zoomout.method == method
        assert zoom_in.method == method
        assert read_field_path(zoomout_payload, query_path) == "hotel room"
        assert read_field_path(zoom_in_payload, query_path) == "site:trip.com What is the room setup?"
        assert read_field_path(zoomout_payload, num_path) == expected_num
        assert read_field_path(zoom_in_payload, num_path) == expected_num

    brave_adapter = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["brave"], "base_url": None, "api_key": "key", "extra": {}})()
    )
    brave_request = brave_adapter.build_request(UnifiedSearchRequest(task="zoomin_search", query="site:trip.com hotel", num_results=3))
    assert brave_request.params["operators"] is True

    search360_adapter = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["360search"], "base_url": None, "api_key": "key", "extra": {}})()
    )
    search360_request = search360_adapter.build_request(
        UnifiedSearchRequest(task="zoomout_search", query="hotel room", num_results=3, trace_context={"request_id": "req-123"})
    )
    assert search360_request.params["ref_prom"] == "360so-s1"
    assert search360_request.params["sid"] == "req-123"


def test_group_b_result_normalization_and_domain_filtering() -> None:
    adapter = GetAdapter(capability=BUILTIN_SEARCH_CAPABILITIES["you"])
    response = adapter.normalize_response(
        payload={
            "results": {
                "web": [
                    {"title": "Keep", "description": "desc", "url": "https://trip.com/a"},
                    {"title": "Drop", "description": "desc", "url": "https://booking.com/b"},
                ]
            }
        },
        request=UnifiedSearchRequest(
            task="zoomin_search",
            query="site:trip.com hotel room",
            num_results=5,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        ),
    )

    assert [item.url for item in response.results] == ["https://trip.com/a"]
    assert response.debug_events[0]["search_query_used"] == "site:trip.com hotel room"
    assert response.debug_events[0]["domain_filtered_result_count"] == 1
    assert any(item.code == "zoom_in_domain_filtered_results" for item in response.normalized_warnings)


def test_group_b_filtered_to_empty_emits_warning() -> None:
    adapter = GetAdapter(capability=BUILTIN_SEARCH_CAPABILITIES["brave"])
    response = adapter.normalize_response(
        payload={
            "web": {
                "results": [
                    {"title": "Drop", "description": "desc", "url": "https://booking.com/b"},
                ]
            }
        },
        request=UnifiedSearchRequest(
            task="zoomin_search",
            query="site:trip.com hotel room",
            num_results=5,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        ),
    )

    assert response.results == []
    assert any(item.code == "zoom_in_filtered_to_empty" for item in response.normalized_warnings)


def test_metasota_zoom_in_mapping_and_best_effort_warning() -> None:
    adapter = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["metasota"], "base_url": None, "api_key": "mk-key", "extra": {}})()
    )
    built = adapter.build_request(
        UnifiedSearchRequest(
            task="zoomin_search",
            query="hotel room",
            num_results=9,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        )
    )
    assert built.json_body == {
        "scope": "webpage",
        "includeSummary": False,
        "q": "site:trip.com hotel room",
        "size": 9,
    }
    assert built.headers["Authorization"] == "Bearer mk-key"

    response = adapter.normalize_response(
        payload={"webpages": [{"title": "Trip", "summary": "desc", "link": "https://trip.com/a"}]},
        request=UnifiedSearchRequest(
            task="zoomin_search",
            query="site:trip.com hotel room",
            num_results=9,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        ),
    )
    warning = next(item for item in response.normalized_warnings if item.code == "low_confidence_site_restriction")
    assert warning.metadata["remaining_result_count"] == 1
    assert "filtered_result_count" not in warning.metadata


def test_searxng_mapping_uses_json_format_and_optional_api_key() -> None:
    adapter = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["searxng"], "base_url": "https://searx.example/search", "api_key": "sx-key", "extra": {}})()
    )
    built = adapter.build_request(
        UnifiedSearchRequest(
            task="zoomin_search",
            query="hotel room",
            num_results=9,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        )
    )

    assert built.method == "GET"
    assert built.endpoint == "https://searx.example/search"
    assert built.params == {"format": "json", "categories": "general", "q": "site:trip.com hotel room"}
    assert built.headers == {"Accept": "application/json", "Authorization": "Bearer sx-key"}


def test_querit_uses_api_key_for_authorization_header() -> None:
    adapter = create_builtin_search_adapter(
        type("Provider", (), {"capability": BUILTIN_SEARCH_CAPABILITIES["querit"], "base_url": None, "api_key": "querit-key", "extra": {}})()
    )
    built = adapter.build_request(
        UnifiedSearchRequest(
            task="zoomin_search",
            query="hotel room",
            num_results=4,
            site_restriction_mode="provider_side",
            site_restriction_domain="trip.com",
        )
    )

    assert built.headers["Authorization"] == "Bearer querit-key"
    assert built.headers["Content-Type"] == "application/json"
    assert built.json_body == {
        "query": "hotel room",
        "filters": {"sites": {"include": ["trip.com"]}},
        "count": 4,
    }


def test_tiangong_sse_normal_completion_and_attribution_normalization() -> None:
    adapter = create_builtin_search_adapter(_FakeTiangongProvider())
    response = adapter.normalize_sse_response(
        lines=[
            'data: {"card_type":"markdown","text":"ignore"}',
            'data: {"card_type":"search_result","arguments":[{"messages":[{"sourceAttributions":[{"title":"A","snippet":"desc A","seeMoreUrl":"https://trip.com/a","showName":"Trip"},{"title":"B","snippet":"desc B","seeMoreUrl":"https://booking.com/b","showName":"Booking"},{"title":"C","snippet":"desc C"}]}]}]}',
            'data: [DONE]',
        ],
        request=UnifiedSearchRequest(
            task="zoomin_search",
            query="site:trip.com hotel room",
            num_results=5,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        ),
    )

    assert [item.url for item in response.results] == ["https://trip.com/a"]
    assert response.results[0].title == "A"
    assert response.results[0].snippet == "desc A"
    assert response.debug_events[0]["done_received"] is True
    assert response.debug_events[0]["search_card_count"] == 1
    assert any(item.code == "zoom_in_domain_filtered_results" for item in response.normalized_warnings)


def test_tiangong_sse_early_end_with_partial_results_emits_warning() -> None:
    adapter = create_builtin_search_adapter(_FakeTiangongProvider())
    response = adapter.normalize_sse_response(
        lines=[
            'data: {"card_type":"search_result","arguments":[{"messages":[{"sourceAttributions":[{"title":"A","snippet":"desc A","seeMoreUrl":"https://trip.com/a"}]}]}]}',
        ],
        request=UnifiedSearchRequest(
            task="zoomin_search",
            query="site:trip.com hotel room",
            num_results=1,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        ),
    )

    assert [item.url for item in response.results] == ["https://trip.com/a"]
    assert response.debug_events[0]["done_received"] is False
    assert any(item.code == "tiangong_sse_stream_ended_early" for item in response.normalized_warnings)


def test_tiangong_empty_attribution_and_missing_card_emit_warnings() -> None:
    adapter = create_builtin_search_adapter(_FakeTiangongProvider())
    response = adapter.normalize_sse_response(
        lines=[
            'data: {"card_type":"search_result","arguments":[{"messages":[{"sourceAttributions":[]}]}]}',
        ],
        request=UnifiedSearchRequest(
            task="zoomin_search",
            query="site:trip.com hotel room",
            num_results=3,
            site_restriction_mode="query_side",
            site_restriction_domain="trip.com",
        ),
    )

    assert response.results == []
    assert any(item.code == "tiangong_empty_source_attributions" for item in response.normalized_warnings)


def test_tiangong_request_contains_signature_headers() -> None:
    adapter = create_builtin_search_adapter(_FakeTiangongProvider())
    built = adapter.build_request(
        UnifiedSearchRequest(task="zoomout_search", query="hotel room", num_results=2)
    )

    assert built.json_body == {"content": "hotel room", "stream_resp_type": "delta"}
    assert built.headers["app_key"] == "app-key"
    assert built.headers["stream"] == "true"
    assert len(built.headers["timestamp"]) >= 10
    assert len(built.headers["sign"]) == 32
