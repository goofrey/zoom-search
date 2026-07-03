"""Unified search interface contracts and generic adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from datetime import UTC
from datetime import datetime
from time import time
from typing import Any
from typing import Protocol
from urllib.parse import parse_qsl
from urllib.parse import urlencode
from urllib.parse import urlparse

from zoom_search.demo import create_demo_search_provider
from zoom_search.errors import call_failure
from zoom_search.errors import extract_openai_error_fields
from zoom_search.errors import search_http_error_reason
from zoom_search.models import ProviderCapability
from zoom_search.models import RawDiagnostics
from zoom_search.models import WarningInfo
from zoom_search.models import RuntimeContext
from zoom_search.models import UnifiedSearchRequest
from zoom_search.models import UnifiedSearchResponse
from zoom_search.models import UnifiedSearchResult
from zoom_search.models import ZoomSearchError
from zoom_search.search.domain import extract_root_domain
from zoom_search.search.domain import normalize_url
from zoom_search.transport import normalize_transport_error


class SearchProvider(Protocol):
    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        ...


def create_search_provider(*, context: RuntimeContext, transport_context=None) -> SearchProvider:
    if context.search_provider.provider_kind == "demo":
        return create_demo_search_provider(seed=context.request.seed)
    if context.search_provider.provider_kind in {"builtin", "custom"}:
        return BuiltinSearchProvider(context=context, transport_context=transport_context)
    return _UnsupportedSearchProvider(context=context)


@dataclass(slots=True)
class BuiltTransportRequest:
    method: str
    endpoint: str
    params: dict[str, Any]
    json_body: dict[str, Any] | None
    headers: dict[str, str]


class SearchCapabilityRegistry:
    def __init__(self, capabilities: dict[str, ProviderCapability]) -> None:
        self._capabilities = dict(capabilities)

    def get(self, engine: str) -> ProviderCapability:
        return self._capabilities[engine]

    def supports_provider_side_site_restriction(self, engine: str) -> bool:
        capability = self.get(engine)
        return capability.supports_site_restriction and capability.site_restriction_mode == "provider_side"

    def recommended_zoom_in_strategy(self, engine: str) -> str:
        return self.get(engine).recommended_zoom_in_strategy


GROUP_B_QUERY_SIDE_ENGINES = {"serper", "brave", "you", "360search"}


class BaseSearchAdapter:
    method = "POST"

    def __init__(self, *, capability: ProviderCapability, endpoint: str = "") -> None:
        self.capability = capability
        self.endpoint = endpoint

    def build_request(self, request: UnifiedSearchRequest) -> BuiltTransportRequest:
        payload = self.build_base_request(request)
        query = self.sanitize_query(request.query, request=request)
        query = self.build_query(query=query, request=request)
        self.write_query(payload, query=query, request=request)
        self.write_site_restriction(payload, request=request)
        self.write_num_results(payload, request=request)
        return self.finalize_transport_request(payload=payload, request=request)

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        return {}

    def get_endpoint(self, request: UnifiedSearchRequest) -> str:
        return self.endpoint

    def get_debug_event(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> dict[str, Any]:
        provider_num_results = read_field_path(payload, self.capability.num_results_param)
        return {
            "provider": self.capability.engine,
            "task": request.task,
            "query_param_path": self.capability.query_param_path,
            "query_value": read_field_path(payload, self.capability.query_param_path),
            "site_restriction_mode": request.site_restriction_mode,
            "site_param_path": self.capability.site_param_path,
            "site_param_value": read_field_path(payload, self.capability.site_param_path),
            "requested_num_results": request.num_results,
            "provider_num_results": provider_num_results,
            "local_truncation": not self.capability.supports_provider_side_num_results,
            "supports_provider_side_site_restriction": self.capability.supports_site_restriction
            and self.capability.site_restriction_mode == "provider_side",
            "supports_query_site_operator": self.capability.supports_query_site_operator,
            "recommended_zoom_in_strategy": self.capability.recommended_zoom_in_strategy,
            "capability_confidence": self.capability.capability_confidence,
        }

    def write_query(self, payload: dict[str, Any], *, query: str, request: UnifiedSearchRequest) -> None:
        write_field_path(payload, self.capability.query_param_path, query)

    def write_site_restriction(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        if request.site_restriction_mode != "provider_side" or not request.site_restriction_domain:
            return
        if not self.capability.site_param_path:
            return
        value = encode_site_restriction_value(
            [request.site_restriction_domain],
            value_type=self.capability.site_value_type,
        )
        write_field_path(payload, self.capability.site_param_path, value)

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        if not self.capability.supports_provider_side_num_results or not self.capability.num_results_param:
            return
        write_field_path(payload, self.capability.num_results_param, request.num_results)

    def sanitize_query(self, query: str, *, request: UnifiedSearchRequest) -> str:
        return clean_search_text(query)

    def build_query(self, *, query: str, request: UnifiedSearchRequest) -> str:
        if request.task == "zoomin_search" and request.site_restriction_mode == "query_side" and request.site_restriction_domain:
            return build_site_query(domain=request.site_restriction_domain, split_question=query)
        return query

    def finalize_transport_request(self, *, payload: dict[str, Any], request: UnifiedSearchRequest) -> BuiltTransportRequest:
        headers = self.get_headers(request)
        endpoint = self.get_endpoint(request)
        if self.method == "GET":
            return BuiltTransportRequest(method="GET", endpoint=endpoint, params=payload, json_body=None, headers=headers)
        return BuiltTransportRequest(method="POST", endpoint=endpoint, params={}, json_body=payload, headers=headers)

    def normalize_response(self, *, payload: Any, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        raw_items = read_result_collection(payload, self.capability.result_collection_path)
        collection_state = inspect_result_collection(payload, self.capability.result_collection_path)
        discarded = 0
        domain_filtered = 0
        results: list[UnifiedSearchResult] = []
        for index, item in enumerate(raw_items):
            normalized = normalize_result_item(item, self.capability.field_candidates, provider_result_index=index)
            if normalized is None:
                discarded += 1
                continue
            if request.task == "zoomin_search" and _should_filter_zoom_in_domain(capability=self.capability, request=request):
                actual_domain = extract_root_domain(normalize_url(normalized.url))
                target_domain = request.site_restriction_domain
                if target_domain and actual_domain and not _domain_matches(target_domain=target_domain, actual_domain=actual_domain):
                    discarded += 1
                    domain_filtered += 1
                    continue
            results.append(normalized)
        results = sort_results_by_provider_rank(results)

        warnings: list[WarningInfo] = []
        debug_event = self.get_debug_event({}, request=request)
        debug_event.update(
            {
                "zoom_in_strategy": self.capability.recommended_zoom_in_strategy,
                "result_collection_path": self.capability.result_collection_path,
                "search_query_used": request.query,
                "raw_result_count": len(raw_items),
                "normalized_result_count": len(results),
                "discarded_result_count": discarded,
                "domain_filtered_result_count": domain_filtered,
            }
        )
        debug_events = [debug_event]
        if request.task == "zoomin_search" and self.capability.recommended_zoom_in_strategy == "best_effort":
            warnings.append(
                WarningInfo(
                    code="low_confidence_site_restriction",
                    message="Search provider uses best-effort site restriction.",
                    phase="zoomin_search",
                    metadata={
                        "provider": self.capability.engine,
                        "source_domain": request.site_restriction_domain,
                        "remaining_result_count": len(results),
                    },
                )
            )
        if request.task == "zoomin_search" and self.capability.engine in GROUP_B_QUERY_SIDE_ENGINES and self.capability.supports_query_site_operator == "unknown":
            warnings.append(
                WarningInfo(
                    code="query_site_operator_support_unknown",
                    message="Search provider query-site operator support is declared as unknown.",
                    phase="zoomin_search",
                    metadata={"provider": self.capability.engine},
                )
            )
        if request.task == "zoomin_search" and domain_filtered > 0:
            warnings.append(
                WarningInfo(
                    code="zoom_in_domain_filtered_results",
                    message="Zoom-in results included non-target domains and were filtered.",
                    phase="zoomin_search",
                    metadata={
                        "provider": self.capability.engine,
                        "source_domain": request.site_restriction_domain,
                        "filtered_result_count": domain_filtered,
                        "remaining_result_count": len(results),
                    },
                )
            )
        if request.task == "zoomin_search" and len(raw_items) > 0 and len(results) == 0:
            warnings.append(
                WarningInfo(
                    code="zoom_in_filtered_to_empty",
                    message="Zoom-in results were filtered to empty after domain validation.",
                    phase="zoomin_search",
                    metadata={
                        "provider": self.capability.engine,
                        "source_domain": request.site_restriction_domain,
                        "raw_result_count": len(raw_items),
                        "filtered_result_count": domain_filtered,
                    },
                )
            )
        if collection_state == "non_list":
            warnings.append(
                WarningInfo(
                    code="result_collection_path_not_list",
                    message="Search provider result collection path resolved to a non-list value.",
                    phase=request.task,
                    metadata={
                        "provider": self.capability.engine,
                        "result_collection_path": self.capability.result_collection_path,
                    },
                )
            )
        if not raw_items:
            warnings.append(
                WarningInfo(
                    code="missing_result_collection",
                    message="Search provider result collection is empty or missing.",
                    phase=request.task,
                    metadata={"provider": self.capability.engine},
                )
            )
        return UnifiedSearchResponse(
            results=results,
            raw_response=payload,
            normalized_warnings=warnings,
            debug_events=debug_events,
        )


class CustomSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        return _bearer_headers(self._api_key)


def _bearer_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class _UnsupportedSearchProvider:
    def __init__(self, *, context: RuntimeContext) -> None:
        self._context = context

    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        raise call_failure(
            category="search_call_failure",
            component="search",
            message="Built-in HTTP search adapters are not implemented in this window.",
            user_message="The selected search provider is not implemented yet.",
            request_id=self._context.request_id,
            provider_engine=self._context.search_provider.engine,
            reason_code="provider_not_implemented",
        )


class BuiltinSearchProvider:
    def __init__(self, *, context: RuntimeContext, transport_context: Any | None) -> None:
        self._context = context
        self._transport_context = transport_context
        self._adapter = create_builtin_search_adapter(context.search_provider)

    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        built = self._adapter.build_request(request)
        payload_for_debug = built.params if built.method == "GET" else built.json_body or {}
        try:
            client = getattr(self._transport_context, "search_client", None) if self._transport_context is not None else None
            if client is None or not hasattr(client, "request"):
                raise RuntimeError("Search transport client is unavailable.")
            if self._context.search_provider.engine == "tiangong":
                normalized = await self._search_tiangong_stream(client=client, built=built, request=request, payload_for_debug=payload_for_debug)
                return normalized
            headers = dict(built.headers)
            response = await client.request(
                built.method,
                built.endpoint,
                params=built.params,
                json=built.json_body,
                headers=headers,
            )
            raise_for_status = getattr(response, "raise_for_status", None)
            if raise_for_status is not None:
                raise_for_status()
            payload = response.json()
            provider_error = extract_provider_error(payload=payload, provider=self._context.search_provider.engine, request_id=self._context.request_id)
            if provider_error is not None:
                raise provider_error
            normalized = self._adapter.normalize_response(payload=payload, request=request)
            adapter_debug = self._adapter.get_debug_event(payload_for_debug, request=request)
            for event in normalized.debug_events:
                event.update(adapter_debug)
            if self._context.search_provider.engine == "firecrawl" and isinstance(payload, dict):
                normalized.provider_metadata.update(
                    {
                        "warning": payload.get("warning"),
                        "creditsUsed": payload.get("creditsUsed"),
                    }
                )
            return normalized
        except Exception as error:  # noqa: BLE001
            raise normalize_search_exception(error=error, context=self._context) from error

    async def _search_tiangong_stream(self, *, client: Any, built: BuiltTransportRequest, request: UnifiedSearchRequest, payload_for_debug: dict[str, Any]) -> UnifiedSearchResponse:
        headers = dict(built.headers)
        if hasattr(client, "stream"):
            lines: list[str] = []
            async with client.stream(
                built.method,
                built.endpoint,
                params=built.params,
                json=built.json_body,
                headers=headers,
            ) as response:
                raise_for_status = getattr(response, "raise_for_status", None)
                if raise_for_status is not None:
                    raise_for_status()
                async for line in response.aiter_lines():
                    if line is not None:
                        lines.append(line)
        else:
            response = await client.request(
                built.method,
                built.endpoint,
                params=built.params,
                json=built.json_body,
                headers=headers,
            )
            raise_for_status = getattr(response, "raise_for_status", None)
            if raise_for_status is not None:
                raise_for_status()
            text = getattr(response, "text", "")
            lines = str(text).splitlines()
        normalized = self._adapter.normalize_sse_response(lines=lines, request=request)
        adapter_debug = self._adapter.get_debug_event(payload_for_debug, request=request)
        for event in normalized.debug_events:
            event.update(adapter_debug)
        return normalized


class LinkupSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"outputType": "searchResults", "depth": "standard"}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers


class PerplexitySearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        return _bearer_headers(self._api_key)

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 20)
        write_field_path(payload, "max_results", value)


class TavilySearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"search_depth": "basic"}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 20)
        write_field_path(payload, "max_results", value)


class GLMSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        return _bearer_headers(self._api_key)

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"search_engine": "search_pro", "search_intent": False}

    def sanitize_query(self, query: str, *, request: UnifiedSearchRequest) -> str:
        cleaned = super().sanitize_query(query, request=request)
        return cleaned[:70]

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 50)
        write_field_path(payload, "count", value)


class ExaSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"type": "auto", "contents": {"highlights": True}}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers


class FirecrawlSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"sources": ["web"]}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        return _bearer_headers(self._api_key)


class MetasotaSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"scope": "webpage", "includeSummary": False}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        write_field_path(payload, "size", min(max(request.num_results, 1), 100))


class SearxngSearchAdapter(BaseSearchAdapter):
    method = "GET"

    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"format": "json", "categories": "general"}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers


class TiangongSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None, provider_extra: dict[str, Any] | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._app_key = api_key or ""
        self._app_secret = str((provider_extra or {}).get("app_secret") or "")

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"content": "", "stream_resp_type": "delta"}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        timestamp = str(int(time()))
        sign = hashlib.md5(f"{self._app_key}{self._app_secret}{timestamp}".encode("utf-8")).hexdigest()
        return {"app_key": self._app_key, "timestamp": timestamp, "sign": sign, "stream": "true", "Content-Type": "application/json"}

    def normalize_sse_response(self, *, lines: list[str], request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        event_count = 0
        search_card_count = 0
        done_received = False
        parse_errors = 0
        source_items: list[dict[str, Any]] = []
        warnings: list[WarningInfo] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text:
                continue
            event_count += 1
            if payload_text == "[DONE]":
                done_received = True
                break
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if payload.get("card_type") != "search_result":
                continue
            search_card_count += 1
            attributions = _extract_tiangong_source_attributions(payload)
            if not attributions:
                warnings.append(WarningInfo(code="tiangong_empty_source_attributions", message="Tiangong search card returned empty source attributions.", phase="zoomin_search", metadata={"provider": self.capability.engine}))
                continue
            source_items.extend(attributions)
        if search_card_count == 0:
            warnings.append(WarningInfo(code="tiangong_missing_search_card", message="Tiangong SSE stream did not contain a search_result card.", phase="zoomin_search", metadata={"provider": self.capability.engine}))
        if not done_received:
            warnings.append(WarningInfo(code="tiangong_sse_stream_ended_early", message="Tiangong SSE stream ended before [DONE].", phase="zoomin_search", metadata={"provider": self.capability.engine, "partial_success": bool(source_items)}))
        normalized = self.normalize_response(payload={"results": source_items}, request=request)
        before_truncate = len(normalized.results)
        debug_event = self.get_debug_event({}, request=request)
        debug_event.update({
            "search_query_used": request.query,
            "sse_event_count": event_count,
            "search_card_count": search_card_count,
            "done_received": done_received,
            "stream_end_reason": "done" if done_received else "stream_closed",
            "sse_parse_error_count": parse_errors,
            "filtering_before_count": before_truncate,
            "filtering_after_count": min(before_truncate, request.num_results),
            "local_truncation_before_count": before_truncate,
            "local_truncation_after_count": min(before_truncate, request.num_results),
        })
        if normalized.debug_events:
            debug_event.update(normalized.debug_events[0])
        return UnifiedSearchResponse(results=normalized.results[: request.num_results], raw_response={"sse_lines": lines}, normalized_warnings=[*normalized.normalized_warnings, *warnings], debug_events=[debug_event])


class BaiduSearchAdapter(BaseSearchAdapter):
    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": ""}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web"}],
        }

    def sanitize_query(self, query: str, *, request: UnifiedSearchRequest) -> str:
        cleaned = super().sanitize_query(query, request=request)
        return cleaned[:72]

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        token = self._token()
        return {
            "Authorization": f"Bearer {token}",
            "X-Appbuilder-Authorization": f"Bearer {token}",
        }

    def _token(self) -> str:
        return ""


class VolcengineSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", provider_extra: dict[str, Any] | None = None, api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._provider_extra = provider_extra or {}
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"SearchType": "web"}

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 50)
        write_field_path(payload, "Count", value)

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def finalize_transport_request(self, *, payload: dict[str, Any], request: UnifiedSearchRequest) -> BuiltTransportRequest:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else self._build_aksk_headers(payload)
        endpoint = self.get_endpoint(request)
        if self.method == "GET":
            return BuiltTransportRequest(method="GET", endpoint=endpoint, params=payload, json_body=None, headers=headers)
        return BuiltTransportRequest(method="POST", endpoint=endpoint, params={}, json_body=payload, headers=headers)

    def get_endpoint(self, request: UnifiedSearchRequest) -> str:
        if self._api_key:
            return self.endpoint or "https://open.feedcoopapi.com/search_api/web_search"
        return "https://mercury.volcengineapi.com?Action=WebSearch&Version=2025-01-01"

    def _build_aksk_headers(self, payload: dict[str, Any]) -> dict[str, str]:
        access_key = str(self._provider_extra.get("access_key", ""))
        secret_key = str(self._provider_extra.get("secret_key", ""))
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        signing_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        digest = hmac.new(secret_key.encode("utf-8"), signing_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return {
            "X-Access-Key": access_key,
            "X-Timestamp": timestamp,
            "X-Signature": digest,
        }


class BochaSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"summary": False, "freshness": "noLimit"}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        return _bearer_headers(self._api_key)

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 50)
        write_field_path(payload, "count", value)


class QueritSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def sanitize_query(self, query: str, *, request: UnifiedSearchRequest) -> str:
        cleaned = super().sanitize_query(query, request=request)
        return cleaned[:72]

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 100)
        write_field_path(payload, "count", value)


class SerpAPISearchAdapter(BaseSearchAdapter):
    method = "GET"

    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"engine": "google", "output": "json"}

    def write_site_restriction(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        super().write_site_restriction(payload, request=request)
        if request.site_restriction_mode == "provider_side" and request.site_restriction_domain:
            payload["as_dt"] = "i"

    def finalize_transport_request(self, *, payload: dict[str, Any], request: UnifiedSearchRequest) -> BuiltTransportRequest:
        payload["api_key"] = self._api_key or ""
        return super().finalize_transport_request(payload=payload, request=request)


class SerperSearchAdapter(BaseSearchAdapter):
    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-KEY"] = self._api_key
        return headers


class BraveSearchAdapter(BaseSearchAdapter):
    method = "GET"

    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        return {"operators": True}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["X-Subscription-Token"] = self._api_key
        return headers

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 20)
        write_field_path(payload, "count", value)


class YouSearchAdapter(BaseSearchAdapter):
    method = "GET"

    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 100)
        write_field_path(payload, "count", value)


class Search360Adapter(BaseSearchAdapter):
    method = "GET"

    def __init__(self, *, capability: ProviderCapability, endpoint: str = "", api_key: str | None = None) -> None:
        super().__init__(capability=capability, endpoint=endpoint)
        self._api_key = api_key

    def build_base_request(self, request: UnifiedSearchRequest) -> dict[str, Any]:
        sid = request.trace_context.get("request_id") or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return {"ref_prom": "360so-s1", "sid": sid}

    def get_headers(self, request: UnifiedSearchRequest) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def write_num_results(self, payload: dict[str, Any], *, request: UnifiedSearchRequest) -> None:
        value = min(max(request.num_results, 1), 20)
        write_field_path(payload, "count", value)


def create_builtin_search_adapter(provider) -> BaseSearchAdapter:
    capability = provider.capability
    endpoint = provider.base_url or capability.default_base_url or ""
    if getattr(provider, "provider_kind", None) == "custom" or capability.provider_kind == "custom":
        return CustomSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "serper":
        return SerperSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "tavily":
        return TavilySearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "linkup":
        return LinkupSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "brave":
        return BraveSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "you":
        return YouSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "360search":
        return Search360Adapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "perplexity":
        return PerplexitySearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "glm":
        return GLMSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "exa":
        return ExaSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "firecrawl":
        return FirecrawlSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "metasota":
        return MetasotaSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "baidu":
        adapter = BaiduSearchAdapter(capability=capability, endpoint=endpoint)
        adapter._token = lambda: provider.api_key or ""  # type: ignore[attr-defined]
        return adapter
    if capability.engine == "volcengine":
        return VolcengineSearchAdapter(
            capability=capability,
            endpoint=endpoint,
            provider_extra=provider.extra,
            api_key=provider.api_key,
        )
    if capability.engine == "bocha":
        return BochaSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "querit":
        return QueritSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "serpapi":
        return SerpAPISearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "searxng":
        return SearxngSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key)
    if capability.engine == "tiangong":
        return TiangongSearchAdapter(capability=capability, endpoint=endpoint, api_key=provider.api_key, provider_extra=provider.extra)
    return BaseSearchAdapter(capability=capability, endpoint=endpoint)


def extract_provider_error(*, payload: Any, provider: str, request_id: str = "pending") -> ZoomSearchError | None:
    if not isinstance(payload, dict):
        return None
    if provider == "bocha" and payload.get("code") not in (None, 200):
        return normalize_search_error(error=RuntimeError(str(payload.get("message") or "bocha provider error")), provider=provider, request_id=request_id, payload=payload)
    if provider == "querit" and payload.get("error_code") not in (None, 200, "200"):
        return normalize_search_error(error=RuntimeError(str(payload.get("error_msg") or payload.get("error_code"))), provider=provider, request_id=request_id, payload=payload)
    if provider == "baidu" and payload.get("error_code"):
        return normalize_search_error(error=RuntimeError(str(payload.get("error_msg") or payload.get("error_code"))), provider=provider, request_id=request_id, payload=payload)
    if provider == "firecrawl" and payload.get("success") is False:
        return normalize_search_error(error=RuntimeError(str(payload.get("error") or payload.get("warning") or "firecrawl provider error")), provider=provider, request_id=request_id, payload=payload)
    return None


def normalize_search_exception(*, error: Exception, context: RuntimeContext) -> ZoomSearchError:
    if isinstance(error, ZoomSearchError):
        return error
    http_error = _normalize_search_http_status_error(error=error, context=context)
    if http_error is not None:
        return http_error
    return normalize_transport_error(
        error=error,
        component="search",
        request_id=context.request_id,
        provider_engine=context.search_provider.engine,
    )


def _normalize_search_http_status_error(*, error: Exception, context: RuntimeContext) -> ZoomSearchError | None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        return None
    payload = _read_error_payload(response)
    fields = _extract_search_error_fields(payload)
    return call_failure(
        category="search_call_failure",
        component="search",
        message=fields["message"] or str(error) or "Search provider HTTP request failed.",
        user_message="Search provider request failed.",
        request_id=context.request_id,
        provider_engine=context.search_provider.engine,
        reason_code=search_http_error_reason(status_code) or "provider_call_failed",
        http_status=status_code,
        retryable=status_code in {429, 500, 502, 503, 504},
        provider_error_code=fields["code"],
        provider_error_type=fields["type"],
        provider_error_param=fields["param"],
        raw_diagnostics=RawDiagnostics(
            http_status=status_code,
            provider_error_body=payload,
            provider_error_code=fields["code"],
            provider_error_type=fields["type"],
            provider_error_param=fields["param"],
            transport_exception_class=error.__class__.__name__,
            transport_exception_message=str(error) or error.__class__.__name__,
        ),
    )


def _read_error_payload(response: Any) -> Any:
    json_reader = getattr(response, "json", None)
    if json_reader is not None:
        try:
            return json_reader()
        except Exception:  # noqa: BLE001
            pass
    return getattr(response, "text", None)


def _extract_search_error_fields(payload: Any) -> dict[str, str | None]:
    fields = extract_openai_error_fields(payload)
    if not isinstance(payload, dict):
        return fields
    return {
        "message": fields["message"] or _coerce_optional_text(payload.get("error_msg") or payload.get("message") or payload.get("error")),
        "code": fields["code"] or _coerce_optional_text(payload.get("error_code") or payload.get("code")),
        "type": fields["type"] or _coerce_optional_text(payload.get("error_type") or payload.get("type")),
        "param": fields["param"] or _coerce_optional_text(payload.get("param")),
    }


def _coerce_optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def encode_site_restriction_value(domains: list[str], *, value_type: str, delimiter: str = "|") -> str | list[str]:
    cleaned = [clean_site_restriction_domain(domain) for domain in domains if clean_site_restriction_domain(domain)]
    if value_type == "string_array":
        return cleaned
    if value_type == "delimiter_string":
        return delimiter.join(cleaned)
    return cleaned[0] if cleaned else ""


def clean_site_restriction_domain(domain: str) -> str:
    normalized = normalize_url(domain)
    if normalized:
        parsed = urlparse(normalized)
        return (parsed.hostname or "").lower()
    cleaned = domain.strip().lower()
    for prefix in ("https://", "http://"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = cleaned.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].strip()
    return cleaned


def write_field_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = target
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        list_mode = part.endswith("[]")
        key = part[:-2] if list_mode else part
        if is_last:
            if list_mode:
                current[key] = value if isinstance(value, list) else [value]
            else:
                current[key] = value
            return
        if list_mode:
            bucket = current.setdefault(key, [])
            if not bucket:
                bucket.append({})
            current = bucket[0]
            continue
        current = current.setdefault(key, {})


def read_field_path(payload: Any, path: str | None) -> Any:
    if path is None:
        return None
    current = payload
    for part in path.split("."):
        list_mode = part.endswith("[]")
        key = part[:-2] if list_mode else part
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
        if list_mode and isinstance(current, list):
            current = current[0] if current else None
    return current


def read_result_collection(payload: Any, path: str | None) -> list[Any]:
    if path is None:
        return []
    current = payload
    for part in path.split("."):
        if current is None:
            return []
        if isinstance(current, list):
            return current
        if not isinstance(current, dict):
            return []
        current = current.get(part)
    return current if isinstance(current, list) else []


def inspect_result_collection(payload: Any, path: str | None) -> str:
    if path is None:
        return "missing"
    current = payload
    for part in path.split("."):
        if current is None:
            return "missing"
        if isinstance(current, list):
            return "list"
        if not isinstance(current, dict):
            return "non_list"
        if part not in current:
            return "missing"
        current = current.get(part)
    return "list" if isinstance(current, list) else "non_list"


def normalize_result_item(
    item: Any,
    field_candidates: dict[str, list[str]],
    *,
    provider_result_index: int,
) -> UnifiedSearchResult | None:
    if not isinstance(item, dict):
        return None
    title = _pick_field(item, field_candidates.get("title", []))
    snippet = clean_search_text(clean_html(_pick_field(item, field_candidates.get("snippet", []))))
    url = clean_search_text(_pick_field(item, field_candidates.get("url", [])))
    normalized_url = normalize_url(url)
    if not title or not snippet or not normalized_url:
        return None
    return UnifiedSearchResult(
        title=clean_search_text(title),
        snippet=snippet,
        url=normalized_url,
        provider_result_index=provider_result_index,
        raw_item=item,
    )


def sort_results_by_provider_rank(results: list[UnifiedSearchResult]) -> list[UnifiedSearchResult]:
    return sorted(results, key=_provider_rank_sort_key)


def _provider_rank_sort_key(result: UnifiedSearchResult) -> tuple[int, float, int]:
    item = result.raw_item if isinstance(result.raw_item, dict) else {}
    position = _coerce_float(item.get("position"))
    if position is not None:
        return (0, position, result.provider_result_index)
    score = _coerce_float(item.get("score") or item.get("RankScore"))
    if score is not None:
        return (1, -score, result.provider_result_index)
    return (2, float(result.provider_result_index), result.provider_result_index)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def clean_html(value: str) -> str:
    text = value or ""
    for source, target in (("<br>", " "), ("<br/>", " "), ("<br />", " ")):
        text = text.replace(source, target)
    result: list[str] = []
    in_tag = False
    for char in text:
        if char == "<":
            in_tag = True
            continue
        if char == ">":
            in_tag = False
            result.append(" ")
            continue
        if not in_tag:
            result.append(char)
    cleaned = "".join(result)
    return cleaned.replace("&nbsp;", " ").replace("&amp;", "&")


def clean_search_text(value: str) -> str:
    return " ".join((value or "").split())


def build_site_query(*, domain: str, split_question: str) -> str:
    return f"site:{clean_site_restriction_domain(domain)} {clean_search_text(split_question)}".strip()


def _should_filter_zoom_in_domain(*, capability: ProviderCapability, request: UnifiedSearchRequest) -> bool:
    return request.task == "zoomin_search" and request.site_restriction_mode == "query_side" and bool(request.site_restriction_domain)


def normalize_search_error(*, error: Exception, provider: str, request_id: str, payload: Any = None) -> ZoomSearchError:
    if isinstance(error, ZoomSearchError):
        return error
    fields = _extract_search_error_fields(payload)
    return call_failure(
        category="search_call_failure",
        component="search",
        message=str(error),
        user_message="Search provider request failed.",
        request_id=request_id,
        provider_engine=provider,
        reason_code="provider_call_failed",
        provider_error_code=fields["code"],
        provider_error_type=fields["type"],
        provider_error_param=fields["param"],
        raw_diagnostics=RawDiagnostics(
            provider_error_body=payload,
            provider_error_code=fields["code"],
            provider_error_type=fields["type"],
            provider_error_param=fields["param"],
            transport_exception_class=error.__class__.__name__,
            transport_exception_message=str(error) or error.__class__.__name__,
        ),
    )


def _pick_field(item: dict[str, Any], candidates: list[str]) -> str:
    for candidate in candidates:
        value = item.get(candidate)
        if value is None:
            continue
        if isinstance(value, list):
            joined = " ".join(str(part).strip() for part in value if str(part).strip())
            if joined:
                return joined
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _domain_matches(*, target_domain: str, actual_domain: str) -> bool:
    target_host = clean_site_restriction_domain(target_domain)
    actual_host = clean_site_restriction_domain(actual_domain)
    return bool(target_host and actual_host) and (actual_host == target_host or actual_host.endswith(f".{target_host}"))


def _extract_tiangong_source_attributions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    arguments = payload.get("arguments")
    if not isinstance(arguments, list) or not arguments:
        return []
    first_argument = arguments[0]
    if not isinstance(first_argument, dict):
        return []
    messages = first_argument.get("messages")
    if not isinstance(messages, list) or not messages:
        return []
    first_message = messages[0]
    if not isinstance(first_message, dict):
        return []
    attributions = first_message.get("sourceAttributions")
    return attributions if isinstance(attributions, list) else []
