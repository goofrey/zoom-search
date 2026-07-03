"""Zoom-in search workflow."""

from __future__ import annotations

import asyncio

from zoom_search.errors import call_failure
from zoom_search.metrics import record_search_attempt
from zoom_search.metrics import record_search_outcome
from zoom_search.models import RuntimeContext
from zoom_search.models import SourceDomainRecord
from zoom_search.models import UnifiedSearchRequest
from zoom_search.models import WarningInfo
from zoom_search.models import ZoomInSearchRequest
from zoom_search.models import ZoomInSearchResult
from zoom_search.models import ZoomSearchError
from zoom_search.providers.search import normalize_search_exception
from zoom_search.providers.search import SearchProvider
from zoom_search.providers.search import _domain_matches
from zoom_search.retry import retry_attempts
from zoom_search.retry import should_retry_error
from zoom_search.retry import sleep_before_retry
from zoom_search.search.domain import extract_root_domain
from zoom_search.search.domain import normalize_url
from zoom_search.traceability import enrich_for_zoom_in_request
from zoom_search.traceability import enrich_for_zoom_in_result


def build_zoom_in_requests(
    *,
    source_domains: list[SourceDomainRecord],
    context: RuntimeContext,
) -> list[ZoomInSearchRequest]:
    requests: list[ZoomInSearchRequest] = []
    num_results = context.request.search_limits.zoomin_num_results
    supports_provider_side = _supports_provider_side_site_restriction(context=context)
    for domain_record in source_domains:
        if supports_provider_side:
            query = domain_record.query1
            provider_site_value = domain_record.source_domain
            site_restriction_mode = "provider_side"
        else:
            query = f"site:{domain_record.source_domain} {domain_record.query1}"
            provider_site_value = None
            site_restriction_mode = "query_side"
        traceability = enrich_for_zoom_in_request(
            domain_record.traceability,
            search_query_used=query,
        )
        requests.append(
            ZoomInSearchRequest(
                source_domain=domain_record.source_domain,
                split_question=domain_record.split_question,
                query=query,
                num_results=num_results,
                site_restriction_mode=site_restriction_mode,
                provider_site_value=provider_site_value,
                traceability=traceability,
            )
        )
    return requests


async def execute_zoom_in_search(
    *,
    requests: list[ZoomInSearchRequest],
    context: RuntimeContext,
    search_provider: SearchProvider,
) -> list[ZoomInSearchResult]:
    semaphore = asyncio.Semaphore(max(int(context.semaphore_limits.get("search_requests", 1)), 1))
    responses = await asyncio.gather(
        *[_execute_limited_zoom_in_request(semaphore=semaphore, request=request, context=context, search_provider=search_provider) for request in requests]
    )
    results: list[ZoomInSearchResult] = []
    for batch in responses:
        results.extend(batch)
    return results


async def _execute_limited_zoom_in_request(
    *,
    semaphore: asyncio.Semaphore,
    request: ZoomInSearchRequest,
    context: RuntimeContext,
    search_provider: SearchProvider,
) -> list[ZoomInSearchResult]:
    async with semaphore:
        return await _execute_single_zoom_in_request(request=request, context=context, search_provider=search_provider)


async def _execute_single_zoom_in_request(
    *,
    request: ZoomInSearchRequest,
    context: RuntimeContext,
    search_provider: SearchProvider,
) -> list[ZoomInSearchResult]:
    last_error: ZoomSearchError | None = None
    for attempt in retry_attempts(context=context, component="search"):
        try:
            record_search_attempt(context=context, phase="zoom_in", attempt=attempt)
            unified_request = UnifiedSearchRequest(
                task="zoomin_search",
                query=request.query,
                num_results=request.num_results,
                site_restriction_mode=request.site_restriction_mode,
                site_restriction_domain=request.provider_site_value,
                trace_context={
                    "request_id": context.request_id,
                    "phase": "Zoom-in Search",
                    "provider_name": context.search_provider.engine,
                    "split_question": request.split_question,
                    "query_variant_id": request.traceability.query_variant_id,
                },
            )
            response = await search_provider.search(unified_request)
            _record_provider_response_metadata(context=context, request=request, response=response)
            if not response.results:
                record_search_outcome(context=context, phase="zoom_in", success=True)
                return []
            normalized_results = [
                _build_zoom_in_result(request=request, title=item.title, snippet=item.snippet, url=item.url, rank=index)
                for index, item in enumerate(response.results, start=1)
                if item.title and item.snippet and item.url
            ]
            normalized_results = _filter_zoom_in_results_by_domain(
                results=normalized_results,
                request=request,
                context=context,
                raw_result_count=len(response.results),
            )
            if not normalized_results:
                record_search_outcome(context=context, phase="zoom_in", success=True)
                return []
            if _should_truncate_results_locally(context=context):
                record_search_outcome(context=context, phase="zoom_in", success=True)
                return normalized_results[: request.num_results]
            record_search_outcome(context=context, phase="zoom_in", success=True)
            return normalized_results
        except Exception as error:  # noqa: BLE001
            normalized = normalize_search_exception(error=error, context=context)
            normalized.retry_attempted = attempt > 0
            last_error = normalized
            if attempt < context.retry_budget.get("search", 0) and should_retry_error(error=normalized, component="search"):
                await sleep_before_retry(attempt=attempt)
                continue
            record_search_outcome(context=context, phase="zoom_in", success=False)
            context.warnings.append(
                WarningInfo(
                    code=normalized.details.reason_code or normalized.category,
                    message=normalized.user_message,
                    phase="zoomin_search",
                    request_id=context.request_id,
                    metadata={
                        "split_question": request.split_question,
                        "source_domain": request.source_domain,
                        "provider_engine": context.search_provider.engine,
                        "retry_attempted": normalized.retry_attempted,
                    },
                )
            )
            return []
    if last_error is not None:
        raise last_error
    return []


def _build_zoom_in_result(*, request: ZoomInSearchRequest, title: str, snippet: str, url: str, rank: int) -> ZoomInSearchResult:
    normalized_url = normalize_url(url)
    actual_domain = extract_root_domain(normalized_url) if normalized_url else None
    traceability = enrich_for_zoom_in_result(request.traceability, rank=rank)
    return ZoomInSearchResult(
        title=title,
        snippet=snippet,
        url=normalized_url or url,
        rank=rank,
        traceability=traceability,
        source_domain=actual_domain or request.source_domain,
    )


def _supports_provider_side_site_restriction(*, context: RuntimeContext) -> bool:
    capability = context.search_provider.capability
    return (
        context.search_provider.provider_kind != "custom"
        and capability.supports_site_restriction
        and capability.site_restriction_mode == "provider_side"
    )


def _should_truncate_results_locally(*, context: RuntimeContext) -> bool:
    capability = context.search_provider.capability
    return (
        context.search_provider.provider_kind == "custom"
        or capability.site_restriction_mode == "query_side"
        or not capability.supports_provider_side_num_results
    )


def _filter_zoom_in_results_by_domain(
    *,
    results: list[ZoomInSearchResult],
    request: ZoomInSearchRequest,
    context: RuntimeContext,
    raw_result_count: int,
) -> list[ZoomInSearchResult]:
    target_domain = request.source_domain
    filtered = [
        item
        for item in results
        if item.source_domain and _domain_matches(target_domain=target_domain, actual_domain=item.source_domain)
    ]
    removed_count = len(results) - len(filtered)
    if removed_count > 0:
        context.warnings.append(
            WarningInfo(
                code="zoom_in_domain_filtered_results",
                message="Zoom-in results included non-target domains and were filtered.",
                phase="zoomin_search",
                request_id=context.request_id,
                metadata={
                    "split_question": request.split_question,
                    "source_domain": request.source_domain,
                    "provider_engine": context.search_provider.engine,
                    "filtered_result_count": removed_count,
                    "remaining_result_count": len(filtered),
                },
            )
        )
    if raw_result_count > 0 and not filtered:
        context.warnings.append(
            WarningInfo(
                code="zoom_in_filtered_to_empty",
                message="Zoom-in results were filtered to empty after domain validation.",
                phase="zoomin_search",
                request_id=context.request_id,
                metadata={
                    "split_question": request.split_question,
                    "source_domain": request.source_domain,
                    "provider_engine": context.search_provider.engine,
                    "raw_result_count": raw_result_count,
                },
            )
        )
    return filtered


def _record_provider_response_metadata(*, context: RuntimeContext, request: ZoomInSearchRequest, response) -> None:
    if response.normalized_warnings:
        context.warnings.extend(response.normalized_warnings)
