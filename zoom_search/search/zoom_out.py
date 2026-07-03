"""Zoom-out search workflow."""

from __future__ import annotations

import asyncio

from zoom_search.errors import call_failure
from zoom_search.metrics import record_search_attempt
from zoom_search.metrics import record_search_outcome
from zoom_search.models import RuntimeContext
from zoom_search.models import WarningInfo
from zoom_search.models import UnifiedSearchRequest
from zoom_search.models import ZoomOutSearchRequest
from zoom_search.models import ZoomOutSearchResult
from zoom_search.models import ZoomSearchError
from zoom_search.providers.search import normalize_search_exception
from zoom_search.providers.search import SearchProvider
from zoom_search.providers.search import _coerce_float
from zoom_search.rewriting import QueryRewriteGroup
from zoom_search.retry import retry_attempts
from zoom_search.retry import should_retry_error
from zoom_search.retry import sleep_before_retry
from zoom_search.traceability import enrich_for_zoom_out_request
from zoom_search.traceability import enrich_for_zoom_out_result


def build_zoom_out_requests(
    *,
    search_groups: list[QueryRewriteGroup],
    context: RuntimeContext,
) -> list[ZoomOutSearchRequest]:
    requests: list[ZoomOutSearchRequest] = []
    num_results = context.request.search_limits.zoomout_num_results
    for search_group in search_groups:
        for query_variant_id, query in ((1, search_group.query1), (2, search_group.query2)):
            traceability = enrich_for_zoom_out_request(
                search_group.traceability,
                query_variant_id=query_variant_id,
                search_query_used=query,
            )
            requests.append(
                ZoomOutSearchRequest(
                    split_question_id=search_group.group,
                    split_question=search_group.split_question,
                    query=query,
                    query_variant_id=query_variant_id,
                    num_results=num_results,
                    traceability=traceability,
                )
            )
    return requests


async def execute_zoom_out_search(
    *,
    requests: list[ZoomOutSearchRequest],
    context: RuntimeContext,
    search_provider: SearchProvider,
) -> list[ZoomOutSearchResult]:
    semaphore = asyncio.Semaphore(max(int(context.semaphore_limits.get("search_requests", 1)), 1))
    responses = await asyncio.gather(
        *[_execute_limited_zoom_out_request(semaphore=semaphore, request=request, context=context, search_provider=search_provider) for request in requests]
    )
    results: list[ZoomOutSearchResult] = []
    for batch in responses:
        results.extend(batch)
    return results


async def _execute_limited_zoom_out_request(
    *,
    semaphore: asyncio.Semaphore,
    request: ZoomOutSearchRequest,
    context: RuntimeContext,
    search_provider: SearchProvider,
) -> list[ZoomOutSearchResult]:
    async with semaphore:
        return await _execute_single_zoom_out_request(request=request, context=context, search_provider=search_provider)


async def _execute_single_zoom_out_request(
    *,
    request: ZoomOutSearchRequest,
    context: RuntimeContext,
    search_provider: SearchProvider,
) -> list[ZoomOutSearchResult]:
    last_error: ZoomSearchError | None = None
    for attempt in retry_attempts(context=context, component="search"):
        try:
            record_search_attempt(context=context, phase="zoom_out", attempt=attempt)
            unified_request = UnifiedSearchRequest(
                task="zoomout_search",
                query=request.query,
                num_results=request.num_results,
                site_restriction_mode="none",
                trace_context={
                    "request_id": context.request_id,
                    "phase": "Zoom-out Search",
                    "provider_name": context.search_provider.engine,
                    "split_question": request.split_question,
                    "query_variant_id": request.query_variant_id,
                },
            )
            response = await search_provider.search(unified_request)
            _record_provider_response_metadata(context=context, response=response)
            if not response.results:
                raise call_failure(
                    category="search_call_failure",
                    component="search",
                    message="Search provider returned empty results.",
                    user_message="Search request returned no results.",
                    request_id=context.request_id,
                    provider_engine=context.search_provider.engine,
                    reason_code="empty_results",
                    retryable=True,
                )
            normalized_results = [
                ZoomOutSearchResult(
                    title=item.title,
                    snippet=item.snippet,
                    url=item.url,
                    rank=index,
                    traceability=enrich_for_zoom_out_result(request.traceability, rank=index),
                    provider_score=_extract_provider_score(item.raw_item),
                )
                for index, item in enumerate(response.results, start=1)
                if item.title and item.snippet and item.url
            ]
            if not normalized_results:
                raise call_failure(
                    category="search_call_failure",
                    component="search",
                    message="Search provider results are missing required fields.",
                    user_message="Search request returned unusable results.",
                    request_id=context.request_id,
                    provider_engine=context.search_provider.engine,
                    reason_code="results_missing_required_fields",
                    retryable=True,
                )
            if _should_truncate_results_locally(context=context):
                record_search_outcome(context=context, phase="zoom_out", success=True)
                return normalized_results[: request.num_results]
            record_search_outcome(context=context, phase="zoom_out", success=True)
            return normalized_results
        except Exception as error:  # noqa: BLE001
            normalized = normalize_search_exception(error=error, context=context)
            normalized.retry_attempted = attempt > 0
            last_error = normalized
            if attempt < context.retry_budget.get("search", 0) and should_retry_error(error=normalized, component="search"):
                await sleep_before_retry(attempt=attempt)
                continue
            record_search_outcome(context=context, phase="zoom_out", success=False)
            context.warnings.append(
                WarningInfo(
                    code=normalized.details.reason_code or normalized.category,
                    message=normalized.user_message,
                    phase="zoomout_search",
                    request_id=context.request_id,
                    metadata={
                        "split_question_id": request.split_question_id,
                        "split_question": request.split_question,
                        "query_variant_id": request.query_variant_id,
                        "provider_engine": context.search_provider.engine,
                        "retry_attempted": normalized.retry_attempted,
                    },
                )
            )
            return []
    if last_error is not None:
        raise last_error
    return []


def _should_truncate_results_locally(*, context: RuntimeContext) -> bool:
    capability = context.search_provider.capability
    return context.search_provider.provider_kind == "custom" or not capability.supports_provider_side_num_results


def _extract_provider_score(raw_item: object) -> float | None:
    if not isinstance(raw_item, dict):
        return None
    return _coerce_float(raw_item.get("score") or raw_item.get("RankScore"))


def _record_provider_response_metadata(*, context: RuntimeContext, response) -> None:
    if response.normalized_warnings:
        context.warnings.extend(response.normalized_warnings)
