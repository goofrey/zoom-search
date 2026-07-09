"""Public API entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from typing import AsyncIterator

from zoom_search.evidence import format_search_evidence
from zoom_search.errors import call_failure
from zoom_search.metrics import record_phase_elapsed
from zoom_search.metrics import record_search_planned
from zoom_search.metrics import snapshot_metrics
from zoom_search.models import SearchRequest
from zoom_search.models import SearchResponse
from zoom_search.models import SearchStreamEvent
from zoom_search.models import SimpleSearchResult
from zoom_search.models import WarningInfo
from zoom_search.models import ZoomSearchError
from zoom_search.orchestration.runtime import build_runtime_context
from zoom_search.orchestration.runtime import create_transport_runtime
from zoom_search.providers.llm import create_llm_provider
from zoom_search.providers.llm import normalize_llm_exception
from zoom_search.providers.resolver import resolve_providers
from zoom_search.providers.search import create_search_provider
from zoom_search.results import build_final_search_results
from zoom_search.rewriting import rewrite_query_groups
from zoom_search.search.domain import attach_source_domains
from zoom_search.search.domain import extract_source_domain_records
from zoom_search.search.zoom_out import build_zoom_out_requests
from zoom_search.search.zoom_out import execute_zoom_out_search
from zoom_search.search.zoomin import build_zoom_in_requests
from zoom_search.search.zoomin import execute_zoom_in_search
from zoom_search.synthesis import stream_synthesize_answer
from zoom_search.synthesis import synthesize_answer
from zoom_search.validation import normalize_and_validate_request


@dataclass(slots=True)
class _SearchRuntime:
    context: Any
    transport_context: Any
    llm_provider: Any
    search_provider: Any


@dataclass(slots=True)
class _SearchArtifacts:
    context: Any
    transport_context: Any
    llm_provider: Any
    final_results: list
    evidence: str


async def search(request: SearchRequest | dict | None = None, **params: Any) -> SearchResponse:
    """Execute the Zoom Search workflow from a request object or keyword parameters."""

    if request is not None and params:
        raise TypeError("search() accepts either a request object or keyword parameters, not both.")
    return await _run_search(request if request is not None else params)


async def astream_search(request: SearchRequest | dict | None = None, **params: Any) -> AsyncIterator[SearchStreamEvent]:
    """Execute Zoom Search and stream answer synthesis events."""

    if request is not None and params:
        raise TypeError("astream_search() accepts either a request object or keyword parameters, not both.")

    runtime = _prepare_search_runtime(request if request is not None else params)
    context = runtime.context
    answer_parts: list[str] = []
    try:
        yield SearchStreamEvent.from_fields(
            type="search_started",
            request_id=context.request_id,
            metrics=snapshot_metrics(context=context),
            warnings=list(context.warnings),
        )
        artifacts = await _collect_search_artifacts_from_runtime(runtime)
        yield SearchStreamEvent.from_fields(
            type="search_completed",
            request_id=context.request_id,
            results=list(artifacts.final_results),
            search_context=artifacts.evidence,
            metrics=snapshot_metrics(context=context),
            warnings=list(context.warnings),
        )
        if context.request.output_mode in {"answer", "answer_with_sources"}:
            yield SearchStreamEvent.from_fields(
                type="answer_started",
                request_id=context.request_id,
                metrics=snapshot_metrics(context=context),
                warnings=list(context.warnings),
            )
            async for text in stream_synthesize_answer(
                context=context,
                llm_provider=artifacts.llm_provider,
                search_context=artifacts.evidence,
            ):
                answer_parts.append(text)
                yield SearchStreamEvent.from_fields(
                    type="answer_delta",
                    request_id=context.request_id,
                    text=text,
                    warnings=list(context.warnings),
                )
            answer = "".join(answer_parts).strip()
            yield SearchStreamEvent.from_fields(
                type="answer_completed",
                request_id=context.request_id,
                answer=answer,
                metrics=snapshot_metrics(context=context),
                warnings=list(context.warnings),
            )
        else:
            answer = None
        response = _build_response(
            context=context,
            answer=answer,
            final_results=artifacts.final_results,
            evidence=artifacts.evidence,
        )
        yield SearchStreamEvent.from_fields(
            type="completed",
            request_id=context.request_id,
            response=response,
            metrics=snapshot_metrics(context=context),
            warnings=list(context.warnings),
        )
    except Exception as error:  # noqa: BLE001
        if isinstance(error, ZoomSearchError):
            raise error
        normalized = normalize_llm_exception(error=error, context=context)
        raise normalized
    finally:
        await runtime.transport_context.aclose()


async def _run_search(request: SearchRequest | dict) -> SearchResponse:
    """Execute the Zoom Search end-to-end workflow."""

    artifacts = await _collect_search_artifacts(request)
    context = artifacts.context
    try:
        answer = None
        if context.request.output_mode in {"answer", "answer_with_sources"}:
            answer = await synthesize_answer(
                context=context,
                llm_provider=artifacts.llm_provider,
                search_context=artifacts.evidence,
            )

        return _build_response(
            context=context,
            answer=answer,
            final_results=artifacts.final_results,
            evidence=artifacts.evidence,
        )
    except Exception as error:  # noqa: BLE001
        if isinstance(error, ZoomSearchError):
            raise error
        normalized = normalize_llm_exception(error=error, context=context)
        raise normalized
    finally:
        await artifacts.transport_context.aclose()


async def _collect_search_artifacts(request: SearchRequest | dict) -> _SearchArtifacts:
    """Run all non-answer-synthesis search phases."""

    runtime = _prepare_search_runtime(request)
    try:
        return await _collect_search_artifacts_from_runtime(runtime)
    except Exception as error:  # noqa: BLE001
        await runtime.transport_context.aclose()
        if isinstance(error, ZoomSearchError):
            raise error
        normalized = normalize_llm_exception(error=error, context=runtime.context)
        raise normalized


def _prepare_search_runtime(request: SearchRequest | dict) -> _SearchRuntime:
    normalized_request = normalize_and_validate_request(request)
    resolved_providers = resolve_providers(normalized_request)
    context = build_runtime_context(
        request=normalized_request,
        llm_provider=resolved_providers.llm,
        search_provider=resolved_providers.search,
    )
    transport_context = create_transport_runtime(context)
    llm_provider = create_llm_provider(context=context, transport_context=transport_context)
    search_provider = create_search_provider(context=context, transport_context=transport_context)
    return _SearchRuntime(
        context=context,
        transport_context=transport_context,
        llm_provider=llm_provider,
        search_provider=search_provider,
    )


async def _collect_search_artifacts_from_runtime(runtime: _SearchRuntime) -> _SearchArtifacts:
    context = runtime.context
    rewrite_result = await rewrite_query_groups(context=context, llm_client=runtime.llm_provider)
    zoom_out_requests = build_zoom_out_requests(search_groups=rewrite_result.search_groups, context=context)
    record_search_planned(context=context, phase="zoom_out", count=len(zoom_out_requests))
    zoom_out_started_at = perf_counter()
    zoom_out_results = await execute_zoom_out_search(
        requests=zoom_out_requests,
        context=context,
        search_provider=runtime.search_provider,
    )
    record_phase_elapsed(context=context, phase="zoom_out_search", started_at=zoom_out_started_at)
    with_domains = attach_source_domains(zoom_out_results)
    source_domains = extract_source_domain_records(
        results=with_domains,
        top_k_domains_per_query=context.request.search_limits.top_k_domains_per_query,
        query1_by_split_question_id={group.group: group.query1 for group in rewrite_result.search_groups},
    )
    zoom_in_requests = build_zoom_in_requests(source_domains=source_domains, context=context)
    record_search_planned(context=context, phase="zoom_in", count=len(zoom_in_requests))
    zoom_in_started_at = perf_counter()
    zoom_in_results = await execute_zoom_in_search(
        requests=zoom_in_requests,
        context=context,
        search_provider=runtime.search_provider,
    )
    record_phase_elapsed(context=context, phase="zoom_in_search", started_at=zoom_in_started_at)
    final_results = build_final_search_results(
        zoom_out_results=with_domains,
        zoom_in_results=zoom_in_results,
    )
    if not final_results:
        raise _all_searches_failed(context=context)

    evidence = format_search_evidence(
        final_results=final_results,
        search_groups=rewrite_result.search_groups,
    )
    return _SearchArtifacts(
        context=context,
        transport_context=runtime.transport_context,
        llm_provider=runtime.llm_provider,
        final_results=final_results,
        evidence=evidence,
    )


def _build_response(*, context, answer: str | None, final_results: list, evidence: str) -> SearchResponse:
    metrics = snapshot_metrics(context=context)
    mode = context.request.output_mode
    raw_diagnostics = dict(context.raw_diagnostics) if context.request.include_raw_diagnostics else None
    optional_fields = {"raw_diagnostics": raw_diagnostics} if raw_diagnostics is not None else {}
    if mode == "answer":
        return SearchResponse.from_fields(
            request_id=context.request_id,
            answer=answer,
            metrics=metrics,
            warnings=list(context.warnings),
            **optional_fields,
        )
    if mode == "answer_with_sources":
        return SearchResponse.from_fields(
            request_id=context.request_id,
            answer=answer,
            results=list(final_results),
            search_context=evidence,
            metrics=metrics,
            warnings=list(context.warnings),
            **optional_fields,
        )
    if mode == "results_simple":
        return SearchResponse.from_fields(
            request_id=context.request_id,
            results=[
                SimpleSearchResult(title=item.title, snippet=item.snippet, url=item.url)
                for item in final_results
            ],
            metrics=metrics,
            warnings=list(context.warnings),
            **optional_fields,
        )
    return SearchResponse.from_fields(
        request_id=context.request_id,
        results=list(final_results),
        metrics=metrics,
        warnings=list(context.warnings),
        **optional_fields,
    )


def _all_searches_failed(*, context) -> Exception:
    last_warning = context.warnings[-1] if context.warnings else None
    reason_code = last_warning.code if isinstance(last_warning, WarningInfo) else "all_search_requests_failed"
    return call_failure(
        category="search_call_failure",
        component="orchestration",
        message="All search requests failed.",
        user_message="All search requests failed.",
        request_id=context.request_id,
        provider_engine=context.search_provider.engine,
        reason_code=reason_code,
        retryable=False,
    )
