"""Final results assembly and deterministic URL deduplication."""

from __future__ import annotations

from zoom_search.models import DuplicateTraceabilityInfo
from zoom_search.models import FinalSearchResult
from zoom_search.models import ZoomInSearchResult
from zoom_search.models import ZoomOutSearchResult
from zoom_search.search.domain import normalize_url
from zoom_search.traceability import enrich_for_final_result


def build_final_search_results(
    *,
    zoom_out_results: list[ZoomOutSearchResult],
    zoom_in_results: list[ZoomInSearchResult],
) -> list[FinalSearchResult]:
    candidates = [
        *[
            FinalSearchResult(
                title=item.title,
                snippet=item.snippet,
                url=item.url,
                source_domain=item.source_domain,
                traceability=enrich_for_final_result(item.traceability),
            )
            for item in zoom_out_results
        ],
        *[
            FinalSearchResult(
                title=item.title,
                snippet=item.snippet,
                url=item.url,
                source_domain=item.source_domain,
                traceability=enrich_for_final_result(item.traceability),
            )
            for item in zoom_in_results
        ],
    ]
    ordered = sorted(candidates, key=_final_result_order_key)
    deduped: list[FinalSearchResult] = []
    seen: dict[str, FinalSearchResult] = {}
    for item in ordered:
        normalized_url = normalize_url(item.url)
        if not normalized_url:
            continue
        existing = seen.get(normalized_url)
        if existing:
            existing.duplicate_traceabilities.append(_duplicate_traceability_from_result(item, normalized_url))
            continue
        result = FinalSearchResult(
            title=item.title,
            snippet=item.snippet,
            url=normalized_url,
            source_domain=item.source_domain,
            traceability=item.traceability,
        )
        seen[normalized_url] = result
        deduped.append(result)
    return deduped


def _final_result_order_key(item: FinalSearchResult) -> tuple[int, int, int, int]:
    traceability = item.traceability
    return (
        0 if traceability.phase == "Zoom-out Search" else 1,
        traceability.split_question_id or 0,
        traceability.query_variant_id or 0,
        traceability.rank or 0,
    )


def _duplicate_traceability_from_result(item: FinalSearchResult, normalized_url: str) -> DuplicateTraceabilityInfo:
    traceability = item.traceability
    return DuplicateTraceabilityInfo(
        phase=traceability.phase,
        split_question=traceability.split_question,
        split_question_id=traceability.split_question_id,
        query_variant_id=traceability.query_variant_id,
        search_query_used=traceability.search_query_used,
        rank=traceability.rank,
        source_domain=item.source_domain,
        title=item.title,
        url=normalized_url,
    )
