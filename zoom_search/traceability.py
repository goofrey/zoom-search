"""Traceability lifecycle helpers."""

from __future__ import annotations

from zoom_search.models import SearchGroup
from zoom_search.models import TraceabilityInfo


def create_initial_traceability(
    *,
    previous_conversation: list[str],
    search_group: SearchGroup,
) -> TraceabilityInfo:
    return TraceabilityInfo(
        previous_conversation=list(previous_conversation),
        original_input=search_group.original_input,
        comparison_question=_normalize_optional_text(search_group.comparison_question),
        split_question=search_group.split_question,
        split_question_id=search_group.group,
    )


def enrich_for_zoom_out_request(
    traceability: TraceabilityInfo,
    *,
    query_variant_id: int,
    search_query_used: str,
) -> TraceabilityInfo:
    return traceability.enriched(
        phase="Zoom-out Search",
        query_variant_id=query_variant_id,
        search_query_used=search_query_used,
    )


def enrich_for_zoom_out_result(
    traceability: TraceabilityInfo,
    *,
    rank: int,
) -> TraceabilityInfo:
    return traceability.enriched(rank=rank)


def enrich_for_zoom_in_request(
    traceability: TraceabilityInfo,
    *,
    search_query_used: str,
) -> TraceabilityInfo:
    return traceability.enriched(
        phase="Zoom-in Search",
        query_variant_id=1,
        search_query_used=search_query_used,
    )


def enrich_for_zoom_in_result(
    traceability: TraceabilityInfo,
    *,
    rank: int,
) -> TraceabilityInfo:
    return traceability.enriched(rank=rank)


def enrich_for_final_result(traceability: TraceabilityInfo) -> TraceabilityInfo:
    return traceability.enriched()


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
