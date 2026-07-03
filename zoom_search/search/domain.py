"""Source domain extraction and deduplication."""

from __future__ import annotations

from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from zoom_search.models import DuplicateTraceabilityInfo
from zoom_search.models import SourceDomainRecord
from zoom_search.models import ZoomOutSearchResult

def attach_source_domains(results: list[ZoomOutSearchResult]) -> list[ZoomOutSearchResult]:
    attached: list[ZoomOutSearchResult] = []
    for result in results:
        normalized_url = normalize_url(result.url)
        source_domain = extract_root_domain(normalized_url) if normalized_url else None
        if source_domain:
            attached.append(
                ZoomOutSearchResult(
                    title=result.title,
                    snippet=result.snippet,
                    url=normalized_url,
                    rank=result.rank,
                    traceability=result.traceability,
                    source_domain=source_domain,
                    provider_score=result.provider_score,
                )
            )
            continue
        attached.append(
            ZoomOutSearchResult(
                title=result.title,
                snippet=result.snippet,
                url=normalized_url or result.url,
                rank=result.rank,
                traceability=result.traceability,
                source_domain=None,
                provider_score=result.provider_score,
            )
        )
    return attached


def extract_source_domain_records(
    *,
    results: list[ZoomOutSearchResult],
    top_k_domains_per_query: int,
    query1_by_split_question_id: dict[int, str],
) -> list[SourceDomainRecord]:
    candidates: list[SourceDomainRecord] = []
    grouped: dict[tuple[int | None, int | None], list[ZoomOutSearchResult]] = {}
    for result in results:
        key = (result.traceability.split_question_id, result.traceability.query_variant_id)
        grouped.setdefault(key, []).append(result)

    for group_results in grouped.values():
        domain_groups: dict[str, list[ZoomOutSearchResult]] = {}
        for result in group_results:
            if result.source_domain:
                domain_groups.setdefault(result.source_domain, []).append(result)

        ranked_domains = sorted(
            (_build_source_domain_record(items, query1_by_split_question_id) for items in domain_groups.values()),
            key=_source_domain_sort_key,
        )
        candidates.extend(ranked_domains[:top_k_domains_per_query])

    ordered = sorted(
        candidates,
        key=lambda item: (
            item.traceability.split_question_id or 0,
            _source_domain_sort_key(item),
            item.traceability.query_variant_id or 0,
            item.source_domain,
        ),
    )
    seen: dict[tuple[str, str], SourceDomainRecord] = {}
    deduped: list[SourceDomainRecord] = []
    for item in ordered:
        key = (item.split_question, item.source_domain)
        existing = seen.get(key)
        if existing:
            existing.duplicate_traceabilities.append(_duplicate_traceability_from_domain(item))
            continue
        seen[key] = item
        deduped.append(item)
    return deduped


def _build_source_domain_record(
    results: list[ZoomOutSearchResult],
    query1_by_split_question_id: dict[int, str],
) -> SourceDomainRecord:
    representative = min(results, key=_zoom_out_result_sort_key)
    duplicate_traceabilities = [_duplicate_traceability_from_result(item) for item in results if item is not representative]
    split_question_id = representative.traceability.split_question_id or 0
    return SourceDomainRecord(
        source_domain=representative.source_domain or "",
        split_question=representative.traceability.split_question or "",
        query1=query1_by_split_question_id.get(split_question_id, representative.traceability.search_query_used or ""),
        rank=min(item.rank for item in results),
        traceability=representative.traceability,
        provider_score=max((item.provider_score for item in results if item.provider_score is not None), default=None),
        duplicate_traceabilities=duplicate_traceabilities,
    )


def _source_domain_sort_key(item: SourceDomainRecord) -> tuple[int, float, int, int, str]:
    score = item.provider_score
    return (
        0 if score is not None else 1,
        -(score if score is not None else 0.0),
        item.rank,
        item.traceability.query_variant_id or 0,
        item.source_domain,
    )


def _zoom_out_result_sort_key(item: ZoomOutSearchResult) -> tuple[int, float, int, str]:
    score = item.provider_score
    return (
        0 if score is not None else 1,
        -(score if score is not None else 0.0),
        item.rank,
        item.source_domain or "",
    )


def _duplicate_traceability_from_result(item: ZoomOutSearchResult) -> DuplicateTraceabilityInfo:
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
        url=item.url,
    )


def _duplicate_traceability_from_domain(item: SourceDomainRecord) -> DuplicateTraceabilityInfo:
    traceability = item.traceability
    return DuplicateTraceabilityInfo(
        phase=traceability.phase,
        split_question=traceability.split_question,
        split_question_id=traceability.split_question_id,
        query_variant_id=traceability.query_variant_id,
        search_query_used=traceability.search_query_used,
        rank=traceability.rank,
        source_domain=item.source_domain,
    )


def normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return ""
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return ""
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = hostname
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    path = parsed.path or ""
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def extract_root_domain(url: str) -> str | None:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return None
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return None
    return hostname
