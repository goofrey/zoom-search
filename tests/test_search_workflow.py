from __future__ import annotations

import pytest

from zoom_search.evidence import format_search_evidence
from zoom_search.models import ProviderCapability
from zoom_search.models import ResolvedProvider
from zoom_search.models import SearchLimits
from zoom_search.models import SearchRequest
from zoom_search.models import UnifiedSearchResponse
from zoom_search.models import UnifiedSearchResult
from zoom_search.models import WarningInfo
from zoom_search.orchestration.runtime import build_runtime_context
from zoom_search.results import build_final_search_results
from zoom_search.retry import retry_attempts
from zoom_search.retry import retry_delay_seconds
from zoom_search.retry import should_retry_error
from zoom_search.search.domain import attach_source_domains
from zoom_search.search.domain import extract_source_domain_records
from zoom_search.search.zoom_out import build_zoom_out_requests
from zoom_search.search.zoom_out import execute_zoom_out_search
from zoom_search.search.zoomin import build_zoom_in_requests
from zoom_search.search.zoomin import execute_zoom_in_search
from zoom_search.traceability import create_initial_traceability
from zoom_search.rewriting import QueryRewriteGroup
from zoom_search.models import SearchGroup


class StubSearchProvider:
    def __init__(self, responses: dict[str, list[UnifiedSearchResult]]) -> None:
        self.responses = responses
        self.calls = []

    async def search(self, request):
        self.calls.append(request)
        return UnifiedSearchResponse(results=list(self.responses.get(request.query, [])))


class WarningSearchProvider:
    async def search(self, request):
        return UnifiedSearchResponse(
            results=[UnifiedSearchResult(title="A", snippet="a", url="https://example.com/a")],
            normalized_warnings=[WarningInfo(code="provider_warning", message="provider warning", phase=request.task)],
        )


class RankedStubSearchProvider:
    def __init__(self, responses: dict[str, list[UnifiedSearchResult]]) -> None:
        self.responses = responses

    async def search(self, request):
        results = sorted(
            self.responses.get(request.query, []),
            key=lambda item: (
                item.raw_item.get("position", item.provider_result_index) if isinstance(item.raw_item, dict) else item.provider_result_index,
                item.provider_result_index,
            ),
        )
        return UnifiedSearchResponse(results=results)


class FlakySearchProvider:
    def __init__(self, results: list[UnifiedSearchResult]) -> None:
        self.results = results
        self.calls = []

    async def search(self, request):
        self.calls.append(request)
        if len(self.calls) == 1:
            return UnifiedSearchResponse(results=[])
        return UnifiedSearchResponse(results=list(self.results))


class FailingSearchProvider:
    async def search(self, request):
        raise RuntimeError("boom")


class ConcurrencyTrackingSearchProvider:
    def __init__(self, results: list[UnifiedSearchResult]) -> None:
        self.results = results
        self.active = 0
        self.max_active = 0

    async def search(self, request):
        import asyncio

        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return UnifiedSearchResponse(results=list(self.results))


def make_context(*, engine: str = "tavily", provider_kind: str = "builtin"):
    request = SearchRequest(
        question="原问题",
        search_limits=SearchLimits(zoomout_num_results=2, zoomin_num_results=2, top_k_domains_per_query=1),
        demo_mode=True,
    )
    llm_provider = ResolvedProvider(
        engine="demo_llm_provider",
        provider_kind="demo",
        component="llm",
        capability=ProviderCapability(
            engine="demo_llm_provider",
            provider_kind="demo",
            protocol="openai_compatible",
        ),
        protocol="openai_compatible",
    )
    search_provider = ResolvedProvider(
        engine=engine,
        provider_kind=provider_kind,
        component="search",
        capability=ProviderCapability(
            engine=engine,
            provider_kind=provider_kind,
            supports_site_restriction=engine == "tavily",
            site_restriction_mode="provider_side" if engine == "tavily" else "query_side",
        ),
    )
    return build_runtime_context(request=request, llm_provider=llm_provider, search_provider=search_provider)


def make_group(*, group: int, original_input: str, comparison_question: str | None, split_question: str, query1: str, query2: str) -> QueryRewriteGroup:
    search_group = SearchGroup(
        group=group,
        original_input=original_input,
        comparison_question=comparison_question,
        split_question=split_question,
        main_term="main",
        key_noun="noun",
        alias1="alias1",
        alias2="alias2",
        query1=query1,
        query2=query2,
    )
    return QueryRewriteGroup(
        search_group=search_group,
        traceability=create_initial_traceability(previous_conversation=[], search_group=search_group),
    )


@pytest.mark.asyncio
async def test_zoom_out_request_count_and_order() -> None:
    context = make_context()
    groups = [
        make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1-a", query2="q2-a"),
        make_group(group=2, original_input="Q2", comparison_question=None, split_question="S2", query1="q1-b", query2="q2-b"),
    ]
    requests = build_zoom_out_requests(search_groups=groups, context=context)

    assert [(item.split_question_id, item.query_variant_id, item.query) for item in requests] == [
        (1, 1, "q1-a"),
        (1, 2, "q2-a"),
        (2, 1, "q1-b"),
        (2, 2, "q2-b"),
    ]


@pytest.mark.asyncio
async def test_source_domain_extraction_and_dedup_prefers_higher_provider_score() -> None:
    context = make_context()
    groups = [
        make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1", query2="q2"),
    ]
    requests = build_zoom_out_requests(search_groups=groups, context=context)
    provider = StubSearchProvider(
        {
            "q1": [
                UnifiedSearchResult(
                    title="A",
                    snippet="a",
                    url="https://www.trip.com/path",
                    provider_result_index=0,
                    raw_item={"score": 0.61},
                ),
                UnifiedSearchResult(
                    title="B",
                    snippet="b",
                    url="https://sub.booking.com/abc",
                    provider_result_index=1,
                    raw_item={"score": 0.82},
                ),
            ],
            "q2": [
                UnifiedSearchResult(
                    title="C",
                    snippet="c",
                    url="https://m.trip.com/other",
                    provider_result_index=0,
                    raw_item={"score": 0.88},
                ),
                UnifiedSearchResult(
                    title="D",
                    snippet="d",
                    url="https://www.agoda.com/x",
                    provider_result_index=1,
                    raw_item={"score": 0.54},
                ),
            ],
        }
    )

    zoom_out = await execute_zoom_out_search(requests=requests, context=context, search_provider=provider)
    with_domains = attach_source_domains(zoom_out)
    domain_records = extract_source_domain_records(
        results=with_domains,
        top_k_domains_per_query=1,
        query1_by_split_question_id={1: "q1"},
    )

    assert [item.source_domain for item in domain_records] == ["m.trip.com", "sub.booking.com"]
    assert [item.provider_score for item in domain_records] == [0.88, 0.82]
    assert [item.query1 for item in domain_records] == ["q1", "q1"]
    assert domain_records[0].traceability.query_variant_id == 2
    assert domain_records[0].duplicate_traceabilities == []
    assert domain_records[1].traceability.query_variant_id == 1
    assert domain_records[1].duplicate_traceabilities == []


def test_source_domain_dedup_across_queries_keeps_best_scored_record() -> None:
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1", query2="q2")
    with_domains = attach_source_domains(
        [
            __import__("zoom_search.models", fromlist=["ZoomOutSearchResult"]).ZoomOutSearchResult(
                title="Q1 docs",
                snippet="a",
                url="https://docs.example.com/a",
                rank=2,
                traceability=group.traceability.enriched(
                    phase="Zoom-out Search",
                    query_variant_id=1,
                    rank=2,
                    search_query_used="q1",
                ),
                provider_score=0.72,
            ),
            __import__("zoom_search.models", fromlist=["ZoomOutSearchResult"]).ZoomOutSearchResult(
                title="Q2 docs",
                snippet="b",
                url="https://docs.example.com/b",
                rank=3,
                traceability=group.traceability.enriched(
                    phase="Zoom-out Search",
                    query_variant_id=2,
                    rank=3,
                    search_query_used="q2",
                ),
                provider_score=0.91,
            ),
        ]
    )

    domain_records = extract_source_domain_records(
        results=with_domains,
        top_k_domains_per_query=1,
        query1_by_split_question_id={1: "q1"},
    )

    assert len(domain_records) == 1
    assert domain_records[0].source_domain == "docs.example.com"
    assert domain_records[0].provider_score == 0.91
    assert domain_records[0].query1 == "q1"
    assert domain_records[0].traceability.query_variant_id == 2
    assert len(domain_records[0].duplicate_traceabilities) == 1


@pytest.mark.asyncio
async def test_zoom_out_ranks_after_provider_ordering_before_local_truncation() -> None:
    context = make_context(engine="serpapi")
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1", query2="q2")
    provider = RankedStubSearchProvider(
        {
            "q1": [
                UnifiedSearchResult(title="Second", snippet="b", url="https://example.com/b", provider_result_index=0, raw_item={"position": 2}),
                UnifiedSearchResult(title="First", snippet="a", url="https://example.com/a", provider_result_index=1, raw_item={"position": 1}),
                UnifiedSearchResult(title="Third", snippet="c", url="https://example.com/c", provider_result_index=2, raw_item={"position": 3}),
            ],
            "q2": [],
        }
    )

    requests = build_zoom_out_requests(search_groups=[group], context=context)
    results = await execute_zoom_out_search(requests=[requests[0]], context=context, search_provider=provider)

    assert [(item.title, item.rank) for item in results] == [("First", 1), ("Second", 2)]


@pytest.mark.asyncio
async def test_zoom_out_uses_retry_budget_for_retryable_empty_results() -> None:
    context = make_context()
    context.retry_budget["search"] = 1
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1", query2="q2")
    requests = build_zoom_out_requests(search_groups=[group], context=context)
    provider = FlakySearchProvider([UnifiedSearchResult(title="Recovered", snippet="ok", url="https://example.com")])

    results = await execute_zoom_out_search(requests=[requests[0]], context=context, search_provider=provider)

    assert len(provider.calls) == 2
    assert [item.title for item in results] == ["Recovered"]


@pytest.mark.asyncio
async def test_zoom_out_records_provider_normalized_warnings() -> None:
    context = make_context()
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1", query2="q2")
    requests = build_zoom_out_requests(search_groups=[group], context=context)

    results = await execute_zoom_out_search(requests=[requests[0]], context=context, search_provider=WarningSearchProvider())

    assert [item.title for item in results] == ["A"]
    assert any(item.code == "provider_warning" for item in context.warnings)


@pytest.mark.asyncio
async def test_zoom_out_respects_search_request_semaphore_limit() -> None:
    context = make_context()
    context.semaphore_limits["search_requests"] = 1
    groups = [
        make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1", query2="q2"),
        make_group(group=2, original_input="Q2", comparison_question=None, split_question="S2", query1="q3", query2="q4"),
    ]
    requests = build_zoom_out_requests(search_groups=groups, context=context)
    provider = ConcurrencyTrackingSearchProvider([UnifiedSearchResult(title="A", snippet="a", url="https://example.com/a")])

    results = await execute_zoom_out_search(requests=requests, context=context, search_provider=provider)

    assert len(results) == 4
    assert provider.max_active == 1


@pytest.mark.asyncio
async def test_failed_search_warning_reports_actual_retry_attempted() -> None:
    context = make_context()
    context.retry_budget["search"] = 0
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1", query2="q2")
    requests = build_zoom_out_requests(search_groups=[group], context=context)

    results = await execute_zoom_out_search(requests=[requests[0]], context=context, search_provider=FailingSearchProvider())

    assert results == []
    assert context.warnings[-1].metadata["retry_attempted"] is False


def test_retry_policy_uses_error_type_and_budget() -> None:
    context = make_context()
    rate_limit = __import__("zoom_search.errors", fromlist=["call_failure"]).call_failure(
        category="llm_call_failure",
        component="llm",
        message="limited",
        user_message="limited",
        request_id=context.request_id,
        reason_code="rate_limited",
    )
    invalid = __import__("zoom_search.errors", fromlist=["call_failure"]).call_failure(
        category="llm_call_failure",
        component="llm",
        message="bad",
        user_message="bad",
        request_id=context.request_id,
        reason_code="invalid_request",
    )

    assert list(retry_attempts(context=context, component="llm")) == [0, 1]
    assert should_retry_error(error=rate_limit, component="llm") is True
    assert should_retry_error(error=invalid, component="llm") is False
    assert retry_delay_seconds(attempt=1) > retry_delay_seconds(attempt=0)


def test_final_url_dedup_order_is_deterministic() -> None:
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="S1", query1="q1", query2="q2")
    zoom_out = attach_source_domains(
        [
            __import__("zoom_search.models", fromlist=["ZoomOutSearchResult"]).ZoomOutSearchResult(
                title="ZoomOut",
                snippet="first",
                url="https://example.com/a#frag",
                rank=1,
                traceability=group.traceability.enriched(phase="Zoom-out Search", query_variant_id=1, rank=1, search_query_used="q1"),
            )
        ]
    )
    zoom_in = [
        __import__("zoom_search.models", fromlist=["ZoomInSearchResult"]).ZoomInSearchResult(
            title="ZoomIn",
            snippet="second",
            url="https://example.com/a",
            rank=1,
            traceability=group.traceability.enriched(phase="Zoom-in Search", query_variant_id=1, rank=1, search_query_used="site:example.com S1"),
            source_domain="example.com",
        )
    ]

    final_results = build_final_search_results(zoom_out_results=zoom_out, zoom_in_results=zoom_in)

    assert len(final_results) == 1
    assert final_results[0].title == "ZoomOut"
    assert len(final_results[0].duplicate_traceabilities) == 1
    assert final_results[0].duplicate_traceabilities[0].phase == "Zoom-in Search"
    assert final_results[0].duplicate_traceabilities[0].title == "ZoomIn"
    assert final_results[0].duplicate_traceabilities[0].url == "https://example.com/a"


@pytest.mark.asyncio
async def test_query_side_site_restriction_records_search_query_used() -> None:
    context = make_context(engine="brave")
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="What is S1?", query1="q1", query2="q2")
    provider = StubSearchProvider({})
    zoom_out_requests = build_zoom_out_requests(search_groups=[group], context=context)
    zoom_out = await execute_zoom_out_search(requests=zoom_out_requests, context=context, search_provider=StubSearchProvider({"q1": [], "q2": []}))
    domain_records = [
        __import__("zoom_search.models", fromlist=["SourceDomainRecord"]).SourceDomainRecord(
            source_domain="trip.com",
            split_question="What is S1?",
            query1="q1",
            rank=1,
            traceability=group.traceability.enriched(
                phase="Zoom-out Search",
                query_variant_id=1,
                rank=1,
                search_query_used="q1",
            ),
        )
    ]

    zoom_in_requests = build_zoom_in_requests(source_domains=domain_records, context=context)
    assert zoom_in_requests[0].query == "site:trip.com q1"
    assert zoom_in_requests[0].traceability.query_variant_id == 1
    assert zoom_in_requests[0].traceability.search_query_used == "site:trip.com q1"

    await execute_zoom_in_search(requests=zoom_in_requests, context=context, search_provider=provider)
    assert provider.calls[0].site_restriction_mode == "query_side"


@pytest.mark.asyncio
async def test_zoom_in_domain_filtering_adds_warning_and_preserves_traceability() -> None:
    context = make_context(engine="you")
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="What is S1?", query1="q1", query2="q2")
    domain_records = [
        __import__("zoom_search.models", fromlist=["SourceDomainRecord"]).SourceDomainRecord(
            source_domain="trip.com",
            split_question="What is S1?",
            query1="q1",
            rank=1,
            traceability=group.traceability.enriched(
                phase="Zoom-out Search",
                query_variant_id=1,
                rank=1,
                search_query_used="q1",
            ),
        )
    ]
    zoom_in_requests = build_zoom_in_requests(source_domains=domain_records, context=context)
    provider = StubSearchProvider(
        {
            "site:trip.com q1": [
                UnifiedSearchResult(title="Keep", snippet="a", url="https://trip.com/ok", provider_result_index=0),
                UnifiedSearchResult(title="Drop", snippet="b", url="https://booking.com/no", provider_result_index=1),
            ]
        }
    )

    results = await execute_zoom_in_search(requests=zoom_in_requests, context=context, search_provider=provider)

    assert [item.url for item in results] == ["https://trip.com/ok"]
    assert results[0].traceability.query_variant_id == 1
    assert results[0].traceability.search_query_used == "site:trip.com q1"
    assert results[0].source_domain == "trip.com"
    assert any(item.code == "zoom_in_domain_filtered_results" for item in context.warnings)


@pytest.mark.asyncio
async def test_zoom_in_filtered_to_empty_continues_global_flow() -> None:
    context = make_context(engine="metasota")
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="What is S1?", query1="q1", query2="q2")
    domain_records = [
        __import__("zoom_search.models", fromlist=["SourceDomainRecord"]).SourceDomainRecord(
            source_domain="trip.com",
            split_question="What is S1?",
            query1="q1",
            rank=1,
            traceability=group.traceability.enriched(
                phase="Zoom-out Search",
                query_variant_id=1,
                rank=1,
                search_query_used="q1",
            ),
        )
    ]
    zoom_in_requests = build_zoom_in_requests(source_domains=domain_records, context=context)
    provider = StubSearchProvider(
        {
            "site:trip.com q1": [
                UnifiedSearchResult(title="Drop", snippet="b", url="https://booking.com/no", provider_result_index=0),
            ]
        }
    )

    results = await execute_zoom_in_search(requests=zoom_in_requests, context=context, search_provider=provider)

    assert results == []
    assert any(item.code == "zoom_in_filtered_to_empty" for item in context.warnings)


@pytest.mark.asyncio
async def test_broad_and_zoom_in_keep_different_query_behavior() -> None:
    context = make_context(engine="serper")
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="What is S1?", query1="zoomout query1", query2="zoomout query2")

    zoom_out_requests = build_zoom_out_requests(search_groups=[group], context=context)
    assert zoom_out_requests[0].query == "zoomout query1"
    assert zoom_out_requests[0].traceability.search_query_used == "zoomout query1"

    domain_records = [
        __import__("zoom_search.models", fromlist=["SourceDomainRecord"]).SourceDomainRecord(
            source_domain="trip.com",
            split_question="What is S1?",
            query1="zoomout query1",
            rank=1,
            traceability=group.traceability.enriched(
                phase="Zoom-out Search",
                query_variant_id=1,
                rank=1,
                search_query_used="zoomout query1",
            ),
        )
    ]
    zoom_in_requests = build_zoom_in_requests(source_domains=domain_records, context=context)

    assert zoom_in_requests[0].query == "site:trip.com zoomout query1"
    assert zoom_in_requests[0].traceability.query_variant_id == 1
    assert zoom_in_requests[0].traceability.search_query_used == "site:trip.com zoomout query1"


@pytest.mark.asyncio
async def test_zoom_in_traceability_resets_query_variant_to_query1() -> None:
    context = make_context(engine="brave")
    group = make_group(group=1, original_input="Q1", comparison_question=None, split_question="What is S1?", query1="q1", query2="q2")
    domain_records = [
        __import__("zoom_search.models", fromlist=["SourceDomainRecord"]).SourceDomainRecord(
            source_domain="trip.com",
            split_question="What is S1?",
            query1="q1",
            rank=1,
            traceability=group.traceability.enriched(
                phase="Zoom-out Search",
                query_variant_id=2,
                rank=1,
                search_query_used="q2",
            ),
        )
    ]

    zoom_in_requests = build_zoom_in_requests(source_domains=domain_records, context=context)

    assert zoom_in_requests[0].traceability.query_variant_id == 1
    assert zoom_in_requests[0].traceability.search_query_used == "site:trip.com q1"


def test_comparison_branch_evidence_grouping_and_empty_branch_output() -> None:
    groups = [
        make_group(group=1, original_input="Compare A and B", comparison_question="A vs B?", split_question="A branch", query1="a1", query2="a2"),
        make_group(group=2, original_input="Compare A and B", comparison_question="A vs B?", split_question="B branch", query1="b1", query2="b2"),
        make_group(group=3, original_input="Single question", comparison_question=None, split_question="Single branch", query1="s1", query2="s2"),
    ]
    final_results = [
        __import__("zoom_search.models", fromlist=["FinalSearchResult"]).FinalSearchResult(
            title="A result",
            snippet="snippet",
            url="https://a.example.com",
            source_domain="example.com",
            traceability=groups[0].traceability.enriched(
                phase="Zoom-out Search",
                query_variant_id=1,
                rank=1,
            ),
        )
    ]

    evidence = format_search_evidence(final_results=final_results, search_groups=groups)

    assert "#### Comparison Question: A vs B?" in evidence
    assert "Comparison Branch 1:" in evidence
    assert "Comparison Branch 2:" in evidence
    assert "Evidence: No evidence found." in evidence
    assert "#### Non-comparison Search Branch" in evidence
