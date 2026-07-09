from __future__ import annotations

import pytest

from zoom_search.api import astream_search
from zoom_search.api import search
from zoom_search.models import SearchRequest
from zoom_search.models import ZoomSearchError


def _assert_metrics_shape(metrics: dict) -> None:
    assert set(metrics) == {"elapsed_ms", "phase_elapsed_ms", "search_requests", "llm_usage"}
    assert isinstance(metrics["elapsed_ms"], int)
    assert set(metrics["phase_elapsed_ms"]) >= {"query_rewriting", "zoom_out_search", "zoom_in_search"}
    assert set(metrics["search_requests"]) >= {
        "planned",
        "attempted",
        "succeeded",
        "failed",
        "retries",
        "zoom_out_planned",
        "zoom_in_planned",
        "zoom_out_attempted",
        "zoom_in_attempted",
        "zoom_out_succeeded",
        "zoom_in_succeeded",
        "zoom_out_failed",
        "zoom_in_failed",
    }
    assert set(metrics["llm_usage"]) == {"query_rewriting", "answer_synthesis", "total"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output_mode", "expected_keys", "result_kind"),
    [
        ("answer", {"request_id", "metrics", "answer", "warnings"}, "answer"),
        ("answer_with_sources", {"request_id", "metrics", "answer", "results", "search_context", "warnings"}, "detailed"),
        ("results_simple", {"request_id", "metrics", "results", "warnings"}, "simple"),
        ("results_detailed", {"request_id", "metrics", "results", "warnings"}, "detailed"),
    ],
)
async def test_demo_mode_supports_all_output_modes(
    output_mode: str,
    expected_keys: set[str],
    result_kind: str,
) -> None:
    response = await search(
        SearchRequest(
            question="What hotels in Shenzhen have rooms with exercise bikes?",
            demo_mode=True,
            output_mode=output_mode,
            seed=7,
        )
    )

    data = response.to_dict()
    assert set(data) == expected_keys
    _assert_metrics_shape(data["metrics"])
    if result_kind == "answer":
        assert response.answer
    elif result_kind == "simple":
        assert isinstance(data["results"], list)
        assert hasattr(data["results"][0], "title")
        assert hasattr(data["results"][0], "snippet")
        assert hasattr(data["results"][0], "url")
    else:
        assert isinstance(data["results"], list)
        assert hasattr(data["results"][0], "traceability")
    if output_mode == "answer_with_sources":
        assert response.search_context
    if hasattr(response, "answer") and response.answer:
        assert "Sources:" in response.answer


@pytest.mark.asyncio
async def test_demo_mode_metrics_include_usage_counts_and_timing() -> None:
    response = await search(
        SearchRequest(
            question="What hotels in Shenzhen have rooms with exercise bikes?",
            demo_mode=True,
            output_mode="answer_with_sources",
            seed=7,
        )
    )

    assert response.metrics is not None
    _assert_metrics_shape(response.metrics)
    assert response.metrics["search_requests"]["planned"] >= 1
    assert response.metrics["search_requests"]["attempted"] >= response.metrics["search_requests"]["planned"]
    assert response.metrics["llm_usage"]["query_rewriting"]["total_tokens"] == 200
    assert response.metrics["llm_usage"]["answer_synthesis"]["total_tokens"] == 350
    assert response.metrics["llm_usage"]["total"]["total_tokens"] == 550


@pytest.mark.asyncio
async def test_raw_diagnostics_are_opt_in() -> None:
    default_response = await search(
        SearchRequest(
            question="What hotels in Shenzhen have rooms with exercise bikes?",
            demo_mode=True,
            output_mode="answer_with_sources",
            seed=7,
        )
    )

    assert not hasattr(default_response, "raw_diagnostics")

    diagnostic_response = await search(
        SearchRequest(
            question="What hotels in Shenzhen have rooms with exercise bikes?",
            demo_mode=True,
            output_mode="answer_with_sources",
            seed=7,
            include_raw_diagnostics=True,
        )
    )

    assert diagnostic_response.raw_diagnostics is not None
    assert set(diagnostic_response.raw_diagnostics["llm_responses"]) == {"query_rewriting", "answer_synthesis"}
    assert diagnostic_response.raw_diagnostics["llm_responses"]["query_rewriting"]["json_payload"]["search_groups"]
    assert diagnostic_response.raw_diagnostics["llm_responses"]["answer_synthesis"]["json_payload"]["answer"]


@pytest.mark.asyncio
async def test_demo_mode_is_deterministic_for_same_seed() -> None:
    request = SearchRequest(
        question="What hotels in Shenzhen have rooms with exercise bikes?",
        demo_mode=True,
        output_mode="answer_with_sources",
        seed=11,
    )

    first = await search(request)
    second = await search(request)

    assert first.answer == second.answer
    assert first.search_context == second.search_context
    assert [item.url for item in first.results] == [item.url for item in second.results]


@pytest.mark.asyncio
async def test_search_accepts_keyword_parameters() -> None:
    response = await search(
        question="What hotels in Shenzhen have rooms with exercise bikes?",
        demo_mode=True,
        output_mode="answer_with_sources",
        seed=7,
    )

    assert response.answer
    assert response.results
    assert response.search_context


@pytest.mark.asyncio
async def test_query_rewriting_failure_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken_rewrite(*, context, llm_client):
        raise Exception("bad llm output")

    monkeypatch.setattr("zoom_search.api.rewrite_query_groups", broken_rewrite)

    with pytest.raises(Exception) as exc_info:
        await search(SearchRequest(question="Q", demo_mode=True))

    assert exc_info.value.category == "llm_call_failure"
    assert exc_info.value.details.error_type == "provider_error"
    assert exc_info.value.details.reason_code == "provider_call_failed"


@pytest.mark.asyncio
async def test_query_rewriting_validation_error_has_reason_and_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    async def invalid_rewrite(*, context, llm_client):
        from zoom_search.rewriting import _build_result_from_payload
        from zoom_search.rewriting import _parse_llm_response

        payload = _parse_llm_response('{"search_groups": [{}]}', context=context)
        return _build_result_from_payload(payload=payload, context=context, fallback_previous_conversation=[])

    monkeypatch.setattr("zoom_search.api.rewrite_query_groups", invalid_rewrite)

    with pytest.raises(Exception) as exc_info:
        await search(SearchRequest(question="Q", demo_mode=True))

    assert exc_info.value.category == "llm_call_failure"
    assert exc_info.value.details.error_type == "invalid_request_error"
    assert exc_info.value.details.reason_code == "missing_required_search_group_fields"
    assert exc_info.value.request_id != "pending"
    assert "search_groups[0].group" in exc_info.value.details.invalid_fields


@pytest.mark.asyncio
async def test_search_all_failure_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def empty_zoom_out(*, requests, context, search_provider):
        return []

    async def empty_zoom_in(*, requests, context, search_provider):
        return []

    monkeypatch.setattr("zoom_search.api.execute_zoom_out_search", empty_zoom_out)
    monkeypatch.setattr("zoom_search.api.execute_zoom_in_search", empty_zoom_in)

    with pytest.raises(Exception) as exc_info:
        await search(
            SearchRequest(
                question="What hotels in Shenzhen have rooms with exercise bikes?",
                demo_mode=True,
            )
        )

    assert exc_info.value.category == "search_call_failure"


@pytest.mark.asyncio
async def test_partial_success_returns_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    original = __import__("zoom_search.search.zoomin", fromlist=["execute_zoom_in_search"]).execute_zoom_in_search

    async def partial_zoom_in(*, requests, context, search_provider):
        if context.warnings:
            return await original(requests=requests, context=context, search_provider=search_provider)
        context.warnings.append(
            __import__("zoom_search.models", fromlist=["WarningInfo"]).WarningInfo(
                code="search_call_failure",
                message="One zoom-in branch failed.",
                phase="zoomin_search",
                request_id=context.request_id,
                metadata={"retry_attempted": True},
            )
        )
        return []

    monkeypatch.setattr("zoom_search.api.execute_zoom_in_search", partial_zoom_in)

    response = await search(
        SearchRequest(
            question="What hotels in Shenzhen have rooms with exercise bikes?",
            demo_mode=True,
            output_mode="results_detailed",
        )
    )

    assert response.results
    assert response.warnings
    assert response.warnings[0].phase == "zoomin_search"


class _MockResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class _MockAsyncClient:
    def __init__(self, responses: list[_MockResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict | None]] = []

    async def post(self, path: str, *, json: dict, headers: dict[str, str]) -> _MockResponse:
        self.calls.append(("POST", path, json))
        return self._responses.pop(0)

    async def request(self, method: str, endpoint: str, *, params=None, json=None, headers=None):
        self.calls.append((method, endpoint, json if json is not None else params))
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


class _MockStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = list(lines)
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _MockStreamingAsyncClient(_MockAsyncClient):
    def __init__(self, responses: list[_MockResponse], stream_lines: list[str]) -> None:
        super().__init__(responses)
        self.stream_lines = list(stream_lines)

    def stream(self, method: str, path: str, *, json: dict, headers: dict[str, str]) -> _MockStreamResponse:
        self.calls.append((method, path, json))
        return _MockStreamResponse(self.stream_lines)


class _MockTransportContext:
    def __init__(self, llm_client: _MockAsyncClient, search_client: _MockAsyncClient) -> None:
        self.llm_client = llm_client
        self.search_client = search_client

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_real_provider_pair_runs_end_to_end_with_mock_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_client = _MockAsyncClient(
        [
            _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"previous_conversation": [], "search_groups": ['
                                    '{"group": 1, "original_input": "Which is better, Python or Java for web development?", '
                                    '"comparison_question": "Which is better, Python or Java for web development?", '
                                    '"split_question": "Is Python better for web development?", '
                                    '"main_term": "Python", "key_noun": "web development advantage", '
                                    '"alias1": "website development benefit", "alias2": "web app strength", '
                                    '"query1": "Python web development advantage", '
                                    '"query2": "Python website development benefit web app strength"}, '
                                    '{"group": 2, "original_input": "Which is better, Python or Java for web development?", '
                                    '"comparison_question": "Which is better, Python or Java for web development?", '
                                    '"split_question": "Is Java better for web development?", '
                                    '"main_term": "Java", "key_noun": "web development advantage", '
                                    '"alias1": "website development benefit", "alias2": "web app strength", '
                                    '"query1": "Java web development advantage", '
                                    '"query2": "Java website development benefit web app strength"}]}'
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
                }
            ),
            _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"answer": "Python offers faster iteration for many web workflows.[1][2] Java offers stronger ecosystem depth for large enterprise stacks.[3][4]", '
                                    '"sources": [{"id": "[1]", "url": "https://www.python.org/about/"}, {"id": "[2]", "url": "https://docs.python.org/3/"}, {"id": "[3]", "url": "https://www.oracle.com/java/"}, {"id": "[4]", "url": "https://jakarta.ee/"}]}'
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 25, "total_tokens": 65},
                }
            ),
        ]
    )
    search_client = _MockAsyncClient(
        [
            _MockResponse({"results": [{"title": "Python", "content": "Python web strengths", "url": "https://www.python.org/about/"}]}),
            _MockResponse({"results": [{"title": "Python Guide", "content": "Python ecosystem", "url": "https://docs.python.org/3/"}]}),
            _MockResponse({"results": [{"title": "Java", "content": "Java web strengths", "url": "https://www.oracle.com/java/"}]}),
            _MockResponse({"results": [{"title": "Jakarta", "content": "Java ecosystem", "url": "https://jakarta.ee/"}]}),
            _MockResponse({"results": [{"title": "Python Zoom In", "content": "Python web framework details", "url": "https://www.python.org/about/apps/"}]}),
            _MockResponse({"results": [{"title": "Java Zoom In", "content": "Java web framework details", "url": "https://www.oracle.com/java/technologies/"}]}),
        ]
    )

    monkeypatch.setattr(
        "zoom_search.api.create_transport_runtime",
        lambda context: _MockTransportContext(llm_client=llm_client, search_client=search_client),
    )

    response = await search(
        question="Which is better, Python or Java for web development?",
        llm_engine="gemini",
        llm_model="gemini-2.5-flash",
        llm_api_key="secret",
        search_engine="tavily",
        search_api_key="search-secret",
        output_mode="answer_with_sources",
    )

    assert response.answer
    assert response.results
    assert response.search_context
    assert any(item.traceability.comparison_question for item in response.results)
    assert len([call for call in llm_client.calls if call[1] == "/chat/completions"]) == 2
    assert "Sources:" in response.answer
    assert "[1] https://www.python.org/about/" in response.answer
    assert "[4] https://jakarta.ee/" in response.answer
    assert response.metrics is not None
    _assert_metrics_shape(response.metrics)
    assert response.metrics["llm_usage"]["total"]["total_tokens"] == 115


@pytest.mark.asyncio
async def test_answer_synthesis_empty_answer_returns_llm_empty_output_error(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_client = _MockAsyncClient(
        [
            _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"previous_conversation": [], "search_groups": ['
                                    '{"group": 1, "original_input": "Which is better, Python or Java for web development?", '
                                    '"comparison_question": "", '
                                    '"split_question": "Is Python better for web development?", '
                                    '"main_term": "Python", "key_noun": "web development advantage", '
                                    '"alias1": "website development benefit", "alias2": "web app strength", '
                                    '"query1": "Python web development advantage", '
                                    '"query2": "Python website development benefit web app strength"}]}'
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
                }
            ),
            _MockResponse(
                {
                    "choices": [
                        {
                            "message": {"content": '{"answer": "", "sources": []}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 1, "total_tokens": 41},
                }
            ),
        ]
    )
    search_client = _MockAsyncClient(
        [
            _MockResponse({"results": [{"title": "Python", "content": "Python web strengths", "url": "https://www.python.org/about/"}]}),
            _MockResponse({"results": [{"title": "Python Guide", "content": "Python ecosystem", "url": "https://docs.python.org/3/"}]}),
            _MockResponse({"results": [{"title": "Python Zoom In", "content": "Python web framework details", "url": "https://www.python.org/about/apps/"}]}),
            _MockResponse({"results": [{"title": "Python Docs Zoom In", "content": "Python docs details", "url": "https://docs.python.org/3/tutorial/"}]}),
        ]
    )
    monkeypatch.setattr(
        "zoom_search.api.create_transport_runtime",
        lambda context: _MockTransportContext(llm_client=llm_client, search_client=search_client),
    )

    with pytest.raises(ZoomSearchError) as caught:
        await search(
            question="Which is better, Python or Java for web development?",
            llm_engine="gemini",
            llm_model="gemini-2.5-flash",
            llm_api_key="secret",
            search_engine="tavily",
            search_api_key="search-secret",
            output_mode="answer",
        )

    assert caught.value.category == "llm_call_failure"
    assert caught.value.details.reason_code == "empty_output"


@pytest.mark.asyncio
async def test_custom_search_provider_runs_end_to_end_with_mock_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_client = _MockAsyncClient(
        [
            _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"previous_conversation": [], "search_groups": ['
                                    '{"group": 1, "original_input": "Which is better, Python or Java for web development?", '
                                    '"comparison_question": "Which is better, Python or Java for web development?", '
                                    '"split_question": "Is Python better for web development?", '
                                    '"main_term": "Python", "key_noun": "web development advantage", '
                                    '"alias1": "website development benefit", "alias2": "web app strength", '
                                    '"query1": "Python web development advantage", '
                                    '"query2": "Python website development benefit web app strength"}, '
                                    '{"group": 2, "original_input": "Which is better, Python or Java for web development?", '
                                    '"comparison_question": "Which is better, Python or Java for web development?", '
                                    '"split_question": "Is Java better for web development?", '
                                    '"main_term": "Java", "key_noun": "web development advantage", '
                                    '"alias1": "website development benefit", "alias2": "web app strength", '
                                    '"query1": "Java web development advantage", '
                                    '"query2": "Java website development benefit web app strength"}]}'
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
                }
            ),
            _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"answer": "Python offers faster iteration for many web workflows.[1][2] Java offers stronger ecosystem depth for large enterprise stacks.[3][4]", '
                                    '"sources": [{"id": "[1]", "url": "https://www.python.org/about/"}, {"id": "[2]", "url": "https://docs.python.org/3/"}, {"id": "[3]", "url": "https://www.oracle.com/java/"}, {"id": "[4]", "url": "https://jakarta.ee/"}]}'
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 25, "total_tokens": 65},
                }
            ),
        ]
    )
    search_client = _MockAsyncClient(
        [
            _MockResponse({"data": {"items": [{"name": "Python", "summary": "Python web strengths", "link": "https://www.python.org/about/"}]}}),
            _MockResponse({"data": {"items": [{"name": "Python Guide", "summary": "Python ecosystem", "link": "https://docs.python.org/3/"}]}}),
            _MockResponse({"data": {"items": [{"name": "Java", "summary": "Java web strengths", "link": "https://www.oracle.com/java/"}]}}),
            _MockResponse({"data": {"items": [{"name": "Jakarta", "summary": "Java ecosystem", "link": "https://jakarta.ee/"}]}}),
            _MockResponse({"data": {"items": [{"name": "Python Zoom In", "summary": "Python web framework details", "link": "https://www.python.org/about/apps/"}]}}),
            _MockResponse({"data": {"items": [{"name": "Java Zoom In", "summary": "Java web framework details", "link": "https://www.oracle.com/java/technologies/"}]}}),
        ]
    )

    monkeypatch.setattr(
        "zoom_search.api.create_transport_runtime",
        lambda context: _MockTransportContext(llm_client=llm_client, search_client=search_client),
    )

    response = await search(
        question="Which is better, Python or Java for web development?",
        llm_engine="gemini",
        llm_model="gemini-2.5-flash",
        llm_api_key="secret",
        search_engine="custom",
        search_base_url="https://search.example.com/api",
        search_api_key="search-secret",
        search_result_collection_path="data.items",
        search_title_fields=["name", "title"],
        search_snippet_fields=["summary", "snippet"],
        search_url_fields=["link", "url"],
        output_mode="answer_with_sources",
    )

    assert response.answer
    assert response.results
    assert response.search_context
    assert any(item.traceability.comparison_question for item in response.results)
    assert all(call[1] == "https://search.example.com/api" for call in search_client.calls)
    assert search_client.calls[0] == (
        "POST",
        "https://search.example.com/api",
        {"query": "Python web development advantage"},
    )
    assert response.results[0].title == "Python"
    assert response.results[0].snippet == "Python web strengths"
    assert response.results[0].url == "https://www.python.org/about/"
    assert response.metrics is not None
    _assert_metrics_shape(response.metrics)
    assert response.metrics["llm_usage"]["total"]["total_tokens"] == 115


@pytest.mark.asyncio
async def test_astream_search_streams_answer_with_sources_demo_mode() -> None:
    events = [
        event
        async for event in astream_search(
            question="What hotels in Shenzhen have rooms with exercise bikes?",
            demo_mode=True,
            output_mode="answer_with_sources",
            seed=7,
        )
    ]

    assert [event.type for event in events] == [
        "search_started",
        "search_completed",
        "answer_started",
        "answer_delta",
        "answer_completed",
        "completed",
    ]
    assert events[1].results
    assert events[1].search_context
    assert events[1].metrics is not None
    _assert_metrics_shape(events[1].metrics)
    assert events[3].text
    assert events[4].metrics is not None
    _assert_metrics_shape(events[4].metrics)
    assert events[-1].response.answer == events[4].answer
    assert events[-1].response.results
    assert events[-1].response.search_context
    assert events[-1].response.metrics is not None
    assert events[-1].response.metrics["llm_usage"]["total"]["total_tokens"] == 550


@pytest.mark.asyncio
async def test_astream_search_uses_provider_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_client = _MockStreamingAsyncClient(
        [
            _MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"previous_conversation": [], "search_groups": ['
                                    '{"group": 1, "original_input": "Which is better, Python or Java for web development?", '
                                    '"comparison_question": "", '
                                    '"split_question": "Is Python better for web development?", '
                                    '"main_term": "Python", "key_noun": "web development advantage", '
                                    '"alias1": "website development benefit", "alias2": "web app strength", '
                                    '"query1": "Python web development advantage", '
                                    '"query2": "Python website development benefit web app strength"}]}'
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
        ],
        [
            'data: {"choices":[{"delta":{"content":"{\\"answer\\":\\"Python wins.[1]\\",\\"sources\\":[{\\"id\\":\\"[1]\\",\\"url\\":\\"https://www.python.org/about/\\"}]}"}}]}',
            'data: {"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}',
            "data: [DONE]",
        ],
    )
    search_client = _MockAsyncClient(
        [
            _MockResponse({"results": [{"title": "Python", "content": "Python web strengths", "url": "https://www.python.org/about/"}]}),
            _MockResponse({"results": [{"title": "Python Guide", "content": "Python ecosystem", "url": "https://docs.python.org/3/"}]}),
            _MockResponse({"results": [{"title": "Python Zoom In", "content": "Python web framework details", "url": "https://www.python.org/about/apps/"}]}),
        ]
    )
    monkeypatch.setattr(
        "zoom_search.api.create_transport_runtime",
        lambda context: _MockTransportContext(llm_client=llm_client, search_client=search_client),
    )

    events = [
        event
        async for event in astream_search(
            question="Which is better, Python or Java for web development?",
            llm_engine="gemini",
            llm_model="gemini-2.5-flash",
            llm_api_key="secret",
            search_engine="tavily",
            search_api_key="search-secret",
            output_mode="answer",
        )
    ]

    deltas = [event.text for event in events if event.type == "answer_delta"]
    assert deltas == ["Python wins.[1]", "\n\nSources:\n[1] https://www.python.org/about/"]
    assert events[-1].response.answer == "Python wins.[1]\n\nSources:\n[1] https://www.python.org/about/"
    assert events[-1].response.metrics is not None
    assert events[-1].response.metrics["llm_usage"]["answer_synthesis"]["total_tokens"] == 18
    assert any(call[0] == "POST" and call[1] == "/chat/completions" and call[2].get("stream") is True for call in llm_client.calls)
