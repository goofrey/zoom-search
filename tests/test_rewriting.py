from __future__ import annotations

import pytest

from zoom_search.models import ProviderCapability
from zoom_search.providers.capabilities import BUILTIN_LLM_CAPABILITIES
from zoom_search.models import ResolvedProvider
from zoom_search.models import SearchRequest
from zoom_search.models import TraceabilityInfo
from zoom_search.orchestration.runtime import build_runtime_context
from zoom_search.rewriting import QueryRewriteResult
from zoom_search.rewriting import build_query_rewriting_prompt
from zoom_search.rewriting import rewrite_query_groups
from zoom_search.traceability import enrich_for_final_result
from zoom_search.traceability import enrich_for_zoom_in_request
from zoom_search.traceability import enrich_for_zoom_in_result
from zoom_search.traceability import enrich_for_zoom_out_request
from zoom_search.traceability import enrich_for_zoom_out_result


class StubLLMClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def generate_json(self, *, prompt: str, context, json_strategy: str) -> object:
        self.calls.append(
            {
                "prompt": prompt,
                "context": context,
                "json_strategy": json_strategy,
            }
        )
        return self._responses.pop(0)


def make_context(*, protocol: str = "openai_compatible"):
    request = SearchRequest(
        question="深圳有什么酒店，房间有健身单车？Python 和 Java 谁更适合 Web 开发？",
        previous_conversation=["中国有什么好的旅游城市?", "深圳", "补充上下文"],
        demo_mode=True,
    )
    capability = ProviderCapability(
        engine="test-llm",
        provider_kind="builtin",
        protocol=protocol,
        supports_structured_output=protocol == "openai_compatible",
        supports_json_mode=protocol == "openai_compatible",
    )
    llm_provider = ResolvedProvider(
        engine="test-llm",
        provider_kind="builtin",
        component="llm",
        capability=capability,
        protocol=protocol,
        model="test-model",
    )
    search_provider = ResolvedProvider(
        engine="demo_search_provider",
        provider_kind="demo",
        component="search",
        capability=ProviderCapability(
            engine="demo_search_provider",
            provider_kind="demo",
            supports_site_restriction=True,
            site_restriction_mode="provider_side",
        ),
    )
    return build_runtime_context(
        request=request,
        llm_provider=llm_provider,
        search_provider=search_provider,
    )


def make_builtin_context(engine: str):
    request = SearchRequest(
        question="深圳有什么酒店，房间有健身单车？",
        previous_conversation=["中国有什么好的旅游城市?", "深圳"],
        demo_mode=True,
    )
    llm_provider = ResolvedProvider(
        engine=engine,
        provider_kind="builtin",
        component="llm",
        capability=BUILTIN_LLM_CAPABILITIES[engine],
        protocol="openai_compatible",
        model="test-model",
    )
    search_provider = ResolvedProvider(
        engine="demo_search_provider",
        provider_kind="demo",
        component="search",
        capability=ProviderCapability(
            engine="demo_search_provider",
            provider_kind="demo",
            supports_site_restriction=True,
            site_restriction_mode="provider_side",
        ),
    )
    return build_runtime_context(request=request, llm_provider=llm_provider, search_provider=search_provider)


def test_build_query_rewriting_prompt_uses_last_two_conversations_in_order() -> None:
    prompt = build_query_rewriting_prompt(
        question="有什么酒店，房间有健身单车？",
        previous_conversation=["第一句", "第二句", "第三句"],
    )

    assert 'previous_conversation: ["第二句", "第三句"]' in prompt
    assert "must produce separate search groups for each compared option or direction" in prompt
    assert "defaulting to English when no target context is clear" in prompt


@pytest.mark.asyncio
async def test_rewrite_query_groups_splits_multiple_questions_and_comparison_question() -> None:
    context = make_context(protocol="openai_compatible")
    llm = StubLLMClient(
        [
            {
                "previous_conversation": ["深圳", "补充上下文"],
                "search_groups": [
                    {
                        "group": 1,
                        "original_input": "深圳有什么酒店，房间有健身单车？Python 和 Java 谁更适合 Web 开发？",
                        "comparison_question": "",
                        "split_question": "深圳有什么酒店房间有健身单车？",
                        "main_term": "深圳酒店",
                        "key_noun": "健身单车房间",
                        "alias1": "动感单车房间",
                        "alias2": "运动客房",
                        "query1": "深圳酒店 健身单车房间",
                        "query2": "深圳酒店 动感单车房间 运动客房",
                    },
                    {
                        "group": 2,
                        "original_input": "深圳有什么酒店，房间有健身单车？Python 和 Java 谁更适合 Web 开发？",
                        "comparison_question": "Python 和 Java 谁更适合 Web 开发？",
                        "split_question": "Python 更适合 Web 开发吗？",
                        "main_term": "Python",
                        "key_noun": "Web 开发优势",
                        "alias1": "网站开发长处",
                        "alias2": "Web 应用优点",
                        "query1": "Python Web 开发优势",
                        "query2": "Python 网站开发长处 Web 应用优点",
                    },
                    {
                        "group": 3,
                        "original_input": "深圳有什么酒店，房间有健身单车？Python 和 Java 谁更适合 Web 开发？",
                        "comparison_question": "Python 和 Java 谁更适合 Web 开发？",
                        "split_question": "Java 更适合 Web 开发吗？",
                        "main_term": "Java",
                        "key_noun": "Web 开发优势",
                        "alias1": "网站开发长处",
                        "alias2": "Web 应用优点",
                        "query1": "Java Web 开发优势",
                        "query2": "Java 网站开发长处 Web 应用优点",
                    },
                ],
            }
        ]
    )

    result = await rewrite_query_groups(context=context, llm_client=llm)

    assert isinstance(result, QueryRewriteResult)
    assert [group.group for group in result.search_groups] == [1, 2, 3]
    assert result.normalized_previous_conversation == ["深圳", "补充上下文"]
    assert result.search_groups[1].comparison_question == "Python 和 Java 谁更适合 Web 开发？"
    assert result.search_groups[2].main_term == "Java"
    assert result.search_groups[0].traceability.previous_conversation == ["深圳", "补充上下文"]
    assert llm.calls[0]["json_strategy"] == "json_schema"


@pytest.mark.asyncio
async def test_rewrite_query_groups_tolerates_field_name_drift() -> None:
    context = make_context(protocol="native")
    llm = StubLLMClient(
        [
            {
                "previousConversation": ["深圳", "补充上下文"],
                "searchGroups": [
                    {
                        "groupId": 1,
                        "originalInput": "原问题",
                        "comparisonQuestion": "",
                        "splitQuestion": "补全后的问题",
                        "mainTerm": "深圳酒店",
                        "keyNoun": "健身单车房间",
                        "aliasOne": "动感单车房间",
                        "aliasTwo": "运动客房",
                        "queryOne": "深圳酒店 健身单车房间",
                        "queryTwo": "深圳酒店 动感单车房间 运动客房",
                    }
                ],
            }
        ]
    )

    result = await rewrite_query_groups(context=context, llm_client=llm)

    assert result.search_groups[0].split_question == "补全后的问题"
    assert result.search_groups[0].query2 == "深圳酒店 动感单车房间 运动客房"
    assert llm.calls[0]["json_strategy"] == "prompt_only_json"


@pytest.mark.asyncio
async def test_rewrite_query_groups_retries_once_then_succeeds() -> None:
    context = make_context(protocol="openai_compatible")
    llm = StubLLMClient(
        [
            "{",
            {
                "previous_conversation": ["深圳", "补充上下文"],
                "search_groups": [
                    {
                        "group": 1,
                        "original_input": "原问题",
                        "comparison_question": "",
                        "split_question": "补全后的问题",
                        "main_term": "深圳酒店",
                        "key_noun": "健身单车房间",
                        "alias1": "动感单车房间",
                        "alias2": "运动客房",
                        "query1": "深圳酒店 健身单车房间",
                        "query2": "深圳酒店 动感单车房间 运动客房",
                    }
                ],
            },
        ]
    )

    result = await rewrite_query_groups(context=context, llm_client=llm)

    assert result.search_groups[0].group == 1
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_query_rewriting_uses_capability_driven_json_mode() -> None:
    context = make_context(protocol="openai_compatible")
    context.llm_provider.engine = "minimax-global"
    context.llm_provider.capability.recommended_query_rewriting_mode = "prompt_only_json"
    llm = StubLLMClient(
        [
            {
                "previous_conversation": ["深圳", "补充上下文"],
                "search_groups": [
                    {
                        "group": 1,
                        "original_input": "原问题",
                        "comparison_question": "",
                        "split_question": "补全后的问题",
                        "main_term": "深圳酒店",
                        "key_noun": "健身单车房间",
                        "alias1": "动感单车房间",
                        "alias2": "运动客房",
                        "query1": "深圳酒店 健身单车房间",
                        "query2": "深圳酒店 动感单车房间 运动客房",
                    }
                ],
            }
        ]
    )

    result = await rewrite_query_groups(context=context, llm_client=llm)

    assert result.search_groups[0].query1 == "深圳酒店 健身单车房间"
    assert llm.calls[0]["json_strategy"] == "prompt_only_json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine", "expected_strategy"),
    [
        ("gemini", "json_schema"),
        ("doubao-global", "json_schema"),
        ("doubao-china", "json_schema"),
        ("qwen-global", "json_object"),
        ("qwen-china", "json_object"),
        ("glm-china", "json_object"),
    ],
)
async def test_query_rewriting_group_a_recommended_modes_are_stable(engine: str, expected_strategy: str) -> None:
    context = make_builtin_context(engine)
    llm = StubLLMClient(
        [
            {
                "previous_conversation": ["中国有什么好的旅游城市?", "深圳"],
                "search_groups": [
                    {
                        "group": 1,
                        "original_input": "深圳有什么酒店，房间有健身单车？",
                        "comparison_question": "",
                        "split_question": "深圳有什么酒店房间有健身单车？",
                        "main_term": "深圳酒店",
                        "key_noun": "健身单车房间",
                        "alias1": "动感单车房间",
                        "alias2": "运动客房",
                        "query1": "深圳酒店 健身单车房间",
                        "query2": "深圳酒店 动感单车房间 运动客房",
                    }
                ],
            }
        ]
    )

    result = await rewrite_query_groups(context=context, llm_client=llm)

    assert result.search_groups[0].query2 == "深圳酒店 动感单车房间 运动客房"
    assert llm.calls[0]["json_strategy"] == expected_strategy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine", "expected_strategy"),
    [
        ("minimax-global", "prompt_only_json"),
        ("minimax-china", "json_schema"),
    ],
)
async def test_query_rewriting_group_c_recommended_modes_are_stable(engine: str, expected_strategy: str) -> None:
    context = make_builtin_context(engine)
    llm = StubLLMClient(
        [
            {
                "previous_conversation": ["中国有什么好的旅游城市?", "深圳"],
                "search_groups": [
                    {
                        "group": 1,
                        "original_input": "深圳有什么酒店，房间有健身单车？",
                        "comparison_question": "",
                        "split_question": "深圳有什么酒店房间有健身单车？",
                        "main_term": "深圳酒店",
                        "key_noun": "健身单车房间",
                        "alias1": "动感单车房间",
                        "alias2": "运动客房",
                        "query1": "深圳酒店 健身单车房间",
                        "query2": "深圳酒店 动感单车房间 运动客房",
                    }
                ],
            }
        ]
    )

    result = await rewrite_query_groups(context=context, llm_client=llm)

    assert result.search_groups[0].query1 == "深圳酒店 健身单车房间"
    assert llm.calls[0]["json_strategy"] == expected_strategy


@pytest.mark.asyncio
async def test_rewrite_query_groups_returns_missing_fields_after_second_failure() -> None:
    context = make_context(protocol="openai_compatible")
    llm = StubLLMClient(
        [
            {"previous_conversation": ["深圳", "补充上下文"], "search_groups": [{}]},
            {"previous_conversation": ["深圳", "补充上下文"], "search_groups": [{}]},
        ]
    )

    with pytest.raises(Exception) as exc_info:
        await rewrite_query_groups(context=context, llm_client=llm)

    error = exc_info.value
    assert len(llm.calls) == 2
    assert error.category == "llm_call_failure"
    assert sorted(error.details.invalid_fields) == [
        "search_groups[0].group",
        "search_groups[0].key_noun",
        "search_groups[0].main_term",
        "search_groups[0].query1",
        "search_groups[0].query2",
        "search_groups[0].split_question",
    ]
    assert error.retry_attempted is True


@pytest.mark.asyncio
async def test_rewrite_query_groups_respects_zero_retry_budget() -> None:
    context = make_context(protocol="openai_compatible")
    context.retry_budget["llm"] = 0
    llm = StubLLMClient(
        [
            {"previous_conversation": ["深圳", "补充上下文"], "search_groups": [{}]},
            {"previous_conversation": ["深圳", "补充上下文"], "search_groups": []},
        ]
    )

    with pytest.raises(Exception) as exc_info:
        await rewrite_query_groups(context=context, llm_client=llm)

    assert len(llm.calls) == 1
    assert exc_info.value.details.error_type == "invalid_request_error"
    assert exc_info.value.retry_attempted is False


def test_traceability_enrich_returns_new_objects() -> None:
    base = TraceabilityInfo(
        previous_conversation=["深圳", "补充上下文"],
        original_input="原问题",
        split_question="补全后的问题",
        split_question_id=1,
    )

    zoom_out_request = enrich_for_zoom_out_request(
        base,
        query_variant_id=1,
        search_query_used="深圳酒店 健身单车房间",
    )
    zoom_out_result = enrich_for_zoom_out_result(zoom_out_request, rank=1)
    zoom_in_request = enrich_for_zoom_in_request(
        zoom_out_result,
        search_query_used="site:trip.com 深圳酒店 健身单车房间",
    )
    zoom_in_result = enrich_for_zoom_in_result(zoom_in_request, rank=2)
    final_result = enrich_for_final_result(zoom_in_result)

    assert base.phase is None
    assert zoom_out_request is not base
    assert zoom_out_result is not zoom_out_request
    assert zoom_in_request is not zoom_out_result
    assert zoom_in_result is not zoom_in_request
    assert final_result is not zoom_in_result
    assert zoom_out_request.phase == "Zoom-out Search"
    assert zoom_out_request.query_variant_id == 1
    assert zoom_in_request.phase == "Zoom-in Search"
    assert zoom_in_request.query_variant_id == 1
    assert zoom_in_request.search_query_used == "site:trip.com 深圳酒店 健身单车房间"
    assert zoom_in_result.rank == 2
