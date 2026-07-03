"""Query rewriting helpers and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from zoom_search.errors import call_failure
from zoom_search.metrics import accumulate_llm_usage
from zoom_search.metrics import record_phase_elapsed
from zoom_search.models import RuntimeContext
from zoom_search.models import SearchGroup
from zoom_search.models import TraceabilityInfo
from zoom_search.models import ZoomSearchError
from zoom_search.providers.llm import BUILTIN_LLM_REGISTRY
from zoom_search.retry import retry_attempts
from zoom_search.retry import should_retry_error
from zoom_search.retry import sleep_before_retry
from zoom_search.traceability import create_initial_traceability

PROMPT_TEMPLATE = """You are a search query rewriting expert. Turn the user's question into structured Search Queries.

Use previous_conversation only to recover omitted context before splitting or rewriting.

## Search Query Structure

Each Search Query joins these parts with spaces:

- **Main term**: the core search object, such as a location + entity, person, event, or concept.
- **Key noun**: a complete noun phrase that combines with the main term to express the full intent, e.g. "exercise bike room" rather than separate words like "exercise bike" or "room".
- **Alias**: an industry/field synonym of key_noun with the same meaning and clearly different wording.

Generate Search Queries according to the required structure: query1 = main_term + key_noun; query2 = main_term + alias1 + alias2.

## Rules

1. Use the search target context's local language for all generated content and aliases: Japanese for Japan-related searches, Chinese for China-related searches, English for English-speaking regions, defaulting to English when no target context is clear.
2. Split the input into independent questions. After context recovery, each split question must be understandable on its own and generate its own search group.
3. Further split questions that need multiple search directions. Comparison questions, e.g. "Which is better, A or B?", must produce separate search groups for each compared option or direction.
4. For comparison questions, every related Search Group must include the full shared comparison_question. Groups with the same original_input and comparison_question must represent different options or directions, use different main_term values, keep the same comparison intent in key_noun, and generate their own query1 and query2.
5. For non-comparison questions, comparison_question may be empty.
6. Generate query1 and query2 exactly according to the Search Query Structure section.

## Output Format

Strictly output JSON, no other content:
{{
  "previous_conversation": {previous_conversation_json},
  "search_groups": [
    {{
      "group": 1,
      "original_input": "original input sentence",
      "comparison_question": "full comparison question shared by comparison branches; empty for non-comparison questions",
      "split_question": "split independent question (with context filled in from previous conversation; comparison-type questions produce one split question per search direction)",
      "main_term": "main term",
      "key_noun": "key noun",
      "alias1": "industry alias1",
      "alias2": "industry alias2",
      "query1": "first complete Search Query",
      "query2": "second complete Search Query"
    }}
  ]
}}

Each split independent question generates one object. If 3 independent questions are split, output 3 objects.

## Example

Input:

previous_conversation: ["What are good travel cities in the UK?", "London"]

question: "What hotels have in-room exercise bikes? Is the weather warmer than Edinburgh?"

Output:

{{
  "previous_conversation": ["What are good travel cities in the UK?", "London"],
  "search_groups": [
    {{
      "group": 1,
      "original_input": "What hotels have in-room exercise bikes? Is the weather warmer than Edinburgh?",
      "comparison_question": "",
      "split_question": "What London hotels have in-room exercise bikes?",
      "main_term": "London hotels",
      "key_noun": "in-room exercise bikes",
      "alias1": "fitness bike suites",
      "alias2": "guest rooms with stationary bikes",
      "query1": "London hotels in-room exercise bikes",
      "query2": "London hotels fitness bike suites guest rooms with stationary bikes"
    }},
    {{
      "group": 2,
      "original_input": "What hotels have in-room exercise bikes? Is the weather warmer than Edinburgh?",
      "comparison_question": "Is London weather warmer than Edinburgh?",
      "split_question": "Is London weather warm?",
      "main_term": "London",
      "key_noun": "warm weather",
      "alias1": "mild climate",
      "alias2": "higher temperatures",
      "query1": "London warm weather",
      "query2": "London mild climate higher temperatures"
    }},
    {{
      "group": 3,
      "original_input": "What hotels have in-room exercise bikes? Is the weather warmer than Edinburgh?",
      "comparison_question": "Is London weather warmer than Edinburgh?",
      "split_question": "Is Edinburgh weather warm?",
      "main_term": "Edinburgh",
      "key_noun": "warm weather",
      "alias1": "mild climate",
      "alias2": "higher temperatures",
      "query1": "Edinburgh warm weather",
      "query2": "Edinburgh mild climate higher temperatures"
    }}
  ]
}}

## Actual Input

previous_conversation: {previous_conversation_json}

question: {question_json}
"""

REQUIRED_FIELDS = (
    "group",
    "split_question",
    "main_term",
    "key_noun",
    "query1",
    "query2",
)
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "previous_conversation": ("previous_conversation", "previousConversation"),
    "search_groups": ("search_groups", "searchGroups"),
    "group": ("group", "groupId", "group_id"),
    "original_input": ("original_input", "originalInput"),
    "comparison_question": ("comparison_question", "comparisonQuestion"),
    "split_question": ("split_question", "splitQuestion"),
    "main_term": ("main_term", "mainTerm"),
    "key_noun": ("key_noun", "keyNoun"),
    "alias1": ("alias1", "aliasOne", "alias_1"),
    "alias2": ("alias2", "aliasTwo", "alias_2"),
    "query1": ("query1", "queryOne", "query_1"),
    "query2": ("query2", "queryTwo", "query_2"),
}


@dataclass(slots=True)
class QueryRewriteGroup:
    search_group: SearchGroup
    traceability: TraceabilityInfo

    @property
    def group(self) -> int:
        return self.search_group.group

    @property
    def original_input(self) -> str:
        return self.search_group.original_input

    @property
    def comparison_question(self) -> str | None:
        return self.search_group.comparison_question

    @property
    def split_question(self) -> str:
        return self.search_group.split_question

    @property
    def main_term(self) -> str:
        return self.search_group.main_term

    @property
    def key_noun(self) -> str:
        return self.search_group.key_noun

    @property
    def alias1(self) -> str:
        return self.search_group.alias1

    @property
    def alias2(self) -> str:
        return self.search_group.alias2

    @property
    def query1(self) -> str:
        return self.search_group.query1

    @property
    def query2(self) -> str:
        return self.search_group.query2


@dataclass(slots=True)
class QueryRewriteResult:
    normalized_previous_conversation: list[str]
    search_groups: list[QueryRewriteGroup]


def build_query_rewriting_prompt(*, question: str, previous_conversation: list[str]) -> str:
    normalized_previous_conversation = _normalize_previous_conversation(previous_conversation)
    return PROMPT_TEMPLATE.format(
        previous_conversation_json=json.dumps(normalized_previous_conversation, ensure_ascii=False),
        question_json=json.dumps(question, ensure_ascii=False),
    )


async def rewrite_query_groups(*, context: RuntimeContext, llm_client: Any) -> QueryRewriteResult:
    started_at = perf_counter()
    prompt = build_query_rewriting_prompt(
        question=context.request.question,
        previous_conversation=context.request.previous_conversation,
    )
    json_strategy = _select_json_strategy(context)
    last_error: ZoomSearchError | None = None

    for attempt in retry_attempts(context=context, component="llm"):
        try:
            response = await llm_client.generate_json(
                prompt=prompt,
                context=context,
                json_strategy=json_strategy,
            )
            if hasattr(response, "usage"):
                accumulate_llm_usage(context=context, phase="query_rewriting", usage=getattr(response, "usage", None))
            response_payload = response
            if hasattr(response, "json_payload") and getattr(response, "json_payload") is not None:
                response_payload = getattr(response, "json_payload")
            elif hasattr(response, "text") and getattr(response, "text") is not None:
                response_payload = getattr(response, "text")
            parsed = _parse_llm_response(response_payload, context=context)
            record_phase_elapsed(context=context, phase="query_rewriting", started_at=started_at)
            return _build_result_from_payload(
                payload=parsed,
                context=context,
                fallback_previous_conversation=context.request.previous_conversation,
            )
        except ZoomSearchError as error:
            last_error = error
            if attempt < context.retry_budget.get("llm", 0) and should_retry_error(error=error, component="llm"):
                await sleep_before_retry(attempt=attempt)
                continue
            error.retry_attempted = attempt > 0
            raise error

    assert last_error is not None
    last_error.retry_attempted = context.retry_budget.get("llm", 0) > 0
    raise last_error


def _select_json_strategy(context: RuntimeContext) -> str:
    provider = context.llm_provider
    if BUILTIN_LLM_REGISTRY.has(provider.engine):
        return BUILTIN_LLM_REGISTRY.recommended_query_rewriting_mode(provider.engine)
    if provider.capability.recommended_query_rewriting_mode:
        if provider.capability.recommended_query_rewriting_mode != "prompt_only_json":
            return provider.capability.recommended_query_rewriting_mode
        if provider.protocol == "openai_compatible" and provider.capability.supports_structured_output:
            return "json_schema"
        if provider.protocol == "openai_compatible" and provider.capability.supports_json_mode:
            return "json_object"
        return provider.capability.recommended_query_rewriting_mode
    return "prompt_only_json"


def _parse_llm_response(response: object, *, context: RuntimeContext) -> dict[str, Any]:
    if response is None:
        raise _llm_failure("LLM returned empty output.", ["search_groups"], context=context, reason_code="empty_output")
    if isinstance(response, str):
        content = response.strip()
        if not content:
            raise _llm_failure("LLM returned empty output.", ["search_groups"], context=context, reason_code="empty_output")
        try:
            response = json.loads(content)
        except json.JSONDecodeError as exc:
            raise _llm_failure(f"LLM returned invalid JSON: {exc}", ["search_groups"], context=context, reason_code="json_parse_failed") from exc
    if not isinstance(response, dict):
        raise _llm_failure("LLM response JSON must be an object.", ["search_groups"], context=context, reason_code="json_parse_failed")
    return response


def _build_result_from_payload(
    *,
    payload: dict[str, Any],
    context: RuntimeContext,
    fallback_previous_conversation: list[str],
) -> QueryRewriteResult:
    normalized_previous_conversation = _normalize_previous_conversation(
        _coerce_string_list(_get_field(payload, "previous_conversation")) or fallback_previous_conversation
    )
    raw_groups = _get_field(payload, "search_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise _llm_failure(
            "LLM response does not contain valid search_groups.",
            ["search_groups"],
            context=context,
            reason_code="missing_required_search_group_fields",
        )

    rewrite_groups: list[QueryRewriteGroup] = []
    invalid_fields: list[str] = []

    for index, raw_group in enumerate(raw_groups):
        group, group_invalid_fields = _map_search_group(raw_group=raw_group, index=index)
        if group is None:
            invalid_fields.extend(group_invalid_fields)
            continue
        traceability = create_initial_traceability(
            previous_conversation=normalized_previous_conversation,
            search_group=group,
        )
        rewrite_groups.append(
            QueryRewriteGroup(search_group=group, traceability=traceability)
        )

    if invalid_fields:
        raise _llm_failure(
            "LLM response is missing required search group fields.",
            invalid_fields,
            context=context,
            reason_code="missing_required_search_group_fields",
        )

    return QueryRewriteResult(
        normalized_previous_conversation=normalized_previous_conversation,
        search_groups=rewrite_groups,
    )


def _map_search_group(*, raw_group: object, index: int) -> tuple[SearchGroup | None, list[str]]:
    if not isinstance(raw_group, dict):
        return None, [f"search_groups[{index}]"]

    values = {field: _get_field(raw_group, field) for field in FIELD_ALIASES if field not in {"previous_conversation", "search_groups"}}
    invalid_fields: list[str] = []

    for field in REQUIRED_FIELDS:
        if _is_missing(values.get(field)):
            invalid_fields.append(f"search_groups[{index}].{field}")

    if invalid_fields:
        return None, invalid_fields

    try:
        group = int(values["group"])
    except (TypeError, ValueError):
        return None, [f"search_groups[{index}].group"]

    return (
        SearchGroup(
            group=group,
            original_input=_coerce_text(values.get("original_input")) or "",
            comparison_question=_coerce_optional_text(values.get("comparison_question")),
            split_question=_coerce_text(values["split_question"]),
            main_term=_coerce_text(values["main_term"]),
            key_noun=_coerce_text(values["key_noun"]),
            alias1=_coerce_text(values.get("alias1")),
            alias2=_coerce_text(values.get("alias2")),
            query1=_coerce_text(values["query1"]),
            query2=_coerce_text(values["query2"]),
        ),
        [],
    )


def _normalize_previous_conversation(previous_conversation: list[str]) -> list[str]:
    values = [str(item).strip() for item in previous_conversation if str(item).strip()]
    return values[-2:]


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_optional_text(value: object) -> str | None:
    normalized = _coerce_text(value)
    return normalized or None


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _get_field(payload: dict[str, Any], field: str) -> object:
    for alias in FIELD_ALIASES[field]:
        if alias in payload:
            return payload[alias]
    return None


def _llm_failure(
    message: str,
    invalid_fields: list[str],
    *,
    context: RuntimeContext,
    reason_code: str,
) -> ZoomSearchError:
    return call_failure(
        category="llm_call_failure",
        component="llm",
        message=message,
        user_message=message,
        request_id=context.request_id,
        provider_engine=context.llm_provider.engine,
        provider_model=context.llm_provider.model,
        reason_code=reason_code,
        retryable=True,
        invalid_fields=sorted(set(invalid_fields)),
    )
