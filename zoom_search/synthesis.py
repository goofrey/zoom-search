"""Answer synthesis helpers."""

from __future__ import annotations

from collections.abc import Iterable
import json
import re
from time import perf_counter
from typing import Any
from typing import AsyncIterator

from zoom_search.errors import call_failure
from zoom_search.metrics import accumulate_llm_usage
from zoom_search.metrics import record_phase_elapsed
from zoom_search.models import RuntimeContext
from zoom_search.models import UnifiedLLMRequest
from zoom_search.models import UnifiedMessage
from zoom_search.providers.llm import LLMProvider


def build_answer_synthesis_prompt(*, question: str, previous_conversation: list[str], search_context: str) -> str:
    normalized_previous_conversation = [str(item).strip() for item in previous_conversation if str(item).strip()][-2:]
    previous_conversation_block = "\n".join(f"- {item}" for item in normalized_previous_conversation) or "- None"
    search_evidence = _strip_search_evidence_heading(search_context)
    return (
        "You are a Q&A assistant. Answer the user's Original Question as a whole using the Search Evidence.\n\n"
        "## Input\n\n"
        "- previous_conversation: latest 2 conversation sentences, possibly empty\n"
        "- question: Original Question\n"
        "- search_evidence: formatted Zoom Search evidence\n\n"
        "Use Previous Conversation only to recover omitted context, then keep the final answer focused on the latest question.\n\n"
        "## Search Evidence Format\n\n"
        "Zoom Search groups evidence by original question, optional comparison question, and split question. "
        "Comparison split questions are marked as comparison branches.\n\n"
        "## Answer Requirements\n\n"
        "- Answer the Original Question directly and as a whole.\n"
        "- Synthesize information across relevant Search Evidence.\n"
        "- Treat split questions as search hints; the final answer answers the Original Question.\n"
        "- If the Search Evidence is insufficient, state the uncertainty briefly in the answer and avoid unsupported conclusions.\n"
        "- Do not fabricate facts, sources, or conclusions.\n"
        "- Return valid JSON only. Do not wrap the JSON in markdown fences.\n"
        "- Write one complete answer in the answer field.\n"
        "- Write the answer in the same language as the user's Original Question.\n"
        "- Use consecutive source markers such as [1], [2], and [3] for source-backed statements. In sources, include only referenced markers with URLs from Search Evidence, reusing the same marker for each URL.\n\n"
        "## Output JSON Schema\n\n"
        '{"answer": "answer text with [1] [2] markers", "sources": [{"id": "[1]", "url": "https://example.com/source"}]}\n\n'
        "## Example\n\n"
        "Example Original Question:\n"
        "What power bank sizes are allowed on passenger flights, and when is airline approval required?\n\n"
        "Example Search Evidence:\n"
        "### Original Question: What power bank sizes are allowed on passenger flights, and when is airline approval required?\n\n"
        "#### Non-comparison Search Branch\n\n"
        "Split Question: What power banks are prohibited on passenger flights?\n\n"
        "Evidence:\n"
        "1. Title: Airline battery policy\n"
        "   Snippet: Power banks above 160Wh are prohibited on passenger flights.\n"
        "   URL: https://example.com/policy\n\n"
        "Split Question: What power banks require airline approval?\n\n"
        "Evidence:\n"
        "1. Title: Carry-on battery limits\n"
        "   Snippet: Power banks from 100Wh to 160Wh require airline approval and may be quantity-limited.\n"
        "   URL: https://example.com/airline\n\n"
        "Example JSON Output:\n"
        '{"answer": "You can usually carry a power bank on a flight when it stays within the airline\'s allowed battery limits. Power banks above 160Wh are prohibited on passenger flights.[1] Power banks from 100Wh to 160Wh usually require airline approval and may be limited in quantity.[2]", '
        '"sources": [{"id": "[1]", "url": "https://example.com/policy"}, {"id": "[2]", "url": "https://example.com/airline"}]}\n\n'
        "## Actual Input\n\n"
        "Previous Conversation:\n"
        f"{previous_conversation_block}\n\n"
        "Search Evidence:\n"
        f"{search_evidence}\n\n"
        "Original Question:\n"
        f"{question}"
    )


def _answer_synthesis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["id", "url"],
                },
            },
        },
        "required": ["answer", "sources"],
    }


def _strip_search_evidence_heading(search_context: str) -> str:
    heading = "## Search Evidence"
    stripped = search_context.strip()
    if stripped.startswith(heading):
        stripped = stripped[len(heading) :].lstrip()
    return stripped


def _extract_allowed_urls(search_context: str) -> set[str]:
    prefix = "URL: "
    return {
        _normalize_source_url(line.strip()[len(prefix) :].strip())
        for line in search_context.splitlines()
        if line.strip().startswith(prefix) and line.strip()[len(prefix) :].strip()
    }


def _normalize_source_url(url: str) -> str:
    normalized = str(url).strip()
    while len(normalized) > len("https://x/") and normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized


def _parse_answer_synthesis_payload(response: Any, *, context: RuntimeContext) -> dict[str, Any]:
    payload = getattr(response, "json_payload", None)
    if isinstance(payload, dict):
        return payload
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise call_failure(
                category="llm_call_failure",
                component="llm",
                message=f"Answer synthesis returned invalid JSON: {exc}",
                user_message="LLM request failed.",
                request_id=context.request_id,
                provider_engine=context.llm_provider.engine,
                provider_model=context.llm_provider.model,
                reason_code="json_parse_failed",
                retryable=True,
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise call_failure(
        category="llm_call_failure",
        component="llm",
        message="Answer synthesis returned no usable JSON payload.",
        user_message="LLM request failed.",
        request_id=context.request_id,
        provider_engine=context.llm_provider.engine,
        provider_model=context.llm_provider.model,
        reason_code="empty_output",
        retryable=True,
    )


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_source_entries(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []
    entries: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        marker = str(item.get("id") or "").strip()
        url = str(item.get("url") or "").strip()
        if marker and url:
            entries.append((marker, url))
    return entries


def _renumber_cited_sources(answer: str, cited_sources: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    url_markers: dict[str, str] = {}
    marker_map: dict[str, str] = {}
    source_by_marker = {marker: url for marker, url in cited_sources}
    renumbered_sources: list[tuple[str, str]] = []
    if not source_by_marker:
        return answer, []
    pattern = re.compile("|".join(re.escape(marker) for marker in sorted(source_by_marker, key=len, reverse=True)))
    for match in pattern.finditer(answer):
        marker = match.group(0)
        if marker in marker_map:
            continue
        url = source_by_marker[marker]
        normalized_url = _normalize_source_url(url)
        assigned_marker = url_markers.get(normalized_url)
        if assigned_marker is None:
            assigned_marker = f"[{len(url_markers) + 1}]"
            url_markers[normalized_url] = assigned_marker
            renumbered_sources.append((assigned_marker, url))
        marker_map[marker] = assigned_marker
    if not marker_map:
        return answer, []
    renumbered_answer = pattern.sub(lambda match: marker_map[match.group(0)], answer)
    return renumbered_answer, renumbered_sources


def _extract_streaming_answer_prefix(text: str) -> str:
    match = re.search(r'"answer"\s*:\s*"', text)
    if not match:
        return ""

    raw_value = text[match.end() :]
    raw_prefix_parts: list[str] = []
    index = 0
    while index < len(raw_value):
        char = raw_value[index]
        if char == '"':
            break
        if char != "\\":
            raw_prefix_parts.append(char)
            index += 1
            continue

        if index + 1 >= len(raw_value):
            break
        escaped = raw_value[index + 1]
        if escaped == "u":
            unicode_escape = raw_value[index : index + 6]
            if len(unicode_escape) < 6 or not re.fullmatch(r"\\u[0-9a-fA-F]{4}", unicode_escape):
                break
            raw_prefix_parts.append(unicode_escape)
            index += 6
            continue
        if escaped not in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
            break
        raw_prefix_parts.append(raw_value[index : index + 2])
        index += 2

    raw_prefix = "".join(raw_prefix_parts)
    if not raw_prefix:
        return ""
    try:
        decoded = json.loads(f'"{raw_prefix}"')
    except json.JSONDecodeError:
        return ""
    return decoded if isinstance(decoded, str) else ""


def _render_answer_synthesis_payload(payload: dict[str, Any], *, search_context: str) -> str:
    allowed_urls = _extract_allowed_urls(search_context)
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        return ""
    cited_sources: list[tuple[str, str]] = []
    seen_markers: set[str] = set()
    for marker, url in _normalize_source_entries(payload.get("sources")):
        if marker in seen_markers:
            continue
        if marker not in answer:
            continue
        if _normalize_source_url(url) not in allowed_urls:
            continue
        cited_sources.append((marker, url))
        seen_markers.add(marker)
    if not cited_sources:
        return answer
    answer, cited_sources = _renumber_cited_sources(answer, cited_sources)
    sources_block = "\n".join(f"{marker} {url}" for marker, url in cited_sources)
    return f"{answer}\n\nSources:\n{sources_block}"


def _raise_empty_answer(context: RuntimeContext) -> None:
    raise call_failure(
        category="llm_call_failure",
        component="llm",
        message="LLM provider returned an empty answer synthesis output.",
        user_message="LLM request failed.",
        request_id=context.request_id,
        provider_engine=context.llm_provider.engine,
        provider_model=context.llm_provider.model,
        reason_code="empty_output",
        retryable=True,
    )


async def synthesize_answer(*, context: RuntimeContext, llm_provider: LLMProvider, search_context: str) -> str:
    started_at = perf_counter()
    prompt = build_answer_synthesis_prompt(
        question=context.request.question,
        previous_conversation=context.request.previous_conversation,
        search_context=search_context,
    )
    response = await llm_provider.generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            messages=[UnifiedMessage(role="user", content=prompt)],
            model=context.llm_provider.model or context.llm_provider.engine,
            temperature=0.2,
            expect_json=True,
            json_schema=_answer_synthesis_schema(),
            stream=False,
            seed=context.request.seed,
            trace_context={
                "request_id": context.request_id,
                "phase": "Answer Synthesis",
                "provider_name": context.llm_provider.engine,
            },
        )
    )
    accumulate_llm_usage(context=context, phase="answer_synthesis", usage=response.usage)
    record_phase_elapsed(context=context, phase="answer_synthesis", started_at=started_at)
    payload = _parse_answer_synthesis_payload(response, context=context)
    rendered = _render_answer_synthesis_payload(payload, search_context=search_context)
    if not rendered.strip():
        _raise_empty_answer(context)
    return rendered


async def stream_synthesize_answer(*, context: RuntimeContext, llm_provider: LLMProvider, search_context: str) -> AsyncIterator[str]:
    started_at = perf_counter()
    prompt = build_answer_synthesis_prompt(
        question=context.request.question,
        previous_conversation=context.request.previous_conversation,
        search_context=search_context,
    )
    final_response = None
    raw_chunks: list[str] = []
    emitted_answer = ""
    async for event in llm_provider.stream_generate(
        UnifiedLLMRequest(
            task="answer_synthesis",
            messages=[UnifiedMessage(role="user", content=prompt)],
            model=context.llm_provider.model or context.llm_provider.engine,
            temperature=0.2,
            expect_json=True,
            json_schema=_answer_synthesis_schema(),
            stream=True,
            seed=context.request.seed,
            trace_context={
                "request_id": context.request_id,
                "phase": "Answer Synthesis",
                "provider_name": context.llm_provider.engine,
            },
        )
    ):
        if event.get("event") == "token":
            text = str(event.get("delta_text") or "")
            if text:
                raw_chunks.append(text)
                answer_prefix = _extract_streaming_answer_prefix("".join(raw_chunks))
                if answer_prefix.startswith(emitted_answer) and len(answer_prefix) > len(emitted_answer):
                    delta = answer_prefix[len(emitted_answer) :]
                    emitted_answer = answer_prefix
                    yield delta
        elif event.get("event") == "done":
            final_response = event.get("response")
    if final_response is not None and hasattr(final_response, "usage"):
        accumulate_llm_usage(context=context, phase="answer_synthesis", usage=final_response.usage)
    record_phase_elapsed(context=context, phase="answer_synthesis", started_at=started_at)
    response_payload = final_response
    if response_payload is None:
        response_payload = type("_Response", (), {"text": "".join(raw_chunks), "json_payload": None})()
    payload = _parse_answer_synthesis_payload(response_payload, context=context)
    rendered = _render_answer_synthesis_payload(payload, search_context=search_context)
    if not rendered.strip():
        _raise_empty_answer(context)
    if rendered.startswith(emitted_answer):
        suffix = rendered[len(emitted_answer) :]
        if suffix:
            yield suffix
    elif rendered:
        yield rendered
