from __future__ import annotations

from zoom_search.synthesis import build_answer_synthesis_prompt
from zoom_search.synthesis import _extract_streaming_answer_prefix
from zoom_search.synthesis import _render_answer_synthesis_payload


def test_answer_synthesis_prompt_includes_few_shot_example() -> None:
    prompt = build_answer_synthesis_prompt(
        question="What are the airline rules for carrying a power bank on a flight?",
        previous_conversation=["I am flying next week.", "This is about carry-on baggage.", "The airline is a budget carrier."],
        search_context="## Search Evidence\n\nURL: https://example.com/policy\nURL: https://example.com/airline",
    )

    assert "## Input" in prompt
    assert "- previous_conversation: latest 2 conversation sentences, possibly empty" in prompt
    assert "- This is about carry-on baggage." in prompt
    assert "- The airline is a budget carrier." in prompt
    assert "I am flying next week." not in prompt
    assert "## Answer Requirements" in prompt
    assert "Use Previous Conversation only to recover omitted context" in prompt
    assert "Return valid JSON only." in prompt
    assert "Write the answer in the same language as the user's Original Question." in prompt
    assert "Use consecutive source markers such as [1], [2], and [3] for source-backed statements." in prompt
    assert "include only referenced markers with URLs from Search Evidence" in prompt
    assert "## Actual Input" in prompt
    assert "Search Evidence:\nURL: https://example.com/policy\nURL: https://example.com/airline" in prompt
    assert '"answer": "answer text with [1] [2] markers", "sources": [{"id": "[1]", "url": "https://example.com/source"}]' in prompt
    assert "### Original Question: What power bank sizes are allowed on passenger flights, and when is airline approval required?" in prompt
    assert "Split Question: What power banks are prohibited on passenger flights?" in prompt
    assert "## Example" in prompt
    assert "Example JSON Output:" in prompt
    assert '"sources": [{"id": "[1]", "url": "https://example.com/policy"}, {"id": "[2]", "url": "https://example.com/airline"}]' in prompt
    assert prompt.endswith("Original Question:\nWhat are the airline rules for carrying a power bank on a flight?")


def test_render_answer_synthesis_payload_filters_unknown_urls() -> None:
    rendered = _render_answer_synthesis_payload(
        {
            "answer": "Known fact.[1] Needs more evidence.",
            "sources": [
                {"id": "[1]", "url": "https://example.com/policy"},
                {"id": "[2]", "url": "https://unknown.example.com"},
            ],
        },
        search_context="## Search Evidence\n\nURL: https://example.com/policy",
    )

    assert rendered == (
        "Known fact.[1] Needs more evidence.\n\n"
        "Sources:\n"
        "[1] https://example.com/policy"
    )


def test_render_answer_synthesis_payload_renumbers_source_markers_sequentially() -> None:
    rendered = _render_answer_synthesis_payload(
        {
            "answer": "Fact A.[2] Fact B.[10]",
            "sources": [
                {"id": "[2]", "url": "https://example.com/a"},
                {"id": "[10]", "url": "https://example.com/b"},
            ],
        },
        search_context="## Search Evidence\n\nURL: https://example.com/a\nURL: https://example.com/b",
    )

    assert rendered == (
        "Fact A.[1] Fact B.[2]\n\n"
        "Sources:\n"
        "[1] https://example.com/a\n"
        "[2] https://example.com/b"
    )


def test_render_answer_synthesis_payload_renumbers_markers_by_first_answer_citation() -> None:
    rendered = _render_answer_synthesis_payload(
        {
            "answer": "Store peaches cold.[1][6] Bring to room temperature before eating.[3] Avoid bags.[12]",
            "sources": [
                {"id": "[1]", "url": "https://example.com/storage"},
                {"id": "[3]", "url": "https://example.com/flavor"},
                {"id": "[6]", "url": "https://example.com/bruising"},
                {"id": "[12]", "url": "https://example.com/bags"},
            ],
        },
        search_context=(
            "## Search Evidence\n\n"
            "URL: https://example.com/storage\n"
            "URL: https://example.com/flavor\n"
            "URL: https://example.com/bruising\n"
            "URL: https://example.com/bags"
        ),
    )

    assert rendered == (
        "Store peaches cold.[1][2] Bring to room temperature before eating.[3] Avoid bags.[4]\n\n"
        "Sources:\n"
        "[1] https://example.com/storage\n"
        "[2] https://example.com/bruising\n"
        "[3] https://example.com/flavor\n"
        "[4] https://example.com/bags"
    )


def test_render_answer_synthesis_payload_merges_duplicate_source_urls() -> None:
    rendered = _render_answer_synthesis_payload(
        {
            "answer": "Fact A.[1] Fact B.[2]",
            "sources": [
                {"id": "[1]", "url": "https://example.com/a"},
                {"id": "[2]", "url": "https://example.com/a"},
            ],
        },
        search_context="## Search Evidence\n\nURL: https://example.com/a",
    )

    assert rendered == (
        "Fact A.[1] Fact B.[1]\n\n"
        "Sources:\n"
        "[1] https://example.com/a"
    )


def test_extract_streaming_answer_prefix_handles_partial_json_string() -> None:
    assert _extract_streaming_answer_prefix('{"answer":"Hello') == "Hello"
    assert _extract_streaming_answer_prefix('{"answer":"Hello\\nwor') == "Hello\nwor"
    assert _extract_streaming_answer_prefix('{"answer":"Hello\\') == "Hello"
    assert _extract_streaming_answer_prefix('{"answer":"Hello\\u4') == "Hello"
    assert _extract_streaming_answer_prefix('{"answer":"Hello\\u4e16') == "Hello世"
