"""Formatted search evidence rendering."""

from __future__ import annotations

from zoom_search.models import FinalSearchResult
from zoom_search.rewriting import QueryRewriteGroup


def format_search_evidence(
    *,
    final_results: list[FinalSearchResult],
    search_groups: list[QueryRewriteGroup],
) -> str:
    result_map: dict[tuple[str, str | None, str], list[FinalSearchResult]] = {}
    for result in final_results:
        traceability = result.traceability
        key = (
            traceability.original_input or "",
            traceability.comparison_question,
            traceability.split_question or "",
        )
        result_map.setdefault(key, []).append(result)

    grouped_search_groups: dict[str, dict[str | None, list[QueryRewriteGroup]]] = {}
    for group in search_groups:
        grouped_search_groups.setdefault(group.original_input, {}).setdefault(group.comparison_question, []).append(group)

    lines = ["## Search Evidence", ""]
    for original_input, comparison_map in grouped_search_groups.items():
        lines.append(f"### Original Question: {original_input}")
        lines.append("")
        non_comparison_groups = comparison_map.get(None, [])
        if non_comparison_groups:
            lines.append("#### Non-comparison Search Branch")
            lines.append("")
            for group in non_comparison_groups:
                _append_split_question_block(lines=lines, group=group, evidence_items=result_map.get((group.original_input, None, group.split_question), []))
        for comparison_question, branches in comparison_map.items():
            if comparison_question is None:
                continue
            lines.append(f"#### Comparison Question: {comparison_question}")
            lines.append("")
            for index, group in enumerate(branches, start=1):
                lines.append(f"Comparison Branch {index}:")
                _append_split_question_block(
                    lines=lines,
                    group=group,
                    evidence_items=result_map.get((group.original_input, comparison_question, group.split_question), []),
                )
    return "\n".join(lines).rstrip()


def _append_split_question_block(*, lines: list[str], group: QueryRewriteGroup, evidence_items: list[FinalSearchResult]) -> None:
    lines.append(f"Split Question: {group.split_question}")
    lines.append("")
    if not evidence_items:
        lines.append("Evidence: No evidence found.")
        lines.append("")
        return
    lines.append("Evidence:")
    for index, item in enumerate(evidence_items, start=1):
        lines.append(f"{index}. Title: {item.title}")
        lines.append(f"   Snippet: {item.snippet}")
        lines.append(f"   URL: {item.url}")
    lines.append("")
