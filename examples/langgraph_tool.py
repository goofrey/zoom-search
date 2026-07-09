"""Use Zoom Search as a LangGraph/LangChain evidence tool.

Install LangChain separately to run this example:

    pip install langchain
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from dataclasses import is_dataclass
from typing import Any

from langchain.tools import tool

from zoom_search import search


def _to_json_like(value: Any) -> Any:
    if is_dataclass(value):
        return _to_json_like(asdict(value))
    if isinstance(value, dict):
        return {key: _to_json_like(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_like(item) for item in value]
    return value


@tool
async def zoom_search_evidence(query: str, include_raw_diagnostics: bool = False) -> dict[str, Any]:
    """Search for grounded web evidence and return answer, sources, warnings, and metrics."""

    response = await search(
        question=query,
        demo_mode=True,
        output_mode="answer_with_sources",
        seed=7,
        include_raw_diagnostics=include_raw_diagnostics,
    )
    payload = {
        "request_id": response.request_id,
        "answer": getattr(response, "answer", None),
        "sources": _to_json_like(getattr(response, "results", []) or []),
        "metrics": _to_json_like(getattr(response, "metrics", {}) or {}),
        "warnings": _to_json_like(response.warnings),
    }
    if include_raw_diagnostics:
        payload["raw_diagnostics"] = _to_json_like(getattr(response, "raw_diagnostics", {}) or {})
    return payload


async def main() -> None:
    result = await zoom_search_evidence.ainvoke(
        {"query": "What hotels in Shenzhen have rooms with exercise bikes?"}
    )
    print(result["answer"])
    print("\nSources:")
    for index, source in enumerate(result["sources"], start=1):
        print(f"{index}. {source['title']} - {source['url']}")
    print("\nMetrics:")
    print(result["metrics"])


if __name__ == "__main__":
    asyncio.run(main())
