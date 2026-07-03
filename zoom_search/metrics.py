"""Runtime metrics helpers for Zoom Search."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from zoom_search.models import RuntimeContext


def create_runtime_metrics() -> dict[str, Any]:
    return {
        "started_at": perf_counter(),
        "elapsed_ms": 0,
        "phase_elapsed_ms": {},
        "search_requests": {
            "planned": 0,
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "retries": 0,
            "zoom_out_planned": 0,
            "zoom_in_planned": 0,
            "zoom_out_attempted": 0,
            "zoom_in_attempted": 0,
            "zoom_out_succeeded": 0,
            "zoom_in_succeeded": 0,
            "zoom_out_failed": 0,
            "zoom_in_failed": 0,
        },
        "llm_usage": {
            "query_rewriting": _empty_usage(),
            "answer_synthesis": _empty_usage(),
            "total": _empty_usage(),
        },
    }


def snapshot_metrics(*, context: RuntimeContext) -> dict[str, Any]:
    metrics = context.metrics
    elapsed_ms = max(0, int((perf_counter() - metrics.get("started_at", perf_counter())) * 1000))
    return {
        "elapsed_ms": elapsed_ms,
        "phase_elapsed_ms": dict(metrics.get("phase_elapsed_ms", {})),
        "search_requests": dict(metrics.get("search_requests", {})),
        "llm_usage": {
            phase: dict(values)
            for phase, values in (metrics.get("llm_usage", {}) or {}).items()
            if isinstance(values, dict)
        },
    }


def record_phase_elapsed(*, context: RuntimeContext, phase: str, started_at: float) -> None:
    context.metrics.setdefault("phase_elapsed_ms", {})[phase] = max(0, int((perf_counter() - started_at) * 1000))


def accumulate_llm_usage(*, context: RuntimeContext, phase: str, usage: dict[str, int | None] | None) -> None:
    target = context.metrics.setdefault("llm_usage", {})
    phase_usage = target.setdefault(phase, _empty_usage())
    total_usage = target.setdefault("total", _empty_usage())
    normalized = usage or {}
    for key in _empty_usage():
        value = normalized.get(key)
        if value is None:
            continue
        phase_usage[key] = (phase_usage.get(key) or 0) + value
        total_usage[key] = (total_usage.get(key) or 0) + value


def record_search_planned(*, context: RuntimeContext, phase: str, count: int) -> None:
    counters = context.metrics.setdefault("search_requests", {})
    counters["planned"] = counters.get("planned", 0) + count
    counters[f"{phase}_planned"] = counters.get(f"{phase}_planned", 0) + count


def record_search_attempt(*, context: RuntimeContext, phase: str, attempt: int) -> None:
    counters = context.metrics.setdefault("search_requests", {})
    counters["attempted"] = counters.get("attempted", 0) + 1
    counters[f"{phase}_attempted"] = counters.get(f"{phase}_attempted", 0) + 1
    if attempt > 0:
        counters["retries"] = counters.get("retries", 0) + 1


def record_search_outcome(*, context: RuntimeContext, phase: str, success: bool) -> None:
    counters = context.metrics.setdefault("search_requests", {})
    status = "succeeded" if success else "failed"
    counters[status] = counters.get(status, 0) + 1
    key = f"{phase}_{status}"
    counters[key] = counters.get(key, 0) + 1


def _empty_usage() -> dict[str, int | None]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
    }
