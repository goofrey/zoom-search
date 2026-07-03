"""Retry policy helpers."""

from __future__ import annotations

import asyncio

from zoom_search.models import RuntimeContext
from zoom_search.models import ZoomSearchError


BASE_RETRY_DELAY_SECONDS = 0.05
MAX_RETRY_DELAY_SECONDS = 1.0


def retry_attempts(*, context: RuntimeContext, component: str) -> range:
    budget = max(int(context.retry_budget.get(component, 0)), 0)
    return range(budget + 1)


def should_retry_error(*, error: ZoomSearchError, component: str) -> bool:
    error_type = error.details.error_type
    if error_type in {"rate_limit_error", "network_error"}:
        return True
    if component == "llm" and error_type == "invalid_request_error":
        return error.details.reason_code in {"json_parse_failed", "missing_required_search_group_fields"} and bool(error.retryable)
    if error_type == "provider_error":
        return bool(error.retryable) or error.details.http_status in {500, 502, 503, 504}
    if error_type == "empty_result_error":
        return component == "search"
    return False


async def sleep_before_retry(*, attempt: int) -> None:
    await asyncio.sleep(retry_delay_seconds(attempt=attempt))


def retry_delay_seconds(*, attempt: int) -> float:
    if attempt < 0:
        attempt = 0
    return min(BASE_RETRY_DELAY_SECONDS * (2**attempt), MAX_RETRY_DELAY_SECONDS)
