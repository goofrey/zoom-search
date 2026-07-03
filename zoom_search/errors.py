"""Error helpers."""

from __future__ import annotations

from typing import Any

from zoom_search.models import ErrorDetails
from zoom_search.models import ErrorType
from zoom_search.models import RawDiagnostics
from zoom_search.models import ZoomSearchError


def configuration_error(
    *,
    category: str,
    component: str,
    message: str,
    user_message: str,
    invalid_fields: list[str] | None = None,
    provider_engine: str | None = None,
    request_id: str = "pending",
) -> ZoomSearchError:
    return ZoomSearchError(
        category=category,
        component=component,
        message=message,
        user_message=user_message,
        provider_engine=provider_engine,
        request_id=request_id,
        details=ErrorDetails(error_type="configuration_error", invalid_fields=invalid_fields or []),
    )


def call_failure(
    *,
    category: str,
    component: str,
    message: str,
    user_message: str,
    request_id: str,
    provider_engine: str | None = None,
    provider_model: str | None = None,
    reason_code: str | None = None,
    http_status: int | None = None,
    retryable: bool = False,
    retry_attempted: bool = False,
    invalid_fields: list[str] | None = None,
    provider_error_code: str | None = None,
    provider_error_type: str | None = None,
    provider_error_param: str | None = None,
    raw_diagnostics: RawDiagnostics | None = None,
) -> ZoomSearchError:
    return ZoomSearchError(
        category=category,
        component=component,
        message=message,
        user_message=user_message,
        request_id=request_id,
        provider_engine=provider_engine,
        provider_model=provider_model,
        retryable=retryable,
        retry_attempted=retry_attempted,
        details=ErrorDetails(
            error_type=normalize_error_type(category=category, reason_code=reason_code, http_status=http_status),
            reason_code=reason_code,
            invalid_fields=invalid_fields or [],
            http_status=http_status,
            provider_error_code=provider_error_code,
            provider_error_type=provider_error_type,
            provider_error_param=provider_error_param,
        ),
        raw_diagnostics=raw_diagnostics,
    )


def llm_http_error_reason(http_status: int | None) -> str | None:
    return _http_error_reason(http_status, is_search=False)


def search_http_error_reason(http_status: int | None) -> str | None:
    return _http_error_reason(http_status, is_search=True)


def normalize_error_type(*, category: str, reason_code: str | None = None, http_status: int | None = None) -> ErrorType:
    if category in {"llm_configuration_error", "search_configuration_error", "proxy_configuration_error"}:
        return "configuration_error"
    if category == "network_connection_failure":
        return "network_error"
    if http_status in {401, 403}:
        return "authentication_error"
    if http_status == 429:
        return "rate_limit_error"
    if reason_code in {"authentication_or_authorization_failed", "auth_error"}:
        return "authentication_error"
    if reason_code in {"invalid_request", "invalid_query_or_request", "json_parse_failed", "missing_required_search_group_fields", "results_missing_required_fields"}:
        return "invalid_request_error"
    if reason_code == "rate_limited":
        return "rate_limit_error"
    if reason_code == "quota_exceeded":
        return "quota_exceeded_error"
    if reason_code == "content_filtered":
        return "content_filtered_error"
    if reason_code in {"empty_output", "empty_results"}:
        return "empty_result_error"
    if reason_code in {"dns_resolution_failed", "connection_timeout", "read_timeout", "connection_refused", "tls_error", "proxy_connection_failed", "network_connection_failed"}:
        return "network_error"
    return "provider_error"


def extract_openai_error_fields(payload: Any) -> dict[str, str | None]:
    if not isinstance(payload, dict):
        return {
            "message": None,
            "code": None,
            "type": None,
            "param": None,
        }
    error = payload.get("error")
    if isinstance(error, str):
        return {
            "message": _coerce_optional_text(error),
            "code": _coerce_optional_text(payload.get("code") or payload.get("reason")),
            "type": _coerce_optional_text(payload.get("type") or payload.get("reason")),
            "param": None,
        }
    if not isinstance(error, dict):
        return {
            "message": _coerce_optional_text(payload.get("message")),
            "code": _coerce_optional_text(payload.get("code")),
            "type": _coerce_optional_text(payload.get("type") or payload.get("reason")),
            "param": _coerce_optional_text(payload.get("param")),
        }
    return {
        "message": _coerce_optional_text(error.get("message")),
        "code": _coerce_optional_text(error.get("code")),
        "type": _coerce_optional_text(error.get("type")),
        "param": _coerce_optional_text(error.get("param")),
    }


def _http_error_reason(http_status: int | None, *, is_search: bool) -> str | None:
    if http_status in {400, 422}:
        return "invalid_query_or_request" if is_search else "invalid_request"
    if http_status in {401, 403}:
        return "authentication_or_authorization_failed"
    if http_status == 429:
        return "rate_limited"
    if http_status in {500, 502, 503, 504}:
        return "provider_server_error"
    return None


def _coerce_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
