"""Request-scoped transport helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zoom_search.errors import call_failure
from zoom_search.models import RawDiagnostics
from zoom_search.models import RuntimeContext

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - covered by fallback tests
    httpx = None


@dataclass(slots=True)
class TransportClientConfig:
    component: str
    proxy: str | None
    headers: dict[str, str]
    base_url: str | None
    timeout: dict[str, float]
    limits: dict[str, int]


@dataclass(slots=True)
class RequestTransportContext:
    llm: TransportClientConfig
    search: TransportClientConfig
    llm_client: Any = None
    search_client: Any = None

    async def aclose(self) -> None:
        for client in (self.llm_client, self.search_client):
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()


def create_transport_context(*, context: RuntimeContext) -> RequestTransportContext:
    timeout = {
        "connect": 10.0,
        "read": 60.0,
        "write": 30.0,
        "pool": 10.0,
    }
    limits = {
        "max_connections": 20,
        "max_keepalive_connections": 10,
    }
    llm_proxy = _resolve_proxy(component="llm", context=context)
    search_proxy = _resolve_proxy(component="search", context=context)
    llm_config = TransportClientConfig(
        component="llm",
        proxy=llm_proxy,
        headers=dict(context.llm_provider.headers),
        base_url=context.llm_provider.base_url,
        timeout=timeout,
        limits=limits,
    )
    search_config = TransportClientConfig(
        component="search",
        proxy=search_proxy,
        headers=dict(context.search_provider.headers),
        base_url=context.search_provider.base_url,
        timeout=timeout,
        limits=limits,
    )
    transport_context = RequestTransportContext(llm=llm_config, search=search_config)
    transport_context.llm_client = _build_async_client(config=llm_config)
    transport_context.search_client = _build_async_client(config=search_config)
    return transport_context


def _resolve_proxy(*, component: str, context: RuntimeContext) -> str | None:
    if component == "llm":
        return context.llm_provider.http_proxy or context.request.proxy.http_proxy
    return context.search_provider.http_proxy or context.request.proxy.http_proxy


def _build_async_client(*, config: TransportClientConfig) -> Any:
    if httpx is None:
        return _FallbackAsyncClient(config=config)
    timeout = httpx.Timeout(
        timeout=60.0,
        connect=config.timeout["connect"],
        read=config.timeout["read"],
        write=config.timeout["write"],
        pool=config.timeout["pool"],
    )
    limits = httpx.Limits(
        max_connections=config.limits["max_connections"],
        max_keepalive_connections=config.limits["max_keepalive_connections"],
    )
    return httpx.AsyncClient(
        base_url=config.base_url or "",
        headers=config.headers,
        timeout=timeout,
        limits=limits,
        proxy=config.proxy,
    )


def normalize_transport_error(
    *,
    error: Exception,
    component: str,
    request_id: str,
    provider_engine: str | None,
    provider_model: str | None = None,
) -> Exception:
    error_name = error.__class__.__name__
    message = str(error) or error_name
    lower_message = message.lower()
    category = "proxy_configuration_error" if "proxy" in lower_message else "network_connection_failure"
    reason_code = _reason_code_from_exception_name(error_name=error_name, message=lower_message)
    return call_failure(
        category=category,
        component="proxy" if category == "proxy_configuration_error" else "transport",
        message=message,
        user_message=f"{component} transport request failed.",
        request_id=request_id,
        provider_engine=provider_engine,
        provider_model=provider_model,
        reason_code=reason_code,
        retryable=True,
        raw_diagnostics=RawDiagnostics(
            transport_exception_class=error_name,
            transport_exception_message=message,
        ),
    )


def _reason_code_from_exception_name(*, error_name: str, message: str) -> str:
    normalized = error_name.lower()
    if "connecttimeout" in normalized:
        return "connection_timeout"
    if "readtimeout" in normalized:
        return "read_timeout"
    if "proxy" in normalized or "proxy" in message:
        return "proxy_connection_failed"
    if "connecterror" in normalized and "refused" in message:
        return "connection_refused"
    if "connecterror" in normalized and "tls" in message:
        return "tls_error"
    if "connecterror" in normalized and "name" in message:
        return "dns_resolution_failed"
    return "network_connection_failed"


class _FallbackAsyncClient:
    def __init__(self, *, config: TransportClientConfig) -> None:
        self.base_url = config.base_url or ""
        self.headers = dict(config.headers)
        self.proxy = config.proxy
        self.timeout = dict(config.timeout)
        self.limits = dict(config.limits)

    async def aclose(self) -> None:
        return None
