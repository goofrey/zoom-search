"""Runtime context helpers."""

from __future__ import annotations

from uuid import uuid4

from zoom_search.metrics import create_runtime_metrics
from zoom_search.models import ResolvedProvider
from zoom_search.models import RuntimeContext
from zoom_search.models import SearchRequest
from zoom_search.transport import create_transport_context


def build_runtime_context(
    *,
    request: SearchRequest,
    llm_provider: ResolvedProvider,
    search_provider: ResolvedProvider,
) -> RuntimeContext:
    request_id = _generate_request_id()
    return RuntimeContext(
        request_id=request_id,
        request=request,
        llm_provider=llm_provider,
        search_provider=search_provider,
        retry_budget={"llm": 1, "search": 1},
        transport_context=create_transport_context_placeholder(
            request=request,
            llm_provider=llm_provider,
            search_provider=search_provider,
        ),
        semaphore_limits={"search_requests": 10},
        metrics=create_runtime_metrics(),
    )


def _generate_request_id() -> str:
    return f"zs_{uuid4().hex}"


def create_transport_runtime(context: RuntimeContext):
    return create_transport_context(context=context)


def create_transport_context_placeholder(*, request: SearchRequest, llm_provider: ResolvedProvider, search_provider: ResolvedProvider) -> dict[str, str | None]:
    return {
        "global_proxy": request.proxy.http_proxy,
        "llm_proxy": llm_provider.http_proxy,
        "search_proxy": search_provider.http_proxy,
    }
