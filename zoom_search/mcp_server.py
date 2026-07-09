"""MCP server adapter for exposing Zoom Search as a tool."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import is_dataclass
import os
from typing import Any
from typing import Literal

from zoom_search import search


OutputModeName = Literal["answer", "answer_with_sources", "results_simple", "results_detailed"]

ENV_LLM_ENGINE = "ZOOM_SEARCH_LLM_ENGINE"
ENV_LLM_MODEL = "ZOOM_SEARCH_LLM_MODEL"
ENV_LLM_API_KEY = "ZOOM_SEARCH_LLM_API_KEY"
ENV_LLM_BASE_URL = "ZOOM_SEARCH_LLM_BASE_URL"
ENV_SEARCH_ENGINE = "ZOOM_SEARCH_SEARCH_ENGINE"
ENV_SEARCH_API_KEY = "ZOOM_SEARCH_SEARCH_API_KEY"
ENV_SEARCH_BASE_URL = "ZOOM_SEARCH_SEARCH_BASE_URL"
ENV_DEMO_MODE = "ZOOM_SEARCH_DEMO_MODE"


def create_server():
    """Create the MCP server instance.

    The MCP SDK is imported here so the base package can be used without the
    optional mcp dependency installed.
    """

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised when optional dependency is absent
        raise RuntimeError("Install MCP support with `pip install zoom-search[mcp]`.") from exc

    mcp = FastMCP(
        "Zoom Search",
        instructions=(
            "Use the zoom_search tool when an agent needs grounded web evidence, "
            "source URLs, warnings, and runtime metrics."
        ),
    )

    @mcp.tool()
    async def zoom_search(
        question: str,
        previous_conversation: list[str] | None = None,
        output_mode: OutputModeName = "answer_with_sources",
        demo_mode: bool | None = None,
        seed: int | None = None,
        zoomout_num_results: int = 5,
        zoomin_num_results: int = 5,
        top_k_domains_per_query: int = 1,
        include_raw_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Run Zoom Search and return answer, sources, warnings, metrics, and evidence."""

        params = _build_search_params(
            question=question,
            previous_conversation=previous_conversation,
            output_mode=output_mode,
            demo_mode=demo_mode,
            seed=seed,
            zoomout_num_results=zoomout_num_results,
            zoomin_num_results=zoomin_num_results,
            top_k_domains_per_query=top_k_domains_per_query,
            include_raw_diagnostics=include_raw_diagnostics,
        )
        response = await search(**params)
        return _to_jsonable(response.to_dict())

    return mcp


def main() -> None:
    """Run the Zoom Search MCP server over stdio."""

    create_server().run("stdio")


def _build_search_params(
    *,
    question: str,
    previous_conversation: list[str] | None,
    output_mode: OutputModeName,
    demo_mode: bool | None,
    seed: int | None,
    zoomout_num_results: int,
    zoomin_num_results: int,
    top_k_domains_per_query: int,
    include_raw_diagnostics: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "question": question,
        "previous_conversation": list(previous_conversation or []),
        "output_mode": output_mode,
        "zoomout_num_results": zoomout_num_results,
        "zoomin_num_results": zoomin_num_results,
        "top_k_domains_per_query": top_k_domains_per_query,
        "include_raw_diagnostics": include_raw_diagnostics,
    }
    if seed is not None:
        params["seed"] = seed
    params["demo_mode"] = _resolve_demo_mode(demo_mode)

    llm_config = _provider_config(
        engine_env=ENV_LLM_ENGINE,
        api_key_env=ENV_LLM_API_KEY,
        base_url_env=ENV_LLM_BASE_URL,
        model_env=ENV_LLM_MODEL,
    )
    if llm_config:
        params["llm"] = llm_config

    search_config = _provider_config(
        engine_env=ENV_SEARCH_ENGINE,
        api_key_env=ENV_SEARCH_API_KEY,
        base_url_env=ENV_SEARCH_BASE_URL,
    )
    if search_config:
        params["search"] = search_config

    return params


def _resolve_demo_mode(value: bool | None) -> bool:
    if value is not None:
        return value
    return _env_bool(os.getenv(ENV_DEMO_MODE), default=False)


def _provider_config(
    *,
    engine_env: str,
    api_key_env: str,
    base_url_env: str,
    model_env: str | None = None,
) -> dict[str, str] | None:
    config: dict[str, str] = {}
    env_map = {
        "engine": engine_env,
        "api_key": api_key_env,
        "base_url": base_url_env,
    }
    if model_env is not None:
        env_map["model"] = model_env
    for field, env_name in env_map.items():
        value = os.getenv(env_name)
        if value:
            config[field] = value
    return config or None


def _env_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
