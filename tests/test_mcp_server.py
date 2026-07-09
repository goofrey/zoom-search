from dataclasses import dataclass

from zoom_search.mcp_server import _build_search_params
from zoom_search.mcp_server import _to_jsonable


def test_build_search_params_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("ZOOM_SEARCH_LLM_ENGINE", "deepseek")
    monkeypatch.setenv("ZOOM_SEARCH_LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("ZOOM_SEARCH_LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("ZOOM_SEARCH_SEARCH_ENGINE", "serpapi")
    monkeypatch.setenv("ZOOM_SEARCH_SEARCH_API_KEY", "test-search-key")

    params = _build_search_params(
        question="What changed?",
        previous_conversation=["Earlier context"],
        output_mode="answer_with_sources",
        demo_mode=None,
        seed=7,
        zoomout_num_results=4,
        zoomin_num_results=3,
        top_k_domains_per_query=2,
        include_raw_diagnostics=True,
    )

    assert params["question"] == "What changed?"
    assert params["previous_conversation"] == ["Earlier context"]
    assert params["output_mode"] == "answer_with_sources"
    assert params["demo_mode"] is False
    assert params["seed"] == 7
    assert params["zoomout_num_results"] == 4
    assert params["zoomin_num_results"] == 3
    assert params["top_k_domains_per_query"] == 2
    assert params["include_raw_diagnostics"] is True
    assert params["llm"] == {
        "engine": "deepseek",
        "model": "deepseek-v4-flash",
        "api_key": "test-llm-key",
    }
    assert params["search"] == {
        "engine": "serpapi",
        "api_key": "test-search-key",
    }


def test_build_search_params_demo_mode_argument_wins(monkeypatch) -> None:
    monkeypatch.setenv("ZOOM_SEARCH_DEMO_MODE", "false")

    params = _build_search_params(
        question="Demo?",
        previous_conversation=None,
        output_mode="answer_with_sources",
        demo_mode=True,
        seed=None,
        zoomout_num_results=5,
        zoomin_num_results=5,
        top_k_domains_per_query=1,
    )

    assert params["demo_mode"] is True
    assert params["include_raw_diagnostics"] is False
    assert params["previous_conversation"] == []
    assert "llm" not in params
    assert "search" not in params


def test_to_jsonable_converts_dataclasses() -> None:
    @dataclass
    class Item:
        name: str
        values: list[int]

    assert _to_jsonable({"item": Item(name="source", values=[1, 2])}) == {
        "item": {"name": "source", "values": [1, 2]}
    }
