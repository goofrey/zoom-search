# Zoom Search

<table>
  <tr>
    <td align="center" colspan="3">
      <strong>Better Answers, Bounded Extra Cost</strong><br />
      Direct search baseline vs Zoom Search workflow
    </td>
  </tr>
  <tr>
    <td align="center"><strong>Useful results</strong></td>
    <td align="center"><strong>Answer quality</strong></td>
    <td align="center"><strong>Extra budget</strong></td>
  </tr>
  <tr>
    <td align="center"><strong>1-5 -&gt; 4-12</strong><br />more good sources</td>
    <td align="center"><strong>2.0-7.2 -&gt; 7.8-8.7</strong><br />stronger final answers</td>
    <td align="center"><strong>+5.9s to +12.2s</strong><br />+2.3k to +5.1k tokens</td>
  </tr>
</table>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-3776AB" alt="Python >=3.10" />
  <img src="https://img.shields.io/badge/license-MIT-0F766E" alt="License: MIT" />
  <img src="https://img.shields.io/badge/package-zoom--search-2563EB" alt="Package: zoom-search" />
  <img src="https://img.shields.io/badge/tests-pytest-0F172A" alt="Tests: pytest" />
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#real-provider-example">Providers</a> ·
  <a href="#streaming">Streaming</a> ·
  <a href="./docs/agent-integration.md">Agents</a> ·
  <a href="./docs/benchmarks.md">Benchmarks</a> ·
  <a href="./docs/advanced-configuration.md">Advanced Configuration</a>
</p>

Zoom Search is a search and evidence tool for AI agents. It helps agents rewrite search questions, gather broader web evidence, zoom into high-value source domains, and return sourced answers with metrics.

It is built for agentic applications that need stronger source discovery, traceability, and answer grounding than a single search call.

## Why Zoom Search

- **Agent search tool**: expose structured answers, sources, warnings, and metrics for tool-calling agents.
- **Better evidence gathering**: rewrite agent questions into stronger search variants.
- **Source-domain zoom-in**: search broadly first, then focus on high-value domains.
- **Traceable outputs**: preserve source domains, duplicate provenance, warnings, and runtime metrics.
- **MCP/LangGraph ready**: use Zoom Search through MCP or LangGraph integrations.
- **Provider-flexible**: use built-in engines or custom OpenAI-compatible and native HTTP providers.

## Install

```bash
pip install zoom-search
```

## Quickstart

Run a deterministic local demo without API keys:

```python
import asyncio

from zoom_search import search


async def main() -> None:
    response = await search(
        question="What hotels in Shenzhen have rooms with exercise bikes?",
        demo_mode=True,
        output_mode="answer_with_sources",
        seed=7,
    )
    print(response.answer)
    print(response.results)


asyncio.run(main())
```

## Real Provider Example

```python
import asyncio

from zoom_search import search


async def main() -> None:
    response = await search(
        question="Which vector databases support hybrid search and metadata filtering for Python apps?",
        llm_engine="gemini",
        llm_model="gemini-2.5-flash",
        llm_api_key="YOUR_GEMINI_API_KEY",
        search_engine="tavily",
        search_api_key="YOUR_TAVILY_API_KEY",
        output_mode="answer_with_sources",
    )
    print(response.answer)
    print(response.search_context)


asyncio.run(main())
```

## Common Usage

Return only normalized search results:

```python
response = await search(
    question="Latest SQLite performance improvements",
    demo_mode=True,
    output_mode="results_simple",
)
```

Use recent conversation context:

```python
response = await search(
    question="What about hotels with in-room fitness equipment?",
    previous_conversation=[
        "I am planning a business trip to Shenzhen.",
        "I prefer hotels with wellness facilities.",
    ],
    demo_mode=True,
    output_mode="answer_with_sources",
)
```

## Streaming

```python
import asyncio

from zoom_search import astream_search


async def main() -> None:
    async for event in astream_search(
        question="What hotels in Shenzhen have rooms with exercise bikes?",
        demo_mode=True,
        output_mode="answer_with_sources",
        seed=7,
    ):
        if event.type == "answer_delta":
            print(event.text, end="")
        if event.type == "completed":
            print(event.response.request_id)


asyncio.run(main())
```

## Benchmarks

Historical evaluations compare direct search against the Zoom Search agent workflow, showing better useful result coverage and stronger final answers with bounded extra time and token cost.

| Case | Good results | Answer quality | Extra time | Extra tokens |
|---|---:|---:|---:|---:|
| Playwright authentication reuse | 5 -> 7 | 6.6 -> 8.7 | +5.89s | +2,324 |
| GitHub Actions secrets inherit | 1 -> 4 | 2.0 -> 7.8 | +8.93s | +2,936 |
| Hydrangea pruning comparison | 4 -> 12 | 7.2 -> 8.4 | +12.17s | +5,073 |

See the full benchmark notes in [`docs/benchmarks.md`](./docs/benchmarks.md).

## Examples

Runnable examples are available in the `examples/` directory:

```bash
python examples/demo_mode.py
python examples/streaming.py
python examples/conversation_history.py
```

For MCP and LangGraph tool usage, see [`docs/agent-integration.md`](./docs/agent-integration.md).

## Documentation

- Advanced configuration: https://github.com/goofrey/zoom-search/blob/main/docs/advanced-configuration.md
- Agent integration: https://github.com/goofrey/zoom-search/blob/main/docs/agent-integration.md
- Development checks: https://github.com/goofrey/zoom-search/blob/main/docs/development.md
- Benchmarks: https://github.com/goofrey/zoom-search/blob/main/docs/benchmarks.md

## License

Zoom Search is open source under the [MIT License](./LICENSE).
