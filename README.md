# Zoom Search

<table>
  <tr>
    <td align="center" colspan="3">
      <h2>Better Answers, Bounded Extra Cost</h2>
      <strong>Direct search baseline vs Zoom Search workflow</strong>
    </td>
  </tr>
  <tr>
    <td align="center"><h3>Useful results</h3></td>
    <td align="center"><h3>Answer quality</h3></td>
    <td align="center"><h3>Extra budget</h3></td>
  </tr>
  <tr>
    <td align="center"><h2>1-5 -&gt; 4-12</h2>more good sources</td>
    <td align="center"><h2>2.0-7.2 -&gt; 7.8-8.7</h2>stronger final answers</td>
    <td align="center"><h2>+5.9s to +12.2s</h2>+2.3k to +5.1k tokens</td>
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
  <a href="#agent-tool-example">Agent Tool</a> ·
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

## Agent Tool Example

Install the MCP extra:

```bash
pip install "zoom-search[mcp]"
```

Add Zoom Search to your MCP client:

```json
{
  "mcpServers": {
    "zoom-search": {
      "command": "zoom-search-mcp",
      "env": {
        "ZOOM_SEARCH_LLM_ENGINE": "gemini",
        "ZOOM_SEARCH_LLM_MODEL": "gemini-2.5-flash",
        "ZOOM_SEARCH_LLM_API_KEY": "YOUR_GEMINI_API_KEY",
        "ZOOM_SEARCH_SEARCH_ENGINE": "tavily",
        "ZOOM_SEARCH_SEARCH_API_KEY": "YOUR_TAVILY_API_KEY"
      }
    }
  }
}
```

Your agent can then call the `zoom_search` tool to get sourced answers, source-domain zoom-in, warnings, and runtime metrics.

Or wrap it as a LangGraph/LangChain tool:

```python
from langchain.tools import tool

from zoom_search import search


@tool
async def zoom_search_evidence(query: str) -> dict:
    response = await search(
        question=query,
        output_mode="answer_with_sources",
    )
    return response.to_dict()
```

See [`docs/agent-integration.md`](./docs/agent-integration.md) for MCP client configuration and provider environment variables.

## Benchmarks

Historical evaluations compare direct search against the Zoom Search agent workflow, showing better useful result coverage and stronger final answers with bounded extra time and token cost.

| Case | Good results | Answer quality | Extra time | Extra tokens |
|---|---:|---:|---:|---:|
| Playwright authentication reuse | 5 -> 7 | 6.6 -> 8.7 | +5.89s | +2,324 |
| GitHub Actions secrets inherit | 1 -> 4 | 2.0 -> 7.8 | +8.93s | +2,936 |
| Hydrangea pruning comparison | 4 -> 12 | 7.2 -> 8.4 | +12.17s | +5,073 |

See the full benchmark notes in [`docs/benchmarks.md`](./docs/benchmarks.md).

Runnable examples for demo mode, streaming, conversation history, and LangGraph are available in the `examples/` directory.

## Documentation

- Advanced configuration: https://github.com/goofrey/zoom-search/blob/main/docs/advanced-configuration.md
- Agent integration: https://github.com/goofrey/zoom-search/blob/main/docs/agent-integration.md
- Development checks: https://github.com/goofrey/zoom-search/blob/main/docs/development.md
- Benchmarks: https://github.com/goofrey/zoom-search/blob/main/docs/benchmarks.md

## License

Zoom Search is open source under the [MIT License](./LICENSE).
