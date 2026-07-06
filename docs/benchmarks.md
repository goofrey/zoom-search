# Benchmarks

These benchmark notes summarize historical evaluation runs comparing a direct search baseline with Zoom Search.

The baseline uses the original question, one search request, and answer synthesis. Zoom Search uses query rewriting, broad search, source-domain zoom-in, deduplication, evidence formatting, and answer synthesis.

The numbers below are representative historical runs, not guaranteed live rerun scores. Search indexes, provider behavior, and LLM outputs can change over time.

## Summary

| Case | Good results | Final answer quality | Extra time | Extra LLM tokens | Extra search requests |
|---|---:|---:|---:|---:|---:|
| Playwright authentication reuse | 5 -> 7 | 6.6 -> 8.7 | +5.89s | +2,324 | +2 |
| GitHub Actions secrets inherit | 1 -> 4 | 2.0 -> 7.8 | +8.93s | +2,936 | +2 |
| Hydrangea pruning comparison | 4 -> 12 | 7.2 -> 8.4 | +12.17s | +5,073 | +7 |

## What the Metrics Mean

- `Good results`: manually judged URLs that are directly useful for answering the exact question after URL deduplication.
- `Final answer quality`: subjective 0-10 score for factual usefulness, grounding, readability, and citation quality.
- `Extra time`: additional wall-clock time spent by Zoom Search compared with the direct search baseline.
- `Extra LLM tokens`: additional total LLM tokens used by Zoom Search compared with the direct search baseline.
- `Extra search requests`: additional search API requests used by Zoom Search.

## Representative Cases

### Playwright Authentication Reuse

Question:

```text
In Playwright Test, how do I authenticate once in a setup project, save the authenticated browser state with storageState, and reuse it across the rest of the test suite?
```

Result:

- Good result count improved from `5` to `7`.
- Final answer quality improved from `6.6 / 10` to `8.7 / 10`.
- Zoom Search added `+5.89s`, `+2,324` LLM tokens, and `+2` search requests.
- The gain came from zooming into `playwright.dev` and adding more authoritative evidence around `storageState`.

### GitHub Actions Secrets Inherit

Question:

```text
In GitHub Actions, how do I call a reusable workflow from another workflow and pass repository secrets with secrets: inherit?
```

Result:

- Good result count improved from `1` to `4`.
- Final answer quality improved from `2.0 / 10` to `7.8 / 10`.
- Zoom Search added `+8.93s`, `+2,936` LLM tokens, and `+2` search requests.
- The gain came from rewriting the natural-language question into syntax-focused search terms that surfaced `secrets: inherit` evidence.

### Hydrangea Pruning Comparison

Question:

```text
For pruning hydrangeas in a home garden, how should the timing and amount of cutting differ between bigleaf hydrangeas and panicle hydrangeas?
```

Result:

- Good result count improved from `4` to `12`.
- Final answer quality improved from `7.2 / 10` to `8.4 / 10`.
- Zoom Search added `+12.17s`, `+5,073` LLM tokens, and `+7` search requests.
- The gain came from splitting the comparison into branch-specific searches for bigleaf and panicle hydrangeas.

## Takeaway

Zoom Search trades a bounded extra budget for better evidence coverage and stronger final answers. It is most useful when the answer depends on exact source discovery, source authority, comparison branches, or syntax-sensitive evidence.
