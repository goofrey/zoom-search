# Development

## Benchmarks

See [Benchmarks](benchmarks.md) for the public methodology, scenarios, and results.

## Test Checks

```bash
# Run tests with uv.
uv run pytest
```

```bash
# Run tests without uv after installing .[dev].
python3 -m pytest
```

```bash
# Check package import.
python3 -c "import zoom_search; print(zoom_search.__version__)"
```
