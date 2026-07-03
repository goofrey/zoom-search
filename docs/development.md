# Development

## Evaluation Assets

- Historical showcase cases: `specs/zoom-search/evaluation/golden-tests.md`
- Regression benchmarks: `specs/zoom-search/evaluation/regression-benchmarks.md`
- Failure tests: `specs/zoom-search/evaluation/failure-tests.md`
- Report library: `specs/zoom-search/evaluation/reports/`
- Regression checker: `python3 tools/check_regression_benchmarks.py`

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
