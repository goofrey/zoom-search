from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "specs/zoom-search/evaluation/regression-benchmarks.md"


def test_regression_benchmark_catalog_checker_passes() -> None:
    if not CATALOG_PATH.exists():
        pytest.skip("regression benchmark catalog is not included in the public package")

    result = subprocess.run(
        [sys.executable, "tools/check_regression_benchmarks.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
