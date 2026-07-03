from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_regression_benchmark_catalog_checker_passes() -> None:
    result = subprocess.run(
        [sys.executable, "tools/check_regression_benchmarks.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
