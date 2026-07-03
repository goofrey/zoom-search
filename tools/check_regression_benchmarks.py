from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "specs/zoom-search/evaluation/regression-benchmarks.md"
CATALOG_BLOCK_PATTERN = (
    r"<!-- regression-benchmark-catalog:start -->\s*```json\s*(.*?)\s*```\s*"
    r"<!-- regression-benchmark-catalog:end -->"
)


@dataclass
class ReportMetrics:
    token_delta: float
    extra_search_requests: float
    good_result_delta: float
    answer_quality_delta: float


def _extract_catalog(markdown_text: str) -> list[dict[str, object]]:
    block_match = re.search(CATALOG_BLOCK_PATTERN, markdown_text, re.DOTALL)
    if not block_match:
        raise ValueError("regression benchmark catalog block not found")
    return json.loads(block_match.group(1))


def _parse_number(value: str) -> float:
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"could not parse number from: {value!r}")
    return float(match.group(0).replace(",", ""))


def _find_table_row_cells(markdown_text: str, label: str) -> list[str] | None:
    row_match = re.search(rf"^\|\s*{re.escape(label)}\s*\|(.+?)\|\s*$", markdown_text, re.MULTILINE)
    if not row_match:
        return None
    return [cell.strip() for cell in row_match.group(1).split("|")]


def _extract_section(markdown_text: str, heading: str) -> str | None:
    section_match = re.search(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        markdown_text,
        re.MULTILINE | re.DOTALL,
    )
    if not section_match:
        return None
    return section_match.group(1)


def _extract_good_result_delta(markdown_text: str) -> float:
    row = _find_table_row_cells(markdown_text, "Good result count")
    if row and len(row) >= 3:
        return _parse_number(row[2])

    baseline_row = _find_table_row_cells(markdown_text, "Baseline good result count")
    zoom_row = _find_table_row_cells(markdown_text, "Zoom Search good result count")
    if baseline_row and zoom_row:
        return _parse_number(zoom_row[0]) - _parse_number(baseline_row[0])

    section = _extract_section(markdown_text, "Good Result Count")
    if section:
        baseline_match = re.search(r"Baseline good result count\s*\|\s*([0-9.]+)", section)
        zoom_match = re.search(r"Zoom Search good result count\s*\|\s*([0-9.]+)", section)
        if baseline_match and zoom_match:
            return float(zoom_match.group(1)) - float(baseline_match.group(1))

    raise ValueError("good result delta not found")


def _extract_answer_quality_delta(markdown_text: str) -> float:
    for label in ("Final answer quality score", "Answer quality score"):
        row = _find_table_row_cells(markdown_text, label)
        if row and len(row) >= 3:
            return _parse_number(row[2])

    for heading in ("Final Answer Quality", "Final Answer Quality Score"):
        section = _extract_section(markdown_text, heading)
        if not section:
            continue
        baseline_match = re.search(r"^- Baseline:\s*`?([0-9.]+)\s*/\s*10`?", section, re.MULTILINE)
        zoom_match = re.search(r"^- Zoom Search:\s*`?([0-9.]+)\s*/\s*10`?", section, re.MULTILINE)
        if baseline_match and zoom_match:
            return float(zoom_match.group(1)) - float(baseline_match.group(1))
        gain_match = re.search(r"^- Gain:\s*`?([+-]?[0-9.]+)`?", section, re.MULTILINE)
        if gain_match:
            return float(gain_match.group(1))

    raise ValueError("answer quality delta not found")


def _extract_token_delta(markdown_text: str) -> float:
    row = _find_table_row_cells(markdown_text, "Total LLM tokens")
    if row and len(row) >= 3:
        return _parse_number(row[2])
    raise ValueError("token delta not found")


def _extract_extra_search_requests(markdown_text: str) -> float:
    for label in (
        "Search requests planned",
        "Search calls observed",
        "Search planned",
        "Search requests attempted",
        "Search attempted",
    ):
        row = _find_table_row_cells(markdown_text, label)
        if row and len(row) >= 3:
            return _parse_number(row[2])
    raise ValueError("extra search requests not found")


def _extract_metrics(markdown_text: str) -> ReportMetrics:
    return ReportMetrics(
        token_delta=_extract_token_delta(markdown_text),
        extra_search_requests=_extract_extra_search_requests(markdown_text),
        good_result_delta=_extract_good_result_delta(markdown_text),
        answer_quality_delta=_extract_answer_quality_delta(markdown_text),
    )


def main() -> int:
    catalog = _extract_catalog(CATALOG_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    for item in catalog:
        case_id = str(item["id"])
        report_path = ROOT / str(item["report"])
        markdown_text = report_path.read_text(encoding="utf-8")
        metrics = _extract_metrics(markdown_text)
        thresholds = item["thresholds"]

        if metrics.good_result_delta < float(thresholds["min_good_result_delta"]):
            failures.append(
                f"{case_id}: good_result_delta={metrics.good_result_delta} < {thresholds['min_good_result_delta']}"
            )
        if metrics.answer_quality_delta < float(thresholds["min_answer_quality_delta"]):
            failures.append(
                f"{case_id}: answer_quality_delta={metrics.answer_quality_delta} < {thresholds['min_answer_quality_delta']}"
            )
        if metrics.token_delta > float(thresholds["max_token_delta"]):
            failures.append(
                f"{case_id}: token_delta={metrics.token_delta} > {thresholds['max_token_delta']}"
            )
        if metrics.extra_search_requests > float(thresholds["max_extra_search_requests"]):
            failures.append(
                f"{case_id}: extra_search_requests={metrics.extra_search_requests} > {thresholds['max_extra_search_requests']}"
            )

        for required_string in item.get("required_strings", []):
            if str(required_string) not in markdown_text:
                failures.append(f"{case_id}: missing required string {required_string!r}")

        print(
            f"PASS {case_id}: token_delta={metrics.token_delta:.0f}, "
            f"extra_search_requests={metrics.extra_search_requests:.0f}, "
            f"good_result_delta={metrics.good_result_delta:.1f}, "
            f"answer_quality_delta={metrics.answer_quality_delta:.1f}"
        )

    if failures:
        print("\nRegression benchmark check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nAll regression benchmarks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
