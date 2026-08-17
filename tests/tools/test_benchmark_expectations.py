"""Tests for synthetic benchmark expectation loading."""

from __future__ import annotations

from src.tools.benchmark_expectations import (
    load_benchmark_expectations,
    resolve_review_expectations,
)
from src.state import ReportedResult


def test_load_benchmark_expectations_optimize_has_table_rows() -> None:
    rows = load_benchmark_expectations("synthetic_optimize")
    assert len(rows) == 20
    sphere_rs = [
        row
        for row in rows
        if row.benchmark == "sphere"
        and row.algorithm == "random_search"
        and row.metric_name == "simple_regret"
    ]
    assert len(sphere_rs) == 1
    assert sphere_rs[0].value == "10.9768"


def test_load_benchmark_expectations_graph_matches_run_graph() -> None:
    rows = load_benchmark_expectations("synthetic_graph")
    dfs_output = next(
        row
        for row in rows
        if row.algorithm == "dfs" and row.metric_name == "output"
    )
    assert dfs_output.benchmark == "simple_undirected"
    assert dfs_output.value == "[0, 1, 3, 2]"


def test_resolve_review_expectations_uses_benchmark_for_synthetic(tmp_path) -> None:
    extraction = tmp_path / "bad.json"
    extraction.write_text(
        '{"merged": {"reported_results": [{"benchmark": "x", "metric_name": "y", "value": "1"}]}}',
        encoding="utf-8",
    )
    rows = resolve_review_expectations("synthetic_optimize", extraction)
    assert len(rows) == 20
    assert all(isinstance(row, ReportedResult) for row in rows)


def test_resolve_review_expectations_falls_back_to_extraction(tmp_path) -> None:
    extraction = tmp_path / "missing.json"
    rows = resolve_review_expectations("some_real_paper", extraction)
    assert rows == []
