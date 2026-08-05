from __future__ import annotations

from datetime import datetime

from src.tools.results_paths import (
    build_run_dir_name,
    is_under_paper_results,
    is_valid_run_dir_name,
    results_contract_for_prompt,
    results_summary_relpath,
    slugify_label,
)


def test_slugify_and_human_readable_run_dir_name() -> None:
    when = datetime(2026, 7, 29, 10, 30, 0)
    name = build_run_dir_name(
        benchmark="Townsend Function (2D)",
        method="BE-CBO",
        seed=1,
        when=when,
    )
    assert name == "2026-07-29_10-30-00__townsend-function-2d__be-cbo__seed-01"
    assert is_valid_run_dir_name(name)
    assert slugify_label("Speed Reducer Design") == "speed-reducer-design"


def test_results_paths_are_paper_scoped() -> None:
    assert results_summary_relpath("boundary_exploration_bo") == (
        "results/boundary_exploration_bo/summary.json"
    )
    assert is_under_paper_results(
        "results/boundary_exploration_bo/summary.json",
        "boundary_exploration_bo",
    )
    assert is_under_paper_results(
        "results/boundary_exploration_bo/2026-07-29_10-30-00__townsend__be-cbo__seed-01/metrics.csv",
        "boundary_exploration_bo",
    )
    assert not is_under_paper_results("/tmp/out.json", "boundary_exploration_bo")
    assert not is_under_paper_results(
        "results/other_paper/summary.json",
        "boundary_exploration_bo",
    )


def test_results_contract_for_prompt_lists_filenames() -> None:
    contract = results_contract_for_prompt("paper_1")
    assert contract["results_root"] == "results/paper_1"
    assert contract["summary_path"] == "results/paper_1/summary.json"
    assert contract["metrics_filename"] == "metrics.csv"
    assert contract["logs_filename"] == "logs.txt"
    assert "benchmark_slug" in contract["run_dir_pattern"]
