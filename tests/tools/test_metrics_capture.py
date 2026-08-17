"""Unit tests for metrics capture from JSON/CSV result files."""

from __future__ import annotations

import json
from pathlib import Path

from src.tools.metrics_capture import (
    load_metrics_from_path,
    load_metrics_from_text,
    merge_unique_metrics,
)
from src.state import CapturedMetric

def test_load_metrics_from_stdout_object() -> None:
    metrics, error = load_metrics_from_text(
        json.dumps(
            {
                "function": "sphere",
                "optimizer": "random_search",
                "seed": 0,
                "simple_regret": 0.4,
                "final_value": 0.4,
            }
        ),
        default_benchmark="sphere",
    )
    assert error is None
    names = {item.metric_name: item.value for item in metrics}
    assert names["simple_regret"] == 0.4
    assert names["final_value"] == 0.4
    assert metrics[0].algorithm == "random_search"
    assert metrics[0].seed == "0"


def test_load_metrics_from_json_list(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            [
                {"benchmark": "lsq", "metric_name": "regret", "value": 0.042, "source": "summary.json"},
                {"benchmark": "lsq", "metric_name": "runtime", "value": "12.5"},
            ]
        ),
        encoding="utf-8",
    )
    metrics, error = load_metrics_from_path(path)
    assert error is None
    assert len(metrics) == 2
    assert metrics[0].benchmark == "lsq"
    assert metrics[0].metric_name == "regret"
    assert metrics[0].value == 0.042


def test_load_metrics_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text(
        "benchmark,metric_name,value\n"
        "gas_compressor,regret,0.1\n"
        "gas_compressor,runtime,3.2\n",
        encoding="utf-8",
    )
    metrics, error = load_metrics_from_path(path)
    assert error is None
    assert len(metrics) == 2
    assert metrics[1].metric_name == "runtime"
    assert metrics[1].value == 3.2


def test_missing_results_path_returns_error(tmp_path: Path) -> None:
    metrics, error = load_metrics_from_path(tmp_path / "missing.json")
    assert metrics == []
    assert error is not None
    assert "does not exist" in error


def test_merge_unique_metrics_replaces_same_key() -> None:
    existing = [CapturedMetric(benchmark="lsq", metric_name="regret", value=0.1, source="a")]
    incoming = [CapturedMetric(benchmark="lsq", metric_name="regret", value=0.2, source="b")]
    merged = merge_unique_metrics(existing, incoming)
    assert len(merged) == 1
    assert merged[0].value == 0.2


def test_merge_keeps_distinct_algorithms() -> None:
    existing = [
        CapturedMetric(benchmark="lsq", algorithm="be-cbo", metric_name="best_objective", value=1.0),
    ]
    incoming = [
        CapturedMetric(benchmark="lsq", algorithm="cei", metric_name="best_objective", value=2.0),
    ]
    merged = merge_unique_metrics(existing, incoming)
    assert len(merged) == 2
    by_algo = {item.algorithm: item.value for item in merged}
    assert by_algo["be-cbo"] == 1.0
    assert by_algo["cei"] == 2.0


def test_optimization_metrics_keep_optimizer_and_seed(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "function": "sphere",
                "optimizer": "random_search",
                "seed": 1,
                "simple_regret": 0.25,
            }
        ),
        encoding="utf-8",
    )

    metrics, error = load_metrics_from_path(path, default_benchmark="sphere")

    assert error is None
    assert [(item.algorithm, item.seed, item.metric_name) for item in metrics] == [
        ("random_search", "1", "simple_regret")
    ]


def test_load_metrics_from_pickle_stem(tmp_path: Path) -> None:
    import pickle

    stem = tmp_path / "results" / "run0"
    stem.parent.mkdir(parents=True)
    payload = {
        "fun_name": "lsq",
        "algo_name": "be-cbo",
        "Y": [-1.0, -0.5, -0.2],
    }
    with (tmp_path / "results" / "run0.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    metrics, error = load_metrics_from_path(stem, default_benchmark="lsq")
    assert error is None
    assert all(item.algorithm == "be-cbo" for item in metrics)
    names = {item.metric_name: item.value for item in metrics}
    assert names["final_objective"] == -0.2
    assert names["best_objective"] == -1.0
    assert names["n_evaluations"] == 3.0
