"""Load benchmark ground-truth metrics for synthetic run-plan papers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from src.config import ROOT_DIR
from src.state import ReportedResult

BENCHMARK_DIR = ROOT_DIR / "benchmark"
SYNTHETIC_BENCHMARK_PAPER_IDS = frozenset({"synthetic_optimize", "synthetic_graph"})

# Table 1 in benchmark/papers/synthetic_optimize.md (mean simple regret, lower is better).
OPTIMIZE_TABLE_MEANS: list[tuple[str, str, float]] = [
    ("sphere", "random_search", 10.9768),
    ("sphere", "bayesian_optimization", 0.0029),
    ("rastrigin", "random_search", 48.4342),
    ("rastrigin", "bayesian_optimization", 32.7766),
    ("ackley", "random_search", 3.6163),
    ("ackley", "bayesian_optimization", 0.1462),
    ("rosenbrock", "random_search", 256.6105),
    ("rosenbrock", "bayesian_optimization", 16.8361),
    ("griewank", "random_search", 38.5694),
    ("griewank", "bayesian_optimization", 0.7424),
]

GRAPH_VALIDATION_TASKS: list[tuple[str, str]] = [
    ("dfs", "simple_undirected"),
    ("bfs", "simple_undirected"),
    ("dijkstra", "weighted_shortest_path"),
    ("floyd_warshall", "weighted_shortest_path"),
    ("kruskal", "minimum_spanning_tree"),
]


def is_benchmark_paper(paper_id: str) -> bool:
    return paper_id in SYNTHETIC_BENCHMARK_PAPER_IDS


def _run_graph_module():
    path = BENCHMARK_DIR / "run_graph.py"
    module_name = "benchmark_run_graph"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load benchmark runner at {path}")
    module = importlib.util.module_from_spec(spec)
    benchmark_path = str(BENCHMARK_DIR)
    if benchmark_path not in sys.path:
        sys.path.insert(0, benchmark_path)
    spec.loader.exec_module(module)
    return module


def _load_graphs() -> dict:
    graphs_path = BENCHMARK_DIR / "graphs.json"
    if not graphs_path.exists():
        raise FileNotFoundError(f"Missing benchmark graphs file: {graphs_path}")
    payload = json.loads(graphs_path.read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if isinstance(value, dict) and "nodes" in value}


def _optimize_expectations() -> list[ReportedResult]:
    rows: list[ReportedResult] = []
    for benchmark, algorithm, value in OPTIMIZE_TABLE_MEANS:
        text = str(value)
        rows.append(
            ReportedResult(
                benchmark=benchmark,
                algorithm=algorithm,
                metric_name="simple_regret",
                value=text,
                source="benchmark/papers/synthetic_optimize.md Table 1",
            )
        )
        rows.append(
            ReportedResult(
                benchmark=benchmark,
                algorithm=algorithm,
                metric_name="final_value",
                value=text,
                source="benchmark/papers/synthetic_optimize.md Table 1",
            )
        )
    return rows


def _graph_expectations() -> list[ReportedResult]:
    run_graph = _run_graph_module()
    graphs = _load_graphs()
    rows: list[ReportedResult] = []
    for algorithm, graph_name in GRAPH_VALIDATION_TASKS:
        if graph_name not in graphs:
            raise KeyError(f"Graph {graph_name!r} missing from benchmark/graphs.json")
        metrics = run_graph.run_algorithm(algorithm, graph_name, graphs[graph_name])
        for item in metrics:
            rows.append(
                ReportedResult(
                    benchmark=str(item["benchmark"]),
                    algorithm=str(item["algorithm"]),
                    metric_name=str(item["metric_name"]),
                    value=json.dumps(item["value"]) if isinstance(item["value"], (list, dict)) else str(item["value"]),
                    source="benchmark/graphs.json ground_truth",
                )
            )
    return rows


def load_benchmark_expectations(paper_id: str) -> list[ReportedResult]:
    """Return reference metrics the Reviewer should grade Engineer output against."""
    if paper_id == "synthetic_optimize":
        return _optimize_expectations()
    if paper_id == "synthetic_graph":
        return _graph_expectations()
    raise ValueError(f"No benchmark expectations configured for paper_id={paper_id!r}")


def resolve_review_expectations(paper_id: str, extraction_path: Path | None) -> list[ReportedResult]:
    """Synthetic benchmarks grade Engineer output against harness ground truth."""
    if is_benchmark_paper(paper_id):
        return load_benchmark_expectations(paper_id)
    from src.persistence import load_reported_results

    return load_reported_results(extraction_path)
