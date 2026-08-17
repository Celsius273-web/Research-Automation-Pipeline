"""Run one graph algorithm and write CapturedMetric JSON for the Executor."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from graph_algorithms import bfs, dfs, dijkstra, floyd_warshall, kruskal_mst_weight
from graph_utils import (
    adjacency_from_graph,
    default_graphs_path,
    load_graphs,
    weighted_adjacency_from_graph,
)

logger = logging.getLogger(__name__)
BENCHMARK_DIR = Path(__file__).resolve().parent


def _metric(benchmark: str, algorithm: str, metric_name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, (list, dict)):
        stored: float | str = json.dumps(value)
    else:
        stored = value
    return {
        "benchmark": benchmark,
        "algorithm": algorithm,
        "metric_name": metric_name,
        "value": stored,
        "source": "run_graph.py",
    }


def run_algorithm(algorithm: str, graph_name: str, graph: dict[str, Any]) -> list[dict[str, Any]]:
    adj = adjacency_from_graph(graph)
    weighted = weighted_adjacency_from_graph(graph)
    nodes = [int(node) for node in graph["nodes"]]
    if algorithm == "dfs":
        order = dfs(adj, 0)
        return [
            _metric(graph_name, algorithm, "output", order),
            _metric(graph_name, algorithm, "n_visited", len(order)),
        ]
    if algorithm == "bfs":
        order = bfs(adj, 0)
        return [
            _metric(graph_name, algorithm, "output", order),
            _metric(graph_name, algorithm, "n_visited", len(order)),
        ]
    if algorithm == "dijkstra":
        length, path = dijkstra(weighted, 0, 9)
        return [
            _metric(graph_name, algorithm, "path_length", round(float(length), 6)),
            _metric(graph_name, algorithm, "output", path),
        ]
    if algorithm == "floyd_warshall":
        distances = floyd_warshall(nodes, weighted)
        serializable = {
            str(u): {str(v): round(float(dist), 6) for v, dist in row.items()}
            for u, row in distances.items()
        }
        sample = distances.get(0, {}).get(9)
        rows = [_metric(graph_name, algorithm, "output", serializable)]
        if sample is not None:
            rows.append(_metric(graph_name, algorithm, "path_length_0_to_9", round(float(sample), 6)))
        return rows
    if algorithm == "kruskal":
        weight = kruskal_mst_weight(nodes, graph["edges"])
        return [_metric(graph_name, algorithm, "mst_weight", round(float(weight), 6))]
    raise ValueError(f"Unknown algorithm {algorithm!r}. Expected dfs, bfs, dijkstra, floyd_warshall, or kruskal.")


def write_metrics(metrics: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run a graph algorithm on graphs.json")
    parser.add_argument("--algorithm", required=True, choices=["dfs", "bfs", "dijkstra", "floyd_warshall", "kruskal"])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True, help="Output CapturedMetric JSON path")
    parser.add_argument("--graphs-path", default=str(default_graphs_path()))
    args = parser.parse_args()

    graphs_path = Path(args.graphs_path)
    if not graphs_path.is_absolute():
        graphs_path = (BENCHMARK_DIR / graphs_path).resolve()
    try:
        graphs = load_graphs(graphs_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if args.graph not in graphs:
        raise SystemExit(f"Unknown test graph {args.graph!r}. Available: {sorted(graphs)}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (BENCHMARK_DIR / out_path).resolve()
    metrics = run_algorithm(args.algorithm, args.graph, graphs[args.graph])
    write_metrics(metrics, out_path)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
