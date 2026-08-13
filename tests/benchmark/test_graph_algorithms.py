"""Reference graph algorithms must match NetworkX ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from graph_algorithms import (  # noqa: E402
    adjacency_from_graph,
    bfs,
    dfs,
    dijkstra,
    floyd_warshall,
    kruskal_mst_weight,
    weighted_adjacency_from_graph,
)
from run_graph import run_algorithm  # noqa: E402
from setup_graph import create_test_graphs  # noqa: E402


def test_dfs_and_bfs_match_simple_undirected_ground_truth() -> None:
    graphs = create_test_graphs()
    simple = graphs["simple_undirected"]
    adj = adjacency_from_graph(simple)
    assert dfs(adj, 0) == simple["ground_truth"]["dfs_order_from_0"]
    assert bfs(adj, 0) == simple["ground_truth"]["bfs_order_from_0"]


def test_dijkstra_and_kruskal_match_numeric_ground_truth() -> None:
    graphs = create_test_graphs()
    weighted = graphs["weighted_shortest_path"]
    mst = graphs["minimum_spanning_tree"]
    length, _path = dijkstra(weighted_adjacency_from_graph(weighted), 0, 9)
    assert length == weighted["ground_truth"]["dijkstra_path_length"]
    assert kruskal_mst_weight(mst["nodes"], mst["edges"]) == mst["ground_truth"]["mst_weight"]


def test_floyd_warshall_matches_dense_graph_sample() -> None:
    graphs = create_test_graphs()
    dense = graphs["dense_graph"]
    distances = floyd_warshall(
        [int(node) for node in dense["nodes"]],
        weighted_adjacency_from_graph(dense),
    )
    expected = dense["ground_truth"]["floyd_warshall"]
    assert round(distances[0][9], 6) == expected["0"]["9"]


def test_run_algorithm_emits_captured_metric_rows() -> None:
    graphs = create_test_graphs()
    rows = run_algorithm("dfs", "simple_undirected", graphs["simple_undirected"])
    assert rows[0]["metric_name"] == "output"
    assert rows[0]["algorithm"] == "dfs"
    assert rows[0]["source"] == "run_graph.py"
