"""Tests for NetworkX graph generation and ground truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from setup_graph import (  # noqa: E402
    GRAPH_NAMES,
    compute_ground_truth,
    create_test_graphs,
    write_graphs_json,
)


def _graph_from_payload(payload: dict) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(payload["nodes"])
    for edge in payload["edges"]:
        if len(edge) == 3:
            graph.add_edge(int(edge[0]), int(edge[1]), weight=float(edge[2]))
        else:
            graph.add_edge(int(edge[0]), int(edge[1]))
    return graph


def test_create_test_graphs_returns_five_named_graphs() -> None:
    graphs = create_test_graphs()
    assert tuple(graphs) == GRAPH_NAMES
    for name, payload in graphs.items():
        assert payload["nodes"], name
        assert "ground_truth" in payload
        assert "dfs_order_from_0" in payload["ground_truth"]
        assert "bfs_order_from_0" in payload["ground_truth"]
        assert "has_cycle" in payload["ground_truth"]
        assert "adjacency_list" in payload


def test_ground_truth_matches_networkx_recompute() -> None:
    graphs = create_test_graphs()
    for name, payload in graphs.items():
        rebuilt = _graph_from_payload(payload)
        target = 9 if 9 in rebuilt else None
        expected = compute_ground_truth(rebuilt, source=0, target=target)
        assert payload["ground_truth"]["dfs_order_from_0"] == expected["dfs_order_from_0"]
        assert payload["ground_truth"]["bfs_order_from_0"] == expected["bfs_order_from_0"]
        assert payload["ground_truth"]["has_cycle"] == expected["has_cycle"]


def test_write_graphs_json_includes_networkx_version(tmp_path: Path) -> None:
    output = tmp_path / "graphs.json"
    write_graphs_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["networkx_version"] == nx.__version__
    assert "simple_undirected" in payload


def test_serialize_graph_rejects_empty_via_payload_builder() -> None:
    empty = nx.Graph()
    with pytest.raises(ValueError, match="has no nodes"):
        from setup_graph import _payload

        _payload("empty", empty)
