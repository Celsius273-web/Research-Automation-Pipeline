"""Tests for Engineer graph task generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from run_all_graphs import generate_graph_tasks, write_graph_tasks  # noqa: E402
from setup_graph import create_test_graphs  # noqa: E402


def test_generate_graph_tasks_has_five_independent_algorithms() -> None:
    tasks = generate_graph_tasks(create_test_graphs())
    algorithms = [task["algorithm"] for task in tasks]
    assert algorithms == ["dfs", "bfs", "dijkstra", "floyd_warshall", "kruskal"]
    for task in tasks:
        assert task["spec"]
        assert task["test_graph"]
        assert task["expected_output"] is not None
        assert task["timeout_seconds"] == 5
        assert task["assertions"]
        assert task["run_command"].startswith("python run_graph.py")


def test_write_graph_tasks_round_trip(tmp_path: Path, monkeypatch) -> None:
    graphs_path = tmp_path / "graphs.json"
    tasks_path = tmp_path / "graph_tasks.json"
    monkeypatch.setattr("run_all_graphs.default_graphs_path", lambda: graphs_path)
    monkeypatch.setattr("run_all_graphs.TASKS_PATH", tasks_path)
    from setup_graph import write_graphs_json

    write_graphs_json(graphs_path)
    output = write_graph_tasks(tasks_path)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 5
    assert payload[0]["task_id"] == "graph_dfs_001"
    assert payload[0]["expected_output"] == [0, 1, 3, 2]
