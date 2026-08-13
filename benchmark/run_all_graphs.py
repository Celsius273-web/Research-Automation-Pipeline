"""Generate Engineer tasks for the five graph algorithms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from setup_graph import default_graphs_path, load_graphs, write_graphs_json

logger = logging.getLogger(__name__)
BENCHMARK_DIR = Path(__file__).resolve().parent
TASKS_PATH = BENCHMARK_DIR / "graph_tasks.json"

TASK_SPECS: list[dict[str, Any]] = [
    {
        "algorithm": "dfs",
        "task_id": "graph_dfs_001",
        "spec": (
            "Implement depth-first search. Input: graph (adjacency list), start node. "
            "Output: list of visited nodes in DFS preorder from node 0."
        ),
        "test_graph": "simple_undirected",
        "truth_key": "dfs_order_from_0",
        "timeout_seconds": 5,
        "assertions": [
            "Output list length should equal number of reachable nodes from 0",
            "First node must be the start node 0",
            "Order must match DFS preorder on the given adjacency list",
        ],
    },
    {
        "algorithm": "bfs",
        "task_id": "graph_bfs_001",
        "spec": (
            "Implement breadth-first search. Input: graph (adjacency list), start node. "
            "Output: list of visited nodes in BFS order from node 0."
        ),
        "test_graph": "simple_undirected",
        "truth_key": "bfs_order_from_0",
        "timeout_seconds": 5,
        "assertions": [
            "Output list length should equal number of reachable nodes from 0",
            "First node must be the start node 0",
            "Nodes at distance k must appear before nodes at distance k+1",
        ],
    },
    {
        "algorithm": "dijkstra",
        "task_id": "graph_dijkstra_001",
        "spec": (
            "Implement Dijkstra shortest path. Input: weighted undirected graph, source 0, target 9. "
            "Output: numeric shortest-path length from 0 to 9."
        ),
        "test_graph": "weighted_shortest_path",
        "truth_key": "dijkstra_path_length",
        "timeout_seconds": 5,
        "assertions": [
            "Output must be a finite number",
            "Length must equal the NetworkX Dijkstra path length from 0 to 9",
        ],
    },
    {
        "algorithm": "floyd_warshall",
        "task_id": "graph_floyd_warshall_001",
        "spec": (
            "Implement Floyd-Warshall all-pairs shortest paths. Input: weighted undirected graph. "
            "Output: nested dict of distances[u][v] for all reachable pairs."
        ),
        "test_graph": "dense_graph",
        "truth_key": "floyd_warshall",
        "timeout_seconds": 5,
        "assertions": [
            "Every node u must have distances[u][u] == 0",
            "Distances must match all-pairs Dijkstra on the same graph",
        ],
    },
    {
        "algorithm": "kruskal",
        "task_id": "graph_kruskal_001",
        "spec": (
            "Implement Kruskal minimum spanning tree. Input: weighted undirected graph. "
            "Output: total MST weight (sum of selected edge weights)."
        ),
        "test_graph": "minimum_spanning_tree",
        "truth_key": "mst_weight",
        "timeout_seconds": 5,
        "assertions": [
            "Output must be a number",
            "Weight must equal NetworkX minimum_spanning_tree(G).size(weight='weight')",
        ],
    },
]


def generate_graph_tasks(graphs: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Generate 5 independent Engineer tasks, one per algorithm."""
    loaded = graphs if graphs is not None else load_graphs()
    tasks: list[dict[str, Any]] = []
    for spec in TASK_SPECS:
        graph_name = spec["test_graph"]
        if graph_name not in loaded:
            raise KeyError(f"Test graph {graph_name!r} missing from graphs.json")
        truth = loaded[graph_name]["ground_truth"]
        tasks.append(
            {
                "algorithm": spec["algorithm"],
                "task_id": spec["task_id"],
                "spec": spec["spec"],
                "test_graph": graph_name,
                "expected_output": truth[spec["truth_key"]],
                "timeout_seconds": spec["timeout_seconds"],
                "assertions": list(spec["assertions"]),
                "run_command": (
                    f"python run_graph.py --algorithm {spec['algorithm']} "
                    f"--graph {graph_name} "
                    f"--out results/synthetic_graph/{spec['algorithm']}.json"
                ),
                "results_path": f"results/synthetic_graph/{spec['algorithm']}.json",
            }
        )
    return tasks


def write_graph_tasks(path: Path | None = None) -> Path:
    graphs_path = default_graphs_path()
    if not graphs_path.exists():
        write_graphs_json(graphs_path)
    output = path or TASKS_PATH
    tasks = generate_graph_tasks()
    output.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = write_graph_tasks()
    logger.info("%s created", output)


if __name__ == "__main__":
    main()
