"""Generate NetworkX test graphs and write benchmark/graphs.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).resolve().parent
GRAPHS_PATH = BENCHMARK_DIR / "graphs.json"
GRAPH_NAMES = (
    "simple_undirected",
    "weighted_shortest_path",
    "minimum_spanning_tree",
    "disconnected",
    "dense_graph",
)


def default_graphs_path() -> Path:
    return GRAPHS_PATH


def _to_native(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, dict):
        return {str(key): _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    return value


def _edge_record(u: Any, v: Any, data: dict[str, Any]) -> list[Any]:
    weight = data.get("weight")
    left, right = int(u), int(v)
    if weight is None:
        return [left, right]
    return [left, right, float(weight)]


def serialize_graph(graph: nx.Graph) -> dict[str, Any]:
    nodes = [int(node) for node in graph.nodes()]
    edges = [_edge_record(u, v, data) for u, v, data in graph.edges(data=True)]
    adjacency: dict[str, list[int]] = {}
    weighted_adjacency: dict[str, list[list[Any]]] = {}
    for node in graph.nodes():
        neighbors: list[int] = []
        weighted: list[list[Any]] = []
        for _, target, data in graph.edges(node, data=True):
            target_id = int(target)
            neighbors.append(target_id)
            weighted.append([target_id, float(data.get("weight", 1.0))])
        adjacency[str(int(node))] = neighbors
        weighted_adjacency[str(int(node))] = weighted
    return {
        "directed": bool(graph.is_directed()),
        "weighted": any("weight" in data for _, _, data in graph.edges(data=True)),
        "nodes": nodes,
        "edges": edges,
        "adjacency_list": adjacency,
        "weighted_adjacency": weighted_adjacency,
    }


def _has_cycle(graph: nx.Graph) -> bool:
    if graph.is_directed():
        return not nx.is_directed_acyclic_graph(graph)
    return bool(nx.cycle_basis(graph))


def compute_ground_truth(graph: nx.Graph, *, source: int = 0, target: int | None = None) -> dict[str, Any]:
    truth: dict[str, Any] = {
        "dfs_order_from_0": _to_native(list(nx.dfs_preorder_nodes(graph, source=source))),
        "bfs_order_from_0": _to_native(list(nx.bfs_tree(graph, source=source).nodes())),
        "has_cycle": _has_cycle(graph),
        "n_components": int(nx.number_connected_components(graph)),
    }
    if target is not None and source in graph and target in graph:
        truth["dijkstra_path_length"] = round(float(nx.dijkstra_path_length(graph, source, target)), 6)
        truth["dijkstra_path"] = _to_native(nx.dijkstra_path(graph, source, target))
    if any("weight" in data for _, _, data in graph.edges(data=True)):
        tree = nx.minimum_spanning_tree(graph)
        truth["mst_weight"] = round(float(tree.size(weight="weight")), 6)
        truth["mst_edges"] = _to_native(
            sorted(
                [int(min(u, v)), int(max(u, v)), float(data.get("weight", 1.0))]
                for u, v, data in tree.edges(data=True)
            )
        )
    pairs = {
        str(int(origin)): {str(int(dest)): round(float(dist), 6) for dest, dist in distances.items()}
        for origin, distances in nx.all_pairs_dijkstra_path_length(graph)
    }
    truth["floyd_warshall"] = pairs
    return truth


def _payload(name: str, graph: nx.Graph, *, target: int | None = None) -> dict[str, Any]:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{name} is not a valid NetworkX graph")
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{name} has no nodes")
    body = serialize_graph(graph)
    body["ground_truth"] = compute_ground_truth(graph, source=0, target=target)
    return body


def build_simple_undirected() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    return graph


def build_weighted_shortest_path() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    weighted_edges = [
        (0, 1, 2.0),
        (1, 2, 2.0),
        (2, 3, 2.0),
        (3, 4, 2.0),
        (4, 5, 2.0),
        (5, 6, 2.0),
        (6, 7, 2.0),
        (7, 8, 2.0),
        (8, 9, 2.0),
        (0, 4, 10.0),
        (2, 7, 8.0),
        (4, 9, 12.0),
        (1, 9, 25.0),
    ]
    graph.add_weighted_edges_from(weighted_edges)
    return graph


def build_minimum_spanning_tree() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_weighted_edges_from(
        [
            (0, 1, 4.0),
            (0, 2, 3.0),
            (1, 2, 1.0),
            (1, 3, 5.0),
            (2, 3, 8.0),
            (2, 4, 6.0),
            (3, 4, 2.0),
            (3, 5, 7.0),
            (4, 5, 3.0),
            (4, 6, 9.0),
            (5, 6, 1.0),
            (5, 7, 4.0),
            (6, 7, 2.0),
        ]
    )
    return graph


def build_disconnected() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 5)])
    return graph


def build_dense_graph() -> nx.Graph:
    graph = nx.erdos_renyi_graph(20, 0.7, seed=42)
    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        for left, right in zip(components, components[1:]):
            graph.add_edge(min(left), min(right))
    for u, v in graph.edges():
        graph[u][v]["weight"] = float((int(u) + int(v)) % 9 + 1)
    return graph


def create_test_graphs() -> dict[str, dict[str, Any]]:
    """Generate 5 test graphs using NetworkX, each with algorithm ground truth."""
    builders = {
        "simple_undirected": (build_simple_undirected, None),
        "weighted_shortest_path": (build_weighted_shortest_path, 9),
        "minimum_spanning_tree": (build_minimum_spanning_tree, None),
        "disconnected": (build_disconnected, None),
        "dense_graph": (build_dense_graph, 9),
    }
    graphs: dict[str, dict[str, Any]] = {}
    for name, (builder, target) in builders.items():
        graphs[name] = _payload(name, builder(), target=target)
    return graphs


def write_graphs_json(path: Path | None = None) -> Path:
    output = path or default_graphs_path()
    graphs = create_test_graphs()
    payload = {"networkx_version": nx.__version__, **graphs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_to_native(payload), indent=2), encoding="utf-8")
    return output


def load_graphs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    target = path or default_graphs_path()
    if not target.exists():
        raise FileNotFoundError(f"Graph file not found at {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {name: payload[name] for name in GRAPH_NAMES if name in payload}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = write_graphs_json()
    logger.info("%s created", output)


if __name__ == "__main__":
    main()
