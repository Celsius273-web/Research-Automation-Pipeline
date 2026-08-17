"""Harness helpers for loading graphs.json payloads (not Engineer-authored)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_graphs_path() -> Path:
    return Path(__file__).resolve().parent / "graphs.json"


def load_graphs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    target = path or default_graphs_path()
    if not target.exists():
        raise FileNotFoundError(f"Graph file not found at {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {
        name: value
        for name, value in payload.items()
        if isinstance(value, dict) and "nodes" in value
    }


def adjacency_from_graph(graph: dict[str, Any]) -> dict[int, list[int]]:
    raw = graph.get("adjacency_list") or {}
    return {int(node): [int(neighbor) for neighbor in neighbors] for node, neighbors in raw.items()}


def weighted_adjacency_from_graph(graph: dict[str, Any]) -> dict[int, list[tuple[int, float]]]:
    """Return {node: [(neighbor, weight), ...]} — not a list of (u, v, weight) triples."""
    raw = graph.get("weighted_adjacency") or {}
    if raw:
        return {
            int(node): [(int(item[0]), float(item[1])) for item in neighbors]
            for node, neighbors in raw.items()
        }
    adj: dict[int, list[tuple[int, float]]] = {int(node): [] for node in graph.get("nodes", [])}
    for edge in graph.get("edges", []):
        if len(edge) == 3:
            left, right, weight = int(edge[0]), int(edge[1]), float(edge[2])
        else:
            left, right, weight = int(edge[0]), int(edge[1]), 1.0
        adj.setdefault(left, []).append((right, weight))
        adj.setdefault(right, []).append((left, weight))
    return adj
