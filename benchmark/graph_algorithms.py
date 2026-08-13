"""Reference graph algorithms used by the Executor (and as Engineer targets)."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any


def adjacency_from_graph(graph: dict[str, Any]) -> dict[int, list[int]]:
    raw = graph.get("adjacency_list") or {}
    return {int(node): [int(neighbor) for neighbor in neighbors] for node, neighbors in raw.items()}


def weighted_adjacency_from_graph(graph: dict[str, Any]) -> dict[int, list[tuple[int, float]]]:
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


def dfs(adj: dict[int, list[int]], start: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()

    def visit(node: int) -> None:
        seen.add(node)
        order.append(node)
        for neighbor in adj.get(node, []):
            if neighbor not in seen:
                visit(neighbor)

    if start in adj:
        visit(start)
    return order


def bfs(adj: dict[int, list[int]], start: int) -> list[int]:
    if start not in adj:
        return []
    order: list[int] = []
    seen = {start}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return order


def dijkstra(
    weighted_adj: dict[int, list[tuple[int, float]]],
    source: int,
    target: int,
) -> tuple[float, list[int]]:
    dist = {node: float("inf") for node in weighted_adj}
    prev: dict[int, int | None] = {node: None for node in weighted_adj}
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist[node]:
            continue
        if node == target:
            break
        for neighbor, weight in weighted_adj.get(node, []):
            candidate = cost + weight
            if candidate < dist[neighbor]:
                dist[neighbor] = candidate
                prev[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))
    if dist.get(target, float("inf")) == float("inf"):
        return float("inf"), []
    path: list[int] = []
    cursor: int | None = target
    while cursor is not None:
        path.append(cursor)
        cursor = prev.get(cursor)
    path.reverse()
    return float(dist[target]), path


def floyd_warshall(
    nodes: list[int],
    weighted_adj: dict[int, list[tuple[int, float]]],
) -> dict[int, dict[int, float]]:
    dist: dict[int, dict[int, float]] = {u: {v: float("inf") for v in nodes} for u in nodes}
    for node in nodes:
        dist[node][node] = 0.0
    for origin, neighbors in weighted_adj.items():
        for target, weight in neighbors:
            if weight < dist[origin][target]:
                dist[origin][target] = float(weight)
    for mid in nodes:
        for origin in nodes:
            for target in nodes:
                candidate = dist[origin][mid] + dist[mid][target]
                if candidate < dist[origin][target]:
                    dist[origin][target] = candidate
    return {
        origin: {target: cost for target, cost in row.items() if cost != float("inf")}
        for origin, row in dist.items()
    }


def kruskal_mst_weight(nodes: list[int], edges: list[list[Any]]) -> float:
    parent = {node: node for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    weighted = []
    for edge in edges:
        if len(edge) < 2:
            continue
        weight = float(edge[2]) if len(edge) > 2 else 1.0
        weighted.append((weight, int(edge[0]), int(edge[1])))
    total = 0.0
    for weight, left, right in sorted(weighted):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        total += weight
    return float(total)
