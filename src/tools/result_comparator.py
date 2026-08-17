"""Deterministic comparison between paper-reported and reproduced metrics."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from src.config import REVIEW_CLOSE_TOLERANCE_PCT, REVIEW_MATCH_TOLERANCE_PCT
from src.state import ComparisonRow, MetricResult, ReportedResult


def normalize_metric_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", name.lower())
    return normalized


def parse_leading_number(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def coerce_numeric(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return parse_leading_number(str(value))


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def parse_sequence(value: object) -> list[Any] | None:
    """Parse list-like metric values (DFS order, MST edges). Scalars return None."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return None
    text = str(value).strip()
    if not text or text[0] not in "[({":
        return None
    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return None


def parse_mapping(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, (int, float, bool)) or value is None:
        return None
    text = str(value).strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, dict):
        return {str(key): item for key, item in parsed.items()}
    return None


def _sequence_as_set(items: list[Any]) -> set[Any] | None:
    frozen: list[Any] = []
    for item in items:
        if isinstance(item, list):
            frozen.append(tuple(item))
        else:
            frozen.append(item)
    try:
        return set(frozen)
    except TypeError:
        return None


def _reconstruct_dfs_order(
    adj: dict[int, list[int]] | None,
    start: int = 0,
) -> list[int]:
    """Reconstruct DFS order from adjacency list."""
    if not adj or start not in adj:
        return []
    order: list[int] = []
    seen: set[int] = set()

    def visit(node: int) -> None:
        seen.add(node)
        order.append(node)
        for neighbor in adj.get(node, []):
            if neighbor not in seen:
                visit(neighbor)

    visit(start)
    return order


def _is_valid_bfs_order(
    order: list[Any],
    adj: dict[int, list[int]] | None,
    start: int = 0,
) -> bool:
    """Check if order is a valid BFS traversal from start on adjacency list adj."""
    if not adj or not order or order[0] != start:
        return False
    from collections import deque

    visited = set([start])
    queue = deque([start])
    bfs_order = [start]
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                bfs_order.append(neighbor)
    return list(order) == bfs_order


def _match_status_from_delta(delta_pct: float) -> str:
    abs_delta = abs(delta_pct)
    if abs_delta <= REVIEW_MATCH_TOLERANCE_PCT:
        return "match"
    if abs_delta <= REVIEW_CLOSE_TOLERANCE_PCT:
        return "close"
    return "diverged"


def compare_metric_values(
    reported_value: object,
    captured_value: object,
) -> tuple[str, float | None, float | None]:
    """Classify one reported/captured pair. Returns (match_status, delta_pct, abs_diff)."""
    if _is_blank(reported_value) and not _is_blank(captured_value):
        return "missing_reported", None, None
    if not _is_blank(reported_value) and _is_blank(captured_value):
        return "missing_captured", None, None

    reported_seq = parse_sequence(reported_value)
    captured_seq = parse_sequence(captured_value)
    if reported_seq is not None and captured_seq is not None:
        if reported_seq == captured_seq:
            return "match", 0.0, 0.0
        reported_set = _sequence_as_set(reported_seq)
        captured_set = _sequence_as_set(captured_seq)
        if reported_set is not None and reported_set == captured_set:
            return "close", 0.0, 0.0
        return "diverged", None, None

    reported_map = parse_mapping(reported_value)
    captured_map = parse_mapping(captured_value)
    if reported_map is not None and captured_map is not None:
        if reported_map == captured_map:
            return "match", 0.0, 0.0
        return "diverged", None, None

    reported_num = coerce_numeric(reported_value)
    captured_num = coerce_numeric(captured_value)
    if reported_num is None or captured_num is None:
        left = str(reported_value).strip()
        right = str(captured_value).strip()
        if left == right:
            return "match", 0.0, 0.0
        return "unparsable", None, None

    absolute_diff = abs(reported_num - captured_num)
    if reported_num == 0:
        delta_pct = 0.0 if captured_num == 0 else 100.0
    else:
        delta_pct = ((captured_num - reported_num) / reported_num) * 100.0
    return _match_status_from_delta(delta_pct), delta_pct, absolute_diff


def _index_metrics(
    values: list[ReportedResult] | list[MetricResult],
) -> dict[tuple[str, str, str], str]:
    out: dict[tuple[str, str, str], str] = {}
    for item in values:
        benchmark = getattr(item, "benchmark", "").strip().lower()
        algorithm = getattr(item, "algorithm", "").strip().lower()
        metric_name = normalize_metric_name(item.metric_name)
        out[(benchmark, algorithm, metric_name)] = item.value
    return out


def compare_results(
    reported: list[ReportedResult],
    captured: list[MetricResult],
) -> tuple[list[ComparisonRow], float]:
    reported_map = _index_metrics(reported)
    captured_map = _index_metrics(captured)

    all_keys = list(dict.fromkeys([*reported_map.keys(), *captured_map.keys()]))
    rows: list[ComparisonRow] = []
    comparable_total = 0
    comparable_success = 0

    for benchmark, algorithm, metric_key in all_keys:
        reported_value = reported_map.get((benchmark, algorithm, metric_key), "")
        captured_value = captured_map.get((benchmark, algorithm, metric_key), "")
        match_status, delta_pct, absolute_diff = compare_metric_values(
            reported_value, captured_value
        )
        if match_status == "missing_captured":
            match_status = "missing_reproduced"
        if match_status in {"match", "close", "diverged"} and delta_pct is not None:
            comparable_total += 1
            if match_status in {"match", "close"}:
                comparable_success += 1

        rows.append(
            ComparisonRow(
                metric_name=metric_key,
                benchmark=benchmark,
                algorithm=algorithm,
                reported_value=str(reported_value),
                reproduced_value=str(captured_value),
                absolute_difference=absolute_diff,
                relative_difference_pct=None if delta_pct is None else abs(delta_pct),
                match_status=match_status,  # type: ignore[arg-type]
            )
        )

    if comparable_total == 0:
        return rows, 0.0
    return rows, comparable_success / comparable_total


def verdict_from_rate(rate: float, has_comparable_rows: bool) -> str:
    if not has_comparable_rows:
        return "inconclusive"
    if rate >= 0.8:
        return "reproduced"
    if rate >= 0.4:
        return "partially_reproduced"
    return "not_reproduced"
