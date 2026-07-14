"""Deterministic comparison between paper-reported and reproduced metrics."""

from __future__ import annotations

import re

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


def _index_metrics(
    values: list[ReportedResult] | list[MetricResult],
) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for item in values:
        benchmark = getattr(item, "benchmark", "").strip().lower()
        metric_name = normalize_metric_name(item.metric_name)
        out[(benchmark, metric_name)] = item.value
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

    for benchmark, metric_key in all_keys:
        reported_value = reported_map.get((benchmark, metric_key), "")
        captured_value = captured_map.get((benchmark, metric_key), "")
        match_status = "unparsable"
        absolute_diff: float | None = None
        relative_diff: float | None = None

        if not reported_value and captured_value:
            match_status = "missing_reported"
        elif reported_value and not captured_value:
            match_status = "missing_reproduced"
        else:
            reported_num = parse_leading_number(reported_value)
            captured_num = parse_leading_number(captured_value)
            if reported_num is None or captured_num is None:
                match_status = "unparsable"
            else:
                comparable_total += 1
                absolute_diff = abs(reported_num - captured_num)
                if reported_num == 0:
                    relative_diff = 0.0 if captured_num == 0 else 100.0
                else:
                    relative_diff = abs((captured_num - reported_num) / reported_num) * 100.0

                if relative_diff <= REVIEW_MATCH_TOLERANCE_PCT:
                    match_status = "match"
                    comparable_success += 1
                elif relative_diff <= REVIEW_CLOSE_TOLERANCE_PCT:
                    match_status = "close"
                    comparable_success += 1
                else:
                    match_status = "diverged"

        rows.append(
            ComparisonRow(
                metric_name=metric_key,
                benchmark=benchmark,
                reported_value=reported_value,
                reproduced_value=captured_value,
                absolute_difference=absolute_diff,
                relative_difference_pct=relative_diff,
                match_status=match_status,
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
