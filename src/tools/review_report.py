"""Deterministic Reviewer report from benchmark or extraction expectations vs Engineer metrics."""

from __future__ import annotations

import json

from src.config import REVIEW_CLOSE_TOLERANCE_PCT, REVIEW_MATCH_TOLERANCE_PCT
from src.state import (
    CapturedMetric,
    ComparisonRow,
    MatchedMetricRow,
    MetricsDocument,
    MissingMetricRow,
    ReportedResult,
    ReviewerRunReport,
)
from src.tools.metric_aliases import (
    PREFERRED_ALGORITHMS,
    canonicalize_algorithm,
    canonicalize_benchmark,
    canonicalize_metric,
)
from src.tools.result_comparator import (
    coerce_numeric,
    compare_metric_values,
    parse_mapping,
    parse_sequence,
)


def _display_value(value: float | str | object) -> float | str:
    if parse_sequence(value) is not None or parse_mapping(value) is not None:
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value)
        return str(value).strip()
    numeric = coerce_numeric(value)
    if numeric is not None:
        return numeric
    return str(value).strip()


def _pair_key(benchmark: str, algorithm: str, metric_name: str) -> tuple[str, str, str]:
    return (
        canonicalize_benchmark(benchmark),
        canonicalize_algorithm(algorithm),
        canonicalize_metric(metric_name),
    )


def _capture_bucket_key(item: CapturedMetric) -> tuple[str, str]:
    return (canonicalize_benchmark(item.benchmark), canonicalize_metric(item.metric_name))


def _index_captured(
    metrics: list[CapturedMetric],
) -> tuple[dict[tuple[str, str, str], list[CapturedMetric]], dict[tuple[str, str], list[CapturedMetric]]]:
    by_triple: dict[tuple[str, str, str], list[CapturedMetric]] = {}
    by_pair: dict[tuple[str, str], list[CapturedMetric]] = {}
    for item in metrics:
        triple = _pair_key(item.benchmark, item.algorithm, item.metric_name)
        pair = _capture_bucket_key(item)
        by_triple.setdefault(triple, []).append(item)
        by_pair.setdefault(pair, []).append(item)
    return by_triple, by_pair


def _pick_captured(
    candidates: list[CapturedMetric],
    reported_algorithm: str,
) -> CapturedMetric | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    wanted = canonicalize_algorithm(reported_algorithm)
    if wanted:
        for item in candidates:
            if canonicalize_algorithm(item.algorithm) == wanted:
                return item

    preferred = {canonicalize_algorithm(name) for name in PREFERRED_ALGORITHMS}
    for item in candidates:
        if canonicalize_algorithm(item.algorithm) in preferred:
            return item

    ranked = sorted(
        candidates,
        key=lambda item: (0 if item.algorithm.strip() else 1, item.algorithm, item.source),
    )
    return ranked[0]


def _resolve_captured(
    reported: ReportedResult,
    by_triple: dict[tuple[str, str, str], list[CapturedMetric]],
    by_pair: dict[tuple[str, str], list[CapturedMetric]],
) -> CapturedMetric | None:
    triple = _pair_key(reported.benchmark, reported.algorithm, reported.metric_name)
    if reported.algorithm.strip():
        if triple in by_triple:
            return _pick_captured(by_triple[triple], reported.algorithm)
        return None
    pair = (
        canonicalize_benchmark(reported.benchmark),
        canonicalize_metric(reported.metric_name),
    )
    return _pick_captured(by_pair.get(pair, []), reported.algorithm)


def assign_confidence(
    run_status: str,
    reported_count: int,
    captured_count: int,
    matched_rows: list[MatchedMetricRow],
) -> str:
    """Assign HIGH / MEDIUM / LOW confidence from run status and match quality."""
    if reported_count <= 0:
        return "LOW" if run_status == "FAILED" else "MEDIUM"

    comparable = [row for row in matched_rows if row.delta_pct is not None]
    capture_ratio = len(comparable) / reported_count if reported_count else 0.0
    deltas = [abs(row.delta_pct or 0.0) for row in comparable]
    all_match = bool(deltas) and all(delta < REVIEW_MATCH_TOLERANCE_PCT for delta in deltas)
    most_close = (
        bool(deltas)
        and (sum(1 for delta in deltas if delta < REVIEW_CLOSE_TOLERANCE_PCT) / len(deltas)) >= 0.5
    )
    many_diverged = (
        bool(deltas)
        and (sum(1 for delta in deltas if delta >= REVIEW_CLOSE_TOLERANCE_PCT) / len(deltas)) > 0.5
    )

    if run_status == "SUCCESS" and capture_ratio >= 1.0 and all_match:
        return "HIGH"
    if run_status == "FAILED" or capture_ratio < 0.5 or many_diverged:
        return "LOW"
    if run_status == "PARTIAL" and capture_ratio >= 0.8 and most_close:
        return "MEDIUM"
    if capture_ratio >= 0.8 and most_close:
        return "MEDIUM"
    return "LOW"


def _build_gaps(
    metrics_doc: MetricsDocument,
    missing: list[MissingMetricRow],
    reported: list[ReportedResult],
) -> list[str]:
    gaps: list[str] = []
    if metrics_doc.phases_failed:
        gaps.append("some phases failed")
    for item in missing:
        if item.reason == "missing_reported":
            continue
        label = f"{item.metric_name}" + (f" on {item.benchmark}" if item.benchmark else "")
        gaps.append(f"{label} not captured")
    for item in reported:
        source = item.source.strip().lower()
        if source.startswith("table ") or source.startswith("figure "):
            key = (
                canonicalize_benchmark(item.benchmark),
                canonicalize_metric(item.metric_name),
            )
            key_present = any(_capture_bucket_key(captured) == key for captured in metrics_doc.metrics)
            if not key_present:
                gaps.append("benchmark data not extracted")
                break
    return list(dict.fromkeys(gaps))


def _build_summary(
    reported_count: int,
    captured_count: int,
    matched: list[MatchedMetricRow],
    gaps: list[str],
) -> str:
    comparable = [row for row in matched if row.delta_pct is not None]
    parts = [f"{len(comparable)} of {reported_count} metrics compared ({captured_count} captured)."]
    if comparable:
        close_or_better = [row for row in comparable if row.match_status in {"match", "close"}]
        if close_or_better:
            worst = max(abs(row.delta_pct or 0.0) for row in close_or_better)
            parts.append(f"Matched metrics within {worst:.0f}%.")
        diverged = [row for row in comparable if row.match_status == "diverged"]
        if diverged:
            parts.append(f"{len(diverged)} metric(s) diverged.")
    if any("phase" in gap for gap in gaps):
        parts.append("Some phases failed.")
    elif any("not captured" in gap for gap in gaps):
        parts.append("Some reported metrics were not captured.")
    return " ".join(parts)


def _to_comparison_row(
    *,
    metric_name: str,
    benchmark: str,
    algorithm: str,
    reported_value: object,
    captured_value: object,
    match_status: str,
    delta_pct: float | None,
    absolute_diff: float | None,
) -> ComparisonRow:
    return ComparisonRow(
        metric_name=metric_name,
        benchmark=benchmark,
        algorithm=algorithm,
        reported_value="" if reported_value is None else str(_display_value(reported_value)),
        reproduced_value="" if captured_value is None else str(_display_value(captured_value)),
        absolute_difference=absolute_diff,
        relative_difference_pct=None if delta_pct is None else abs(delta_pct),
        match_status=match_status,  # type: ignore[arg-type]
    )


def _append_matched(
    matched: list[MatchedMetricRow],
    table: list[ComparisonRow],
    reported: ReportedResult,
    captured: CapturedMetric,
) -> None:
    status, delta_pct, abs_diff = compare_metric_values(reported.value, captured.value)
    if status == "unparsable":
        status = "diverged"
    if status in {"missing_captured", "missing_reported"}:
        status = "diverged"
    algorithm = captured.algorithm or reported.algorithm
    matched.append(
        MatchedMetricRow(
            metric_name=reported.metric_name,
            benchmark=reported.benchmark,
            algorithm=algorithm,
            reported_value=_display_value(reported.value),
            captured_value=_display_value(captured.value),
            delta_pct=delta_pct,
            match_status=status,  # type: ignore[arg-type]
        )
    )
    table.append(
        _to_comparison_row(
            metric_name=reported.metric_name,
            benchmark=reported.benchmark,
            algorithm=algorithm,
            reported_value=reported.value,
            captured_value=captured.value,
            match_status=status,
            delta_pct=delta_pct,
            absolute_diff=abs_diff,
        )
    )


def _record_reported_rows(
    reported_results: list[ReportedResult],
    by_triple: dict[tuple[str, str, str], list[CapturedMetric]],
    by_pair: dict[tuple[str, str], list[CapturedMetric]],
) -> tuple[list[MatchedMetricRow], list[MissingMetricRow], list[ComparisonRow], set[int]]:
    matched: list[MatchedMetricRow] = []
    missing: list[MissingMetricRow] = []
    table: list[ComparisonRow] = []
    used_ids: set[int] = set()
    for reported in reported_results:
        key = (
            canonicalize_benchmark(reported.benchmark),
            canonicalize_metric(reported.metric_name),
        )
        if not key[0] or not key[1]:
            missing.append(
                MissingMetricRow(
                    metric_name=reported.metric_name,
                    benchmark=reported.benchmark,
                    algorithm=reported.algorithm,
                    reason="not_comparable",
                )
            )
            continue
        captured = _resolve_captured(reported, by_triple, by_pair)
        if captured is None:
            missing.append(
                MissingMetricRow(
                    metric_name=reported.metric_name,
                    benchmark=reported.benchmark,
                    algorithm=reported.algorithm,
                    reason="missing_captured",
                )
            )
            table.append(
                _to_comparison_row(
                    metric_name=reported.metric_name,
                    benchmark=reported.benchmark,
                    algorithm=reported.algorithm,
                    reported_value=reported.value,
                    captured_value="",
                    match_status="missing_captured",
                    delta_pct=None,
                    absolute_diff=None,
                )
            )
            continue
        used_ids.add(id(captured))
        _append_matched(matched, table, reported, captured)
    return matched, missing, table, used_ids


def _record_unmatched_captured(
    metrics: list[CapturedMetric],
    used_ids: set[int],
    missing: list[MissingMetricRow],
    table: list[ComparisonRow],
) -> None:
    for captured in metrics:
        if id(captured) in used_ids:
            continue
        missing.append(
            MissingMetricRow(
                metric_name=captured.metric_name,
                benchmark=captured.benchmark,
                algorithm=captured.algorithm,
                reason="missing_reported",
            )
        )
        table.append(
            _to_comparison_row(
                metric_name=captured.metric_name,
                benchmark=captured.benchmark,
                algorithm=captured.algorithm,
                reported_value="",
                captured_value=captured.value,
                match_status="missing_reported",
                delta_pct=None,
                absolute_diff=None,
            )
        )


def build_reviewer_run_report(
    paper_id: str,
    reported_results: list[ReportedResult],
    metrics_doc: MetricsDocument,
) -> ReviewerRunReport:
    """Compare reported_results to captured metrics by (benchmark, algorithm, metric_name)."""
    by_triple, by_pair = _index_captured(metrics_doc.metrics)
    matched, missing, table, used_ids = _record_reported_rows(
        reported_results, by_triple, by_pair
    )
    _record_unmatched_captured(metrics_doc.metrics, used_ids, missing, table)
    gaps = _build_gaps(metrics_doc, missing, reported_results)
    confidence = assign_confidence(
        run_status=metrics_doc.run_status,
        reported_count=len(reported_results),
        captured_count=len(metrics_doc.metrics),
        matched_rows=matched,
    )
    summary = _build_summary(len(reported_results), len(metrics_doc.metrics), matched, gaps)
    return ReviewerRunReport(
        paper_id=paper_id,
        reported_count=len(reported_results),
        captured_count=len(metrics_doc.metrics),
        metrics_matched=matched,
        metrics_missing=missing,
        comparison_table=table,
        confidence=confidence,  # type: ignore[arg-type]
        gaps=gaps,
        summary=summary,
    )
