"""Deterministic Reviewer report from extraction reported_results vs Engineer metrics."""

from __future__ import annotations

from src.config import REVIEW_CLOSE_TOLERANCE_PCT, REVIEW_MATCH_TOLERANCE_PCT
from src.state import (
    CapturedMetric,
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
from src.tools.result_comparator import parse_leading_number


def _coerce_numeric(value: float | str | object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return parse_leading_number(str(value))


def _display_value(value: float | str | object) -> float | str:
    numeric = _coerce_numeric(value)
    if numeric is not None:
        return numeric
    return str(value).strip()


def _match_status(delta_pct: float) -> str:
    abs_delta = abs(delta_pct)
    if abs_delta < REVIEW_MATCH_TOLERANCE_PCT:
        return "match"
    if abs_delta < REVIEW_CLOSE_TOLERANCE_PCT:
        return "close"
    return "diverged"


def _capture_bucket_key(item: CapturedMetric) -> tuple[str, str]:
    return (canonicalize_benchmark(item.benchmark), canonicalize_metric(item.metric_name))


def _index_captured(
    metrics: list[CapturedMetric],
) -> dict[tuple[str, str], list[CapturedMetric]]:
    indexed: dict[tuple[str, str], list[CapturedMetric]] = {}
    for item in metrics:
        key = _capture_bucket_key(item)
        indexed.setdefault(key, []).append(item)
    return indexed


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

    # Fall back to first stable order: prefer non-empty algorithm, then name.
    ranked = sorted(
        candidates,
        key=lambda item: (0 if item.algorithm.strip() else 1, item.algorithm, item.source),
    )
    return ranked[0]


def assign_confidence(
    run_status: str,
    reported_count: int,
    captured_count: int,
    matched_rows: list[MatchedMetricRow],
) -> str:
    """Assign HIGH / MEDIUM / LOW confidence from run status and match quality."""
    if reported_count <= 0:
        return "LOW" if run_status == "FAILED" else "MEDIUM"

    comparable = [
        row
        for row in matched_rows
        if row.delta_pct is not None
    ]
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
        close_or_better = [
            row for row in comparable if row.match_status in {"match", "close"}
        ]
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


def build_reviewer_run_report(
    paper_id: str,
    reported_results: list[ReportedResult],
    metrics_doc: MetricsDocument,
) -> ReviewerRunReport:
    """Compare Analyst reported_results to Engineer metrics.json without assigning pass/fail."""
    captured_index = _index_captured(metrics_doc.metrics)
    matched: list[MatchedMetricRow] = []
    missing: list[MissingMetricRow] = []

    for reported in reported_results:
        key = (
            canonicalize_benchmark(reported.benchmark),
            canonicalize_metric(reported.metric_name),
        )
        # Skip non-comparable qualitative / unmapped metric labels (no alias, no digit-ready name).
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

        candidates = captured_index.get(key, [])
        captured = _pick_captured(candidates, reported.algorithm)
        if captured is None:
            missing.append(
                MissingMetricRow(
                    metric_name=reported.metric_name,
                    benchmark=reported.benchmark,
                    algorithm=reported.algorithm,
                    reason="not_captured",
                )
            )
            continue

        reported_num = _coerce_numeric(reported.value)
        captured_num = _coerce_numeric(captured.value)
        delta_pct: float | None = None
        status = "diverged"
        if reported_num is not None and captured_num is not None:
            if reported_num == 0:
                delta_pct = 0.0 if captured_num == 0 else 100.0
            else:
                delta_pct = ((captured_num - reported_num) / reported_num) * 100.0
            status = _match_status(delta_pct)

        matched.append(
            MatchedMetricRow(
                metric_name=reported.metric_name,
                benchmark=reported.benchmark,
                algorithm=captured.algorithm or reported.algorithm,
                reported_value=_display_value(reported.value),
                captured_value=_display_value(captured.value),
                delta_pct=delta_pct,
                match_status=status,  # type: ignore[arg-type]
            )
        )

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
        confidence=confidence,  # type: ignore[arg-type]
        gaps=gaps,
        summary=summary,
    )
