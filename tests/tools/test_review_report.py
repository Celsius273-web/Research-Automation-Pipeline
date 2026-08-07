"""Unit tests for deterministic Reviewer run reports."""

from __future__ import annotations

from src.state import CapturedMetric, MetricsDocument, ReportedResult
from src.tools.review_report import assign_confidence, build_reviewer_run_report


def test_build_reviewer_run_report_matches_and_missing() -> None:
    reported = [
        ReportedResult(benchmark="lsq", metric_name="regret", value="0.035", source="Table 1"),
        ReportedResult(benchmark="gas_compressor", metric_name="runtime", value="10", source="Table 2"),
    ]
    metrics_doc = MetricsDocument(
        run_status="PARTIAL",
        exit_code=0,
        attempts=1,
        metrics=[
            CapturedMetric(benchmark="lsq", metric_name="regret", value=0.042, source="summary.json"),
        ],
        phases_completed=["setup", "smoke"],
        phases_failed=["real_world"],
    )
    report = build_reviewer_run_report("boundary_exploration_bo", reported, metrics_doc)
    assert report.reported_count == 2
    assert report.captured_count == 1
    assert len(report.metrics_matched) == 1
    assert report.metrics_matched[0].match_status == "close"
    assert abs((report.metrics_matched[0].delta_pct or 0.0) - 20.0) < 0.01
    assert len(report.metrics_missing) == 1
    assert report.metrics_missing[0].reason == "not_captured"
    assert "some phases failed" in report.gaps
    assert report.confidence in {"MEDIUM", "LOW"}


def test_delta_pct_match_status_boundaries() -> None:
    reported = [ReportedResult(benchmark="lsq", metric_name="regret", value="100")]
    metrics_doc = MetricsDocument(
        run_status="SUCCESS",
        metrics=[CapturedMetric(benchmark="lsq", metric_name="regret", value=101.0, source="s")],
        phases_completed=["all"],
    )
    report = build_reviewer_run_report("paper", reported, metrics_doc)
    assert report.metrics_matched[0].match_status == "match"


def test_assign_confidence_high_requires_full_success() -> None:
    from src.state import MatchedMetricRow

    matched = [
        MatchedMetricRow(
            metric_name="regret",
            benchmark="lsq",
            reported_value=1.0,
            captured_value=1.0,
            delta_pct=0.5,
            match_status="match",
        )
    ]
    assert assign_confidence("SUCCESS", reported_count=1, captured_count=1, matched_rows=matched) == "HIGH"
    assert assign_confidence("FAILED", reported_count=1, captured_count=1, matched_rows=matched) == "LOW"


def test_reviewer_aliases_table1_fx_to_best_objective() -> None:
    reported = [
        ReportedResult(
            benchmark="Three-bar truss design problem",
            algorithm="be-cbo",
            metric_name="f(x*)",
            value="263.89",
            source="Table 1",
        ),
        ReportedResult(
            benchmark="LSQ Function",
            metric_name="Global Optimum Discovery & Boundary Accuracy",
            value="BE-CBO discovers global optimum at 100 evaluations",
            source="Figure 7 analysis",
        ),
    ]
    metrics_doc = MetricsDocument(
        run_status="SUCCESS",
        metrics=[
            CapturedMetric(
                benchmark="3bar",
                algorithm="cei",
                metric_name="best_objective",
                value=-300.0,
                source="a",
            ),
            CapturedMetric(
                benchmark="3bar",
                algorithm="be-cbo",
                metric_name="best_objective",
                value=-280.0,
                source="b",
            ),
        ],
        phases_completed=["real_world"],
    )
    report = build_reviewer_run_report("boundary_exploration_bo", reported, metrics_doc)
    assert len(report.metrics_matched) == 1
    assert report.metrics_matched[0].algorithm == "be-cbo"
    assert report.metrics_matched[0].captured_value == -280.0
    assert any(row.reason == "not_captured" for row in report.metrics_missing)
