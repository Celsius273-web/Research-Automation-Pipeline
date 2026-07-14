from __future__ import annotations

from src.state import MetricResult, ReportedResult
from src.tools.result_comparator import compare_results, normalize_metric_name, parse_leading_number, verdict_from_rate


def test_normalize_metric_name() -> None:
    assert normalize_metric_name("Best-Regret (%)") == "bestregret"


def test_parse_leading_number() -> None:
    assert parse_leading_number("12.3 ± 0.4") == 12.3
    assert parse_leading_number("n/a") is None


def test_compare_results_classification() -> None:
    reported = [
        ReportedResult(metric_name="regret", value="10.0"),
        ReportedResult(metric_name="accuracy", value="90"),
        ReportedResult(metric_name="missing", value="1"),
    ]
    captured = [
        MetricResult(metric_name="regret", value="10.2"),
        MetricResult(metric_name="accuracy", value="60"),
    ]
    rows, rate = compare_results(reported, captured)
    status = {row.metric_name: row.match_status for row in rows}
    assert status["regret"] in {"match", "close"}
    assert status["accuracy"] == "diverged"
    assert status["missing"] == "missing_reproduced"
    assert 0.0 <= rate <= 1.0


def test_verdict_from_rate() -> None:
    assert verdict_from_rate(0.85, True) == "reproduced"
    assert verdict_from_rate(0.5, True) == "partially_reproduced"
    assert verdict_from_rate(0.1, True) == "not_reproduced"
    assert verdict_from_rate(0.0, False) == "inconclusive"
