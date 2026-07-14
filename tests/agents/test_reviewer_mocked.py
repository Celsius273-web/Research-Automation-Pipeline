from __future__ import annotations

from src.agents.reviewer import PaperReviewer, ReviewerNarrativeOutput
from src.state import MetricResult, ReportedResult


def test_reviewer_generate_report_uses_deterministic_verdict(monkeypatch) -> None:
    reviewer = PaperReviewer()

    def fake_call(_payload):
        return ReviewerNarrativeOutput(
            summary="Looks good",
            risks=["single-seed run"],
            notes=["check confidence intervals"],
        )

    monkeypatch.setattr(reviewer, "_call_narrative_json", fake_call)
    report = reviewer.generate_report(
        paper_id="paper_1",
        domain="optimization",
        reported_results=[ReportedResult(metric_name="regret", value="10")],
        captured_metrics=[MetricResult(metric_name="regret", value="10.1")],
        run_summary={"final_executor_status": "success"},
    )

    assert report.verdict in {"reproduced", "partially_reproduced"}
    assert report.summary == "Looks good"
    assert report.comparison_table
