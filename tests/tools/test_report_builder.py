from __future__ import annotations

from pathlib import Path

from src.state import ComparisonRow, ReviewerReport
from src.tools import report_builder


def test_report_builder_writes_artifacts(tmp_path: Path) -> None:
    rows = [
        ComparisonRow(
            metric_name="regret",
            benchmark="bbob",
            reported_value="10",
            reproduced_value="11",
            absolute_difference=1.0,
            relative_difference_pct=10.0,
            match_status="close",
        )
    ]
    report = ReviewerReport(
        paper_id="paper_1",
        domain="optimization",
        summary="Summary",
        verdict="partially_reproduced",
        reproduction_rate=0.5,
        comparison_table=rows,
        run_summary={"final_executor_status": "success"},
    )

    csv_path = tmp_path / "comparison_table.csv"
    png_path = tmp_path / "comparison_chart.png"
    md_path = tmp_path / "report.md"

    report_builder.render_comparison_csv(rows, csv_path)
    report_builder.render_comparison_chart(rows, png_path)
    report_builder.render_report_markdown(report, png_path, md_path)

    assert csv_path.exists()
    assert png_path.exists()
    assert md_path.exists()
