"""Build reviewer report artifacts (CSV, chart, markdown)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.state import ComparisonRow, ReviewerReport
from src.tools.result_comparator import parse_leading_number


def render_comparison_csv(rows: list[ComparisonRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.model_dump() for row in rows])
    frame.to_csv(output_path, index=False)
    return output_path


def render_comparison_chart(rows: list[ComparisonRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_rows = [row for row in rows if row.absolute_difference is not None]
    if not numeric_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No comparable numeric rows", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    labels = [f"{row.benchmark}:{row.metric_name}" if row.benchmark else row.metric_name for row in numeric_rows]
    reported = [parse_leading_number(row.reported_value) or 0.0 for row in numeric_rows]
    reproduced = [parse_leading_number(row.reproduced_value) or 0.0 for row in numeric_rows]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    width = 0.4
    ax.bar([i - width / 2 for i in x], reported, width=width, label="Reported")
    ax.bar([i + width / 2 for i in x], reproduced, width=width, label="Reproduced")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Reported vs Reproduced Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_report_markdown(report: ReviewerReport, chart_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Reproduction Report: {report.paper_id}",
        "",
        f"Paper ID: {report.paper_id}",
        f"Domain: {report.domain}",
        f"Verdict: {report.verdict} (reproduction rate: {round(report.reproduction_rate * 100, 2)}%)",
        "",
        "## Executive Summary",
        report.summary or "No summary generated.",
        "",
        "## Comparison Table",
        "Metric | Benchmark | Reported | Reproduced | Relative Diff | Status",
        "--- | --- | --- | --- | --- | ---",
    ]
    for row in report.comparison_table:
        diff = "" if row.relative_difference_pct is None else f"{round(row.relative_difference_pct, 3)}%"
        lines.append(
            f"{row.metric_name} | {row.benchmark} | {row.reported_value} | {row.reproduced_value} | {diff} | {row.match_status}"
        )
    lines.extend(
        [
            "",
            "## Plots",
            chart_path.name,
            "",
            "## Run Details",
        ]
    )
    for key, value in report.run_summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Risks and Caveats"])
    if report.risks:
        lines.extend([f"- {item}" for item in report.risks])
    else:
        lines.append("- None reported.")
    lines.extend(["", "## Notes"])
    if report.notes:
        lines.extend([f"- {item}" for item in report.notes])
    else:
        lines.append("- None.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
