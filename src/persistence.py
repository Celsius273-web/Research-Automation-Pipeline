"""Read and write of JSON artifacts produced by each pipeline phase."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import EXTRACTIONS_DIR, PLANS_DIR, REPORTS_DIR, RUNS_DIR
from src.state import (
    ExtractionBundle,
    ExecutionPlan,
    ExecutorResult,
    PaperMetadata,
    PlanReviewRecord,
    ReportedResult,
    RepoContext,
    ReviewerReport,
    ReviewRecord,
    SECTION_NAMES,
    SectionExtraction,
)
from src.tools import report_builder


def _format_bundle_as_text(paper: PaperMetadata, bundle: ExtractionBundle) -> str:
    """Render per-section extractions as human-readable text matching the examples format."""
    lines: list[str] = [f"Paper: {paper.title}", f"ID:    {paper.paper_id}", ""]
    for section in SECTION_NAMES:
        ext = bundle.by_section.get(section)
        if ext is None:
            continue
        lines.append(f"Section: {section.capitalize()}")
        lines.append("")
        lines.append(json.dumps(ext.model_dump(), indent=2))
        lines.append("")
    lines.append("=== Merged ===")
    lines.append("")
    lines.append(json.dumps(bundle.merged.model_dump(), indent=2))
    return "\n".join(lines)


def persist_extraction_bundle(
    paper: PaperMetadata,
    bundle: ExtractionBundle,
    review: ReviewRecord,
) -> Path:
    """Save per-section and merged extraction to JSON and human-readable text.

    Returns the path of the JSON file. The .txt file is written alongside it
    and can be opened directly to inspect section-by-section output.
    """
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EXTRACTIONS_DIR / f"{paper.paper_id}.json"
    txt_path = EXTRACTIONS_DIR / f"{paper.paper_id}_sections.txt"

    payload = {
        "paper": paper.model_dump(),
        "review": review.model_dump(),
        "by_section": {k: v.model_dump() for k, v in bundle.by_section.items()},
        "merged": bundle.merged.model_dump(),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path.write_text(_format_bundle_as_text(paper, bundle), encoding="utf-8")
    return json_path


def persist_extraction(paper: PaperMetadata, extraction: SectionExtraction, review: ReviewRecord) -> Path:
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output = EXTRACTIONS_DIR / f"{paper.paper_id}.json"
    payload = {
        "paper": paper.model_dump(),
        "review": review.model_dump(),
        "approved_extraction": extraction.model_dump(),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def persist_plan(
    paper: PaperMetadata,
    plan: ExecutionPlan,
    plan_review: PlanReviewRecord,
    source_extraction_path: str | None = None,
) -> Path:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    output = PLANS_DIR / f"{paper.paper_id}.json"
    payload = {
        "paper": paper.model_dump(),
        "plan_review": plan_review.model_dump(),
        "execution_plan": plan.model_dump(),
        "source_extraction_path": source_extraction_path or "",
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def persist_run_summary(
    paper: PaperMetadata,
    execution_plan: ExecutionPlan,
    repo_context: RepoContext,
    executor_result: ExecutorResult,
) -> Path:
    run_dir = RUNS_DIR / paper.paper_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "run_summary.json"
    payload = {
        "paper": paper.model_dump(),
        "execution_plan": execution_plan.model_dump(),
        "repo_context": repo_context.model_dump(),
        "executor_result": executor_result.model_dump(),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def persist_report(paper: PaperMetadata, report: ReviewerReport) -> Path:
    report_dir = REPORTS_DIR / paper.paper_id
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "report.json"
    md_path = report_dir / "report.md"
    csv_path = report_dir / "comparison_table.csv"
    png_path = report_dir / "comparison_chart.png"

    report_builder.render_comparison_csv(report.comparison_table, csv_path)
    report_builder.render_comparison_chart(report.comparison_table, png_path)
    report_builder.render_report_markdown(report=report, chart_path=png_path, output_path=md_path)

    report.artifacts = [
        str(json_path),
        str(md_path),
        str(csv_path),
        str(png_path),
    ]
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return json_path


def resolve_extraction_path(extraction_path: str | None, paper_id: str | None) -> Path:
    if extraction_path:
        return Path(extraction_path).resolve()
    if paper_id:
        return (EXTRACTIONS_DIR / f"{paper_id}.json").resolve()
    raise ValueError("Provide --extraction-path or --paper-id.")


def resolve_plan_path(plan_path: str | None, paper_id: str | None) -> Path:
    if plan_path:
        return Path(plan_path).resolve()
    if paper_id:
        return (PLANS_DIR / f"{paper_id}.json").resolve()
    raise ValueError("Provide --plan-path or --paper-id.")


def resolve_run_summary_path(run_path: str | None, paper_id: str | None) -> Path:
    if run_path:
        return Path(run_path).resolve()
    if paper_id:
        return (RUNS_DIR / paper_id / "run_summary.json").resolve()
    raise ValueError("Provide --run-path or --paper-id.")


def load_reported_results(extraction_path: Path | None) -> list[ReportedResult]:
    if extraction_path is None or not extraction_path.exists():
        return []
    payload = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction = SectionExtraction.model_validate(payload.get("approved_extraction", {}))
    return extraction.reported_results
