"""Read and write of JSON artifacts produced by each pipeline phase."""

from __future__ import annotations

import json
from pathlib import Path

from src.bundle import PaperBundle
from src.agents.planner_debug import refresh_planner_debug_with_saved_plan
from src.config import EXTRACTIONS_DIR, PLANS_DIR, REPORTS_DIR, RUNS_DIR
from src.state import (
    AgentEnvelope,
    ExtractionBundle,
    ExecutionPlan,
    ExecutorResult,
    PaperMetadata,
    PlannerPayload,
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
    
    Primary path: data/{paper_id}/extraction.json
    Fallback: data/extractions/{paper_id}.json (for backward compatibility)
    """
    # Try bundle-aware persistence first - always prioritize bundles
    paper_bundle = PaperBundle(paper.paper_id)
    paper_bundle.create_bundle_dir()  # Ensure bundle directory exists
    paper_bundle.save_extraction(bundle, review, paper)
    
    # Also save human-readable sections file alongside extraction
    sections_txt_path = paper_bundle.bundle_dir / "extraction_sections.txt"
    sections_txt_path.write_text(_format_bundle_as_text(paper, bundle), encoding="utf-8")
    
    return paper_bundle.extraction_path


def persist_extraction(paper: PaperMetadata, extraction: SectionExtraction, review: ReviewRecord) -> Path:
    """Legacy extraction persistence - converts to bundle format when possible."""
    # Convert single extraction to bundle format
    bundle = ExtractionBundle(by_section={}, merged=extraction)
    return persist_extraction_bundle(paper, bundle, review)


def persist_plan(
    paper: PaperMetadata,
    plan: ExecutionPlan | AgentEnvelope[PlannerPayload],
    plan_review: PlanReviewRecord,
    source_extraction_path: str | None = None,
) -> Path:
    """Save execution plan to bundle directory.

    Primary path: data/{paper_id}/plan.json
    Fallback: data/plans/{paper_id}.json (for backward compatibility)
    """
    # Always use bundle persistence - create bundle if needed
    paper_bundle = PaperBundle(paper.paper_id)
    paper_bundle.create_bundle_dir()
    paper_bundle.save_plan(plan, ReviewRecord(status=plan_review.status, notes=plan_review.notes), paper, source_extraction_path)
    refresh_planner_debug_with_saved_plan(paper_bundle.bundle_dir, paper_bundle.plan_path)
    return paper_bundle.plan_path


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
    """Save reviewer report with bundle-aware persistence when possible."""
    # Try bundle-aware persistence first
    paper_bundle = PaperBundle(paper.paper_id)
    if paper_bundle.exists():
        paper_bundle.save_report(report, paper)
        # Also create legacy artifacts in bundle runs directory
        bundle_runs_dir = paper_bundle.runs_dir
        bundle_runs_dir.mkdir(exist_ok=True)
        md_path = bundle_runs_dir / "report.md"
        csv_path = bundle_runs_dir / "comparison_table.csv"
        png_path = bundle_runs_dir / "comparison_chart.png"
        
        report_builder.render_comparison_csv(report.comparison_table, csv_path)
        report_builder.render_comparison_chart(report.comparison_table, png_path)
        report_builder.render_report_markdown(report=report, chart_path=png_path, output_path=md_path)
        
        return paper_bundle.report_path
    
    # Fall back to legacy persistence
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
        # Try bundle first (data/papers/{paper_id}/extraction.json)
        bundle = PaperBundle(paper_id)
        if bundle.extraction_path.exists():
            return bundle.extraction_path.resolve()
        
        # Check for manually organized papers with standard name (data/{paper_id}/extraction.json)
        manual_paper_dir = EXTRACTIONS_DIR.parent / paper_id / "extraction.json"
        if manual_paper_dir.exists():
            return manual_paper_dir.resolve()
        
        # Check for manually organized papers with legacy name (data/{paper_id}/{paper_id}.json)
        legacy_manual_name = EXTRACTIONS_DIR.parent / paper_id / f"{paper_id}.json"
        if legacy_manual_name.exists():
            return legacy_manual_name.resolve()
        
        # Fall back to legacy location (data/extractions/{paper_id}.json)
        legacy_path = EXTRACTIONS_DIR / f"{paper_id}.json"
        if legacy_path.exists():
            return legacy_path.resolve()
        
        # Return bundle path as the canonical location (even if doesn't exist yet)
        return bundle.extraction_path.resolve()
    raise ValueError("Provide --extraction-path or --paper-id.")


def resolve_plan_path(plan_path: str | None, paper_id: str | None) -> Path:
    if plan_path:
        return Path(plan_path).resolve()
    if paper_id:
        # Try bundle first (data/papers/{paper_id}/{paper_id}_plan.json) - new naming
        bundle = PaperBundle(paper_id)
        if bundle.plan_path.exists():
            return bundle.plan_path.resolve()
        
        # Fallback to old bundle naming (data/papers/{paper_id}/plan.json)
        old_bundle_plan = bundle.bundle_dir / "plan.json"
        if old_bundle_plan.exists():
            return old_bundle_plan.resolve()
        
        # Check for manually organized papers with new naming (data/{paper_id}/{paper_id}_plan.json)
        manual_paper_new = PLANS_DIR.parent / paper_id / f"{paper_id}_plan.json"
        if manual_paper_new.exists():
            return manual_paper_new.resolve()
        
        # Check for manually organized papers with old naming (data/{paper_id}/plan.json)
        manual_paper_old = PLANS_DIR.parent / paper_id / "plan.json"
        if manual_paper_old.exists():
            return manual_paper_old.resolve()
        
        # Fall back to legacy location (data/plans/{paper_id}.json)
        legacy_path = PLANS_DIR / f"{paper_id}.json"
        if legacy_path.exists():
            return legacy_path.resolve()
        
        # Return bundle path as the canonical location (even if doesn't exist yet)
        return bundle.plan_path.resolve()
    raise ValueError("Provide --plan-path or --paper-id.")


def resolve_run_summary_path(run_path: str | None, paper_id: str | None) -> Path:
    if run_path:
        return Path(run_path).resolve()
    if paper_id:
        # Try bundle first (data/papers/{paper_id}/runs/run_summary.json)
        bundle = PaperBundle(paper_id)
        bundle_run_summary = bundle.runs_dir / "run_summary.json"
        if bundle_run_summary.exists():
            return bundle_run_summary.resolve()
        
        # Check for manually organized papers (data/{paper_id}/runs/run_summary.json)
        manual_run_summary = RUNS_DIR.parent / paper_id / "runs" / "run_summary.json"
        if manual_run_summary.exists():
            return manual_run_summary.resolve()
        
        # Fall back to legacy location (data/runs/{paper_id}/run_summary.json)
        legacy_run_summary = RUNS_DIR / paper_id / "run_summary.json"
        if legacy_run_summary.exists():
            return legacy_run_summary.resolve()
        
        # Return bundle path as the canonical location
        return bundle_run_summary.resolve()
    raise ValueError("Provide --run-path or --paper-id.")


def load_reported_results(extraction_path: Path | None) -> list[ReportedResult]:
    if extraction_path is None or not extraction_path.exists():
        return []
    
    try:
        payload = json.loads(extraction_path.read_text(encoding="utf-8"))
        
        # Handle bundle format
        if "merged" in payload:
            extraction = SectionExtraction.model_validate(payload["merged"])
        elif "approved_extraction" in payload:
            # Legacy format
            extraction = SectionExtraction.model_validate(payload["approved_extraction"])
        else:
            return []
            
        return extraction.reported_results
    except (json.JSONDecodeError, KeyError, ValueError):
        return []
