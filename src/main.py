"""CLI entrypoint for the Research Assistant pipeline (Analyst -> Planner -> Engineer/Executor -> Reviewer)."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
# Planner I/O goes to planner_debug.md; keep a single INFO line for that path.
logging.getLogger("src.agents.planner").setLevel(logging.INFO)
logging.getLogger("src.agents.experiment_runner").setLevel(logging.INFO)
logging.getLogger("src.plan_runner").setLevel(logging.INFO)

from src.config import EXTRACTIONS_DIR, MAX_RETRY_ATTEMPTS, ROOT_DIR
from src.db import DatabaseError, get_paper_by_id
from src.graphs.research_graph import (
    build_phase1_graph,
    build_phase2_graph,
    build_phase3_graph,
    build_phase4_graph,
)
from src.persistence import (
    load_metrics_document,
    load_planner_envelope,
    load_reported_results,
    persist_extraction_bundle,
    persist_plan,
    persist_report,
    persist_reviewer_run_report,
    persist_run_summary,
    resolve_extraction_path,
    resolve_plan_path,
    resolve_run_dir,
    resolve_run_summary_path,
)
from src.planner_input import load_unified_planner_input
from src.review_prompts import ReviewCancelledError
from src.pipeline_nodes import (
    default_runtime_constraints,
    make_engineer_executor_nodes,
    make_phase1_nodes,
    make_planner_node,
    make_reviewer_node,
)
from src.agents.experiment_runner import ExperimentRunner
from src.plan_runner import run_saved_plan
from src.state import (
    AgentEnvelope,
    ExtractionBundle,
    ExecutorResult,
    PaperMetadata,
    PlannerPayload,
    RepoContext,
    ResearchState,
    ReviewRecord,
    SectionExtraction,
    make_initial_state,
)
from src.tools.docker_executor import DockerExecutor
from src.tools.language_detect import detect_language
from src.tools.benchmark_expectations import resolve_review_expectations
from src.tools.review_report import build_reviewer_run_report


def slugify(value: str) -> str:
    base = value.lower().strip()
    base = re.sub(r"\.pdf$", "", base)
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base or "paper"


def _noop_node(state: ResearchState) -> ResearchState:
    return state


def _handle_review_cancelled() -> int:
    print("Review cancelled. No artifacts were saved.")
    return 130


def _handle_keyboard_interrupt() -> int:
    print("\nInterrupted by user.")
    return 130


def _print_errors_and_fail(errors: list[str], prefix: str) -> int:
    if not errors:
        return 0
    print(prefix)
    for item in errors:
        print(f"- {item}")
    return 1


def _has_real_plan(plan: AgentEnvelope[PlannerPayload]) -> bool:
    return bool(plan.payload.plan_summary.strip() or plan.payload.phases)


def run_analyst(paper_id: str, non_interactive: bool, with_plan: bool) -> int:
    """Run analyst only (extraction), optionally followed by planner."""
    try:
        ingested_paper = get_paper_by_id(paper_id)
    except DatabaseError as e:
        print(f"Database error: {e}")
        return 1
    
    if ingested_paper is None:
        print(f"Paper '{paper_id}' not found. Please ingest the paper first using:")
        print(f"  python scripts/ingest_paper.py --pdf-path <pdf_path> --paper-id {paper_id}")
        return 1
    
    pdf_path = Path(ingested_paper.pdf_path)
    if not pdf_path.exists():
        print(f"Ingested PDF file missing: {pdf_path}")
        return 1
    
    paper = PaperMetadata(
        paper_id=ingested_paper.paper_id,
        title=ingested_paper.title,
        pdf_path=ingested_paper.pdf_path,
        arxiv_id=ingested_paper.arxiv_id,
    )
    state = make_initial_state(paper)

    parse_node, analyst_node, review_node = make_phase1_nodes(non_interactive=non_interactive)
    if with_plan:
        planner_node = make_planner_node(non_interactive=non_interactive)
        graph = build_phase2_graph(
            parse_node=parse_node,
            analyst_node=analyst_node,
            review_node=review_node,
            planner_node=planner_node,
        )
    else:
        graph = build_phase1_graph(parse_node=parse_node, analyst_node=analyst_node, review_node=review_node)
    try:
        out = graph.invoke(state)
    except ReviewCancelledError:
        return _handle_review_cancelled()
    error_code = _print_errors_and_fail(out.get("errors", []), "Analyze failed:")
    if error_code:
        return error_code

    saved_extraction = persist_extraction_bundle(
        paper=out["paper"],
        bundle=out["extraction"],
        review=out["review"],
    )
    print(f"Saved extraction: {saved_extraction}")
    print(f"Review status: {out['review'].status}")
    plan_review = out.get("plan_review")
    if (
        with_plan
        and out["review"].status == "approved"
        and plan_review is not None
        and plan_review.status == "approved"
        and _has_real_plan(out["planner_output"])
    ):
        saved_plan = persist_plan(
            paper=out["paper"],
            plan=out["planner_output"],
            plan_review=plan_review,
            source_extraction_path=str(saved_extraction),
        )
        print(f"Saved plan: {saved_plan}")
        print(f"Plan review status: {plan_review.status}")
    return 0


def run_analyze(paper_id: str, non_interactive: bool, with_plan: bool) -> int:
    try:
        ingested_paper = get_paper_by_id(paper_id)
    except DatabaseError as e:
        print(f"Database error: {e}")
        return 1
    
    if ingested_paper is None:
        print(f"Paper '{paper_id}' not found. Please ingest the paper first using:")
        print(f"  python scripts/ingest_paper.py --pdf-path <pdf_path> --paper-id {paper_id}")
        return 1
    
    pdf_path = Path(ingested_paper.pdf_path)
    if not pdf_path.exists():
        print(f"Ingested PDF file missing: {pdf_path}")
        print("The paper may have been moved or deleted after ingestion.")
        return 1
    
    paper = PaperMetadata(
        paper_id=ingested_paper.paper_id,
        title=ingested_paper.title,
        pdf_path=ingested_paper.pdf_path,
        arxiv_id=ingested_paper.arxiv_id,
    )
    state = make_initial_state(paper)

    parse_node, analyst_node, review_node = make_phase1_nodes(non_interactive=non_interactive)
    if with_plan:
        planner_node = make_planner_node(non_interactive=non_interactive)
        graph = build_phase2_graph(
            parse_node=parse_node,
            analyst_node=analyst_node,
            review_node=review_node,
            planner_node=planner_node,
        )
    else:
        graph = build_phase1_graph(parse_node=parse_node, analyst_node=analyst_node, review_node=review_node)
    try:
        out = graph.invoke(state)
    except ReviewCancelledError:
        return _handle_review_cancelled()
    error_code = _print_errors_and_fail(out.get("errors", []), "Analyze failed:")
    if error_code:
        return error_code

    saved_extraction = persist_extraction_bundle(
        paper=out["paper"],
        bundle=out["extraction"],
        review=out["review"],
    )
    print(f"Saved extraction: {saved_extraction}")
    print(f"Review status: {out['review'].status}")
    plan_review = out.get("plan_review")
    if (
        with_plan
        and out["review"].status == "approved"
        and plan_review is not None
        and plan_review.status == "approved"
        and _has_real_plan(out["planner_output"])
    ):
        saved_plan = persist_plan(
            paper=out["paper"],
            plan=out["planner_output"],
            plan_review=plan_review,
            source_extraction_path=str(saved_extraction),
        )
        print(f"Saved plan: {saved_plan}")
        print(f"Plan review status: {plan_review.status}")
    return 0


def run_plan(
    extraction_path: str | None,
    paper_id: str | None,
    non_interactive: bool,
    input_json: str | None = None,
) -> int:
    if input_json:
        if extraction_path or paper_id:
            print("--input-json cannot be combined with --extraction-path or --paper-id.")
            return 1
        source_path = Path(input_json).resolve()
        if not source_path.exists():
            print(f"Planner input file does not exist: {source_path}")
            return 1
        try:
            unified_input = load_unified_planner_input(source_path)
        except (json.JSONDecodeError, OSError, ValidationError) as exc:
            print(f"Invalid unified Planner input: {exc}")
            return 1

        state = make_initial_state(unified_input.paper_context)
        planner_node = make_planner_node(
            non_interactive=non_interactive,
            unified_input=unified_input,
        )
        try:
            out = planner_node(state)
        except ReviewCancelledError:
            return _handle_review_cancelled()
        error_code = _print_errors_and_fail(out["errors"], "Planner failed:")
        if error_code:
            return error_code
        saved_plan = persist_plan(
            paper=out["paper"],
            plan=out["planner_output"],
            plan_review=out["plan_review"],
            source_extraction_path=str(source_path),
        )
        print(f"Saved plan: {saved_plan}")
        print(f"Plan review status: {out['plan_review'].status}")
        return 0

    try:
        source_path = resolve_extraction_path(extraction_path=extraction_path, paper_id=paper_id)
    except ValueError as exc:
        print(str(exc))
        return 1
    if not source_path.exists():
        print(f"Extraction file does not exist: {source_path}")
        return 1

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    paper = PaperMetadata.model_validate(payload.get("paper", {}))
    
    # Load extraction - handle both bundle format (merged) and legacy format (approved_extraction)
    if "merged" in payload:
        approved_extraction = SectionExtraction.model_validate(payload.get("merged", {}))
        extraction = ExtractionBundle.model_validate(
            {
                "by_section": payload.get("by_section", {}),
                "merged": payload.get("merged", {}),
            }
        )
    else:
        approved_extraction = SectionExtraction.model_validate(payload.get("approved_extraction", {}))
        extraction = ExtractionBundle(by_section={}, merged=approved_extraction)
    
    review = ReviewRecord.model_validate(payload.get("review", {}))
    if review.status != "approved":
        print("Extraction review is not approved. Planner cannot run.")
        return 1

    state = make_initial_state(paper)
    state["extraction"] = extraction
    state["approved_extraction"] = approved_extraction
    state["review"] = review

    planner_node = make_planner_node(non_interactive=non_interactive)
    try:
        out = planner_node(state)
    except ReviewCancelledError:
        return _handle_review_cancelled()
    if out["errors"]:
        print("Planner failed:")
        for item in out["errors"]:
            print(f"- {item}")
        return 1

    saved_plan = persist_plan(
        paper=out["paper"],
        plan=out["planner_output"],
        plan_review=out["plan_review"],
        source_extraction_path=str(source_path),
    )
    print(f"Saved plan: {saved_plan}")
    print(f"Plan review status: {out['plan_review'].status}")
    return 0


def _build_execute_graph(
    with_review: bool,
    engineer_node: Callable[[ResearchState], ResearchState],
    engineer_review_node: Callable[[ResearchState], ResearchState],
    executor_node: Callable[[ResearchState], ResearchState],
    reviewer_node: Callable[[ResearchState], ResearchState] | None,
    max_retries: int,
):
    common = {
        "parse_node": _noop_node,
        "analyst_node": _noop_node,
        "review_node": _noop_node,
        "planner_node": _noop_node,
        "engineer_node": engineer_node,
        "engineer_review_node": engineer_review_node,
        "executor_node": executor_node,
        "max_retries": max_retries,
    }
    if with_review:
        return build_phase4_graph(reviewer_node=reviewer_node, **common)
    return build_phase3_graph(**common)


def run_execute(
    plan_path: str | None,
    paper_id: str | None,
    repo_path: str | None,
    non_interactive: bool,
    with_review: bool,
) -> int:
    try:
        source_path = resolve_plan_path(plan_path=plan_path, paper_id=paper_id)
    except ValueError as exc:
        print(str(exc))
        return 1
    if not source_path.exists():
        print(f"Plan file does not exist: {source_path}")
        return 1

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    paper = PaperMetadata.model_validate(payload.get("paper", {}))
    execution_plan = load_planner_envelope(payload)
    source_extraction_path = payload.get("source_extraction_path", "")
    
    # Resolve repository path - use ingested repo if no explicit path provided
    resolved_repo_path = repo_path
    if resolved_repo_path is None:
        try:
            ingested_paper = get_paper_by_id(paper.paper_id)
            if ingested_paper and ingested_paper.code_path:
                resolved_repo_path = ingested_paper.code_path
                print(f"Using ingested repository: {resolved_repo_path}")
            else:
                print(f"No repository path provided and no ingested repository found for paper '{paper.paper_id}'")
                return 1
        except DatabaseError as e:
            print(f"Database error looking up ingested repository: {e}")
            return 1
    
    if not Path(resolved_repo_path).exists():
        print(f"Repository path does not exist: {resolved_repo_path}")
        return 1
    
    state = make_initial_state(paper)
    state["planner_output"] = execution_plan
    state["planner_output_json"] = execution_plan.model_dump()
    state["review"] = ReviewRecord(status="approved", notes="Loaded from approved plan artifact.")
    state["repo_context"] = detect_language(repo_path=resolved_repo_path)

    runtime_constraints = default_runtime_constraints()
    max_retries = int(runtime_constraints.get("max_retry_attempts", str(MAX_RETRY_ATTEMPTS)))
    try:
        engineer_node, engineer_review_node, executor_node = make_engineer_executor_nodes(
            execution_plan=execution_plan,
            runtime_constraints=runtime_constraints,
            non_interactive=non_interactive,
        )
    except RuntimeError as exc:
        print(f"Failed to initialize execution tools: {exc}")
        return 1

    reviewer_node = None
    if with_review:
        extraction_path = Path(source_extraction_path).resolve() if source_extraction_path else None
        reviewer_node = make_reviewer_node(load_reported_results(extraction_path))

    graph = _build_execute_graph(
        with_review=with_review,
        engineer_node=engineer_node,
        engineer_review_node=engineer_review_node,
        executor_node=executor_node,
        reviewer_node=reviewer_node,
        max_retries=max_retries,
    )
    try:
        out = graph.invoke(state)
    except ReviewCancelledError:
        return _handle_review_cancelled()
    error_code = _print_errors_and_fail(out.get("errors", []), "Execution errors:")
    if error_code:
        return error_code

    summary_path = persist_run_summary(
        paper=out["paper"],
        execution_plan=out["planner_output"],
        repo_context=out["repo_context"],
        executor_result=out["executor_result"],
    )
    print(f"Run summary saved: {summary_path}")
    print(f"Final status: {out['executor_result'].final_status}")
    if with_review:
        report_path = persist_report(out["paper"], out["reviewer_report"])
        print(f"Reviewer report saved: {report_path}")
    return 0 if out["executor_result"].final_status == "success" else 1


def run_review(run_path: str | None, paper_id: str | None, extraction_path: str | None) -> int:
    try:
        summary_path = resolve_run_summary_path(run_path=run_path, paper_id=paper_id)
    except ValueError as exc:
        print(str(exc))
        return 1
    if not summary_path.exists():
        print(f"Run summary does not exist: {summary_path}")
        return 1

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    paper = PaperMetadata.model_validate(payload.get("paper", {}))
    execution_plan = load_planner_envelope(payload)
    repo_context = RepoContext.model_validate(payload.get("repo_context", {}))
    executor_result = ExecutorResult.model_validate(payload.get("executor_result", {}))

    resolved_extraction_path = (
        Path(extraction_path).resolve() if extraction_path else (EXTRACTIONS_DIR / f"{paper.paper_id}.json")
    )
    reported_results = load_reported_results(resolved_extraction_path)

    state = make_initial_state(paper)
    state["planner_output"] = execution_plan
    state["planner_output_json"] = execution_plan.model_dump()
    state["repo_context"] = repo_context
    state["executor_result"] = executor_result
    state["retry_count"] = executor_result.total_attempts

    reviewer_node = make_reviewer_node(reported_results)
    out = reviewer_node(state)
    error_code = _print_errors_and_fail(out.get("errors", []), "Reviewer errors:")
    if error_code:
        return error_code

    report_path = persist_report(out["paper"], out["reviewer_report"])
    print(f"Reviewer report saved: {report_path}")
    return 0


def run_engineer(paper_id: str, non_interactive: bool, repo_path: str | None = None) -> int:
    """Run plan-driven Engineer: execute phases via Docker and write metrics.json."""
    _ = non_interactive  # Engineer has no interactive patch review in this path.
    try:
        runner = ExperimentRunner(docker_executor=DockerExecutor(project_root=ROOT_DIR))
        metrics_doc, run_dir = runner.execute_paper(paper_id=paper_id, repo_path=repo_path)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Engineer failed: {exc}")
        return 1
    except ModuleNotFoundError as exc:
        print(
            f"Engineer failed: missing host package {exc.name!r}. "
            f"Install project deps with: pip install -r requirements.txt"
        )
        return 1
    except Exception as exc:
        print(f"Engineer failed with unexpected error: {exc}")
        return 1

    print(f"Run directory: {run_dir}")
    print(f"RUN_ID={run_dir.name}")
    print(f"metrics.json status: {metrics_doc.run_status} exit_code={metrics_doc.exit_code}")
    return 0 if metrics_doc.run_status in {"SUCCESS", "PARTIAL"} else 1


def run_reviewer(
    paper_id: str,
    non_interactive: bool,
    run_id: str,
    extraction_path: str | None = None,
) -> int:
    """Compare benchmark or extraction expectations to Engineer metrics.json."""
    _ = non_interactive
    try:
        resolved_run_dir = resolve_run_dir(paper_id=paper_id, run_id=run_id)
        metrics_doc = load_metrics_document(resolved_run_dir)
        resolved_extraction = resolve_extraction_path(
            extraction_path=extraction_path,
            paper_id=paper_id,
        )
        if not resolved_extraction.exists():
            print(f"extraction.json not found: {resolved_extraction}")
            return 1
        reported = resolve_review_expectations(paper_id, resolved_extraction)
        report = build_reviewer_run_report(
            paper_id=paper_id,
            reported_results=reported,
            metrics_doc=metrics_doc,
        )
        report_path = persist_reviewer_run_report(
            resolved_run_dir, report, paper_id=paper_id
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Reviewer failed: {exc}")
        return 1
    except Exception as exc:
        print(f"Reviewer failed with unexpected error: {exc}")
        return 1

    print(f"Reviewer report saved: {report_path}")
    print(f"run_id={run_id} confidence={report.confidence} matched={len(report.metrics_matched)} missing={len(report.metrics_missing)}")
    print(report.summary)
    return 0


def run_full_pipeline(paper_id: str, non_interactive: bool) -> int:
    """Run the complete pipeline: analyze -> plan -> execute -> review."""
    print(f"Starting full pipeline for paper: {paper_id}")
    
    # Phase 1: Analyze
    print("=== Phase 1: Analysis ===")
    exit_code = run_analyze(paper_id=paper_id, non_interactive=non_interactive, with_plan=False)
    if exit_code != 0:
        print("Analysis phase failed, stopping pipeline.")
        return exit_code
    
    # Phase 2: Plan
    print("\n=== Phase 2: Planning ===")
    exit_code = run_plan(extraction_path=None, paper_id=paper_id, non_interactive=non_interactive)
    if exit_code != 0:
        print("Planning phase failed, stopping pipeline.")
        return exit_code
    
    # Phase 3: Execute
    print("\n=== Phase 3: Execution ===")
    exit_code = run_execute(
        plan_path=None,
        paper_id=paper_id,
        repo_path=None,
        non_interactive=non_interactive,
        with_review=False
    )
    if exit_code != 0:
        print("Execution phase failed, stopping pipeline.")
        return exit_code
    
    # Phase 4: Review
    print("\n=== Phase 4: Review ===")
    exit_code = run_review(run_path=None, paper_id=paper_id, extraction_path=None)
    if exit_code != 0:
        print("Review phase failed.")
        return exit_code
    
    print(f"\n=== Pipeline Complete for {paper_id} ===")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous Research Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Full pipeline command
    run_parser = subparsers.add_parser("run", help="Run complete pipeline: analyze -> plan -> execute -> review")
    run_parser.add_argument("paper_id", help="Paper ID of an ingested paper")
    run_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-approve all phases without CLI prompts.",
    )
    analyze = subparsers.add_parser("analyze", help="Run Phase 1 analyst flow for an ingested paper")
    analyze.add_argument("paper_id", help="Paper ID of an ingested paper")
    analyze.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-approve extraction without CLI prompts.",
    )
    analyze.add_argument(
        "--with-plan",
        action="store_true",
        help="Run Planner after approved extraction and persist plan output.",
    )
    analyst = subparsers.add_parser("analyst", help="Run analyst only (extraction)")
    analyst.add_argument("--paper-id", required=True, help="Paper ID of an ingested paper")
    analyst.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-approve extraction without CLI prompts.",
    )
    analyst.add_argument(
        "--with-plan",
        action="store_true",
        help="Also run Planner after approved extraction.",
    )
    plan = subparsers.add_parser("plan", help="Run Planner from an existing extraction artifact")
    plan.add_argument("--extraction-path", help="Path to extraction JSON artifact")
    plan.add_argument("--paper-id", help="Paper id to load from data/extractions/<paper_id>.json")
    plan.add_argument("--input-json", help="Path to unified four-key Planner input JSON")
    plan.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-approve plan without CLI prompts.",
    )
    execute = subparsers.add_parser("execute", help="Run Engineer/Executor flow from an existing plan")
    execute.add_argument("--plan-path", help="Path to plan JSON artifact")
    execute.add_argument("--paper-id", help="Paper id to load from data/plans/<paper_id>.json")
    execute.add_argument("--repo-path", help="Path to cloned repository for execution (defaults to ingested repository)")
    execute.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-approve engineer patch without CLI prompts.",
    )
    execute.add_argument(
        "--with-review",
        action="store_true",
        help="Run Reviewer after execution and persist report artifacts.",
    )
    review = subparsers.add_parser("review", help="Run Reviewer from existing run summary")
    review.add_argument("--run-path", help="Path to run_summary.json")
    review.add_argument("--paper-id", help="Paper id to load data/runs/<paper_id>/run_summary.json")
    review.add_argument("--extraction-path", help="Optional extraction artifact path for reported results")

    engineer = subparsers.add_parser(
        "engineer",
        help="Run plan-driven Engineer: execute plan phases in Docker and write metrics.json",
    )
    engineer.add_argument("--paper-id", required=True, help="Paper ID with an approved plan artifact")
    engineer.add_argument(
        "--non-interactive",
        action="store_true",
        help="Log progress to stdout (no interactive prompts in this path).",
    )
    engineer.add_argument("--repo-path", help="Override repository path (defaults to ingested code/)")

    reviewer = subparsers.add_parser(
        "reviewer",
        help="Compare extraction reported_results to Engineer metrics.json",
    )
    reviewer.add_argument("--paper-id", required=True, help="Paper ID with Engineer run artifacts")
    reviewer.add_argument(
        "--run-id",
        required=True,
        help="Run directory name under data/papers/{paper_id}/runs/ (R1, R2, …)",
    )
    reviewer.add_argument(
        "--non-interactive",
        action="store_true",
        help="Log progress to stdout (no interactive prompts in this path).",
    )
    reviewer.add_argument("--extraction-path", help="Optional extraction artifact path")

    run_plan_cmd = subparsers.add_parser(
        "run-plan",
        help="Run Engineer/Executor/Reviewer from a saved plan JSON (skips Analyst, Planner, and ingestion)",
    )
    run_plan_cmd.add_argument("--plan-path", required=True, help="Path to AgentEnvelope[PlannerPayload] JSON")
    run_plan_cmd.add_argument("--paper-id", required=True, help="Paper id used for data/papers/{paper_id}/runs/")
    run_plan_cmd.add_argument(
        "--repo-path",
        help="Optional construction input directory (defaults to the paper code/ directory)",
    )
    run_plan_cmd.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive prompts (required for this PoC path).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        if args.command == "run":
            return run_full_pipeline(
                paper_id=args.paper_id,
                non_interactive=args.non_interactive,
            )
        if args.command == "analyze":
            return run_analyze(
                paper_id=args.paper_id,
                non_interactive=args.non_interactive,
                with_plan=args.with_plan,
            )
        if args.command == "analyst":
            return run_analyst(
                paper_id=args.paper_id,
                non_interactive=args.non_interactive,
                with_plan=args.with_plan,
            )
        if args.command == "plan":
            return run_plan(
                extraction_path=args.extraction_path,
                paper_id=args.paper_id,
                non_interactive=args.non_interactive,
                input_json=args.input_json,
            )
        if args.command == "execute":
            return run_execute(
                plan_path=args.plan_path,
                paper_id=args.paper_id,
                repo_path=args.repo_path,
                non_interactive=args.non_interactive,
                with_review=args.with_review,
            )
        if args.command == "review":
            return run_review(
                run_path=args.run_path,
                paper_id=args.paper_id,
                extraction_path=args.extraction_path,
            )
        if args.command == "engineer":
            return run_engineer(
                paper_id=args.paper_id,
                non_interactive=args.non_interactive,
                repo_path=args.repo_path,
            )
        if args.command == "reviewer":
            return run_reviewer(
                paper_id=args.paper_id,
                non_interactive=args.non_interactive,
                run_id=args.run_id,
                extraction_path=args.extraction_path,
            )
        if args.command == "run-plan":
            return run_saved_plan(
                plan_path=args.plan_path,
                paper_id=args.paper_id,
                repo_path=args.repo_path,
                non_interactive=args.non_interactive,
            )
        return 0
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt()


if __name__ == "__main__":
    raise SystemExit(main())
