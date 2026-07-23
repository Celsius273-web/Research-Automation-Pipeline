"""CLI entrypoint for the Research Assistant pipeline (Analyst -> Planner -> Engineer/Executor -> Reviewer)."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

# #region DEBUG: Setup console logging for visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console
    ]
)
# #endregion

from src.config import EXTRACTIONS_DIR, MAX_RETRY_ATTEMPTS
from src.db import DatabaseError, get_paper_by_id
from src.graphs.research_graph import (
    build_phase1_graph,
    build_phase2_graph,
    build_phase3_graph,
    build_phase4_graph,
)
from src.persistence import (
    load_reported_results,
    persist_extraction,
    persist_plan,
    persist_report,
    persist_run_summary,
    resolve_extraction_path,
    resolve_plan_path,
    resolve_run_summary_path,
)
from src.review_prompts import ReviewCancelledError
from src.pipeline_nodes import (
    default_runtime_constraints,
    make_engineer_executor_nodes,
    make_phase1_nodes,
    make_planner_node,
    make_reviewer_node,
)
from src.state import (
    ExecutionPlan,
    ExecutorResult,
    PaperMetadata,
    RepoContext,
    ResearchState,
    ReviewRecord,
    SectionExtraction,
    make_initial_state,
)
from src.tools.language_detect import detect_language


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


def _has_real_plan(plan: ExecutionPlan) -> bool:
    return bool(plan.model_dump(exclude_defaults=True, exclude_none=True))


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

    saved_extraction = persist_extraction(
        paper=out["paper"],
        extraction=out["approved_extraction"],
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

    saved_extraction = persist_extraction(
        paper=out["paper"],
        extraction=out["approved_extraction"],
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


def run_plan(extraction_path: str | None, paper_id: str | None, non_interactive: bool) -> int:
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
    else:
        approved_extraction = SectionExtraction.model_validate(payload.get("approved_extraction", {}))
    
    review = ReviewRecord.model_validate(payload.get("review", {}))
    if review.status != "approved":
        print("Extraction review is not approved. Planner cannot run.")
        return 1

    state = make_initial_state(paper)
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
    execution_plan = ExecutionPlan.model_validate(payload.get("execution_plan", {}))
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
    execution_plan = ExecutionPlan.model_validate(payload.get("execution_plan", {}))
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
        return 0
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt()


if __name__ == "__main__":
    raise SystemExit(main())
