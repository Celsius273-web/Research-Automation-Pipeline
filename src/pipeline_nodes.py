"""LangGraph node factories: translate ResearchState into agent calls and back."""

from __future__ import annotations

from src.agents.analyst import PaperAnalyst
from src.agents.engineer import PaperEngineer
from src.agents.executor import ExecutorAgent
from src.agents.planner import PaperPlanner
from src.agents.reviewer import PaperReviewer
from src.bundle import PaperBundle
from src.config import EXECUTOR_TIMEOUT_SECONDS, MAX_RETRY_ATTEMPTS, ROOT_DIR, RUNS_DIR
from src.review_prompts import run_cli_engineer_review, run_cli_plan_review, run_cli_review
from src.state import (
    AgentEnvelope,
    EngineerInputContext,
    EngineerOutput,
    EngineerReviewRecord,
    ExecutionPlan,
    ExecutorResult,
    FailureContext,
    PlannerInputContext,
    PlannerPayload,
    PlanReviewRecord,
    PlanStep,
    PlanStepCore,
    ReportedResult,
    ResearchState,
    ReviewRecord,
)
from src.tools.docker_executor import DockerExecutor
from src.tools.pdf_parser import parse_pdf_sections


def default_runtime_constraints() -> dict[str, str]:
    return {
        "hardware": "cpu_only",
        "max_retry_attempts": str(MAX_RETRY_ATTEMPTS),
        "model_loading": "single_model_at_a_time",
    }


def make_phase1_nodes(non_interactive: bool):
    analyst = PaperAnalyst()

    def parse_node(state: ResearchState) -> ResearchState:
        paper = state["paper"]
        state["section_texts"] = parse_pdf_sections(paper.pdf_path)
        return state

    def analyst_node(state: ResearchState) -> ResearchState:
        extraction = analyst.extract(state["section_texts"])
        state["extraction"] = extraction
        return state

    def review_node(state: ResearchState) -> ResearchState:
        merged = state["extraction"].merged
        if non_interactive:
            state["approved_extraction"] = merged
            state["review"] = ReviewRecord(status="approved", notes="Auto-approved (non-interactive mode).")
            return state
        approved, review = run_cli_review(merged)
        state["approved_extraction"] = approved
        state["review"] = review
        return state

    return parse_node, analyst_node, review_node


def make_planner_node(non_interactive: bool):
    planner = PaperPlanner()

    def planner_node(state: ResearchState) -> ResearchState:
        paper = state["paper"]
        paper_bundle = PaperBundle(paper.paper_id)
        
        # Load extraction from state, or from disk if plan-only flow
        if "extraction" not in state:
            extraction = paper_bundle.get_extraction()
            if not extraction:
                state["errors"].append("No extraction found for paper; run analyst first")
                return state
        else:
            extraction = state["extraction"]
        
        # #region DEBUG: Log extraction state (H1)
        import json
        import logging
        logger = logging.getLogger(__name__)
        logger.info("PLANNER_NODE_EXTRACTION_DEBUG: extraction type=%s, merged=%s, by_section keys=%s", 
                    type(extraction).__name__,
                    "None" if extraction.merged is None else f"research_q={bool(extraction.merged.research_question)}, vars={len(extraction.merged.variables)}, hyper={len(extraction.merged.hyperparameters)}",
                    list(extraction.by_section.keys()) if extraction.by_section else "None")
        # #endregion
        
        # Load rich context from paper bundle
        repo_context = paper_bundle.get_repo_info()
        repo_setup_guide = paper_bundle.get_setup_guide()
        hyperparameter_reference = paper_bundle.get_hyperparameter_reference()
        
        # Convert repo_context to dict for backwards compatibility
        repo_context_dict = {
            "repo_url": repo_context.repo_url,
            "repo_path": repo_context.repo_path,
            "language": repo_context.language,
            "build_system": repo_context.build_system,
            "notes": repo_context.notes,
        }
        
        # #region DEBUG: Log context being built (H2, H3)
        logger.info("PLANNER_CONTEXT_BUILD: merged has research_q=%s, %d vars, %d hyper; by_section has %d sections",
                    bool(extraction.merged.research_question) if extraction.merged else "None",
                    len(extraction.merged.variables) if extraction.merged else 0,
                    len(extraction.merged.hyperparameters) if extraction.merged else 0,
                    len(extraction.by_section) if extraction.by_section else 0)
        # #endregion
        
        context = PlannerInputContext(
            paper=paper,
            approved_extraction=extraction.merged,
            extraction_sections=extraction.by_section,
            runtime_constraints=default_runtime_constraints(),
            repo_context=repo_context_dict,
            repo_setup_guide=repo_setup_guide,
            hyperparameter_reference=hyperparameter_reference,
            extraction_file_path=str(paper_bundle.extraction_path),
            paper_bundle_path=str(paper_bundle.bundle_dir),
        )
        
        # #region DEBUG: Log context passed to planner (H3)
        logger.info("PLANNER_CONTEXT_READY: extraction_sections in context=%s, has %d sections",
                    bool(context.extraction_sections),
                    len(context.extraction_sections) if context.extraction_sections else 0)
        # #endregion
        try:
            plan = planner.build_plan(context)
        except RuntimeError as exc:
            state["errors"].append(str(exc))
            return state

        if non_interactive:
            state["planner_output"] = plan
            state["planner_output_json"] = plan.model_dump()
            state["plan_review"] = PlanReviewRecord(
                status="approved",
                notes="Auto-approved (non-interactive mode).",
            )
            return state

        reviewed_plan, plan_review = run_cli_plan_review(plan)
        state["planner_output"] = reviewed_plan
        state["planner_output_json"] = reviewed_plan.model_dump()
        state["plan_review"] = plan_review
        return state

    return planner_node


def make_reviewer_node(reported_results: list[ReportedResult]):
    reviewer = PaperReviewer()

    def reviewer_node(state: ResearchState) -> ResearchState:
        execution_plan = state["planner_output"]
        run_summary = {
            "final_executor_status": state["executor_result"].final_status,
            "total_attempts": str(state["executor_result"].total_attempts),
            "retry_count": str(state.get("retry_count", 0)),
            "language": state["repo_context"].language,
            "build_system": state["repo_context"].build_system,
        }
        
        if isinstance(execution_plan, ExecutionPlan):
            domain = execution_plan.domain
        else:
            domain = execution_plan.payload.core.domain
            
        try:
            report = reviewer.generate_report(
                paper_id=state["paper"].paper_id,
                domain=domain,
                reported_results=reported_results,
                captured_metrics=state["executor_result"].captured_metrics,
                run_summary=run_summary,
            )
        except RuntimeError as exc:
            state["errors"].append(str(exc))
            return state
        state["reviewer_report"] = report
        return state

    return reviewer_node


def _build_failure_context(state: ResearchState) -> FailureContext | None:
    attempts = state["executor_result"].attempts
    if not attempts:
        return None
    last = attempts[-1]
    return FailureContext(
        stage=last.stage,
        exit_code=last.exit_code,
        log_excerpt=(last.stderr_excerpt or last.stdout_excerpt),
        prior_patch_summary=state["engineer_output"].rationale,
    )


def make_engineer_executor_nodes(
    execution_plan: ExecutionPlan | AgentEnvelope[PlannerPayload],
    runtime_constraints: dict[str, str],
    non_interactive: bool,
):
    engineer = PaperEngineer()
    docker_executor = DockerExecutor(project_root=ROOT_DIR)
    executor = ExecutorAgent(project_root=ROOT_DIR, runs_dir=RUNS_DIR, docker_executor=docker_executor)

    def engineer_node(state: ResearchState) -> ResearchState:
        if isinstance(execution_plan, ExecutionPlan):
            steps = execution_plan.steps
        else:
            steps = execution_plan.payload.core.steps
            
        if not steps:
            state["errors"].append("Execution plan has no steps to run.")
            return state
        step = steps[0]
        context = EngineerInputContext(
            paper=state["paper"],
            execution_plan=execution_plan,
            plan_step=step,
            repo_context=state["repo_context"],
            runtime_constraints=runtime_constraints,
            failure_context=_build_failure_context(state),
        )
        try:
            output = engineer.propose_patch(context)
        except RuntimeError as exc:
            state["errors"].append(str(exc))
            return state

        if not output.step_id:
            output.step_id = step.step_id or "step_1"
        state["engineer_output"] = output
        return state

    def engineer_review_node(state: ResearchState) -> ResearchState:
        if non_interactive or int(state.get("retry_count", 0)) > 0:
            state["engineer_review"] = EngineerReviewRecord(
                status="approved",
                notes="Auto-approved for non-interactive or retry flow.",
            )
            return state
        state["engineer_review"] = run_cli_engineer_review(state["engineer_output"])
        return state

    def executor_node(state: ResearchState) -> ResearchState:
        if state["engineer_review"].status != "approved":
            state["executor_result"].needs_manual_patch = True
            state["executor_result"].final_status = "failed"
            return state

        output = state["engineer_output"]
        if isinstance(execution_plan, ExecutionPlan):
            step = execution_plan.steps[0]
            commands = step.verification or output.verification_commands
            results_path = step.results_path
            step_id = step.step_id
        else:
            step = execution_plan.payload.core.steps[0]
            # Convert run_command string to a list, or use engineer's verification commands
            commands = [step.run_command] if step.run_command else output.verification_commands
            results_path = step.results_path
            step_id = step.step_id

        if not commands:
            state["errors"].append("No verification commands available for executor.")
            state["executor_result"].final_status = "failed"
            return state

        attempt_number = int(state.get("retry_count", 0)) + 1
        
        try:
            executor.apply_patches(state["repo_context"], output)
            result = executor.execute_step(
                paper_id=state["paper"].paper_id,
                step_id=step_id or "step_1",
                repo_context=state["repo_context"],
                verification_commands=commands,
                current_attempt=attempt_number,
                results_path=results_path,
                timeout_seconds=EXECUTOR_TIMEOUT_SECONDS,
            )
        except RuntimeError as exc:
            state["errors"].append(str(exc))
            result = ExecutorResult(final_status="failed", total_attempts=attempt_number)

        existing = state["executor_result"].attempts
        result.attempts = existing + result.attempts
        result.total_attempts = len(result.attempts)
        if result.final_status != "success":
            state["retry_count"] = attempt_number
            max_retries = int(runtime_constraints.get("max_retry_attempts", str(MAX_RETRY_ATTEMPTS)))
            if state["retry_count"] >= max_retries:
                result.final_status = "exhausted_retries"
        elif not result.captured_metrics:
            state["errors"].append(
                "Step succeeded but no results metrics were captured"
                f" (results_path='{results_path or '<unset>'}')."
            )
        state["executor_result"] = result
        return state

    return engineer_node, engineer_review_node, executor_node
