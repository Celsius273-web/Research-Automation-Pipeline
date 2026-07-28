"""Build and load the Planner's fixed unified input contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import PLANNER_DEFAULT_SETUP_MINUTES
from src.state import (
    PaperMetadata,
    PlannerAnalystOutput,
    PlannerFlags,
    PlannerInputContext,
    PlannerRepoContext,
    UnifiedPlannerInput,
)


def _is_present_text(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and not text.lower().startswith("unknown:")


def infer_paper_type(analyst_output: PlannerAnalystOutput) -> str:
    """Classify the paper using deterministic language in the Analyst output."""
    description = f"{analyst_output.methodology} {analyst_output.notes}".lower()
    toolkit_terms = ("library", "toolkit", "software package", "framework implementing")
    if any(term in description for term in toolkit_terms):
        return "toolkit"

    empirical_terms = ("empirical study", "evaluate", "evaluation", "benchmark", "comparison")
    if analyst_output.datasets_or_benchmarks and any(term in description for term in empirical_terms):
        return "empirical"
    return "methods"


def derive_planner_flags(
    analyst_output: PlannerAnalystOutput,
    repo_context: PlannerRepoContext,
) -> PlannerFlags:
    """Derive Planner routing flags from validated context."""
    return PlannerFlags(
        has_research_question=_is_present_text(analyst_output.research_question),
        has_methodology=_is_present_text(analyst_output.methodology),
        has_code_repo=repo_context.has_code,
        has_datasets=bool(analyst_output.datasets_or_benchmarks),
        paper_type=infer_paper_type(analyst_output),
    )


def _build_analyst_output(context: PlannerInputContext) -> PlannerAnalystOutput:
    extraction = context.approved_extraction
    return PlannerAnalystOutput(
        research_question=extraction.research_question,
        methodology=extraction.methodology,
        datasets_or_benchmarks=extraction.datasets_or_benchmarks,
        variables=extraction.variables,
        hyperparameters=dict(extraction.hyperparameters),
        evaluation_metrics=extraction.evaluation_metrics,
        reported_results=extraction.reported_results,
        notes=extraction.notes,
    )


def _build_repo_context(context: PlannerInputContext) -> PlannerRepoContext:
    repo: dict[str, Any] = context.repo_context
    language = str(repo.get("language") or "unknown")
    has_code = bool(repo.get("has_code", language != "unknown"))
    setup_minutes = repo.get("setup_time_minutes")
    if setup_minutes is None:
        setup_minutes = PLANNER_DEFAULT_SETUP_MINUTES if has_code else 0
    example_commands = repo.get("example_commands") or []
    if not isinstance(example_commands, list):
        example_commands = []
    return PlannerRepoContext(
        url=str(repo.get("url") or repo.get("repo_url") or ""),
        language=language,
        build_system=str(repo.get("build_system") or "unknown"),
        has_code=has_code,
        setup_time_minutes=float(setup_minutes),
        file_tree=str(repo.get("file_tree") or ""),
        readme_summary=str(repo.get("readme_summary") or context.repo_setup_guide or "No README"),
        example_commands=[str(item).strip() for item in example_commands if str(item).strip()],
    )


def build_unified_planner_input(context: PlannerInputContext) -> UnifiedPlannerInput:
    """Translate internal pipeline context into the four-key Planner contract."""
    analyst_output = _build_analyst_output(context)
    repo_context = _build_repo_context(context)
    return UnifiedPlannerInput(
        analyst_output=analyst_output,
        repo_context=repo_context,
        paper_context=context.paper,
        flags=derive_planner_flags(analyst_output, repo_context),
    )


def load_unified_planner_input(path: Path) -> UnifiedPlannerInput:
    """Load unified Planner JSON; always re-derive flags from analyst + repo."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    analyst_output = PlannerAnalystOutput.model_validate(payload.get("analyst_output", {}))
    repo_context = PlannerRepoContext.model_validate(payload.get("repo_context", {}))
    paper_context = PaperMetadata.model_validate(payload.get("paper_context", {}))

    return UnifiedPlannerInput(
        analyst_output=analyst_output,
        repo_context=repo_context,
        paper_context=paper_context,
        flags=derive_planner_flags(analyst_output, repo_context),
    )
