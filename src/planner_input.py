"""Build and load the Planner's fixed unified input contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import (
    PLANNER_DEFAULT_SETUP_MINUTES,
    PLANNER_MAX_CONTEXT_ITEMS,
    PLANNER_MAX_NOTE_ITEM_CHARS,
    PLANNER_MAX_NOTE_ITEMS_PER_CATEGORY,
)
from src.state import (
    PaperMetadata,
    PlannerAnalystOutput,
    PlannerFlags,
    PlannerInputContext,
    PlannerRepoContext,
    UnifiedPlannerInput,
)
from src.tools.repo_context import is_experiment_command
from src.tools.repo_exploration import explore_repository

_PROMPT_LEAK_MARKERS = (
    "words_max",
    "provided_text_only",
    "no_tables",
    "output_field",
    "answer_should",
    "must_be_based_on",
    "no_external_knowledge",
)
_NOTE_CATEGORIES = (
    "what_to_know",
    "implementation_details",
    "issues",
    "warnings",
)
_ISSUE_TERMS = (
    "limitation",
    "missing",
    "unavailable",
    "not explicitly",
    "cut off",
    "only ",
    "resource constraint",
    "negative transfer",
    "duplicate",
)
_WARNING_TERMS = (
    "warning",
    "risk",
    "must ",
    "requires ",
    "careful",
    "assumes ",
    "stability",
    "version",
)
_IMPLEMENTATION_TERMS = (
    "implement",
    "install",
    "optimizer",
    "preprocess",
    "warping",
    "training",
    "architecture",
    "hyperparameter",
    "dataset",
    "dependency",
    "command",
)


def _is_present_text(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and not text.lower().startswith("unknown:")


def infer_paper_type(analyst_output: PlannerAnalystOutput) -> str:
    """Classify the paper using deterministic language in the Analyst output."""
    description = (
        f"{analyst_output.methodology} {analyst_output.paper_overview} {analyst_output.notes}"
    ).lower()
    toolkit_terms = ("library", "toolkit", "software package", "framework implementing")
    if any(term in description for term in toolkit_terms):
        return "toolkit"

    # Prefer strong empirical cues; bare "evaluation"/"benchmark" appear in methods papers too.
    empirical_terms = (
        "empirical study",
        "large-scale evaluation",
        "benchmark comparison",
        "comparative study",
    )
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
        paper_overview=extraction.paper_overview,
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
    entrypoint_hints = repo.get("entrypoint_hints") or []
    if not isinstance(entrypoint_hints, list):
        entrypoint_hints = []
    return PlannerRepoContext(
        url=str(repo.get("url") or repo.get("repo_url") or ""),
        language=language,
        build_system=str(repo.get("build_system") or "unknown"),
        has_code=has_code,
        setup_time_minutes=float(setup_minutes),
        file_tree=str(repo.get("file_tree") or ""),
        readme_summary=str(repo.get("readme_summary") or context.repo_setup_guide or "No README"),
        example_commands=[str(item).strip() for item in example_commands if str(item).strip()],
        entrypoint_hints=[str(item).strip() for item in entrypoint_hints if str(item).strip()],
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


def _is_useful_planner_text(value: object) -> bool:
    text = str(value).strip()
    lowered = text.lower()
    if not text or lowered.startswith("unknown:"):
        return False
    return not any(marker in lowered for marker in _PROMPT_LEAK_MARKERS)


def _clean_text_items(values: list[str]) -> list[str]:
    """Deduplicate Planner list context and remove extraction prompt leakage."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if not _is_useful_planner_text(text) or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= PLANNER_MAX_CONTEXT_ITEMS:
            break
    return cleaned


def _split_note_items(notes: str) -> list[str]:
    items: list[str] = []
    for line in notes.splitlines():
        text = re.sub(r"^[\s*+-]+", "", line).strip()
        if not text:
            continue
        chunks = (
            re.split(r"(?<=[.!?])\s+", text)
            if len(text) > PLANNER_MAX_NOTE_ITEM_CHARS
            else [text]
        )
        items.extend(chunk.strip() for chunk in chunks if chunk.strip())
    return items


def _note_category(note: str) -> str:
    lowered = note.lower()
    if any(term in lowered for term in _ISSUE_TERMS):
        return "issues"
    if any(term in lowered for term in _WARNING_TERMS):
        return "warnings"
    if any(term in lowered for term in _IMPLEMENTATION_TERMS):
        return "implementation_details"
    return "what_to_know"


def categorize_planner_notes(notes: str) -> dict[str, list[str]]:
    """Convert free-form Analyst notes into compact Planner-only bullet groups."""
    categorized = {name: [] for name in _NOTE_CATEGORIES}
    seen: set[str] = set()
    for item in _split_note_items(notes):
        if not _is_useful_planner_text(item):
            continue
        clipped = item[:PLANNER_MAX_NOTE_ITEM_CHARS].strip()
        key = clipped.casefold()
        if key in seen:
            continue
        category = _note_category(clipped)
        if len(categorized[category]) >= PLANNER_MAX_NOTE_ITEMS_PER_CATEGORY:
            continue
        seen.add(key)
        categorized[category].append(clipped)
    return categorized


def build_planner_prompt_context(
    unified_input: UnifiedPlannerInput,
    *,
    repo_path: str | Path | None = None,
) -> dict[str, object]:
    """Build a compact prompt view without changing saved Analyst artifacts."""
    analyst = unified_input.analyst_output
    analyst_context = analyst.model_dump()
    analyst_context["datasets_or_benchmarks"] = _clean_text_items(
        analyst.datasets_or_benchmarks
    )
    analyst_context["variables"] = _clean_text_items(analyst.variables)
    analyst_context["evaluation_metrics"] = _clean_text_items(analyst.evaluation_metrics)
    analyst_context["hyperparameters"] = {
        str(key).strip(): value
        for key, value in analyst.hyperparameters.items()
        if _is_useful_planner_text(key) and _is_useful_planner_text(value)
    }
    analyst_context["reported_results"] = [
        item.model_dump()
        for item in analyst.reported_results
        if _is_useful_planner_text(item.benchmark)
        and _is_useful_planner_text(item.metric_name)
        and _is_useful_planner_text(item.value)
    ]
    analyst_context["notes"] = categorize_planner_notes(analyst.notes)

    resolved_repo_path = _resolve_repo_path(unified_input, repo_path)
    exploration = explore_repository(resolved_repo_path)

    repo_context = unified_input.repo_context.model_dump()
    runnable_commands = [
        command
        for command in unified_input.repo_context.example_commands
        if is_experiment_command(command)
    ]
    explored_commands = exploration.get("example_commands") if isinstance(exploration, dict) else []
    if isinstance(explored_commands, list):
        for command in explored_commands:
            text = str(command).strip()
            if text and is_experiment_command(text) and text not in runnable_commands:
                runnable_commands.append(text)
    explored_hints = exploration.get("entrypoint_hints") if isinstance(exploration, dict) else []
    if isinstance(explored_hints, list):
        merged_hints = list(repo_context.get("entrypoint_hints") or [])
        for hint in explored_hints:
            text = str(hint).strip()
            if text and text not in merged_hints:
                merged_hints.append(text)
        repo_context["entrypoint_hints"] = merged_hints

    repo_context["example_commands"] = runnable_commands
    repo_context["has_runnable_experiment_command"] = bool(runnable_commands)
    if isinstance(exploration, dict):
        surface = str(exploration.get("execution_surface") or "")
        library_cmds = exploration.get("library_verification_commands") or []
        native = exploration.get("native_build")
        native_available = bool(isinstance(native, dict) and native.get("available"))
        repo_context["execution_surface"] = surface or (
            "cli" if runnable_commands else "unknown"
        )
        repo_context["has_library_verification"] = bool(library_cmds) or surface == "library"
        repo_context["has_script_entrypoints"] = bool(exploration.get("script_entrypoints"))
        repo_context["has_native_build"] = native_available or bool(
            exploration.get("native_tests")
        )
        repo_context["has_config_files"] = bool(exploration.get("config_files"))
        repo_context["has_container_files"] = bool(exploration.get("container_files"))
        repo_context["has_artifact_dirs"] = bool(exploration.get("artifact_dirs"))
    if resolved_repo_path is not None:
        repo_context["repo_path"] = str(resolved_repo_path)

    return {
        "analyst_output": analyst_context,
        "repo_context": repo_context,
        "repo_exploration": exploration,
        "paper_context": unified_input.paper_context.model_dump(),
        "flags": unified_input.flags.model_dump(),
    }


def _resolve_repo_path(
    unified_input: UnifiedPlannerInput,
    repo_path: str | Path | None,
) -> Path | None:
    if repo_path is not None:
        path = Path(repo_path)
        return path if path.is_dir() else None
    paper_id = unified_input.paper_context.paper_id
    from src.config import PAPER_BUNDLES_DIR

    candidate = PAPER_BUNDLES_DIR / paper_id / "code"
    return candidate if candidate.is_dir() else None
