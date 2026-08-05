"""Planner agent: generate a structured phase-DAG plan from approved extraction."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

from pydantic import ValidationError

from src.agents.planner_debug import PlannerDebugTrace, write_planner_debug_files
from src.agents.prompts.planner_prompt import build_planner_system_prompt
from src.config import (
    MODEL_TEMPERATURE,
    OLLAMA_HOST,
    PLANNER_MAX_RETRIES,
    PLANNER_MODEL,
    PLANNER_NUM_PREDICT,
)
from src.planner_input import build_planner_prompt_context, build_unified_planner_input
from src.state import (
    AgentEnvelope,
    PlannerEnvelope,
    PlannerInputContext,
    PlannerPayload,
    UnifiedPlannerInput,
    UnknownItem,
)
from src.tools.phase_builder import build_plan_phases, collect_surface_context_notes
from src.tools.repo_context import is_experiment_command
from src.tools.plan_verification import verify_and_filter_phases
from src.tools.results_paths import (
    is_under_paper_results,
    results_contract_for_prompt,
    results_summary_relpath,
)

logger = logging.getLogger(__name__)

STRICT_RETRY_REMINDER = (
    "Your previous response did not match the Planner schema. Return only a Planner envelope with "
    "top-level keys schema_version, agent, status, unknowns, warnings, and payload. "
    "Do not summarize or reformat the input."
)
AIM_GROUNDING_RETRY_REMINDER = (
    "Research question was provided in analyst_output; do not mark it unknown. "
    "Restate the paper aim in objective, then plan repo phases for the Engineer."
)
METHODOLOGY_AIM_RETRY_REMINDER = (
    "flags.has_research_question is false but methodology is present. "
    "Do not block. Synthesize objective from methodology, leave unknown entrypoints as empty "
    "run_command with unknowns, and use status partial or ok."
)
BLOCKED_SOFTEN_RETRY_REMINDER = (
    "Do not use status blocked when methodology or research_question is present. "
    "Use status partial, keep unknowns, and continue planning from available context."
)
GROUNDING_RETRY_REMINDER = (
    "Ground payload.phases in analyst_output, repo_context, and repo_exploration only. "
    "Use a phase DAG (setup → smoke → experimental groups → summarize) with compact axes "
    "and a few example matrix rows containing run_command, code_refs, and verify. "
    "Do not invent benchmarks, hyperparameters, metrics, entrypoints, or file paths."
)
RUNNABLE_MATRIX_RETRY_REMINDER = (
    "has_runnable_experiment_command is true. Do not leave phases empty. Include setup, smoke, "
    "and at least one experiment phase with axes + example matrix rows. Fill planned_actions "
    "with short Engineer attack notes. Matrix rows need run_command/code_refs/verify."
)
OUTPUT_SKELETON = (
    '{"schema_version":"2.0","agent":"planner","status":"ok|partial|blocked",'
    '"unknowns":[],"warnings":[],"payload":{"plan_summary":"","domain":"","objective":"",'
    '"phases":[],"assumptions":[],"constraints":[],'
    '"missing_context":[],"verification_checks":[],"risks":[],"organization":[],'
    '"execution":[],"repo_usage":[],"engineer_notes":[],"results_summary_path":""}}'
)


def _clean_json_response(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = raw.rstrip("`").strip()
    return raw


def _schema_retry_reminder(
    exc: Exception,
    parsed: dict[str, object] | None,
) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_input=False)
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in errors[:6]
        )
    else:
        details = str(exc)[:500]
    returned_keys = sorted(parsed) if parsed else []
    return (
        f"{STRICT_RETRY_REMINDER} Validation failure: {details}. "
        f"Your returned top-level keys were {returned_keys}. "
        f"Required skeleton: {OUTPUT_SKELETON}"
    )


def _ensure_list_str(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_axis_values(value: object) -> list[str | int | float | bool]:
    if not isinstance(value, list):
        return []
    out: list[str | int | float | bool] = []
    for item in value:
        if isinstance(item, (int, float, bool)):
            out.append(item)
        else:
            text = str(item).strip()
            if text:
                out.append(text)
    return out


def _normalize_run_variables(value: object) -> dict[str, str | int | float | bool]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        name = str(key).strip()
        if not name:
            continue
        if isinstance(item, (int, float, bool)):
            out[name] = item
        else:
            out[name] = str(item)
    return out


def _normalize_phase_matrix(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, object]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        run_command = str(
            item.get("run_command") or item.get("execution_pattern") or ""
        ).strip()
        if not run_command:
            impl = item.get("implementation_steps")
            if isinstance(impl, list) and impl:
                run_command = str(impl[0]).strip()
        variables = item.get("variables")
        if isinstance(variables, list):
            variables_map: dict[str, str | int | float | bool] = {}
        else:
            variables_map = _normalize_run_variables(variables)
        if not variables_map and isinstance(item.get("hyperparameters"), dict):
            variables_map = _normalize_run_variables(item.get("hyperparameters"))
        rows.append(
            {
                "name": str(item.get("name", f"run_{idx}")).strip() or f"run_{idx}",
                "variables": variables_map,
                "run_command": run_command,
                "code_refs": _ensure_list_str(item.get("code_refs")),
                "verify": _ensure_list_str(item.get("verify")),
                "results_path": str(item.get("results_path", "")).strip(),
                "metrics": _ensure_list_str(item.get("metrics")),
                "source": str(item.get("source") or item.get("source_section") or "").strip(),
            }
        )
    return rows


def _normalize_phases(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    phases: list[dict[str, object]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        axes_raw = item.get("axes") if isinstance(item.get("axes"), dict) else {}
        axes = {
            str(key).strip(): _normalize_axis_values(values)
            for key, values in axes_raw.items()
            if str(key).strip()
        }
        phases.append(
            {
                "phase_id": str(item.get("phase_id", f"phase_{idx}")).strip() or f"phase_{idx}",
                "title": str(item.get("title", f"Phase {idx}")).strip() or f"Phase {idx}",
                "goal": str(item.get("goal", "")).strip(),
                "depends_on": _ensure_list_str(item.get("depends_on")),
                "variables": _ensure_list_str(item.get("variables")),
                "axes": axes,
                "run_template": str(item.get("run_template", "")).strip(),
                "matrix": _normalize_phase_matrix(item.get("matrix")),
                "planned_actions": str(item.get("planned_actions", "")).strip(),
                "results_path": str(item.get("results_path", "")).strip(),
            }
        )
    return phases


def _legacy_steps_to_phases(data: dict[str, object]) -> list[dict[str, object]]:
    """Best-effort migration when the model still emits steps/experiment_matrix."""
    steps = data.get("steps")
    experiments = data.get("experiment_matrix")
    phases: list[dict[str, object]] = []
    if isinstance(steps, list) and steps:
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            phases.append(
                {
                    "phase_id": str(step.get("step_id", f"phase_{idx}")).strip()
                    or f"phase_{idx}",
                    "title": str(step.get("title", f"Phase {idx}")).strip() or f"Phase {idx}",
                    "goal": str(step.get("goal", "")).strip(),
                    "depends_on": _ensure_list_str(step.get("depends_on")),
                    "variables": [],
                    "axes": {},
                    "run_template": str(step.get("run_command", "")).strip(),
                    "matrix": [],
                    "planned_actions": str(step.get("goal", "")).strip(),
                    "results_path": str(step.get("results_path", "")).strip(),
                }
            )
    if isinstance(experiments, list) and experiments:
        matrix = _normalize_phase_matrix(experiments)
        if phases:
            phases[-1]["matrix"] = matrix
            if not phases[-1].get("variables"):
                keys: list[str] = []
                for row in matrix:
                    keys.extend(str(key) for key in row.get("variables", {}))
                phases[-1]["variables"] = list(dict.fromkeys(keys))
        else:
            phases.append(
                {
                    "phase_id": "experiments",
                    "title": "Experiments",
                    "goal": "Run experiment matrix",
                    "depends_on": [],
                    "variables": [],
                    "axes": {},
                    "run_template": "",
                    "matrix": matrix,
                    "planned_actions": "",
                    "results_path": "",
                }
            )
    return phases


def _normalize_planner_payload(payload: dict[str, object]) -> dict[str, object]:
    """Coerce LLM JSON into the phase-DAG PlannerPayload shape."""
    data = dict(payload)
    if "core" in data and isinstance(data["core"], dict):
        core = data.pop("core")
        for key, value in core.items():
            data.setdefault(key, value)
    if "extensions" in data and isinstance(data["extensions"], dict):
        extensions = data.pop("extensions")
        for key, value in extensions.items():
            data.setdefault(key, value)

    for field in (
        "assumptions",
        "constraints",
        "missing_context",
        "verification_checks",
        "risks",
        "organization",
        "execution",
        "repo_usage",
        "engineer_notes",
    ):
        data[field] = _ensure_list_str(data.get(field))

    phases = _normalize_phases(data.get("phases"))
    if not phases:
        phases = _legacy_steps_to_phases(data)
    data["phases"] = phases
    data.pop("steps", None)
    data.pop("experiment_matrix", None)
    data["plan_summary"] = str(data.get("plan_summary", "")).strip()
    data["domain"] = str(data.get("domain", "")).strip()
    data["objective"] = str(data.get("objective", "")).strip()
    data["results_summary_path"] = str(data.get("results_summary_path", "")).strip()
    return data


def _normalize_envelope_dict(parsed: dict[str, object]) -> dict[str, object]:
    data = dict(parsed)
    payload = data.get("payload")
    if isinstance(payload, dict):
        data["payload"] = _normalize_planner_payload(payload)
    return data


def _extraction_to_analyst_dict(extraction: object) -> dict[str, object]:
    return {
        "research_question": extraction.research_question,
        "paper_overview": extraction.paper_overview,
        "methodology": extraction.methodology,
        "datasets_or_benchmarks": extraction.datasets_or_benchmarks,
        "variables": extraction.variables,
        "hyperparameters": dict(extraction.hyperparameters),
        "evaluation_metrics": extraction.evaluation_metrics,
        "reported_results": [item.model_dump() for item in extraction.reported_results],
        "notes": extraction.notes,
    }


def _is_present_research_question(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and not text.startswith("unknown:")


def _unknown_mentions_aim(unknowns: list[object]) -> bool:
    for item in unknowns:
        field = ""
        if hasattr(item, "field"):
            field = str(getattr(item, "field", "")).lower()
        elif isinstance(item, dict):
            field = str(item.get("field", "")).lower()
        if "research_question" in field or field.endswith(".aim") or field == "aim":
            return True
    return False


def _plan_misses_aim(
    research_question: str,
    methodology: str,
    plan: AgentEnvelope[PlannerPayload],
) -> bool:
    has_rq = _is_present_research_question(research_question)
    has_method = bool((methodology or "").strip()) and not (
        methodology or ""
    ).strip().lower().startswith("unknown:")
    if not has_rq and not has_method:
        return False
    if has_rq and _unknown_mentions_aim(list(plan.unknowns)):
        return True
    payload = plan.payload
    return not (payload.objective or "").strip() or not (payload.plan_summary or "").strip()


def _should_soften_blocked(
    research_question: str,
    methodology: str,
    plan: AgentEnvelope[PlannerPayload],
) -> bool:
    if plan.status != "blocked":
        return False
    has_rq = _is_present_research_question(research_question)
    has_method = bool((methodology or "").strip()) and not (
        methodology or ""
    ).strip().lower().startswith("unknown:")
    return has_rq or has_method


def _soften_blocked_envelope(
    plan: AgentEnvelope[PlannerPayload],
) -> AgentEnvelope[PlannerPayload]:
    warnings = list(plan.warnings)
    note = (
        "status downgraded from blocked to partial because methodology or "
        "research_question is present"
    )
    if note not in warnings:
        warnings.append(note)
    return plan.model_copy(update={"status": "partial", "warnings": warnings})


def _command_stem(command: str) -> str:
    text = command.strip()
    if not text:
        return ""
    tokens = text.split()
    for token in tokens:
        if token.endswith(".py") or "/" in token:
            return token
    if len(tokens) >= 2 and tokens[0] in {"python", "python3", "pip", "pip3"}:
        return tokens[1]
    return tokens[0]


def _runnable_commands(
    unified_input: UnifiedPlannerInput,
    repo_exploration: dict[str, object] | None,
) -> list[str]:
    commands = [
        command
        for command in unified_input.repo_context.example_commands
        if is_experiment_command(command)
    ]
    explored = (repo_exploration or {}).get("example_commands") if repo_exploration else []
    if isinstance(explored, list):
        for command in explored:
            text = str(command).strip()
            if text and is_experiment_command(text) and text not in commands:
                commands.append(text)
    exploration = repo_exploration if isinstance(repo_exploration, dict) else {}
    for key in (
        "library_verification_commands",
        "verification_commands",
    ):
        extra = exploration.get(key) or []
        if isinstance(extra, list):
            for command in extra:
                text = str(command).strip()
                if text and text not in commands:
                    commands.append(text)
    for script in exploration.get("script_entrypoints") or []:
        text = f"python {script}".strip()
        if text not in commands:
            commands.append(text)
    native = exploration.get("native_build")
    if isinstance(native, dict):
        for command in native.get("commands") or []:
            text = str(command).strip()
            if text and text not in commands:
                commands.append(text)
    return commands


_RUNNABLE_SURFACES = frozenset(
    {"cli", "script", "library", "native", "config", "container", "artifact"}
)
_EXPERIMENT_PHASE_IDS = frozenset(
    {
        "smoke",
        "synthetic",
        "real_world",
        "experiments",
        "library_smoke",
        "reproduce_similar",
        "native_smoke",
        "container_smoke",
        "verify_artifacts",
        "missing_context",
        "deps_check",
        "generate_inputs",
    }
)


def _surface_is_runnable(surface: str, exploration: dict[str, object]) -> bool:
    if surface in _RUNNABLE_SURFACES:
        if surface == "cli":
            return True
        if surface == "script":
            return bool(
                exploration.get("script_entrypoints") or exploration.get("example_commands")
            )
        if surface == "library":
            return bool(
                exploration.get("test_files")
                or exploration.get("notebooks")
                or exploration.get("library_verification_commands")
            )
        if surface == "native":
            native = exploration.get("native_build")
            return bool(
                (isinstance(native, dict) and native.get("available"))
                or exploration.get("native_tests")
                or exploration.get("make_targets")
            )
        if surface == "config":
            return bool(exploration.get("config_files"))
        if surface == "container":
            return bool(exploration.get("container_files"))
        if surface == "artifact":
            return bool(exploration.get("artifact_dirs"))
    return False


def _collect_grounding_issues(
    envelope: AgentEnvelope[PlannerPayload],
    unified_input: UnifiedPlannerInput,
    *,
    repo_exploration: dict[str, object] | None = None,
) -> list[str]:
    """Return grounding problems that should trigger a soft retry."""
    issues: list[str] = []
    paper_id = unified_input.paper_context.paper_id
    expected_summary = results_summary_relpath(paper_id)
    payload = envelope.payload

    if payload.results_summary_path and payload.results_summary_path != expected_summary:
        issues.append(
            f"results_summary_path must be {expected_summary}, got {payload.results_summary_path}"
        )

    for phase in payload.phases:
        if phase.results_path and not is_under_paper_results(phase.results_path, paper_id):
            issues.append(
                f"phase {phase.phase_id} results_path must be under results/{paper_id}/: "
                f"{phase.results_path}"
            )
        for row in phase.matrix:
            if row.results_path and not is_under_paper_results(row.results_path, paper_id):
                issues.append(
                    f"phase {phase.phase_id} run {row.name} results_path must be under "
                    f"results/{paper_id}/: {row.results_path}"
                )

    phase_ids = {phase.phase_id for phase in payload.phases}
    for phase in payload.phases:
        for dep in phase.depends_on:
            if dep not in phase_ids:
                issues.append(f"phase {phase.phase_id} depends_on unknown phase_id: {dep}")

    runnable = _runnable_commands(unified_input, repo_exploration)
    exploration = repo_exploration if isinstance(repo_exploration, dict) else {}
    surface = str(exploration.get("execution_surface") or "")
    multi_surface = _surface_is_runnable(surface, exploration) or any(
        phase.phase_id in _EXPERIMENT_PHASE_IDS for phase in payload.phases
    )
    if runnable or multi_surface:
        if not payload.phases:
            issues.append(
                "runnable/multi-surface evidence is present but phases is empty; "
                "emit setup plus smoke/experiment phases for the detected surface"
            )
        else:
            if not any(phase.phase_id == "setup" for phase in payload.phases):
                issues.append("runnable plans require a setup phase")
            experiment_phases = [
                phase
                for phase in payload.phases
                if phase.phase_id not in {"setup", "summarize"}
                and (
                    phase.matrix
                    or phase.axes
                    or phase.phase_id in _EXPERIMENT_PHASE_IDS
                )
            ]
            if not experiment_phases:
                issues.append(
                    "runnable plans require at least one experiment/surface phase with axes, "
                    "matrix rows, or a scaffold phase_id"
                )
            for phase in experiment_phases:
                if phase.matrix and not any((row.run_command or "").strip() for row in phase.matrix):
                    issues.append(
                        f"phase {phase.phase_id} matrix rows need concrete run_command values"
                    )
                if phase.matrix and not any(row.code_refs for row in phase.matrix):
                    issues.append(f"phase {phase.phase_id} matrix rows need code_refs")
                if phase.matrix and not any(row.verify for row in phase.matrix):
                    issues.append(f"phase {phase.phase_id} matrix rows need verify checks")
        if not payload.repo_usage:
            issues.append("runnable plans require repo_usage detailing README/commands/files used")
        if not payload.execution:
            issues.append("runnable plans require execution notes describing reuse of repo scripts")

    build_command = unified_input.repo_context.build_system.strip()
    runnable_stems = {_command_stem(command) for command in runnable}
    grounded_paths = set()
    for key in ("test_files", "script_entrypoints", "native_tests", "config_files"):
        for item in exploration.get(key) or []:
            grounded_paths.add(str(item))
    for phase in payload.phases:
        for row in phase.matrix:
            command = (row.run_command or "").strip()
            if not command or command == build_command or command.startswith("#"):
                continue
            command = command.split("  #", 1)[0].strip()
            if multi_surface and (
                command.startswith(("python ", "python3 ", "make ", "cmake ", "ctest ", "docker "))
                or command.startswith("make")
            ):
                if " -c " in f" {command} " or any(
                    path and path in command for path in grounded_paths
                ):
                    continue
                if any(
                    token in command.lower()
                    for token in ("ctest", "cmake", "make test", "docker")
                ):
                    continue
            stem = _command_stem(command)
            if runnable and stem not in runnable_stems and not any(
                stem in cmd or cmd.endswith(stem) for cmd in runnable
            ):
                issues.append(
                    f"phase {phase.phase_id} run {row.name} run_command is not grounded in "
                    f"example_commands: {row.run_command}"
                )
    return issues


def _merge_planned_actions(deterministic, llm_phases):
    """Keep deterministic axes/matrix/goals; only polish titles on structure-owned phases.

    LLM goals/planned_actions previously overwrote scaffolds with invented ablation values
    (e.g. goal says [20,40] while axes are [1,2,3,4]). Experiment phases with axes or
    matrix keep scaffold text; setup/summarize may still adopt LLM prose.
    """
    by_id = {phase.phase_id: phase for phase in llm_phases}
    merged = []
    for phase in deterministic:
        llm = by_id.get(phase.phase_id)
        if llm is None:
            merged.append(phase)
            continue
        if phase.axes or phase.matrix:
            merged.append(
                phase.model_copy(
                    update={"title": llm.title or phase.title}
                )
            )
            continue
        merged.append(
            phase.model_copy(
                update={
                    "title": llm.title or phase.title,
                    "goal": llm.goal or phase.goal,
                    "planned_actions": llm.planned_actions or phase.planned_actions,
                }
            )
        )
    return merged


def _ensure_phases_from_builder(
    envelope: AgentEnvelope[PlannerPayload],
    unified_input: UnifiedPlannerInput,
    *,
    repo_exploration: dict[str, object] | None,
) -> AgentEnvelope[PlannerPayload]:
    """Replace thin/invented phases with deterministic DAG; keep useful LLM notes."""
    paper_id = unified_input.paper_context.paper_id
    exploration = repo_exploration if isinstance(repo_exploration, dict) else {}
    built = build_plan_phases(
        paper_id=paper_id,
        build_system=unified_input.repo_context.build_system,
        exploration=exploration,
        analyst=unified_input.analyst_output,
    )
    has_experiment = any(
        phase.phase_id not in {"setup", "summarize"}
        and (phase.matrix or phase.axes or phase.phase_id in _EXPERIMENT_PHASE_IDS)
        for phase in built
    )
    built_matrix_rows = sum(len(phase.matrix) for phase in built)
    llm_matrix_rows = sum(len(phase.matrix) for phase in envelope.payload.phases)

    if not built or not has_experiment:
        # No scaffold available; keep LLM phases if present, else keep builder setup.
        if envelope.payload.phases and llm_matrix_rows >= built_matrix_rows:
            return envelope
        warnings = list(envelope.warnings)
        note = (
            "only setup/missing_context available; repo evidence incomplete for experiment matrix."
        )
        if built and note not in warnings:
            warnings.append(note)
        context_notes = collect_surface_context_notes(
            exploration=exploration,
            analyst=unified_input.analyst_output,
            phases=built,
        )
        return envelope.model_copy(
            update={
                "payload": envelope.payload.model_copy(
                    update={
                        "phases": built,
                        "missing_context": _unique_text(
                            list(envelope.payload.missing_context)
                            + context_notes["missing_context"]
                        ),
                        "risks": _unique_text(
                            list(envelope.payload.risks) + context_notes["risks"]
                        ),
                        "verification_checks": _unique_text(
                            list(envelope.payload.verification_checks)
                            + context_notes["verification_checks"]
                        ),
                    }
                ),
                "warnings": warnings,
            }
        )

    # Prefer deterministic scaffold whenever it carries a real matrix / surface DAG.
    phases = _merge_planned_actions(built, list(envelope.payload.phases))
    organization = list(envelope.payload.organization) or [
        " → ".join(phase.phase_id for phase in phases)
    ]
    execution = list(envelope.payload.execution)
    if not execution or built_matrix_rows > llm_matrix_rows:
        sample = next((row.run_command for phase in phases for row in phase.matrix), "")
        execution = [
            "Follow phases in depends_on order.",
            f"Example run: {sample}" if sample else "Use each phase run_template / matrix run_command.",
        ]
    repo_usage = list(envelope.payload.repo_usage) or [
        "README + exploration evidence + phase axes from repo_exploration / Analyst ablations",
    ]
    if exploration.get("script_entrypoints"):
        repo_usage = _unique_text(
            repo_usage + [f"script:{path}" for path in exploration.get("script_entrypoints") or []][:6]
        )
    if exploration.get("test_files"):
        repo_usage = _unique_text(
            repo_usage + [f"test:{path}" for path in exploration.get("test_files") or []][:6]
        )
    native = exploration.get("native_build")
    if isinstance(native, dict) and native.get("files"):
        repo_usage = _unique_text(repo_usage + [f"native:{path}" for path in native.get("files") or []])

    context_notes = collect_surface_context_notes(
        exploration=exploration,
        analyst=unified_input.analyst_output,
        phases=phases,
    )
    warnings = list(envelope.warnings)
    # Drop stale grounding complaints that the scaffold just fixed.
    warnings = [
        item
        for item in warnings
        if "phases is empty" not in item
        and "require at least one experiment" not in item
        and "require execution notes" not in item
        and "require repo_usage" not in item
        and "require a setup phase" not in item
    ]
    note = "phases rebuilt deterministically from execution_surface scaffolds and Analyst lists."
    if note not in warnings:
        warnings.append(note)
    surface = str(exploration.get("execution_surface") or "")
    if surface and f"execution_surface={surface}" not in warnings:
        warnings.append(f"execution_surface={surface}")

    unknowns = [
        item
        for item in envelope.unknowns
        if not (
            item.field == "grounding"
            and (
                "phases is empty" in item.reason
                or "require at least one experiment" in item.reason
                or "require execution notes" in item.reason
                or "require repo_usage" in item.reason
            )
        )
    ]

    payload = envelope.payload.model_copy(
        update={
            "phases": phases,
            "organization": organization,
            "execution": execution,
            "repo_usage": repo_usage,
            "missing_context": _unique_text(
                list(envelope.payload.missing_context) + context_notes["missing_context"]
            ),
            "risks": _unique_text(list(envelope.payload.risks) + context_notes["risks"]),
            "verification_checks": _unique_text(
                list(envelope.payload.verification_checks)
                + context_notes["verification_checks"]
            ),
            "results_summary_path": results_summary_relpath(paper_id),
        }
    )
    status = envelope.status
    if status == "partial" and not unknowns and has_experiment and built_matrix_rows:
        status = "ok"
    return envelope.model_copy(
        update={"payload": payload, "warnings": warnings, "unknowns": unknowns, "status": status}
    )

def _unique_text(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _apply_results_contract(
    envelope: AgentEnvelope[PlannerPayload],
    paper_id: str,
) -> AgentEnvelope[PlannerPayload]:
    expected = results_summary_relpath(paper_id)
    payload = envelope.payload.model_copy(update={"results_summary_path": expected})
    return envelope.model_copy(update={"payload": payload})


def _mark_partial_with_grounding(
    envelope: AgentEnvelope[PlannerPayload],
    issues: list[str],
) -> AgentEnvelope[PlannerPayload]:
    unknowns = list(envelope.unknowns)
    warnings = list(envelope.warnings)
    for issue in issues:
        unknowns.append(
            UnknownItem(
                field="grounding",
                reason=issue,
                severity="medium",
            )
        )
        if issue not in warnings:
            warnings.append(issue)
    status = envelope.status if envelope.status == "blocked" else "partial"
    return envelope.model_copy(update={"status": status, "unknowns": unknowns, "warnings": warnings})


def _apply_runnable_repo_contract(
    envelope: AgentEnvelope[PlannerPayload],
    unified_input: UnifiedPlannerInput,
    *,
    repo_exploration: dict[str, object] | None = None,
) -> AgentEnvelope[PlannerPayload]:
    """Keep multi-surface phases; never wipe to empty when has_code."""
    repo = unified_input.repo_context
    runnable_commands = [
        command for command in repo.example_commands if is_experiment_command(command)
    ]
    exploration = repo_exploration if isinstance(repo_exploration, dict) else {}
    surface = str(exploration.get("execution_surface") or "")
    if (
        not repo.has_code
        or runnable_commands
        or _surface_is_runnable(surface, exploration)
    ):
        # Never leave an empty phase list when code exists.
        if repo.has_code and not envelope.payload.phases:
            rebuilt = build_plan_phases(
                paper_id=unified_input.paper_context.paper_id,
                build_system=repo.build_system,
                exploration=exploration or {"execution_surface": "unknown"},
                analyst=unified_input.analyst_output,
            )
            missing = list(envelope.payload.missing_context)
            note = (
                "Repository code is present but phases were empty; "
                "inserted deterministic surface scaffold."
            )
            if note not in missing:
                missing.append(note)
            payload = envelope.payload.model_copy(
                update={"phases": rebuilt, "missing_context": missing}
            )
            return envelope.model_copy(update={"payload": payload, "status": "partial"})
        return envelope

    missing_note = (
        "Repository code is present, but no grounded runnable surface evidence was found."
    )
    engineer_note = (
        "Do not invent CLIs; obtain documented scripts, tests, cmake/make targets, or configs "
        "before execution."
    )
    missing_context = list(envelope.payload.missing_context)
    engineer_notes = list(envelope.payload.engineer_notes)
    if missing_note not in missing_context:
        missing_context.append(missing_note)
    if engineer_note not in engineer_notes:
        engineer_notes.append(engineer_note)

    unknowns = list(envelope.unknowns)
    if not any(item.field == "repo_context.example_commands" for item in unknowns):
        unknowns.append(
            UnknownItem(
                field="repo_context.example_commands",
                reason="No documented runnable experiment command was found.",
                severity="high",
            )
        )

    # Prefer unknown scaffold over silently emptying phases.
    rebuilt = build_plan_phases(
        paper_id=unified_input.paper_context.paper_id,
        build_system=repo.build_system,
        exploration={
            **exploration,
            "execution_surface": "unknown",
        },
        analyst=unified_input.analyst_output,
    )
    keep = rebuilt or [phase for phase in envelope.payload.phases if phase.phase_id == "setup"]
    payload = envelope.payload.model_copy(
        update={
            "phases": keep,
            "missing_context": missing_context,
            "engineer_notes": engineer_notes,
        }
    )
    return envelope.model_copy(
        update={"status": "partial", "unknowns": unknowns, "payload": payload}
    )


def _apply_plan_verification(
    envelope: AgentEnvelope[PlannerPayload],
    *,
    paper_id: str,
    repo_path: str | Path | None,
    repo_exploration: dict[str, object] | None,
    analyst_metrics: list[str] | None = None,
) -> AgentEnvelope[PlannerPayload]:
    """Deterministic verification: keep only grounded runnable matrix rows."""
    phases, missing, warnings, all_ok = verify_and_filter_phases(
        list(envelope.payload.phases),
        repo_path=repo_path,
        exploration=repo_exploration,
        paper_id=paper_id,
        analyst_metrics=analyst_metrics,
    )
    payload = envelope.payload.model_copy(
        update={
            "phases": phases,
            "missing_context": _unique_text(
                list(envelope.payload.missing_context) + missing
            ),
            "verification_checks": _unique_text(
                list(envelope.payload.verification_checks)
                + [
                    "entrypoint exists for each kept run_command",
                    "CLI flags documented in entrypoint argparse (when flags used)",
                    "manual OrderedDict / # set edits demoted to missing_context",
                ]
            ),
            "engineer_notes": _unique_text(
                list(envelope.payload.engineer_notes)
                + (
                    [
                        "Verification demoted unverified matrix rows; do not invent wrappers "
                        "or CLIs listed only under missing_context."
                    ]
                    if missing
                    else []
                )
            ),
        }
    )
    merged_warnings = _unique_text(list(envelope.warnings) + warnings)
    status = envelope.status
    if status == "ok" and (not all_ok or missing):
        status = "partial"
    return envelope.model_copy(
        update={"payload": payload, "warnings": merged_warnings, "status": status}
    )


def _resolve_debug_dir(context: PlannerInputContext | UnifiedPlannerInput) -> Path:
    if isinstance(context, PlannerInputContext) and context.paper_bundle_path:
        return Path(context.paper_bundle_path)
    paper = context.paper if isinstance(context, PlannerInputContext) else context.paper_context
    return Path("data") / "papers" / paper.paper_id


def _write_debug_trace(
    context: PlannerInputContext | UnifiedPlannerInput,
    trace: PlannerDebugTrace,
) -> None:
    output_dir = _resolve_debug_dir(context)
    json_path, md_path = write_planner_debug_files(trace, output_dir, saved_plan=None)
    logger.info("Planner debug written to %s (and %s)", md_path, json_path)


@dataclass
class PaperPlanner:
    model: str = PLANNER_MODEL
    max_parse_retries: int = PLANNER_MAX_RETRIES

    def _call_ollama_json(
        self,
        context: PlannerInputContext | UnifiedPlannerInput,
    ) -> AgentEnvelope[PlannerPayload]:
        unified_input = (
            context
            if isinstance(context, UnifiedPlannerInput)
            else build_unified_planner_input(context)
        )
        research_question = unified_input.analyst_output.research_question
        methodology = unified_input.analyst_output.methodology
        paper_id = unified_input.paper_context.paper_id
        results_contract = results_contract_for_prompt(paper_id)
        system_prompt = build_planner_system_prompt()
        repo_path: str | Path | None = None
        if isinstance(context, PlannerInputContext):
            repo_path = context.repo_context.get("repo_path")
            if not repo_path and context.paper_bundle_path:
                repo_path = str(Path(context.paper_bundle_path) / "code")
        payload = build_planner_prompt_context(unified_input, repo_path=repo_path)
        payload["results_contract"] = results_contract
        deterministic_phases = build_plan_phases(
            paper_id=paper_id,
            build_system=unified_input.repo_context.build_system,
            exploration=payload.get("repo_exploration")
            if isinstance(payload.get("repo_exploration"), dict)
            else None,
            analyst=unified_input.analyst_output,
        )
        payload["phase_scaffold"] = [phase.model_dump() for phase in deterministic_phases]
        repo_exploration = (
            payload.get("repo_exploration")
            if isinstance(payload.get("repo_exploration"), dict)
            else {}
        )
        enriched_commands = payload.get("repo_context", {}).get("example_commands")
        if isinstance(enriched_commands, list) and enriched_commands != list(
            unified_input.repo_context.example_commands
        ):
            unified_input = unified_input.model_copy(
                update={
                    "repo_context": unified_input.repo_context.model_copy(
                        update={
                            "example_commands": [str(item) for item in enriched_commands],
                            "entrypoint_hints": list(
                                payload.get("repo_context", {}).get("entrypoint_hints")
                                or unified_input.repo_context.entrypoint_hints
                            ),
                        }
                    )
                }
            )
        output_schema = PlannerEnvelope.model_json_schema()

        trace = PlannerDebugTrace(
            paper_id=paper_id,
            model=self.model,
            received_context=payload,
            system_prompt=system_prompt,
        )

        prompt = f"""
        Create an Engineer-ready phase-DAG plan from this unified Planner input.
        Use analyst_output as paper ground truth.
        Prefer phase_scaffold for axes/matrix/goals on experiment phases; enrich
        objective, plan_summary, setup/summarize notes, organization, execution,
        and repo_usage. Do not expand full cartesian products.
        Honor repo_context, flags, and the code-owned results_contract.

        Context JSON:
        {json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
        END_CONTEXT

        Do not summarize the context. Return the Planner envelope now.
        Required top-level skeleton:
        {OUTPUT_SKELETON}
        """.strip()

        last_error: Exception | None = None
        raw_response: str = ""
        parse_failures = 0
        aim_retry_used = False
        blocked_retry_used = False
        grounding_retry_used = False
        use_aim_reminder = False
        use_blocked_reminder = False
        use_method_aim_reminder = False
        use_grounding_reminder = False
        schema_retry_feedback = STRICT_RETRY_REMINDER
        grounding_retry_feedback = GROUNDING_RETRY_REMINDER

        try:
            while parse_failures <= self.max_parse_retries:
                reminder = "none"
                attempt_prompt = prompt
                if use_aim_reminder:
                    attempt_prompt = f"{prompt}\n\n{AIM_GROUNDING_RETRY_REMINDER}"
                    reminder = "aim"
                    use_aim_reminder = False
                elif use_method_aim_reminder:
                    attempt_prompt = f"{prompt}\n\n{METHODOLOGY_AIM_RETRY_REMINDER}"
                    reminder = "method_aim"
                    use_method_aim_reminder = False
                elif use_blocked_reminder:
                    attempt_prompt = f"{prompt}\n\n{BLOCKED_SOFTEN_RETRY_REMINDER}"
                    reminder = "blocked"
                    use_blocked_reminder = False
                elif use_grounding_reminder:
                    attempt_prompt = f"{prompt}\n\n{grounding_retry_feedback}"
                    reminder = "grounding"
                    use_grounding_reminder = False
                elif parse_failures > 0:
                    attempt_prompt = f"{prompt}\n\n{schema_retry_feedback}"
                    reminder = "strict"

                attempt = trace.add_attempt(
                    reminder=reminder,
                    user_prompt=attempt_prompt,
                    system_prompt=system_prompt,
                )

                req_payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": attempt_prompt},
                    ],
                    "stream": False,
                    "think": False,
                    "format": output_schema,
                    "options": {
                        "temperature": MODEL_TEMPERATURE,
                        "num_predict": PLANNER_NUM_PREDICT,
                    },
                }
                parsed_response: dict[str, object] | None = None
                try:
                    req = request.Request(
                        f"{OLLAMA_HOST}/api/chat",
                        method="POST",
                        data=json.dumps(req_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    with request.urlopen(req, timeout=180) as response:
                        body = json.loads(response.read().decode("utf-8"))
                    raw_response = str(body.get("message", {}).get("content", ""))
                    attempt.raw_response = raw_response

                    if not raw_response.strip():
                        raise json.JSONDecodeError("Empty model response", "", 0)
                    parsed = json.loads(_clean_json_response(raw_response))
                    if not isinstance(parsed, dict):
                        raise ValueError("Planner response must be a JSON object")
                    parsed = _normalize_envelope_dict(parsed)
                    parsed_response = parsed
                    attempt.parsed = parsed

                    envelope = PlannerEnvelope.model_validate(parsed)

                    if _should_soften_blocked(research_question, methodology, envelope):
                        if not blocked_retry_used:
                            attempt.outcome = "blocked_retry"
                            blocked_retry_used = True
                            use_blocked_reminder = True
                            continue
                        envelope = _soften_blocked_envelope(envelope)

                    if (
                        _plan_misses_aim(research_question, methodology, envelope)
                        and not aim_retry_used
                    ):
                        attempt.outcome = "aim_retry"
                        aim_retry_used = True
                        if _is_present_research_question(research_question):
                            use_aim_reminder = True
                        else:
                            use_method_aim_reminder = True
                        continue

                    # Apply deterministic scaffolds before grounding so thin LLM DAGs
                    # do not block or stick when exploration already has a surface.
                    envelope = _ensure_phases_from_builder(
                        envelope,
                        unified_input,
                        repo_exploration=repo_exploration
                        if isinstance(repo_exploration, dict)
                        else None,
                    )
                    grounding_issues = _collect_grounding_issues(
                        envelope,
                        unified_input,
                        repo_exploration=repo_exploration
                        if isinstance(repo_exploration, dict)
                        else None,
                    )
                    if grounding_issues and not grounding_retry_used:
                        attempt.outcome = "grounding_retry"
                        grounding_retry_used = True
                        grounding_retry_feedback = (
                            f"{GROUNDING_RETRY_REMINDER} {RUNNABLE_MATRIX_RETRY_REMINDER} "
                            f"Problems found: " + "; ".join(grounding_issues[:8])
                        )
                        use_grounding_reminder = True
                        continue

                    remaining_issues = _collect_grounding_issues(
                        envelope,
                        unified_input,
                        repo_exploration=repo_exploration
                        if isinstance(repo_exploration, dict)
                        else None,
                    )
                    if remaining_issues:
                        envelope = _mark_partial_with_grounding(envelope, remaining_issues)

                    envelope = _apply_runnable_repo_contract(
                        envelope,
                        unified_input,
                        repo_exploration=repo_exploration
                        if isinstance(repo_exploration, dict)
                        else None,
                    )
                    envelope = _apply_plan_verification(
                        envelope,
                        paper_id=paper_id,
                        repo_path=repo_path
                        or (
                            repo_exploration.get("repo_path")
                            if isinstance(repo_exploration, dict)
                            else None
                        ),
                        repo_exploration=repo_exploration
                        if isinstance(repo_exploration, dict)
                        else None,
                        analyst_metrics=list(
                            unified_input.analyst_output.evaluation_metrics
                        ),
                    )
                    envelope = _apply_results_contract(envelope, paper_id)
                    attempt.outcome = "accepted"
                    trace.final_output = envelope.model_dump()
                    return envelope
                except (
                    json.JSONDecodeError,
                    error.URLError,
                    error.HTTPError,
                    TimeoutError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    last_error = exc
                    attempt.error = str(exc)
                    attempt.outcome = "error"
                    parse_failures += 1
                    schema_retry_feedback = _schema_retry_reminder(exc, parsed_response)
                    time.sleep(1.5 * parse_failures)

            trace.final_error = (
                f"Failed to build planner JSON after retry. last_error={last_error}. "
                f"raw_response={raw_response!r}"
            )
            raise RuntimeError(trace.final_error)
        finally:
            _write_debug_trace(context, trace)

    def build_plan(
        self,
        context: PlannerInputContext | UnifiedPlannerInput,
    ) -> AgentEnvelope[PlannerPayload]:
        return self._call_ollama_json(context)
