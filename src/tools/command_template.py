"""Deterministic command templating and optional axes→matrix expansion for Planner phases.

By default Engineer runs concrete ``matrix`` rows only. Set
``ENGINEER_EXPAND_FULL_AXES=1`` to expand ``phase.axes`` into the full cartesian product.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

from src.config import ENGINEER_EXPAND_FULL_AXES
from src.state import PhaseRunSpec, PlanPhase
from src.tools.repo_exploration import (
    _PLACEHOLDER_ALGO,
    _PLACEHOLDER_CLS,
    _PLACEHOLDER_FUN,
    _PLACEHOLDER_LOG,
    _PLACEHOLDER_REG,
    _replace_placeholders,
)


def substitute_command_variables(command: str, variables: dict[str, object]) -> str:
    """Replace {name} placeholders; leave unknown braces untouched."""
    if not command or not variables:
        return command

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        return match.group(0)

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", replace, command)


def _parse_flag(command: str, flag: str, default: str) -> str:
    match = re.search(rf"{re.escape(flag)}\s+(\S+)", command)
    if not match:
        return default
    return match.group(1).strip()


def _infer_paper_id(phase: PlanPhase, paper_id: str = "") -> str:
    if paper_id.strip():
        return paper_id.strip()
    candidates = [phase.results_path, *[row.results_path for row in phase.matrix]]
    for path in candidates:
        parts = [part for part in path.replace("\\", "/").split("/") if part]
        if len(parts) >= 2 and parts[0] == "results":
            return parts[1]
    return "paper"


def _infer_reg_cls(phase: PlanPhase) -> tuple[str, str]:
    for row in phase.matrix:
        if row.run_command.strip():
            return (
                _parse_flag(row.run_command, "--reg-type", "gp"),
                _parse_flag(row.run_command, "--cls-type", "de"),
            )
    return "gp", "de"


def _template_for_phase(phase: PlanPhase) -> str:
    if phase.run_template.strip():
        return phase.run_template.strip()
    for row in phase.matrix:
        if row.run_command.strip():
            return row.run_command.strip()
    return ""


def _fill_run_command(
    template: str,
    *,
    benchmark: str,
    algorithm: str,
    reg_type: str,
    cls_type: str,
    seed: int,
    log_path: str,
) -> str:
    mapping = {
        **{key: benchmark for key in _PLACEHOLDER_FUN},
        **{key: algorithm for key in _PLACEHOLDER_ALGO},
        **{key: reg_type for key in _PLACEHOLDER_REG},
        **{key: cls_type for key in _PLACEHOLDER_CLS},
        **{key: log_path for key in _PLACEHOLDER_LOG},
        "LOG_DIR": str(log_path).rsplit("/", 1)[0] if "/" in log_path else log_path,
        "N_SEED": str(seed),
        "NUM_PROC": "1",
        "benchmark": benchmark,
        "algorithm": algorithm,
        "seed": str(seed),
    }
    command = _replace_placeholders(template, mapping)
    command = substitute_command_variables(
        command,
        {"benchmark": benchmark, "algorithm": algorithm, "seed": seed},
    )
    if re.search(r"run_exp\.py\b", command) and " --seed " not in command:
        command = f"{command} --seed {seed}"
    return command


def _axis_combinations(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = [key for key, values in axes.items() if values]
    if not keys:
        return []
    value_lists = [list(axes[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def expand_phase_axes(
    phase: PlanPhase,
    *,
    paper_id: str = "",
) -> list[PhaseRunSpec]:
    """Expand phase.axes into one PhaseRunSpec per cartesian cell."""
    if not phase.axes:
        return []
    template = _template_for_phase(phase)
    if not template:
        return []

    combos = _axis_combinations(phase.axes)
    if not combos:
        return []

    resolved_paper_id = _infer_paper_id(phase, paper_id)
    reg_type, cls_type = _infer_reg_cls(phase)
    code_refs = list(phase.matrix[0].code_refs) if phase.matrix else []
    metrics = list(phase.matrix[0].metrics) if phase.matrix else []
    source = phase.matrix[0].source if phase.matrix else "axes_expand"

    rows: list[PhaseRunSpec] = []
    for factors in combos:
        benchmark = str(factors.get("benchmark") or factors.get("fun") or factors.get("function") or "")
        algorithm = str(factors.get("algorithm") or factors.get("algo") or factors.get("method") or "")
        seed_raw = factors.get("seed", 0)
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            seed = 0

        # Non CLI-grid phases (script/config) still expand using {placeholders}.
        variables: dict[str, str | int | float | bool] = {}
        for key, value in factors.items():
            if isinstance(value, bool):
                variables[key] = value
            elif isinstance(value, (int, float)):
                variables[key] = value
            else:
                variables[key] = str(value)

        if benchmark and algorithm:
            log_path = (
                f"results/{resolved_paper_id}/{phase.phase_id}/{benchmark}/"
                f"{reg_type}_{cls_type}/{algorithm}/{seed}"
            )
            run_command = _fill_run_command(
                template,
                benchmark=benchmark,
                algorithm=algorithm,
                reg_type=str(variables.get("reg_type", reg_type)),
                cls_type=str(variables.get("cls_type", cls_type)),
                seed=seed,
                log_path=log_path,
            )
            name = f"{phase.phase_id}__{benchmark}__{algorithm}__seed{seed}"
        else:
            log_path = phase.results_path.strip() or f"results/{resolved_paper_id}/{phase.phase_id}"
            run_command = substitute_command_variables(template, dict(variables))
            name_bits = [phase.phase_id, *[f"{k}-{v}" for k, v in variables.items()]]
            name = "__".join(str(bit) for bit in name_bits)

        rows.append(
            PhaseRunSpec(
                name=name,
                variables=variables,
                run_command=run_command,
                code_refs=code_refs,
                verify=[f"exists:{log_path}"] if log_path else [],
                results_path=log_path,
                metrics=metrics,
                source=source,
            )
        )
    return rows


def phase_commands(
    phase: PlanPhase,
    *,
    paper_id: str = "",
    expand_axes: bool | None = None,
) -> list[tuple[str, PhaseRunSpec | None]]:
    """Return (command, matrix_row) pairs for a phase.

    When axes are present and expansion is enabled, expand the full cartesian
    product from axes + run_template (Planner example matrix is only a sample).
    """
    should_expand = ENGINEER_EXPAND_FULL_AXES if expand_axes is None else expand_axes
    if should_expand and phase.axes:
        expanded = expand_phase_axes(phase, paper_id=paper_id)
        if expanded:
            return [
                (substitute_command_variables(row.run_command, dict(row.variables)), row)
                for row in expanded
                if row.run_command.strip()
            ]

    rows: list[tuple[str, PhaseRunSpec | None]] = []
    for row in phase.matrix:
        command = substitute_command_variables(row.run_command, dict(row.variables))
        if command.strip():
            rows.append((command, row))
    if not rows and phase.run_template.strip():
        rows.append((phase.run_template.strip(), None))
    return rows
