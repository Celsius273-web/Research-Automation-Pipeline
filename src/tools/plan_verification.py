"""Deterministic Planner verification stage.

Filters phase matrices to runnable, grounded rows before Engineer handoff:
- entrypoint file exists
- CLI flags documented in the entrypoint (or known build tools)
- no unresolved `# set …` / comment-only manual edits
- library unit tests are not treated as paper reproduction
- native deps / generated inputs are called out explicitly
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from src.config import ROOT_DIR
from src.state import PhaseRunSpec, PlanPhase
from src.tools.plan_repair import repair_cleared_phases
from src.tools.repo_exploration import _extract_cli_flags

_MANUAL_EDIT_RE = re.compile(
    r"#\s*set\s+\S+|in script tunables|in code/config|OrderedDict|manual edit",
    re.IGNORECASE,
)
_PYTHON_LAUNCHERS = frozenset({"python", "python3"})
_BUILD_TOOLS = frozenset(
    {"cmake", "ctest", "make", "ninja", "docker", "pip", "pip3", "bash", "sh"}
)
_FLAG_IN_COMMAND_RE = re.compile(r"--([A-Za-z0-9-]+)(?:\s|=|$)")
_TEST_MODULE_RE = re.compile(r"(?:^|/)(?:test_[^/]+|[^/]+_test)\.py$", re.IGNORECASE)


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _strip_comment(command: str) -> str:
    text = (command or "").strip()
    if not text:
        return ""
    if text.startswith("#"):
        return ""
    return text.split("  #", 1)[0].strip()


def _tokenize(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _entrypoint_from_command(command: str) -> str | None:
    tokens = _tokenize(command)
    if not tokens:
        return None
    if tokens[0] in _PYTHON_LAUNCHERS and len(tokens) >= 2:
        candidate = tokens[1]
        if candidate.endswith(".py") or "/" in candidate:
            return candidate[2:] if candidate.startswith("./") else candidate
        return None
    for token in tokens:
        if token.endswith(".py"):
            return token[2:] if token.startswith("./") else token
    first = tokens[0]
    if first.startswith("./"):
        first = first[2:]
    if first.endswith((".sh", ".py", ".exe")) or "/" in first:
        return first
    return None


def _flags_in_command(command: str) -> list[str]:
    return _unique(_FLAG_IN_COMMAND_RE.findall(command))


def _normalize_flag(flag: str) -> str:
    return flag.strip().lstrip("-").replace("-", "_").lower()


def _resolve_repo_file(repo_path: Path | None, relative: str) -> Path | None:
    if not relative:
        return None

    candidates: list[Path] = []
    if repo_path is not None:
        # Repo-relative and sibling planner_stubs via ../planner_stubs/...
        candidates.append((repo_path / relative).resolve())
    # Workspace-root paths: data/papers/<id>/planner_stubs/...
    candidates.append((ROOT_DIR / relative).resolve())

    for candidate in candidates:
        if not candidate.is_file():
            continue
        if repo_path is None:
            try:
                candidate.relative_to(ROOT_DIR / "data" / "papers")
                return candidate
            except ValueError:
                continue
        bundle_root = repo_path.resolve().parent
        try:
            candidate.relative_to(bundle_root)
            return candidate
        except ValueError:
            pass
        try:
            candidate.relative_to(repo_path.resolve())
            return candidate
        except ValueError:
            pass
        try:
            candidate.relative_to(ROOT_DIR / "data" / "papers")
            return candidate
        except ValueError:
            continue
    return None


def _documented_flags(repo_path: Path | None, entrypoint: str | None) -> set[str]:
    if not repo_path or not entrypoint:
        return set()
    path = _resolve_repo_file(repo_path, entrypoint)
    if path is None:
        return set()
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    return {_normalize_flag(flag) for flag in _extract_cli_flags(content)}


def _looks_like_unit_test(command: str, entrypoint: str | None) -> bool:
    target = entrypoint or command
    return bool(_TEST_MODULE_RE.search(target.replace("\\", "/")))


def _native_dep_check_commands() -> list[str]:
    return [
        (
            "bash -lc 'test -f /usr/include/eigen3/Eigen/Dense "
            "|| test -f /usr/local/include/eigen3/Eigen/Dense "
            "|| test -n \"${EIGEN3_INCLUDE_DIR:-}\" "
            "|| (echo \"Eigen headers not found; install Eigen/Spectra before cmake\" >&2 && exit 1)'"
        ),
        (
            "bash -lc 'test -d /usr/include/spectra "
            "|| test -d /usr/local/include/spectra "
            "|| test -n \"${SPECTRA_INCLUDE_DIR:-}\" "
            "|| (echo \"Spectra headers not found; install Spectra before cmake\" >&2 && exit 1)'"
        ),
    ]


def verify_run_command(
    command: str,
    *,
    repo_path: Path | None,
    phase_id: str,
    exploration: dict[str, object] | None = None,
) -> tuple[bool, list[str]]:
    """Return (is_verified, reasons). reasons empty when verified."""
    exploration = exploration if isinstance(exploration, dict) else {}
    raw = (command or "").strip()
    reasons: list[str] = []
    if not raw:
        return False, ["empty run_command"]
    if raw.startswith("#"):
        return False, ["comment-only command; human wrapper or README sample required"]
    if _MANUAL_EDIT_RE.search(raw):
        reasons.append(
            "requires in-file / OrderedDict edit (no documented CLI for varied factors)"
        )

    executable = _strip_comment(raw)
    if not executable:
        return False, reasons or ["no executable portion before comment"]

    tokens = _tokenize(executable)
    if not tokens:
        return False, ["unparseable run_command"]

    launcher = tokens[0]
    entrypoint = _entrypoint_from_command(executable)

    if launcher in _PYTHON_LAUNCHERS or (entrypoint and entrypoint.endswith(".py")):
        if not entrypoint:
            reasons.append("python command missing script path")
        elif repo_path is not None and _resolve_repo_file(repo_path, entrypoint) is None:
            reasons.append(f"entrypoint does not exist: {entrypoint}")
        else:
            documented = _documented_flags(repo_path, entrypoint)
            used = [_normalize_flag(flag) for flag in _flags_in_command(executable)]
            if used and documented:
                unknown = [flag for flag in used if flag not in documented]
                if unknown:
                    reasons.append(
                        f"undocumented CLI flags for {entrypoint}: {', '.join(unknown)}"
                    )
            elif used and not documented and entrypoint:
                reasons.append(
                    f"entrypoint {entrypoint} has no documented argparse flags for "
                    f"{', '.join(used)}"
                )
            if phase_id == "reproduce_similar" and _looks_like_unit_test(
                executable, entrypoint
            ):
                if entrypoint and "planner_stubs" not in entrypoint.replace("\\", "/"):
                    reasons.append(
                        "unit test is not paper reproduction; need demo/notebook port + data"
                    )
    elif launcher in {"cmake", "ctest", "make", "ninja"}:
        native = exploration.get("native_build")
        if not (isinstance(native, dict) and native.get("available")) and not exploration.get(
            "native_tests"
        ):
            reasons.append("native build evidence missing for cmake/ctest/make command")
    elif launcher == "docker":
        if not exploration.get("container_files"):
            reasons.append("no container files discovered for docker command")
    elif launcher in {"pip", "pip3", "bash", "sh"}:
        pass
    elif entrypoint and repo_path is not None and _resolve_repo_file(repo_path, entrypoint) is None:
        if not entrypoint.startswith("build/") and not entrypoint.startswith("./build/"):
            reasons.append(f"entrypoint does not exist: {entrypoint}")

    return (len(reasons) == 0), reasons


def _axes_without_unverified_factors(
    axes: dict[str, list[str | int | float | bool]],
    kept_rows: list[PhaseRunSpec],
) -> dict[str, list[str | int | float | bool]]:
    if not kept_rows:
        return {}
    present_keys: set[str] = set()
    for row in kept_rows:
        present_keys.update(str(key) for key in row.variables)
    if not present_keys:
        return {}
    return {
        key: values
        for key, values in axes.items()
        if key in present_keys
    }


def ensure_native_dependency_checks(
    phases: list[PlanPhase],
    *,
    paper_id: str,
    exploration: dict[str, object] | None,
) -> list[PlanPhase]:
    """Insert an explicit Eigen/Spectra check phase before native_smoke when missing."""
    exploration = exploration if isinstance(exploration, dict) else {}
    surface = str(exploration.get("execution_surface") or "")
    if surface != "native":
        return phases
    if any(phase.phase_id == "deps_check" for phase in phases):
        return phases
    if not any(phase.phase_id == "native_smoke" for phase in phases):
        return phases

    results_root = f"results/{paper_id}/deps_check"
    code_refs = _unique(
        [
            str(exploration.get("readme_file") or "README.md"),
            "INSTALL",
            *[
                str(item)
                for item in (
                    (exploration.get("native_build") or {}).get("files")
                    if isinstance(exploration.get("native_build"), dict)
                    else []
                )
            ],
        ]
    )
    matrix = [
        PhaseRunSpec(
            name="deps_check__eigen",
            variables={"dependency": "eigen"},
            run_command=_native_dep_check_commands()[0],
            code_refs=code_refs,
            verify=["exit_code:0"],
            results_path=results_root,
            metrics=[],
            source="repo",
        ),
        PhaseRunSpec(
            name="deps_check__spectra",
            variables={"dependency": "spectra"},
            run_command=_native_dep_check_commands()[1],
            code_refs=code_refs,
            verify=["exit_code:0"],
            results_path=results_root,
            metrics=[],
            source="repo",
        ),
    ]
    deps_phase = PlanPhase(
        phase_id="deps_check",
        title="Native dependency checks",
        goal="Fail fast if Eigen/Spectra headers are missing before cmake.",
        depends_on=["setup"],
        variables=["dependency"],
        axes={"dependency": ["eigen", "spectra"]},
        run_template=matrix[0].run_command,
        matrix=matrix,
        planned_actions=(
            "Verify Eigen and Spectra include paths (or EIGEN3_INCLUDE_DIR / "
            "SPECTRA_INCLUDE_DIR). Do not run cmake until both checks pass."
        ),
        results_path=results_root,
    )

    revised: list[PlanPhase] = []
    for phase in phases:
        if phase.phase_id == "native_smoke":
            revised.append(deps_phase)
            revised.append(
                phase.model_copy(
                    update={
                        "depends_on": ["deps_check"]
                        if phase.depends_on == ["setup"]
                        else _unique([*phase.depends_on, "deps_check"])
                    }
                )
            )
        elif phase.phase_id == "setup":
            revised.append(
                phase.model_copy(
                    update={
                        "planned_actions": (
                            (phase.planned_actions + " ").strip()
                            + " Then run deps_check for Eigen/Spectra before cmake."
                        ).strip()
                    }
                )
            )
        else:
            revised.append(phase)
    if not any(phase.phase_id == "deps_check" for phase in revised):
        revised.insert(1, deps_phase)
    return revised


def ensure_input_generation_steps(
    phases: list[PlanPhase],
    *,
    paper_id: str,
    exploration: dict[str, object] | None,
) -> list[PlanPhase]:
    """For STAG SBM: make edgelist generation an explicit precede step."""
    exploration = exploration if isinstance(exploration, dict) else {}
    if str(exploration.get("execution_surface") or "") != "native":
        return phases
    reproduce = next((phase for phase in phases if phase.phase_id == "reproduce_similar"), None)
    if reproduce is None or any(phase.phase_id == "generate_inputs" for phase in phases):
        return phases

    sbm_rows = [
        row
        for row in reproduce.matrix
        if "stag_sbm" in (row.run_command or "") and row.variables.get("benchmark") == "sbm"
    ]
    if not sbm_rows:
        return phases

    edgelist = f"results/{paper_id}/generate_inputs/sbm.edgelist"
    generate_cmd = (
        "cmake --build build --target stag_sbm 2>/dev/null; "
        f"./build/stagtools/stag_sbm {edgelist} 200 2 0.6 0.1"
    )
    generate = PlanPhase(
        phase_id="generate_inputs",
        title="Generate native input graphs",
        goal="Create SBM edgelist input via stagtools before reproduce_similar.",
        depends_on=["native_smoke"],
        variables=["benchmark"],
        axes={"benchmark": ["sbm"]},
        run_template=generate_cmd,
        matrix=[
            PhaseRunSpec(
                name="generate__sbm_edgelist",
                variables={"benchmark": "sbm", "output": edgelist},
                run_command=generate_cmd,
                code_refs=_unique([*sbm_rows[0].code_refs, "stagtools/sbm.cpp"]),
                verify=["exit_code:0", f"exists:{edgelist}"],
                results_path=f"results/{paper_id}/generate_inputs",
                metrics=[],
                source="repo",
            )
        ],
        planned_actions=(
            f"Build stag_sbm and write {edgelist}. Fail if the edgelist file is missing."
        ),
        results_path=f"results/{paper_id}/generate_inputs",
    )

    # reproduce_similar keeps a verification-oriented follow-up that consumes the file.
    follow_cmd = f"bash -lc 'test -s {edgelist} && ls -l {edgelist}'"
    updated_reproduce = reproduce.model_copy(
        update={
            "depends_on": ["generate_inputs"],
            "variables": ["benchmark"],
            "axes": {"benchmark": ["sbm"]},
            "run_template": follow_cmd,
            "matrix": [
                PhaseRunSpec(
                    name="reproduce__sbm_input_ready",
                    variables={"benchmark": "sbm", "input": edgelist},
                    run_command=follow_cmd,
                    code_refs=sbm_rows[0].code_refs,
                    verify=["exit_code:0", f"exists:{edgelist}"],
                    results_path=f"results/{paper_id}/reproduce_similar/sbm",
                    metrics=list(sbm_rows[0].metrics),
                    source="repo",
                )
            ],
            "planned_actions": (
                "Confirm generated SBM edgelist exists. Clustering against paper metrics "
                "still needs a handwritten driver (no grounded cluster CLI) — see missing_context."
            ),
        }
    )

    revised: list[PlanPhase] = []
    for phase in phases:
        if phase.phase_id == "reproduce_similar":
            revised.append(generate)
            revised.append(updated_reproduce)
        else:
            revised.append(phase)
    return revised


def verify_and_filter_phases(
    phases: list[PlanPhase],
    *,
    repo_path: Path | str | None,
    exploration: dict[str, object] | None,
    paper_id: str,
    analyst_metrics: list[str] | None = None,
) -> tuple[list[PlanPhase], list[str], list[str], bool]:
    """Filter matrices to verified rows; return phases, missing, warnings, all_ok."""
    root = Path(repo_path) if repo_path else None
    if root is not None and not root.is_dir():
        root = None
    exploration = exploration if isinstance(exploration, dict) else {}

    working = ensure_native_dependency_checks(
        list(phases), paper_id=paper_id, exploration=exploration
    )
    working = ensure_input_generation_steps(
        working, paper_id=paper_id, exploration=exploration
    )

    missing: list[str] = []
    warnings: list[str] = []
    revised: list[PlanPhase] = []
    demoted = 0
    kept = 0
    # Preserve Analyst metrics from demoted reproduce rows for stub repair.
    preserved_metrics: list[str] = [
        str(item).strip() for item in (analyst_metrics or []) if str(item).strip()
    ]

    for phase in working:
        if not phase.matrix:
            revised.append(phase)
            continue
        kept_rows: list[PhaseRunSpec] = []
        demote_reasons: list[str] = []
        for row in phase.matrix:
            ok, reasons = verify_run_command(
                row.run_command,
                repo_path=root,
                phase_id=phase.phase_id,
                exploration=exploration,
            )
            if ok:
                kept_rows.append(row)
                kept += 1
                continue
            demoted += 1
            if not preserved_metrics and row.metrics:
                preserved_metrics = [str(item).strip() for item in row.metrics if str(item).strip()][:5]
            detail = "; ".join(reasons) if reasons else "unverified"
            demote_reasons.append(f"{row.name} ({detail})")

        if demote_reasons:
            preview = "; ".join(demote_reasons[:3])
            extra = (
                f" (+{len(demote_reasons) - 3} more)"
                if len(demote_reasons) > 3
                else ""
            )
            missing.append(
                f"phase {phase.phase_id}: demoted {len(demote_reasons)} matrix row(s) "
                f"from verification: {preview}{extra}"
            )

        if kept_rows:
            revised.append(
                phase.model_copy(
                    update={
                        "matrix": kept_rows,
                        "axes": _axes_without_unverified_factors(phase.axes, kept_rows),
                        "variables": [
                            key
                            for key in phase.variables
                            if key in {k for r in kept_rows for k in r.variables}
                        ]
                        or list(
                            dict.fromkeys(key for row in kept_rows for key in row.variables)
                        ),
                    }
                )
            )
            continue

        # Leave an empty matrix shell for repair_cleared_phases to refill or collapse.
        missing.append(
            f"phase {phase.phase_id}: no verified runnable matrix rows after first pass."
        )
        revised.append(
            phase.model_copy(
                update={
                    "matrix": [],
                }
            )
        )

    if demoted:
        warnings.append(
            f"plan verification demoted {demoted} unverified matrix row(s); kept {kept}."
        )
    else:
        warnings.append(f"plan verification kept {kept} matrix row(s); none demoted.")

    repaired, repair_missing, repair_warnings, stub_paths = repair_cleared_phases(
        revised,
        repo_path=root,
        exploration=exploration,
        paper_id=paper_id,
        analyst_metrics=preserved_metrics,
    )
    missing.extend(repair_missing)
    warnings.extend(repair_warnings)
    if stub_paths:
        warnings.append(
            "planner stubs written: " + ", ".join(Path(path).name for path in stub_paths)
        )

    # Re-verify repaired matrices; collapse anything still empty/unverified.
    from src.tools.plan_repair import _collapse_phases

    verified_by_id: dict[str, PlanPhase] = {}
    drop_ids: set[str] = set()
    for phase in repaired:
        if phase.phase_id in {"setup", "summarize"}:
            verified_by_id[phase.phase_id] = phase
            continue
        if not phase.matrix:
            drop_ids.add(phase.phase_id)
            missing.append(
                f"phase {phase.phase_id}: collapsed after repair — matrix still empty."
            )
            continue
        kept_rows: list[PhaseRunSpec] = []
        for row in phase.matrix:
            ok, reasons = verify_run_command(
                row.run_command,
                repo_path=root,
                phase_id=phase.phase_id,
                exploration=exploration,
            )
            if ok:
                kept_rows.append(row)
            else:
                missing.append(
                    f"phase {phase.phase_id} row {row.name}: stub/fallback still unverified "
                    f"({'; '.join(reasons)})"
                )
        if kept_rows:
            verified_by_id[phase.phase_id] = phase.model_copy(update={"matrix": kept_rows})
        else:
            drop_ids.add(phase.phase_id)
            missing.append(
                f"phase {phase.phase_id}: collapsed — repaired rows failed verification."
            )

    working = [verified_by_id.get(phase.phase_id, phase) for phase in repaired]
    if drop_ids:
        working = _collapse_phases(working, drop_ids)

    empty_experiment = [
        phase.phase_id
        for phase in working
        if phase.phase_id not in {"setup", "summarize"} and not phase.matrix
    ]
    all_ok = demoted == 0 and not empty_experiment and not drop_ids
    return working, _unique(missing), _unique(warnings), all_ok


# Re-export for tests.
__all__ = [
    "ensure_input_generation_steps",
    "ensure_native_dependency_checks",
    "verify_and_filter_phases",
    "verify_run_command",
]
