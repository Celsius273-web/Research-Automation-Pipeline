"""Deterministic phase-DAG builder for Planner plans.

Builds compact axes + a few example matrix rows from repo exploration and
Analyst hyperparameter lists. Full cartesian expansion is left to Executor.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from src.config import (
    PLANNER_LIBRARY_SMOKE_TESTS,
    PLANNER_PHASE_ABLATION_VALUES_MAX,
    PLANNER_PHASE_ALGO_MAX,
    PLANNER_PHASE_EXAMPLE_ROWS,
    PLANNER_PHASE_REALWORLD_MAX,
    PLANNER_PHASE_SEED_COUNT,
    PLANNER_PHASE_SYNTHETIC_MAX,
)
from src.state import PhaseRunSpec, PlanPhase, SectionExtraction
from src.tools.repo_exploration import (
    _PLACEHOLDER_ALGO,
    _PLACEHOLDER_CLS,
    _PLACEHOLDER_FUN,
    _PLACEHOLDER_LOG,
    _PLACEHOLDER_REG,
    _infer_model_type_options,
    _pick_run_command_template,
    _replace_placeholders,
    infer_execution_surface,
)

_SYNTHETIC_PREFIXES = ("lsq", "sim", "tow")
_ABLATION_KEYS = (
    "hidden_layers",
    "neurons_factor_c",
    "neuron_scale_c",
    "learning_rates",
    "learning_rate",
    "mlp_count",
)
_LIST_VALUE_RE = re.compile(r"\[([^\]]+)\]")


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


def split_benchmarks(functions: list[str]) -> tuple[list[str], list[str]]:
    """Split registry functions into synthetic vs real-world groups."""
    synthetic: list[str] = []
    real: list[str] = []
    for name in _unique(functions):
        lowered = name.lower()
        if any(
            lowered == prefix or lowered.startswith(f"{prefix}-")
            for prefix in _SYNTHETIC_PREFIXES
        ):
            synthetic.append(name)
        else:
            real.append(name)
    return synthetic, real


def rank_algorithms(algorithms: list[str]) -> list[str]:
    return sorted(
        _unique(algorithms),
        key=lambda name: (
            0 if any(token in name.lower() for token in ("be-cbo", "becbo", "proposed")) else 1,
            0 if not name.lower().startswith("random") else 2,
            name,
        ),
    )


def parse_analyst_list_values(raw: object) -> list[str | int | float]:
    """Parse Analyst hyperparameter values like '[1, 2, 3]' into a short list."""
    if isinstance(raw, list):
        values = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return []
        match = _LIST_VALUE_RE.search(text)
        candidate = match.group(0) if match else text
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            parsed = [
                part.strip().strip("'\"")
                for part in candidate.strip("[]").split(",")
                if part.strip()
            ]
        values = parsed if isinstance(parsed, list) else [parsed]

    out: list[str | int | float] = []
    for item in values:
        if isinstance(item, bool):
            out.append(item)
        elif isinstance(item, (int, float)):
            out.append(item)
        else:
            text = str(item).strip()
            if not text:
                continue
            try:
                if "." in text:
                    out.append(float(text))
                else:
                    out.append(int(text))
            except ValueError:
                out.append(text)
        if len(out) >= PLANNER_PHASE_ABLATION_VALUES_MAX:
            break
    return out


def extract_ablation_axes(
    hyperparameters: dict[str, Any],
) -> list[tuple[str, list[str | int | float]]]:
    """Pull ablation-style lists from Analyst hyperparameters only."""
    found: list[tuple[str, list[str | int | float]]] = []
    lowered_map = {
        str(key).strip().lower(): (str(key).strip(), value)
        for key, value in hyperparameters.items()
    }
    for key in _ABLATION_KEYS:
        if key not in lowered_map:
            continue
        original_key, raw = lowered_map[key]
        values = parse_analyst_list_values(raw)
        if len(values) < 2:
            continue
        axis_name = "neuron_scale_C" if "neuron" in original_key.lower() else original_key
        found.append((axis_name, values))
    return found


def _fill_run_command(
    template: str,
    *,
    benchmark: str,
    algorithm: str,
    reg_type: str,
    cls_type: str,
    seed: int,
    log_path: str,
    extra: dict[str, str] | None = None,
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
    }
    if extra:
        mapping.update(extra)
    command = _replace_placeholders(template, mapping)
    if re.search(r"run_exp\.py\b", command) and " --seed " not in command:
        command = f"{command} --seed {seed}"
    return command


def _example_rows(
    *,
    phase_id: str,
    axes: dict[str, list[str | int | float | bool]],
    variables: list[str],
    template: str,
    paper_id: str,
    code_refs: list[str],
    metrics: list[str],
    reg_type: str,
    cls_type: str,
    source: str,
    max_rows: int = PLANNER_PHASE_EXAMPLE_ROWS,
) -> list[PhaseRunSpec]:
    benchmarks = [str(item) for item in axes.get("benchmark", [])]
    algorithms = [str(item) for item in axes.get("algorithm", [])]
    seeds = [int(item) for item in axes.get("seed", [0]) if str(item).strip() != ""]
    if not seeds:
        seeds = [0]

    rows: list[PhaseRunSpec] = []
    if not benchmarks or not algorithms or not template:
        return rows

    for benchmark in benchmarks:
        for algorithm in algorithms:
            seed = seeds[0]
            factors: dict[str, str | int | float | bool] = {
                "benchmark": benchmark,
                "algorithm": algorithm,
                "seed": seed,
            }
            for key in variables:
                if key in factors:
                    continue
                values = axes.get(key) or []
                if values:
                    factors[key] = values[0]
            log_path = (
                f"results/{paper_id}/{phase_id}/{benchmark}/"
                f"{reg_type}_{cls_type}/{algorithm}/{seed}"
            )
            run_command = _fill_run_command(
                template,
                benchmark=benchmark,
                algorithm=algorithm,
                reg_type=str(factors.get("reg_type", reg_type)),
                cls_type=str(factors.get("cls_type", cls_type)),
                seed=seed,
                log_path=log_path,
            )
            rows.append(
                PhaseRunSpec(
                    name=f"{phase_id}__{benchmark}__{algorithm}__seed{seed}",
                    variables={key: factors[key] for key in variables if key in factors},
                    run_command=run_command,
                    code_refs=list(code_refs),
                    verify=[
                        f"exists:{log_path}",
                        f"exists:results/{paper_id}/summary.json",
                    ],
                    results_path=log_path,
                    metrics=list(metrics[:3]),
                    source=source,
                )
            )
            if len(rows) >= max_rows:
                return rows
    return rows


def _build_library_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object],
    analyst: SectionExtraction,
) -> list[PlanPhase]:
    """Setup + repo tests that prove the library, then a results-oriented reproduction phase."""
    results_root = f"results/{paper_id}"
    test_files = [str(item) for item in (exploration.get("test_files") or [])]
    notebooks = [str(item) for item in (exploration.get("notebooks") or [])]
    code_refs = _unique(
        [
            str(exploration.get("readme_file") or "README.md"),
            *notebooks[:2],
            *test_files[:4],
            *[str(item) for item in (exploration.get("entrypoint_hints") or [])[:4]],
        ]
    )
    metrics = [item.strip() for item in analyst.evaluation_metrics if item.strip()][:5]
    benchmarks = selectable_benchmark_labels(list(analyst.datasets_or_benchmarks))
    reported = [
        f"{item.benchmark}:{item.metric_name}"
        for item in analyst.reported_results[:6]
        if item.benchmark.strip() and item.metric_name.strip()
    ]

    phases: list[PlanPhase] = [
        PlanPhase(
            phase_id="setup",
            title="Environment setup",
            goal="Install the library package and confirm imports from README/setup.",
            depends_on=[],
            variables=[],
            axes={},
            run_template=build_system.strip(),
            matrix=[],
            planned_actions=(
                "Create venv if needed, install with build_system / setup.py, import the package, "
                "open README + demo notebook for the intended API."
            ),
            results_path=results_root,
        )
    ]

    smoke_pool = [
        path
        for path in test_files
        if not any(token in path.lower() for token in ("bayesopt", "objectives"))
    ] or list(test_files)
    smoke_tests = smoke_pool[:PLANNER_LIBRARY_SMOKE_TESTS]
    if smoke_tests:
        matrix = [
            PhaseRunSpec(
                name=f"library_smoke__{Path(path).stem}",
                variables={"test_module": path},
                run_command=f"python {path}",
                code_refs=_unique([path, *code_refs[:3]]),
                verify=["exit_code:0"],
                results_path=f"{results_root}/library_smoke",
                # Paper targets for later reproduce — smoke itself only checks exit_code.
                metrics=list(metrics),
                source="repo",
            )
            for path in smoke_tests
        ]
        phases.append(
            PlanPhase(
                phase_id="library_smoke",
                title="Library correctness smoke tests",
                goal="Run repository unit tests to prove core GP/BO library behavior.",
                depends_on=["setup"],
                variables=["test_module"],
                axes={"test_module": smoke_tests},
                run_template="python {test_module}",
                matrix=matrix,
                planned_actions=(
                    "Execute listed test modules for correctness (exit 0). Smoke does NOT produce "
                    "regret/evaluation counts—those come from the reproduce_similar phase. "
                    "If a test outputs numeric values (e.g., kernel eigenvalues, GP likelihood), "
                    "capture them under results/; otherwise smoke = API validation only. "
                    f"Matrix metrics list paper targets for later "
                    f"({', '.join(metrics[:3]) or 'Analyst metrics'}), not smoke outputs."
                ),
                results_path=f"{results_root}/library_smoke",
            )
        )
        previous = "library_smoke"
    else:
        previous = "setup"

    # Prefer BO-oriented tests for "similar results"; fall back to remaining tests / notebook.
    reproduce_tests = [
        path
        for path in test_files
        if any(token in path.lower() for token in ("bayesopt", "objectives"))
    ]
    if not reproduce_tests:
        reproduce_tests = [path for path in test_files if path not in smoke_tests][:2]
    if not reproduce_tests and notebooks:
        reproduce_tests = []

    # Only vary what the command actually changes (test_module / notebook).
    # Analyst benchmarks are guidance in planned_actions + missing_context, not fake axes.
    reproduce_matrix: list[PhaseRunSpec] = []
    target_note = ", ".join(benchmarks[:4]) if benchmarks else "Analyst datasets_or_benchmarks"
    for path in reproduce_tests[:3]:
        log_path = f"{results_root}/reproduce_similar/{Path(path).stem}"
        reproduce_matrix.append(
            PhaseRunSpec(
                name=f"reproduce__{Path(path).stem}",
                variables={"test_module": path},
                run_command=f"python {path}",
                code_refs=_unique([path, *notebooks[:1], *code_refs[:3]]),
                verify=["exit_code:0"],
                results_path=log_path,
                metrics=metrics,
                source="repo",
            )
        )
    if notebooks and not reproduce_matrix:
        notebook = notebooks[0]
        log_path = f"{results_root}/reproduce_similar/demo"
        reproduce_matrix.append(
            PhaseRunSpec(
                name="reproduce__demo_notebook",
                variables={"notebook": notebook},
                run_command=(
                    f"# port key cells from {notebook} into a script; "
                    f"write metrics under {log_path}/"
                ),
                code_refs=_unique([notebook, *code_refs[:4]]),
                verify=[f"exists:{notebook}"],
                results_path=log_path,
                metrics=metrics,
                source="repo",
            )
        )

    if reproduce_matrix:
        claimed = ", ".join(reported[:4]) if reported else "Analyst evaluation_metrics"
        variables = ["test_module"] if reproduce_tests else ["notebook"]
        axes: dict[str, list[str | int | float | bool]] = (
            {"test_module": reproduce_tests[:3]}
            if reproduce_tests
            else {"notebook": notebooks[:1]}
        )
        phases.append(
            PlanPhase(
                phase_id="reproduce_similar",
                title="Reproduce paper-similar library results",
                goal=(
                    "Use library tests/demo API to produce metrics comparable to Analyst "
                    "reported_results (direction/scale), writing aggregates under results/."
                ),
                depends_on=[previous],
                variables=variables,
                axes=axes,
                run_template="python {test_module}" if reproduce_tests else "",
                matrix=reproduce_matrix,
                planned_actions=(
                    "1) Run listed BO/GP tests or port demo-notebook cells into a small script. "
                    "2) Capture objective/regret-style values under results/ when the API allows. "
                    f"3) Compare directionally to Analyst claims ({claimed}); do not invent numbers "
                    f"not produced by the run. 4) Paper targets ({target_note}) need external data — "
                    "record gaps in missing_context rather than inventing CLI flags."
                ),
                results_path=f"{results_root}/reproduce_similar",
            )
        )
        previous = "reproduce_similar"

    phases.append(
        PlanPhase(
            phase_id="summarize",
            title="Aggregate results",
            goal="Write/update summary.json from library smoke + reproduction metrics.",
            depends_on=[previous],
            variables=[],
            axes={},
            run_template="",
            matrix=[],
            planned_actions=(
                f"Merge metrics under {results_root} into {results_root}/summary.json; "
                "include which Analyst benchmarks were approximated vs blocked by missing data."
            ),
            results_path=f"{results_root}/summary.json",
        )
    )
    return phases


def collect_surface_context_notes(
    *,
    exploration: dict[str, object],
    analyst: SectionExtraction,
    phases: list[PlanPhase],
) -> dict[str, list[str]]:
    """Derive missing_context / risks / verification_checks from surface evidence."""
    surface = str(exploration.get("execution_surface") or "unknown")
    missing: list[str] = []
    risks: list[str] = []
    checks: list[str] = []

    for phase in phases:
        for row in phase.matrix:
            command = (row.run_command or "").strip()
            if command.startswith("#"):
                missing.append(
                    f"phase {phase.phase_id}: no grounded runnable command; "
                    "README/API sample must be supplied before Engineer execution."
                )
            for item in row.verify:
                if item and item not in checks:
                    checks.append(item)

    if surface == "script":
        tunables = exploration.get("script_tunables") if isinstance(
            exploration.get("script_tunables"), dict
        ) else {}
        if not tunables:
            missing.append(
                "Script tunables (dataset/method) were not scraped; confirm in-file defaults."
            )
        datasets = [str(item) for item in (tunables.get("dataset") or tunables.get("benchmark") or [])]
        gated = [name for name in datasets if name.lower() not in {"synth", "synthetic"}]
        if gated:
            missing.append(
                "External citation/graph datasets may be required "
                f"({', '.join(gated[:5])}); confirm download/cache paths before full matrix."
            )
        risks.append("In-file OrderedDict edits are required when scripts expose no CLI flags.")
    elif surface == "native":
        missing.append(
            "Native README samples are often C++ API snippets (not shell CLIs); "
            "use built test binaries / stagtools after cmake."
        )
        if not exploration.get("example_commands"):
            missing.append(
                "No shell experiment commands in README; reproduce_similar depends on "
                "compiled tools (e.g. stagtools/sbm) or a handwritten driver."
            )
        risks.append("Eigen/Spectra (or other native deps) must be installed before cmake.")
        grounded: set[str] = set()
        for phase in phases:
            if phase.phase_id != "reproduce_similar":
                continue
            grounded = {
                str(row.variables.get("benchmark", "")).strip()
                for row in phase.matrix
                if (row.run_command or "").strip() and not (row.run_command or "").startswith("#")
            }
        for name in selectable_benchmark_labels(list(analyst.datasets_or_benchmarks)):
            if name not in grounded:
                missing.append(
                    f"No grounded shell command for Analyst/native benchmark '{name}'; "
                    "need stagtools driver or README sample before Engineer execution."
                )
    elif surface == "library":
        benchmarks = selectable_benchmark_labels(list(analyst.datasets_or_benchmarks))
        external = [
            name
            for name in benchmarks
            if name.lower() in {"hpo_b", "pd1", "imagenet", "cifar10", "cifar100"}
        ]
        if external:
            missing.append(
                "External benchmark data may be unavailable locally "
                f"({', '.join(external[:4])}); unit tests still validate the library API — "
                "do not invent CLI flags for paper targets on test modules."
            )
        risks.append(
            "Library smoke proves API correctness; paper-similar metrics need data + demo port."
        )
    elif surface == "unknown":
        missing.append("Insufficient grounded execution evidence for an experiment matrix.")

    return {
        "missing_context": _unique(missing),
        "risks": _unique(risks),
        "verification_checks": _unique(checks)[:12],
    }


_KNOWN_BENCHMARK_HINTS: tuple[tuple[str, str], ...] = (
    ("stochastic block", "sbm"),
    ("sbm", "sbm"),
    ("edgelist", "edgelist"),
    ("hpo-b", "hpo_b"),
    ("hpo_b", "hpo_b"),
    ("pd1", "pd1"),
    ("imagenet", "imagenet"),
    ("cifar100", "cifar100"),
    ("cifar10", "cifar10"),
    ("cifar-100", "cifar100"),
    ("cifar-10", "cifar10"),
    ("mnist", "mnist"),
    ("cora", "cora"),
    ("citeseer", "citeseer"),
    ("pubmed", "pubmed"),
    ("wiki-topcats", "wiki-topcats"),
)
_JUNK_BENCHMARK_LABELS = frozenset(
    {
        "real-world",
        "real_world",
        "multi-task",
        "multi_task",
        "generic",
        "various",
        "sample",
        "these",
        "the",
        "specific",
        "no",
        "modified",
        "neo4j",
        "unknown",
        "n_a",
        "none",
    }
)


def _known_benchmark_label(raw: str) -> str | None:
    """Return a canonical label only when a known benchmark hint matches."""
    lowered = str(raw or "").strip().lower()
    if not lowered:
        return None
    for hint, label in _KNOWN_BENCHMARK_HINTS:
        if hint in lowered:
            return label
    return None


def _short_benchmark_label(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    known = _known_benchmark_label(text)
    if known:
        return known
    # Reject prose-like Analyst blurbs; only keep short concrete tokens.
    if len(text.split()) > 4:
        return ""
    token = re.split(r"[\s(,/:]+", text)[0]
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", token).strip("_")
    if not cleaned or cleaned.lower() in _JUNK_BENCHMARK_LABELS:
        return ""
    return cleaned[:48]


def selectable_benchmark_labels(
    raw_items: list[str],
    *,
    max_items: int = 6,
) -> list[str]:
    """Compact Analyst/dataset names into runnable axis labels; drop junk prose."""
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        label = _short_benchmark_label(item)
        if not label:
            continue
        key = label.lower()
        if key in _JUNK_BENCHMARK_LABELS or key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= max_items:
            break
    return out


def command_grounds_factor(command: str, axis_name: str, value: object) -> bool:
    """True when factor appears as a CLI/flag assignment, not only inside a results path."""
    text = (command or "").split("  #", 1)[0].strip()
    if not text or text.startswith("#"):
        return False
    # Log paths often embed axis=value; those must not count as grounding.
    stripped = re.sub(r"results/\S+", "", text)
    stripped = re.sub(r"\./build/\S+", "", stripped)
    axis = str(axis_name).strip()
    if not axis:
        return False
    dashed = axis.replace("_", "-")
    value_text = str(value).strip()
    flag_forms = (f"--{axis}", f"--{dashed}")
    if any(flag in stripped for flag in flag_forms):
        return True
    assignment_forms = (
        f"{axis}={value_text}",
        f"{dashed}={value_text}",
        f"{axis} = {value_text}",
    )
    return any(form in stripped for form in assignment_forms)


def annotate_ungrounded_factor(
    command: str,
    *,
    axis_name: str,
    value: object,
) -> str:
    """Append an explicit code/config edit hint when the CLI does not take the factor."""
    text = (command or "").strip()
    note = f"# set {axis_name}={value} in code/config"
    if note in text:
        return text
    if command_grounds_factor(text, axis_name, value):
        return text
    if not text or text.startswith("#"):
        return text or note
    return f"{text}  {note}"


def _setup_phase(
    *,
    paper_id: str,
    build_system: str,
    planned_actions: str,
) -> PlanPhase:
    return PlanPhase(
        phase_id="setup",
        title="Environment setup",
        goal="Install repository dependencies and confirm entrypoints from README.",
        depends_on=[],
        variables=[],
        axes={},
        run_template=build_system.strip(),
        matrix=[],
        planned_actions=planned_actions,
        results_path=f"results/{paper_id}",
    )


def _summarize_phase(*, paper_id: str, depends_on: list[str], note: str) -> PlanPhase:
    results_root = f"results/{paper_id}"
    return PlanPhase(
        phase_id="summarize",
        title="Aggregate results",
        goal="Write/update summary.json aggregates from phase logs.",
        depends_on=depends_on,
        variables=[],
        axes={},
        run_template="",
        matrix=[],
        planned_actions=note,
        results_path=f"{results_root}/summary.json",
    )


def _build_cli_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object],
    analyst: SectionExtraction,
) -> list[PlanPhase]:
    """Registry-backed setup → smoke → synthetic → real-world → ablations → summarize."""
    registry = (
        exploration.get("registry_ids")
        if isinstance(exploration.get("registry_ids"), dict)
        else {}
    )
    functions = [str(item) for item in (registry.get("functions_or_benchmarks") or [])]
    algorithms = rank_algorithms(
        [str(item) for item in (registry.get("algorithms_or_methods") or [])]
    )[:PLANNER_PHASE_ALGO_MAX]
    commands = [str(item) for item in (exploration.get("example_commands") or [])]
    template = _pick_run_command_template(commands) or ""
    readme = str(exploration.get("readme_full") or "")
    source_blob = "\n".join(
        str(item.get("excerpt") or "")
        for item in (exploration.get("source_excerpts") or [])
        if isinstance(item, dict)
    )
    reg_types, cls_types = _infer_model_type_options(f"{readme}\n{source_blob}")
    reg_type = reg_types[0] if reg_types else "gp"
    primary_cls = cls_types[-1] if cls_types else "gp"

    code_refs = _unique(
        [
            str(exploration.get("readme_file") or "README.md"),
            *[str(item) for item in (exploration.get("entrypoint_hints") or [])[:6]],
        ]
    )
    metrics = [item.strip() for item in analyst.evaluation_metrics if item.strip()][:5]
    seeds = list(range(PLANNER_PHASE_SEED_COUNT))
    synthetic, real = split_benchmarks(functions)
    synthetic = synthetic[:PLANNER_PHASE_SYNTHETIC_MAX]
    real = real[:PLANNER_PHASE_REALWORLD_MAX]
    results_root = f"results/{paper_id}"

    phases: list[PlanPhase] = [
        _setup_phase(
            paper_id=paper_id,
            build_system=build_system,
            planned_actions=(
                "Install with build_system, open README and registry __init__ files, "
                "confirm FUN/ALGO IDs before any experiment phase."
            ),
        )
    ]

    if not template or not algorithms or not (synthetic or real or functions):
        return phases

    paper_algo = algorithms[0]
    smoke_benchmark = (synthetic or real or functions)[0]
    smoke_axes: dict[str, list[str | int | float | bool]] = {
        "benchmark": [smoke_benchmark],
        "algorithm": [paper_algo],
        "seed": [0],
        "reg_type": [reg_type],
        "cls_type": [primary_cls],
    }
    smoke_vars = ["benchmark", "algorithm", "seed"]
    phases.append(
        PlanPhase(
            phase_id="smoke",
            title="Smoke run",
            goal="One cheap serial run to validate install, CLI IDs, and logging paths.",
            depends_on=["setup"],
            variables=smoke_vars,
            axes={key: smoke_axes[key] for key in smoke_vars},
            run_template=template,
            matrix=_example_rows(
                phase_id="smoke",
                axes=smoke_axes,
                variables=smoke_vars,
                template=template,
                paper_id=paper_id,
                code_refs=code_refs,
                metrics=metrics,
                reg_type=reg_type,
                cls_type=primary_cls,
                source="repo",
                max_rows=1,
            ),
            planned_actions=(
                f"Run one {smoke_benchmark} × {paper_algo} seed-0 command; verify log path "
                "exists before launching larger phases."
            ),
            results_path=f"{results_root}/smoke",
        )
    )

    previous = "smoke"
    if synthetic:
        axes: dict[str, list[str | int | float | bool]] = {
            "benchmark": synthetic,
            "algorithm": algorithms[:3] or algorithms,
            "seed": seeds,
        }
        variables = ["benchmark", "algorithm", "seed"]
        example_axes = {
            **axes,
            "reg_type": [reg_type],
            "cls_type": [primary_cls],
        }
        phases.append(
            PlanPhase(
                phase_id="synthetic",
                title="Synthetic benchmarks",
                goal="Serial runs on synthetic benchmarks × key algorithms × seeds.",
                depends_on=[previous],
                variables=variables,
                axes=axes,
                run_template=template,
                matrix=_example_rows(
                    phase_id="synthetic",
                    axes=example_axes,
                    variables=variables,
                    template=template,
                    paper_id=paper_id,
                    code_refs=code_refs,
                    metrics=metrics,
                    reg_type=reg_type,
                    cls_type=primary_cls,
                    source="repo",
                ),
                planned_actions=(
                    "Expand axes into serial runs (or repo parallel helper). "
                    "Keep seeds identical across algorithms for fair comparison."
                ),
                results_path=f"{results_root}/synthetic",
            )
        )
        previous = "synthetic"

    if real:
        axes = {
            "benchmark": real,
            "algorithm": algorithms,
            "seed": seeds,
        }
        variables = ["benchmark", "algorithm", "seed"]
        example_axes = {
            **axes,
            "reg_type": [reg_type],
            "cls_type": [primary_cls],
        }
        phases.append(
            PlanPhase(
                phase_id="real_world",
                title="Real-world problems",
                goal="Serial/batch runs on real-world registry benchmarks × algorithms × seeds.",
                depends_on=[previous],
                variables=variables,
                axes=axes,
                run_template=template,
                matrix=_example_rows(
                    phase_id="real_world",
                    axes=example_axes,
                    variables=variables,
                    template=template,
                    paper_id=paper_id,
                    code_refs=code_refs,
                    metrics=metrics,
                    reg_type=reg_type,
                    cls_type=primary_cls,
                    source="repo",
                ),
                planned_actions=(
                    "Only start after smoke (and synthetic if present) succeed. "
                    "Use parallel runner if documented; otherwise serialize by benchmark."
                ),
                results_path=f"{results_root}/real_world",
            )
        )
        previous = "real_world"

    ablation_parent = previous
    ablation_ids: list[str] = []
    for index, (axis_name, values) in enumerate(
        extract_ablation_axes(analyst.hyperparameters), start=1
    ):
        phase_id = f"ablation_{index}_{axis_name}"
        ablation_ids.append(phase_id)
        benchmarks = (synthetic + real) or functions[
            : PLANNER_PHASE_SYNTHETIC_MAX + PLANNER_PHASE_REALWORLD_MAX
        ]
        axes = {
            axis_name: list(values),
            "benchmark": benchmarks,
            "seed": [0],
            "algorithm": [paper_algo],
        }
        variables = [axis_name, "benchmark", "seed"]
        matrix = []
        for benchmark in benchmarks[:PLANNER_PHASE_EXAMPLE_ROWS]:
            value = values[0]
            log_path = f"{results_root}/{phase_id}/{benchmark}/{axis_name}={value}/0"
            factors = {
                axis_name: value,
                "benchmark": benchmark,
                "seed": 0,
                "algorithm": paper_algo,
            }
            run_command = _fill_run_command(
                template,
                benchmark=benchmark,
                algorithm=paper_algo,
                reg_type=reg_type,
                cls_type=primary_cls,
                seed=0,
                log_path=log_path,
                extra={axis_name: str(value)},
            )
            run_command = annotate_ungrounded_factor(
                run_command,
                axis_name=axis_name,
                value=value,
            )
            matrix.append(
                PhaseRunSpec(
                    name=f"{phase_id}__{benchmark}__{axis_name}={value}",
                    variables={key: factors[key] for key in variables},
                    run_command=run_command,
                    code_refs=code_refs,
                    verify=[f"exists:{log_path}", f"exists:{results_root}/summary.json"],
                    results_path=log_path,
                    metrics=list(metrics[:3]),
                    source="analyst",
                )
            )
        phases.append(
            PlanPhase(
                phase_id=phase_id,
                title=f"Ablation: {axis_name}",
                goal=(
                    f"Sweep {axis_name} over Analyst values on all selected benchmarks "
                    "with one seed (serial)."
                ),
                depends_on=[ablation_parent],
                variables=variables,
                axes={key: axes[key] for key in variables},
                run_template=template,
                matrix=matrix,
                planned_actions=(
                    f"For each {axis_name} value, run all benchmarks at seed 0 using paper "
                    f"method {paper_algo}. Do not invent values outside Analyst list {values}."
                ),
                results_path=f"{results_root}/{phase_id}",
            )
        )

    summarize_deps = ablation_ids or [previous]
    phases.append(
        _summarize_phase(
            paper_id=paper_id,
            depends_on=summarize_deps,
            note=(
                f"Collect metrics under {results_root} into {results_root}/summary.json; "
                "do not invent measured values."
            ),
        )
    )
    return phases


def _build_script_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object],
    analyst: SectionExtraction,
) -> list[PlanPhase]:
    """Script repos without registries: setup → smoke → experiments → summarize."""
    results_root = f"results/{paper_id}"
    scripts = _unique(
        [str(item) for item in (exploration.get("script_entrypoints") or [])]
        + [
            token.strip("`'")
            for command in (exploration.get("example_commands") or [])
            for token in str(command).split()
            if token.endswith(".py")
        ]
    )
    tunables = (
        exploration.get("script_tunables")
        if isinstance(exploration.get("script_tunables"), dict)
        else {}
    )
    benchmarks = [str(item) for item in (tunables.get("benchmark") or tunables.get("dataset") or [])]
    algorithms = [str(item) for item in (tunables.get("algorithm") or tunables.get("method") or [])]
    # Prefer a tunable-bearing script (e.g. Clustering.py) over unrelated README peers.
    primary_script = scripts[0] if scripts else ""
    if benchmarks and algorithms:
        preferred = [
            path
            for path in scripts
            if any(token in Path(path).stem.lower() for token in ("cluster", "train", "run", "experiment"))
        ]
        if preferred:
            primary_script = preferred[0]
            scripts = _unique([primary_script, *scripts])
    metrics = [item.strip() for item in analyst.evaluation_metrics if item.strip()][:5]
    code_refs = _unique(
        [
            str(exploration.get("readme_file") or "README.md"),
            *scripts[:6],
        ]
    )
    phases = [
        _setup_phase(
            paper_id=paper_id,
            build_system=build_system,
            planned_actions=(
                "Install dependencies from README/requirements, confirm script entrypoints exist, "
                "and note in-file tunables (dataset/method) before running experiments."
            ),
        )
    ]
    if not scripts:
        phases.append(
            _summarize_phase(
                paper_id=paper_id,
                depends_on=["setup"],
                note=f"No runnable scripts found; record gaps under {results_root}/summary.json.",
            )
        )
        return phases

    smoke_script = primary_script or scripts[0]
    smoke_path = f"{results_root}/smoke"
    phases.append(
        PlanPhase(
            phase_id="smoke",
            title="Script smoke run",
            goal="Run one README/script entrypoint to validate install and outputs.",
            depends_on=["setup"],
            variables=["script"],
            axes={"script": [smoke_script]},
            run_template="python {script}",
            matrix=[
                PhaseRunSpec(
                    name=f"smoke__{Path(smoke_script).stem}",
                    variables={"script": smoke_script},
                    run_command=f"python {smoke_script}",
                    code_refs=_unique([smoke_script, *code_refs[:3]]),
                    verify=["exit_code:0", f"exists:{smoke_path}"],
                    results_path=smoke_path,
                    metrics=metrics[:2],
                    source="repo",
                )
            ],
            planned_actions=f"Execute `python {smoke_script}` once; confirm it starts and writes logs.",
            results_path=smoke_path,
        )
    )

    experiment_matrix: list[PhaseRunSpec] = []
    max_example_rows = max(PLANNER_PHASE_EXAMPLE_ROWS, 6)
    if benchmarks and algorithms:
        variables = ["benchmark", "algorithm", "script"]
        axes: dict[str, list[str | int | float | bool]] = {
            "benchmark": benchmarks[:PLANNER_PHASE_REALWORLD_MAX],
            "algorithm": algorithms[:PLANNER_PHASE_ALGO_MAX],
            "script": [smoke_script],
        }
        for benchmark in axes["benchmark"]:
            for algorithm in axes["algorithm"]:
                log_path = f"{results_root}/experiments/{benchmark}/{algorithm}"
                experiment_matrix.append(
                    PhaseRunSpec(
                        name=f"exp__{benchmark}__{algorithm}",
                        variables={
                            "benchmark": str(benchmark),
                            "algorithm": str(algorithm),
                            "script": smoke_script,
                        },
                        run_command=(
                            f"python {smoke_script}  # set dataset={benchmark}, method={algorithm} "
                            "in script tunables OrderedDict"
                        ),
                        code_refs=_unique([smoke_script, *code_refs[:3]]),
                        verify=[
                            "exit_code:0",
                            f"exists:{log_path}",
                            f"exists:{results_root}/summary.json",
                        ],
                        results_path=log_path,
                        metrics=metrics[:3],
                        source="repo",
                    )
                )
                if len(experiment_matrix) >= max_example_rows:
                    break
            if len(experiment_matrix) >= max_example_rows:
                break
        # Also include peer scripts as additional rows when space remains.
        for script in scripts[1:]:
            if len(experiment_matrix) >= max_example_rows:
                break
            if script == smoke_script:
                continue
            log_path = f"{results_root}/experiments/{Path(script).stem}"
            experiment_matrix.append(
                PhaseRunSpec(
                    name=f"exp__{Path(script).stem}",
                    variables={
                        "benchmark": str(benchmarks[0]),
                        "algorithm": str(algorithms[0]),
                        "script": script,
                    },
                    run_command=f"python {script}",
                    code_refs=_unique([script, *code_refs[:3]]),
                    verify=["exit_code:0", f"exists:{log_path}"],
                    results_path=log_path,
                    metrics=metrics[:3],
                    source="repo",
                )
            )
        axes["script"] = _unique([smoke_script, *[str(r.variables.get('script')) for r in experiment_matrix]])
        run_template = f"python {smoke_script}"
    else:
        variables = ["script"]
        axes = {"script": scripts[:max_example_rows]}
        run_template = "python {script}"
        for script in scripts[:max_example_rows]:
            log_path = f"{results_root}/experiments/{Path(script).stem}"
            experiment_matrix.append(
                PhaseRunSpec(
                    name=f"exp__{Path(script).stem}",
                    variables={"script": script},
                    run_command=f"python {script}",
                    code_refs=_unique([script, *code_refs[:3]]),
                    verify=["exit_code:0", f"exists:{log_path}"],
                    results_path=log_path,
                    metrics=metrics[:3],
                    source="repo",
                )
            )

    phases.append(
        PlanPhase(
            phase_id="experiments",
            title="Script experiments",
            goal="Run grounded script entrypoints (or scraped dataset×method grid) serially.",
            depends_on=["smoke"],
            variables=variables,
            axes=axes,
            run_template=run_template,
            matrix=experiment_matrix,
            planned_actions=(
                "Edit in-file tunables when axes come from script scrape; do not invent CLI flags. "
                "Record dataset availability gaps in missing_context."
            ),
            results_path=f"{results_root}/experiments",
        )
    )
    phases.append(
        _summarize_phase(
            paper_id=paper_id,
            depends_on=["experiments"],
            note=(
                f"Collect metrics under {results_root} into {results_root}/summary.json; "
                "do not invent measured values."
            ),
        )
    )
    return phases


def _build_native_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object],
    analyst: SectionExtraction,
) -> list[PlanPhase]:
    """CMake/Make native repos: setup → native_smoke → reproduce_similar → summarize."""
    results_root = f"results/{paper_id}"
    native = (
        exploration.get("native_build")
        if isinstance(exploration.get("native_build"), dict)
        else {}
    )
    native_commands = [str(item) for item in (native.get("commands") or []) if str(item).strip()]
    native_tests = [str(item) for item in (exploration.get("native_tests") or [])]
    make_targets = [str(item) for item in (exploration.get("make_targets") or [])]
    metrics = [item.strip() for item in analyst.evaluation_metrics if item.strip()][:5]
    raw_benchmarks = [item.strip() for item in analyst.datasets_or_benchmarks if item.strip()]
    benchmarks = selectable_benchmark_labels(raw_benchmarks) or ["sbm"]
    ungrounded_analyst = [
        item
        for item in raw_benchmarks
        if item and not _known_benchmark_label(item) and not _short_benchmark_label(item)
    ]
    code_refs = _unique(
        [
            str(exploration.get("readme_file") or "README.md"),
            *[str(item) for item in (native.get("files") or [])],
            *native_tests[:4],
        ]
    )
    tree = str(exploration.get("file_tree_deep") or "")
    has_sbm_tool = any(
        token in tree or token in " ".join(code_refs)
        for token in ("stagtools/", "stagtools/sbm", "sbm.cpp")
    )
    if has_sbm_tool:
        code_refs = _unique([*code_refs, "stagtools/sbm.cpp"])
    setup_cmd = build_system.strip()
    if native_commands:
        setup_cmd = native_commands[0]
    elif "cmake" in setup_cmd.lower() or setup_cmd in {"", "unknown"}:
        setup_cmd = "cmake -S . -B build && cmake --build build"

    phases = [
        _setup_phase(
            paper_id=paper_id,
            build_system=setup_cmd,
            planned_actions=(
                "Follow INSTALL/README: install Eigen/Spectra, configure with CMake or Make, "
                "build binaries and stagtools, confirm test targets before smoke."
            ),
        )
    ]

    smoke_commands: list[str] = []
    for command in native_commands:
        if "ctest" in command.lower() or command.startswith("make test"):
            smoke_commands.append(command)
    if not smoke_commands and make_targets:
        for target in make_targets:
            if "test" in target.lower():
                smoke_commands.append(f"make {target}")
                break
    if not smoke_commands:
        smoke_commands = ["ctest --test-dir build --output-on-failure"]

    smoke_matrix = [
        PhaseRunSpec(
            name="native_smoke__build_tests",
            variables={"verify_command": smoke_commands[0]},
            run_command=smoke_commands[0],
            code_refs=code_refs,
            verify=["exit_code:0"],
            results_path=f"{results_root}/native_smoke",
            metrics=[],
            source="repo",
        )
    ]
    # Prefer named compiled tests when discovery found them.
    for test_path in native_tests[:2]:
        stem = Path(test_path).stem
        smoke_matrix.append(
            PhaseRunSpec(
                name=f"native_smoke__{stem}",
                variables={"verify_command": f"ctest --test-dir build -R {stem} --output-on-failure"},
                run_command=f"ctest --test-dir build -R {stem} --output-on-failure",
                code_refs=_unique([test_path, *code_refs[:4]]),
                verify=["exit_code:0"],
                results_path=f"{results_root}/native_smoke/{stem}",
                metrics=[],
                source="repo",
            )
        )
    phases.append(
        PlanPhase(
            phase_id="native_smoke",
            title="Native build/test smoke",
            goal="Compile and run ctest/make test to prove the native library builds.",
            depends_on=["setup"],
            variables=["verify_command"],
            axes={
                "verify_command": _unique(
                    [row.variables["verify_command"] for row in smoke_matrix if "verify_command" in row.variables]
                )
            },
            run_template=smoke_commands[0],
            matrix=smoke_matrix,
            planned_actions=(
                "After cmake/make build, run ctest or make test. "
                f"Known test sources: {', '.join(native_tests[:4]) or 'see test/'}."
            ),
            results_path=f"{results_root}/native_smoke",
        )
    )

    reproduce_matrix: list[PhaseRunSpec] = []
    skipped_benchmarks: list[str] = list(ungrounded_analyst)
    runnable_benchmarks: list[str] = []
    for benchmark in benchmarks[:4]:
        log_path = f"{results_root}/reproduce_similar/{benchmark}"
        if has_sbm_tool and benchmark == "sbm":
            run_command = (
                "cmake --build build --target stag_sbm 2>/dev/null; "
                "./build/stagtools/stag_sbm /tmp/stag_sbm.edgelist 200 2 0.6 0.1"
            )
            runnable_benchmarks.append(benchmark)
            reproduce_matrix.append(
                PhaseRunSpec(
                    name=f"reproduce__{benchmark}",
                    variables={"benchmark": benchmark},
                    run_command=run_command,
                    code_refs=code_refs,
                    verify=["exit_code:0"],
                    results_path=log_path,
                    metrics=metrics,
                    source="repo",
                )
            )
        else:
            skipped_benchmarks.append(benchmark)

    phases.append(
        PlanPhase(
            phase_id="reproduce_similar",
            title="Reproduce paper-similar native sample",
            goal="Run grounded native samples after successful build/tests.",
            depends_on=["native_smoke"],
            variables=["benchmark"] if runnable_benchmarks else [],
            axes={"benchmark": runnable_benchmarks} if runnable_benchmarks else {},
            run_template=reproduce_matrix[0].run_command if reproduce_matrix else "",
            matrix=reproduce_matrix,
            planned_actions=(
                "Prefer compiled stagtools (sbm) and ctest cluster suites over inventing CLIs. "
                "README C++ snippets need a small driver that writes metrics under results/. "
                + (
                    f"Ungrounded Analyst benchmarks deferred to missing_context: "
                    f"{', '.join(skipped_benchmarks[:6])}."
                    if skipped_benchmarks
                    else ""
                )
            ),
            results_path=f"{results_root}/reproduce_similar",
        )
    )
    phases.append(
        _summarize_phase(
            paper_id=paper_id,
            depends_on=["reproduce_similar"],
            note=(
                f"Merge native smoke + sample metrics into {results_root}/summary.json; "
                "note any missing datasets or CLI wrappers."
            ),
        )
    )
    return phases


def _build_config_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object],
    analyst: SectionExtraction,
) -> list[PlanPhase]:
    results_root = f"results/{paper_id}"
    configs = [str(item) for item in (exploration.get("config_files") or [])]
    scripts = [str(item) for item in (exploration.get("script_entrypoints") or [])]
    commands = [str(item) for item in (exploration.get("example_commands") or [])]
    template = _pick_run_command_template(commands) or (
        f"python {scripts[0]}" if scripts else ""
    )
    metrics = [item.strip() for item in analyst.evaluation_metrics if item.strip()][:5]
    code_refs = _unique(
        [
            str(exploration.get("readme_file") or "README.md"),
            *configs[:4],
            *scripts[:3],
        ]
    )
    phases = [
        _setup_phase(
            paper_id=paper_id,
            build_system=build_system,
            planned_actions="Install deps; list configs/ and confirm train/run entrypoint.",
        )
    ]
    if not configs:
        phases.append(
            _summarize_phase(
                paper_id=paper_id,
                depends_on=["setup"],
                note="No config files discovered; record missing_context.",
            )
        )
        return phases

    smoke_cfg = configs[0]
    smoke_cmd = (
        f"{template} --config {smoke_cfg}" if template else f"# apply config {smoke_cfg}"
    )
    phases.append(
        PlanPhase(
            phase_id="smoke",
            title="Config smoke run",
            goal="Run one config to validate the train/eval entrypoint.",
            depends_on=["setup"],
            variables=["config"],
            axes={"config": [smoke_cfg]},
            run_template=smoke_cmd,
            matrix=[
                PhaseRunSpec(
                    name=f"smoke__{Path(smoke_cfg).stem}",
                    variables={"config": smoke_cfg},
                    run_command=smoke_cmd,
                    code_refs=_unique([smoke_cfg, *code_refs[:3]]),
                    verify=["exit_code:0"],
                    results_path=f"{results_root}/smoke",
                    metrics=metrics[:2],
                    source="repo",
                )
            ],
            planned_actions=f"Launch one run with config {smoke_cfg}.",
            results_path=f"{results_root}/smoke",
        )
    )
    matrix = []
    for config in configs[:PLANNER_PHASE_EXAMPLE_ROWS]:
        cmd = f"{template} --config {config}" if template else f"# apply config {config}"
        log_path = f"{results_root}/experiments/{Path(config).stem}"
        matrix.append(
            PhaseRunSpec(
                name=f"exp__{Path(config).stem}",
                variables={"config": config},
                run_command=cmd,
                code_refs=_unique([config, *code_refs[:3]]),
                verify=[f"exists:{log_path}"],
                results_path=log_path,
                metrics=metrics[:3],
                source="repo",
            )
        )
    phases.append(
        PlanPhase(
            phase_id="experiments",
            title="Config matrix",
            goal="Sweep grounded config files through the documented train/run entrypoint.",
            depends_on=["smoke"],
            variables=["config"],
            axes={"config": configs[:PLANNER_PHASE_REALWORLD_MAX]},
            run_template=template or "",
            matrix=matrix,
            planned_actions="Do not invent Hydra overrides beyond listed config files.",
            results_path=f"{results_root}/experiments",
        )
    )
    phases.append(
        _summarize_phase(
            paper_id=paper_id,
            depends_on=["experiments"],
            note=f"Aggregate config-run metrics into {results_root}/summary.json.",
        )
    )
    return phases


def _build_container_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object],
    analyst: SectionExtraction,
) -> list[PlanPhase]:
    results_root = f"results/{paper_id}"
    containers = [str(item) for item in (exploration.get("container_files") or [])]
    code_refs = _unique(
        [str(exploration.get("readme_file") or "README.md"), *containers]
    )
    phases = [
        _setup_phase(
            paper_id=paper_id,
            build_system=build_system,
            planned_actions=(
                "Review Dockerfile/compose; prefer CPU-safe documented run. "
                "Note GPU-only images in missing_context."
            ),
        )
    ]
    smoke_cmd = (
        "docker compose up --build"
        if any("compose" in name for name in containers)
        else "docker build -t paper-repro ."
    )
    phases.append(
        PlanPhase(
            phase_id="container_smoke",
            title="Container smoke",
            goal="Build/run the documented container entry once if CPU-safe.",
            depends_on=["setup"],
            variables=["container_file"],
            axes={"container_file": containers[:2] or ["Dockerfile"]},
            run_template=smoke_cmd,
            matrix=[
                PhaseRunSpec(
                    name="container_smoke__build",
                    variables={"container_file": containers[0] if containers else "Dockerfile"},
                    run_command=smoke_cmd,
                    code_refs=code_refs,
                    verify=["exit_code:0"],
                    results_path=f"{results_root}/container_smoke",
                    metrics=[],
                    source="repo",
                )
            ],
            planned_actions=(
                "Only run if README documents a CPU path; otherwise leave phases and "
                "record GPU/data requirements in missing_context."
            ),
            results_path=f"{results_root}/container_smoke",
        )
    )
    phases.append(
        _summarize_phase(
            paper_id=paper_id,
            depends_on=["container_smoke"],
            note=f"Record container smoke outcome in {results_root}/summary.json.",
        )
    )
    return phases


def _build_artifact_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object],
    analyst: SectionExtraction,
) -> list[PlanPhase]:
    results_root = f"results/{paper_id}"
    artifacts = [str(item) for item in (exploration.get("artifact_dirs") or [])]
    code_refs = _unique(
        [str(exploration.get("readme_file") or "README.md"), *artifacts]
    )
    phases = [
        _setup_phase(
            paper_id=paper_id,
            build_system=build_system,
            planned_actions="Install minimal deps needed to load shipped figures/results.",
        ),
        PlanPhase(
            phase_id="verify_artifacts",
            title="Verify shipped artifacts",
            goal="Confirm results/figures exist and document how they map to Analyst claims.",
            depends_on=["setup"],
            variables=["artifact_dir"],
            axes={"artifact_dir": artifacts or ["results/"]},
            run_template=f"ls {' '.join(artifacts) if artifacts else 'results'}",
            matrix=[
                PhaseRunSpec(
                    name="verify_artifacts__list",
                    variables={"artifact_dir": artifacts[0] if artifacts else "results/"},
                    run_command=f"ls {' '.join(artifacts) if artifacts else 'results'}",
                    code_refs=code_refs,
                    verify=[f"exists:{artifacts[0] if artifacts else 'results'}"],
                    results_path=f"{results_root}/verify_artifacts",
                    metrics=[],
                    source="repo",
                )
            ],
            planned_actions=(
                "Re-run is weak; inventory artifacts, map to reported_results, "
                "and note missing re-execution commands in missing_context."
            ),
            results_path=f"{results_root}/verify_artifacts",
        ),
        _summarize_phase(
            paper_id=paper_id,
            depends_on=["verify_artifacts"],
            note=f"Write artifact inventory notes into {results_root}/summary.json.",
        ),
    ]
    return phases


def _build_unknown_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object],
) -> list[PlanPhase]:
    """Never return empty phases when code exists — setup + explicit missing_context hooks."""
    results_root = f"results/{paper_id}"
    missing_bits = []
    if not exploration.get("example_commands"):
        missing_bits.append("no README experiment commands")
    if not exploration.get("script_entrypoints"):
        missing_bits.append("no script entrypoints")
    native = exploration.get("native_build")
    if not (isinstance(native, dict) and native.get("available")):
        missing_bits.append("no native build files")
    if not exploration.get("test_files"):
        missing_bits.append("no Python tests")
    if not exploration.get("config_files"):
        missing_bits.append("no config matrix")
    reason = ", ".join(missing_bits) or "insufficient grounded execution evidence"
    return [
        _setup_phase(
            paper_id=paper_id,
            build_system=build_system,
            planned_actions=(
                "Install whatever deps README documents. Do not invent CLIs. "
                f"Evidence gaps: {reason}."
            ),
        ),
        PlanPhase(
            phase_id="missing_context",
            title="Document missing execution context",
            goal="Capture what evidence is required before an Engineer can run experiments.",
            depends_on=["setup"],
            variables=[],
            axes={},
            run_template="",
            matrix=[],
            planned_actions=(
                f"Record missing grounded commands/files ({reason}) under missing_context; "
                "block experiment invention until README/CLI/tests appear."
            ),
            results_path=f"{results_root}/missing_context",
        ),
        _summarize_phase(
            paper_id=paper_id,
            depends_on=["missing_context"],
            note=f"Summarize blockers in {results_root}/summary.json.",
        ),
    ]


def build_plan_phases(
    *,
    paper_id: str,
    build_system: str,
    exploration: dict[str, object] | None,
    analyst: SectionExtraction,
) -> list[PlanPhase]:
    """Route to a surface-specific phase DAG from exploration evidence."""
    exploration = exploration if isinstance(exploration, dict) else {}
    surface = str(exploration.get("execution_surface") or "").strip()
    if not surface or surface == "unknown":
        # Infer from evidence when callers omit classification (tests / partial packs).
        registry = (
            exploration.get("registry_ids")
            if isinstance(exploration.get("registry_ids"), dict)
            else {}
        )
        native = exploration.get("native_build")
        surface = infer_execution_surface(
            example_commands=[str(item) for item in (exploration.get("example_commands") or [])],
            test_files=[str(item) for item in (exploration.get("test_files") or [])],
            notebooks=[str(item) for item in (exploration.get("notebooks") or [])],
            has_setup_py_or_pyproject=False,
            script_entrypoints=[str(item) for item in (exploration.get("script_entrypoints") or [])],
            native_build=native if isinstance(native, dict) else {},
            native_tests=[str(item) for item in (exploration.get("native_tests") or [])],
            config_files=[str(item) for item in (exploration.get("config_files") or [])],
            container_files=[str(item) for item in (exploration.get("container_files") or [])],
            make_targets=[str(item) for item in (exploration.get("make_targets") or [])],
            artifact_dirs=[str(item) for item in (exploration.get("artifact_dirs") or [])],
            registry_functions=[str(item) for item in (registry.get("functions_or_benchmarks") or [])],
            registry_algorithms=[str(item) for item in (registry.get("algorithms_or_methods") or [])],
        )
        exploration = {**exploration, "execution_surface": surface}

    if surface == "cli":
        phases = _build_cli_phases(
            paper_id=paper_id,
            build_system=build_system,
            exploration=exploration,
            analyst=analyst,
        )
        # Registry-backed CLI that lacks matrix evidence falls through to script/library.
        if any(phase.phase_id not in {"setup", "summarize"} and (phase.matrix or phase.axes) for phase in phases):
            return phases
        if exploration.get("script_entrypoints") or exploration.get("example_commands"):
            return _build_script_phases(
                paper_id=paper_id,
                build_system=build_system,
                exploration=exploration,
                analyst=analyst,
            )
        if exploration.get("test_files") or exploration.get("notebooks"):
            return _build_library_phases(
                paper_id=paper_id,
                build_system=build_system,
                exploration=exploration,
                analyst=analyst,
            )
        return phases

    if surface == "script":
        return _build_script_phases(
            paper_id=paper_id,
            build_system=build_system,
            exploration=exploration,
            analyst=analyst,
        )
    if surface == "library":
        return _build_library_phases(
            paper_id=paper_id,
            build_system=build_system,
            exploration=exploration,
            analyst=analyst,
        )
    if surface == "native":
        return _build_native_phases(
            paper_id=paper_id,
            build_system=build_system,
            exploration=exploration,
            analyst=analyst,
        )
    if surface == "config":
        return _build_config_phases(
            paper_id=paper_id,
            build_system=build_system,
            exploration=exploration,
            analyst=analyst,
        )
    if surface == "container":
        return _build_container_phases(
            paper_id=paper_id,
            build_system=build_system,
            exploration=exploration,
            analyst=analyst,
        )
    if surface == "artifact":
        return _build_artifact_phases(
            paper_id=paper_id,
            build_system=build_system,
            exploration=exploration,
            analyst=analyst,
        )
    return _build_unknown_phases(
        paper_id=paper_id,
        build_system=build_system,
        exploration=exploration,
    )
