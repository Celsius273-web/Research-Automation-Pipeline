"""Repair empty phase matrices after verification.

When verification clears all rows from a phase, either:
- emit a Planner-generated wrapper/driver stub and refill a grounded matrix, or
- collapse the phase into missing_context and rewire depends_on.
"""

from __future__ import annotations

from pathlib import Path

from src.config import PLANNER_STUB_EXAMPLE_ROWS, PLANNER_STUBS_DIRNAME
from src.state import PhaseRunSpec, PlanPhase

_SCRIPT_WRAPPER_NAME = "run_script_experiment.py"
_NATIVE_DRIVER_CPP = "stag_spectral_cluster_driver.cpp"
_NATIVE_DRIVER_PY = "run_stag_cluster_driver.py"
_LIBRARY_PORT_NAME = "port_demo_metrics.py"


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


def stubs_dir_for_repo(repo_path: Path | None) -> Path | None:
    """Resolve <paper_bundle>/planner_stubs from a .../code repo path."""
    if repo_path is None:
        return None
    root = Path(repo_path)
    if not root.is_dir():
        return None
    # Prefer sibling of code/: data/papers/<id>/planner_stubs
    if root.name == "code":
        return root.parent / PLANNER_STUBS_DIRNAME
    return root / PLANNER_STUBS_DIRNAME


def stub_relpath(stub_name: str, repo_path: Path | None = None) -> str:
    """Path used in run_command when Engineer cwd is the cloned repo."""
    if repo_path is not None:
        stubs = stubs_dir_for_repo(repo_path)
        if stubs is not None:
            target = (stubs / stub_name).resolve()
            try:
                return target.relative_to(Path(repo_path).resolve()).as_posix()
            except ValueError:
                return f"../{PLANNER_STUBS_DIRNAME}/{stub_name}"
    return f"../{PLANNER_STUBS_DIRNAME}/{stub_name}"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_script_experiment_wrapper(stubs_dir: Path) -> Path:
    """CLI wrapper that patches OrderedDict tunables then runs a script."""
    path = stubs_dir / _SCRIPT_WRAPPER_NAME
    _write_text(
        path,
        '''#!/usr/bin/env python3
"""Planner-generated CLI wrapper for scripts that only expose in-file OrderedDict tunables.

Usage (cwd = paper code/):
  python ../planner_stubs/run_script_experiment.py --script Clustering.py \\
      --dataset cora --method mincut_pool
"""
from __future__ import annotations

import argparse
import os
import re
import runpy
import sys
import tempfile
from pathlib import Path


def _patch_tunables(source: str, dataset: str, method: str) -> str:
    patched = re.sub(
        r"\\(\\s*['\\\"]dataset['\\\"]\\s*,\\s*\\[[^\\]]*\\]",
        f"('dataset', ['{dataset}']",
        source,
        count=1,
    )
    patched = re.sub(
        r"\\(\\s*['\\\"]method['\\\"]\\s*,\\s*\\[[^\\]]*\\]",
        f"('method', ['{method}']",
        patched,
        count=1,
    )
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch dataset/method OrderedDict entries and run a repo script."
    )
    parser.add_argument("--script", required=True, help="Repo-relative script path")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    repo = Path.cwd()
    script_path = (repo / args.script).resolve()
    if not script_path.is_file():
        raise SystemExit(f"script not found: {script_path}")

    original = script_path.read_text(encoding="utf-8")
    patched = _patch_tunables(original, args.dataset, args.method)
    if patched == original:
        print(
            "warning: did not find OrderedDict dataset/method lists to patch; "
            "running original script",
            file=sys.stderr,
        )

    sys.path.insert(0, str(repo))
    with tempfile.TemporaryDirectory(prefix="planner_wrap_") as tmp:
        tmp_script = Path(tmp) / script_path.name
        tmp_script.write_text(patched, encoding="utf-8")
        os.chdir(repo)
        runpy.run_path(str(tmp_script), run_name="__main__")


if __name__ == "__main__":
    main()
''',
    )
    return path


def write_native_cluster_driver(stubs_dir: Path) -> tuple[Path, Path]:
    """Write README-based C++ driver + Python runner that uses built STAG artifacts."""
    cpp_path = stubs_dir / _NATIVE_DRIVER_CPP
    _write_text(
        cpp_path,
        '''/**
 * Planner-generated STAG driver (from README spectral_cluster sample).
 * Build via run_stag_cluster_driver.py after cmake has produced libstag.
 */
#include <fstream>
#include <iostream>
#include <string>

#include "cluster.h"
#include "graph.h"
#include "graphio.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: stag_spectral_cluster_driver <edgelist> <k> <metrics_csv>\\n";
    return 2;
  }
  const std::string filename = argv[1];
  const int k = std::stoi(argv[2]);
  const std::string metrics_path = argv[3];

  stag::Graph graph = stag::load_edgelist(filename);
  auto labels = stag::spectral_cluster(&graph, k);

  std::ofstream out(metrics_path);
  out << "vertex,cluster\\n";
  for (size_t i = 0; i < labels.size(); ++i) {
    out << i << "," << labels[i] << "\\n";
  }
  std::cout << "wrote " << metrics_path << " clusters=" << k
            << " vertices=" << labels.size() << "\\n";
  return 0;
}
''',
    )
    py_path = stubs_dir / _NATIVE_DRIVER_PY
    _write_text(
        py_path,
        '''#!/usr/bin/env python3
"""Compile/run Planner STAG spectral_cluster driver against the cmake build tree.

Usage (cwd = paper code/):
  python ../planner_stubs/run_stag_cluster_driver.py \\
      --edgelist results/<paper>/generate_inputs/sbm.edgelist \\
      --out results/<paper>/reproduce_similar/sbm --k 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edgelist", required=True)
    parser.add_argument("--out", required=True, help="Directory for metrics.csv")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--build-dir", default="build")
    args = parser.parse_args()

    repo = Path.cwd()
    stubs = Path(__file__).resolve().parent
    edgelist = Path(args.edgelist)
    if not edgelist.is_file():
        # Allow repo-relative paths.
        edgelist = repo / args.edgelist
    if not edgelist.is_file():
        raise SystemExit(f"edgelist not found: {args.edgelist}")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / "metrics.csv"
    driver_bin = Path(args.build_dir) / "planner_stag_spectral_cluster_driver"
    if not driver_bin.is_absolute():
        driver_bin = repo / driver_bin

    cxx = os.environ.get("CXX", "c++")
    include_flags = [
        f"-I{repo / 'stag_lib'}",
        f"-I{repo / args.build_dir}",
    ]
    # Link against the cmake-built stag library when present.
    lib_dir = repo / args.build_dir / "stag_lib"
    link_flags = [f"-L{lib_dir}", "-lstag"] if lib_dir.is_dir() else []
    cpp = stubs / "stag_spectral_cluster_driver.cpp"
    compile_cmd = [
        cxx,
        "-std=c++17",
        *include_flags,
        str(cpp),
        "-o",
        str(driver_bin),
        *link_flags,
        "-Wl,-rpath," + str(lib_dir) if link_flags else "",
    ]
    compile_cmd = [tok for tok in compile_cmd if tok]
    print("compile:", " ".join(compile_cmd), flush=True)
    compiled = subprocess.run(compile_cmd, cwd=repo)
    if compiled.returncode != 0:
        # Grounded fallback: run cluster unit tests and record proxy metrics.
        print(
            "driver compile failed; falling back to ctest -R cluster_test proxy metrics",
            file=sys.stderr,
        )
        proxy = subprocess.run(
            ["ctest", "--test-dir", args.build_dir, "-R", "cluster_test", "--output-on-failure"],
            cwd=repo,
        )
        metrics.write_text(
            "metric,value\\n"
            f"edgelist,{edgelist}\\n"
            f"ctest_cluster_test_exit,{proxy.returncode}\\n"
            "note,compiled_driver_unavailable_used_ctest_proxy\\n",
            encoding="utf-8",
        )
        if proxy.returncode != 0:
            raise SystemExit(proxy.returncode)
        return

    run = subprocess.run(
        [str(driver_bin), str(edgelist), str(args.k), str(metrics)],
        cwd=repo,
    )
    if run.returncode != 0:
        raise SystemExit(run.returncode)
    if not metrics.is_file():
        raise SystemExit(f"metrics not written: {metrics}")


if __name__ == "__main__":
    main()
''',
    )
    return cpp_path, py_path


def write_library_demo_port_stub(stubs_dir: Path, notebook: str | None) -> Path:
    """Stub that executes a demo notebook when possible, else fails clearly."""
    path = stubs_dir / _LIBRARY_PORT_NAME
    notebook_literal = repr(notebook or "")
    content = '''#!/usr/bin/env python3
"""Planner-generated demo-port helper for library papers without a paper CLI.

Stub auto-generated by Planner. Attempts nbconvert on the demo notebook and writes
metrics.csv with status only (not paper regret/error numbers).

Usage (from workspace root):
  python data/papers/<paper_id>/planner_stubs/port_demo_metrics.py \\
      --repo-root data/papers/<paper_id>/code \\
      --out results/<paper_id>/reproduce_similar/demo

Usage (from data/papers/<paper_id>/):
  python planner_stubs/port_demo_metrics.py --repo-root code \\
      --out ../../results/<paper_id>/reproduce_similar/demo

CSV columns: metric_name,value,source,notes
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

DEFAULT_NOTEBOOK = __NOTEBOOK__
CSV_FIELDS = ("metric_name", "value", "source", "notes")


def _write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def _resolve_repo_root(raw: str) -> Path:
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"--repo-root is not a directory: {root}")
    # Accept paper bundle root that contains code/.
    if not (root / "setup.py").is_file() and not (root / "pyproject.toml").is_file():
        nested = root / "code"
        if nested.is_dir():
            return nested
    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Planner stub: nbconvert demo notebook and write status metrics.csv"
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Cloned paper repo root (directory with setup.py), or paper bundle containing code/",
    )
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for metrics.csv (created if missing)",
    )
    args = parser.parse_args()

    repo = _resolve_repo_root(args.repo_root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"

    notebook = args.notebook
    nb_path = repo / notebook if notebook else None
    if nb_path is not None and not nb_path.is_file():
        # Common layout: notebook under package dir already covered; try bare name.
        alt = repo / Path(notebook).name
        if alt.is_file():
            nb_path = alt

    if nb_path is not None and nb_path.is_file() and importlib.util.find_spec("nbformat") is not None:
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(nb_path),
            "--output",
            str(out_dir / "demo_executed.ipynb"),
        ]
        result = subprocess.run(cmd, cwd=repo)
        try:
            nb_note = str(nb_path.relative_to(repo))
        except ValueError:
            nb_note = str(nb_path)
        _write_metrics(
            metrics_path,
            [
                {
                    "metric_name": "nbconvert_exit",
                    "value": str(result.returncode),
                    "source": "planner_stub",
                    "notes": f"executed {nb_note}",
                },
                {
                    "metric_name": "demo_port_status",
                    "value": "ok" if result.returncode == 0 else "failed",
                    "source": "planner_stub",
                    "notes": "status only — not paper regret/validation metrics",
                },
            ],
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        return

    _write_metrics(
        metrics_path,
        [
            {
                "metric_name": "demo_port_status",
                "value": "blocked",
                "source": "planner_stub",
                "notes": "demo_notebook_or_nbconvert_unavailable",
            },
            {
                "metric_name": "fallback",
                "value": "manual_port_required",
                "source": "planner_stub",
                "notes": (
                    "Hand-code metric extraction: read demo cells, run key BO loops "
                    "(Branin or synthetic), capture regret curves into this CSV schema"
                ),
            },
        ],
    )
    raise SystemExit(
        "Demo notebook execution unavailable. Install jupyter/nbconvert, or hand-code "
        "metric extraction from the demo notebook (Branin/synthetic BO loops → regret) "
        f"into {metrics_path} using columns metric_name,value,source,notes."
    )


if __name__ == "__main__":
    main()
'''.replace("__NOTEBOOK__", notebook_literal)
    _write_text(path, content)
    return path


def _collapse_phases(
    phases: list[PlanPhase],
    remove_ids: set[str],
) -> list[PlanPhase]:
    """Remove phases and rewire depends_on through removed nodes."""
    by_id = {phase.phase_id: phase for phase in phases}
    replacements: dict[str, list[str]] = {}
    for phase_id in remove_ids:
        phase = by_id.get(phase_id)
        if phase is None:
            continue
        replacements[phase_id] = list(phase.depends_on)

    def resolve(deps: list[str]) -> list[str]:
        resolved: list[str] = []
        for dep in deps:
            if dep in remove_ids:
                resolved.extend(resolve(replacements.get(dep, [])))
            else:
                resolved.append(dep)
        return _unique(resolved)

    kept: list[PlanPhase] = []
    for phase in phases:
        if phase.phase_id in remove_ids:
            continue
        new_deps = resolve(phase.depends_on)
        if new_deps != phase.depends_on:
            kept.append(phase.model_copy(update={"depends_on": new_deps}))
        else:
            kept.append(phase)
    return kept


def _repair_script_experiments(
    phase: PlanPhase,
    *,
    stubs_dir: Path,
    paper_id: str,
    repo_path: Path,
) -> tuple[PlanPhase | None, list[str], list[str]]:
    """Refill experiments matrix via generated CLI wrapper, or collapse."""
    notes: list[str] = []
    warnings: list[str] = []
    benchmarks = [str(item) for item in (phase.axes.get("benchmark") or [])]
    algorithms = [str(item) for item in (phase.axes.get("algorithm") or [])]
    scripts = [str(item) for item in (phase.axes.get("script") or [])]
    script = scripts[0] if scripts else "Clustering.py"
    if not benchmarks or not algorithms:
        notes.append(
            f"phase {phase.phase_id}: collapsed — no benchmark/algorithm axes to wrap."
        )
        return None, notes, warnings

    write_script_experiment_wrapper(stubs_dir)
    stub = stub_relpath(_SCRIPT_WRAPPER_NAME, repo_path)
    rows: list[PhaseRunSpec] = []
    for benchmark in benchmarks:
        for algorithm in algorithms:
            log_path = f"results/{paper_id}/experiments/{benchmark}/{algorithm}"
            rows.append(
                PhaseRunSpec(
                    name=f"exp__{benchmark}__{algorithm}",
                    variables={
                        "benchmark": benchmark,
                        "algorithm": algorithm,
                        "script": script,
                    },
                    run_command=(
                        f"python {stub} --script {script} "
                        f"--dataset {benchmark} --method {algorithm}"
                    ),
                    code_refs=_unique([stub, script]),
                    verify=["exit_code:0"],
                    results_path=log_path,
                    metrics=[],
                    source="planner_stub",
                )
            )
            if len(rows) >= PLANNER_STUB_EXAMPLE_ROWS:
                break
        if len(rows) >= PLANNER_STUB_EXAMPLE_ROWS:
            break

    warnings.append(
        f"phase {phase.phase_id}: refilled {len(rows)} row(s) via Planner stub {stub}."
    )
    notes.append(
        f"Planner wrote {PLANNER_STUBS_DIRNAME}/{_SCRIPT_WRAPPER_NAME} to expose "
        f"--dataset/--method for in-file OrderedDict scripts (primary script={script})."
    )
    return (
        phase.model_copy(
            update={
                "matrix": rows,
                "variables": ["benchmark", "algorithm", "script"],
                "axes": {
                    "benchmark": benchmarks[:PLANNER_STUB_EXAMPLE_ROWS],
                    "algorithm": algorithms[:PLANNER_STUB_EXAMPLE_ROWS],
                    "script": [script],
                },
                "run_template": (
                    f"python {stub} --script {script} --dataset {{benchmark}} "
                    "--method {algorithm}"
                ),
                "planned_actions": (
                    f"Use Planner stub `{stub}` (patches OrderedDict then runs {script}). "
                    "Do not hand-edit the script for matrix factors."
                ),
            }
        ),
        notes,
        warnings,
    )


def _repair_native_reproduce(
    phase: PlanPhase,
    *,
    stubs_dir: Path,
    paper_id: str,
    repo_path: Path,
) -> tuple[PlanPhase | None, list[str], list[str]]:
    notes: list[str] = []
    warnings: list[str] = []
    write_native_cluster_driver(stubs_dir)
    stub = stub_relpath(_NATIVE_DRIVER_PY, repo_path)
    cpp = stub_relpath(_NATIVE_DRIVER_CPP, repo_path)
    edgelist = f"results/{paper_id}/generate_inputs/sbm.edgelist"
    out_dir = f"results/{paper_id}/reproduce_similar/sbm"
    row = PhaseRunSpec(
        name="reproduce__sbm_spectral_driver",
        variables={"benchmark": "sbm", "input": edgelist},
        run_command=(
            f"python {stub} --edgelist {edgelist} --out {out_dir} --k 2"
        ),
        code_refs=_unique([stub, cpp, "stagtools/sbm.cpp", "README.md"]),
        verify=["exit_code:0", f"exists:{out_dir}/metrics.csv"],
        results_path=out_dir,
        metrics=list(phase.matrix[0].metrics) if phase.matrix else [],
        source="planner_stub",
    )
    warnings.append(f"phase {phase.phase_id}: replaced weak row with stub driver {stub}.")
    notes.append(
        f"Planner wrote {PLANNER_STUBS_DIRNAME}/{_NATIVE_DRIVER_CPP} and "
        f"{_NATIVE_DRIVER_PY}; runner compiles the README spectral_cluster sample or "
        "falls back to ctest -R cluster_test proxy metrics."
    )
    return (
        phase.model_copy(
            update={
                "matrix": [row],
                "variables": ["benchmark"],
                "axes": {"benchmark": ["sbm"]},
                "run_template": row.run_command,
                "planned_actions": (
                    f"After generate_inputs, run `{stub}` to produce metrics.csv. "
                    "Prefer compiled driver; ctest proxy is a fallback only."
                ),
            }
        ),
        notes,
        warnings,
    )


def _repair_library_reproduce(
    phase: PlanPhase,
    *,
    stubs_dir: Path,
    paper_id: str,
    exploration: dict[str, object],
    repo_path: Path,
    analyst_metrics: list[str] | None = None,
) -> tuple[PlanPhase | None, list[str], list[str]]:
    notes: list[str] = []
    warnings: list[str] = []
    notebooks = [str(item) for item in (exploration.get("notebooks") or [])]
    notebook = notebooks[0] if notebooks else None
    write_library_demo_port_stub(stubs_dir, notebook)
    stub_rel = stub_relpath(_LIBRARY_PORT_NAME, repo_path)
    # Engineer-facing command assumes workspace root (ResearchAssistant/).
    workspace_stub = f"data/papers/{paper_id}/{PLANNER_STUBS_DIRNAME}/{_LIBRARY_PORT_NAME}"
    repo_root_arg = f"data/papers/{paper_id}/code"
    out_dir = f"results/{paper_id}/reproduce_similar/demo"
    metrics = [item.strip() for item in (analyst_metrics or []) if str(item).strip()][:5]
    # Prefer metrics already present on the phase shell / prior rows if repair was given none.
    if not metrics:
        for row in phase.matrix:
            metrics = [item for item in row.metrics if str(item).strip()][:5]
            if metrics:
                break
    run_command = (
        f"python {workspace_stub} --repo-root {repo_root_arg} --out {out_dir}"
        + (f" --notebook {notebook}" if notebook else "")
    )
    code_refs = _unique(
        [
            workspace_stub,
            stub_rel,
            *([f"{repo_root_arg}/{notebook}", notebook] if notebook else []),
        ]
    )
    row = PhaseRunSpec(
        name="reproduce__demo_port",
        # Single demo path — no varying factors; notebook lives in code_refs only.
        variables={},
        run_command=run_command,
        code_refs=code_refs,
        verify=["exit_code:0", f"exists:{out_dir}/metrics.csv"],
        results_path=out_dir,
        metrics=metrics,
        source="planner_stub",
    )
    demo_name = notebook or "demo notebook"
    metric_note = ", ".join(metrics[:3]) if metrics else "Analyst evaluation_metrics"
    warnings.append(f"phase {phase.phase_id}: refilled via demo-port stub {workspace_stub}.")
    notes.append(
        f"Planner wrote {PLANNER_STUBS_DIRNAME}/{_LIBRARY_PORT_NAME} as a demo-port gate "
        f"(nbconvert on {demo_name}). CSV columns: metric_name,value,source,notes — "
        f"status only, not paper numbers. Paper-similar targets to seek later: {metric_note}. "
        "External benchmark data (PD1/HPO-B/ImageNet/…) may still be required. "
        f"Layout: repo under {repo_root_arg}/; stub under data/papers/{paper_id}/"
        f"{PLANNER_STUBS_DIRNAME}/; outputs under {out_dir}/."
    )
    return (
        phase.model_copy(
            update={
                "matrix": [row],
                "variables": [],
                "axes": {},
                "run_template": row.run_command,
                "goal": (
                    "Demo-port gate: execute the repo demo notebook via Planner stub; "
                    "confirm metrics.csv appears. This is not full paper reproduction."
                ),
                "planned_actions": (
                    f"Run `{workspace_stub}` (stub auto-generated by Planner) with "
                    f"`--repo-root {repo_root_arg} --out {out_dir}` (cwd = workspace root). "
                    f"Stub attempts nbconvert on {demo_name} → writes metrics.csv "
                    "(columns metric_name,value,source,notes) with status only. "
                    "If notebook port fails, hand-code metric extraction: read the demo cells, "
                    "run key BO loops (Branin or synthetic), capture regret curves into the same CSV. "
                    f"Paper-similar metrics to pursue after a real port: {metric_note}."
                ),
            }
        ),
        notes,
        warnings,
    )


def repair_cleared_phases(
    phases: list[PlanPhase],
    *,
    repo_path: Path | str | None,
    exploration: dict[str, object] | None,
    paper_id: str,
    analyst_metrics: list[str] | None = None,
) -> tuple[list[PlanPhase], list[str], list[str], list[str]]:
    """Repair or collapse phases left with empty matrices after verification.

    Returns (phases, missing_notes, warnings, stub_paths_written).
    """
    root = Path(repo_path) if repo_path else None
    if root is not None and not root.is_dir():
        root = None
    exploration = exploration if isinstance(exploration, dict) else {}
    surface = str(exploration.get("execution_surface") or "")
    stubs_dir = stubs_dir_for_repo(root)
    metrics = [str(item).strip() for item in (analyst_metrics or []) if str(item).strip()]

    missing: list[str] = []
    warnings: list[str] = []
    stub_paths: list[str] = []
    collapse_ids: set[str] = set()
    rebuilt: dict[str, PlanPhase] = {}

    for phase in phases:
        # Always upgrade weak native reproduce (ls-only) when we can write a driver.
        weak_native = (
            surface == "native"
            and phase.phase_id == "reproduce_similar"
            and phase.matrix
            and all(
                "test -s" in (row.run_command or "") or (row.run_command or "").startswith("#")
                for row in phase.matrix
            )
        )
        needs_repair = (not phase.matrix and phase.phase_id not in {"setup", "summarize"}) or weak_native
        if not needs_repair:
            continue

        if stubs_dir is None:
            collapse_ids.add(phase.phase_id)
            missing.append(
                f"phase {phase.phase_id}: collapsed — no repo path to write Planner stubs."
            )
            continue

        if phase.phase_id.startswith("ablation_"):
            collapse_ids.add(phase.phase_id)
            axes_preview = ", ".join(
                f"{key}={values[:4]}" for key, values in list(phase.axes.items())[:3]
            )
            missing.append(
                f"phase {phase.phase_id}: collapsed (blocked) — ablation factors lack CLI "
                f"flags. Intent retained in missing_context only: {axes_preview}"
            )
            continue

        if phase.phase_id == "experiments" and surface == "script":
            repaired, notes, warns = _repair_script_experiments(
                phase, stubs_dir=stubs_dir, paper_id=paper_id, repo_path=root
            )
            missing.extend(notes)
            warnings.extend(warns)
            if repaired is None:
                collapse_ids.add(phase.phase_id)
            else:
                rebuilt[phase.phase_id] = repaired
                stub_paths.append(str(stubs_dir / _SCRIPT_WRAPPER_NAME))
            continue

        if phase.phase_id == "reproduce_similar" and surface == "native":
            repaired, notes, warns = _repair_native_reproduce(
                phase, stubs_dir=stubs_dir, paper_id=paper_id, repo_path=root
            )
            missing.extend(notes)
            warnings.extend(warns)
            if repaired is None:
                collapse_ids.add(phase.phase_id)
            else:
                rebuilt[phase.phase_id] = repaired
                stub_paths.extend(
                    [
                        str(stubs_dir / _NATIVE_DRIVER_CPP),
                        str(stubs_dir / _NATIVE_DRIVER_PY),
                    ]
                )
            continue

        if phase.phase_id == "reproduce_similar" and surface == "library":
            repaired, notes, warns = _repair_library_reproduce(
                phase,
                stubs_dir=stubs_dir,
                paper_id=paper_id,
                exploration=exploration,
                repo_path=root,
                analyst_metrics=metrics,
            )
            missing.extend(notes)
            warnings.extend(warns)
            if repaired is None:
                collapse_ids.add(phase.phase_id)
            else:
                rebuilt[phase.phase_id] = repaired
                stub_paths.append(str(stubs_dir / _LIBRARY_PORT_NAME))
            continue

        # Default: collapse empty non-runnable phase shells.
        collapse_ids.add(phase.phase_id)
        missing.append(
            f"phase {phase.phase_id}: collapsed (blocked) — verification left an empty "
            "matrix and no stub repair applies for this surface."
        )

    working = [rebuilt.get(phase.phase_id, phase) for phase in phases]
    if collapse_ids:
        working = _collapse_phases(working, collapse_ids)
        warnings.append(
            f"collapsed {len(collapse_ids)} blocked empty phase(s): "
            + ", ".join(sorted(collapse_ids))
        )

    return working, _unique(missing), _unique(warnings), _unique(stub_paths)
