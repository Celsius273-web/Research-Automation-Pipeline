"""Tests for Planner empty-matrix repair and stub emission."""

from __future__ import annotations

from pathlib import Path

from src.state import SectionExtraction
from src.tools.phase_builder import build_plan_phases
from src.tools.plan_repair import (
    repair_cleared_phases,
    stubs_dir_for_repo,
    write_script_experiment_wrapper,
)
from src.tools.plan_verification import verify_and_filter_phases, verify_run_command
from src.tools.repo_exploration import explore_repository


def test_script_wrapper_stub_is_verified(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "Clustering.py").write_text(
        "from collections import OrderedDict\n"
        "tunables = OrderedDict([\n"
        "    ('dataset', ['cora']),\n"
        "    ('method', ['mincut_pool']),\n"
        "])\n"
        "if __name__ == '__main__':\n"
        "    print('ok')\n",
        encoding="utf-8",
    )
    (code / "README.md").write_text("Run Clustering.py\n", encoding="utf-8")
    stubs = stubs_dir_for_repo(code)
    assert stubs is not None
    write_script_experiment_wrapper(stubs)
    ok, reasons = verify_run_command(
        "python ../planner_stubs/run_script_experiment.py "
        "--script Clustering.py --dataset cora --method mincut_pool",
        repo_path=code,
        phase_id="experiments",
    )
    assert ok, reasons


def test_verify_and_filter_refills_script_experiments(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "README.md").write_text("Run [Clustering.py](Clustering.py)\n", encoding="utf-8")
    (code / "Clustering.py").write_text(
        "from collections import OrderedDict\n"
        "tunables = OrderedDict([\n"
        "    ('dataset', ['cora']),  # 'citeseer'\n"
        "    ('method', ['mincut_pool']),  # 'diff_pool'\n"
        "])\n"
        "if __name__ == '__main__':\n"
        "    pass\n",
        encoding="utf-8",
    )
    exploration = explore_repository(code)
    phases = build_plan_phases(
        paper_id="spectral",
        build_system="pip install .",
        exploration=exploration,
        analyst=SectionExtraction(
            research_question="Q?",
            methodology="M",
            evaluation_metrics=["NMI"],
            hyperparameters={},
        ),
    )
    revised, missing, warnings, _ = verify_and_filter_phases(
        phases,
        repo_path=code,
        exploration=exploration,
        paper_id="spectral",
    )
    by_id = {phase.phase_id: phase for phase in revised}
    assert "smoke" in by_id
    assert by_id["smoke"].matrix
    assert "experiments" in by_id
    assert by_id["experiments"].matrix, missing
    assert all(
        "planner_stubs/run_script_experiment.py" in row.run_command
        for row in by_id["experiments"].matrix
    )
    assert (tmp_path / "planner_stubs" / "run_script_experiment.py").is_file()
    assert any("stub" in item.lower() for item in warnings)


def test_verify_and_filter_collapses_empty_ablations(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "exp").mkdir()
    (code / "exp" / "run_exp.py").write_text(
        "from argparse import ArgumentParser\n"
        "p = ArgumentParser()\n"
        "p.add_argument('--fun')\n"
        "p.add_argument('--algo')\n"
        "p.add_argument('--reg-type')\n"
        "p.add_argument('--cls-type')\n"
        "p.add_argument('--seed')\n"
        "p.add_argument('--log-path')\n",
        encoding="utf-8",
    )
    (code / "README.md").write_text(
        "```\npython exp/run_exp.py --fun FUN_NAME --algo ALGO_NAME "
        "--reg-type REGRESSOR --cls-type CLASSIFIER --log-path LOG_PATH\n```\n",
        encoding="utf-8",
    )
    (code / "algorithms").mkdir()
    (code / "test_functions").mkdir()
    (code / "algorithms" / "__init__.py").write_text(
        "algorithms = {'be-cbo': BECBO, 'cei': CEI}\n",
        encoding="utf-8",
    )
    (code / "test_functions" / "__init__.py").write_text(
        "test_functions = {'lsq': LSQ, '3bar': ThreeBar}\n",
        encoding="utf-8",
    )
    exploration = explore_repository(code)
    phases = build_plan_phases(
        paper_id="becbo",
        build_system="pip install .",
        exploration=exploration,
        analyst=SectionExtraction(
            research_question="Q?",
            methodology="M",
            evaluation_metrics=["obj"],
            hyperparameters={"hidden_layers": "[1, 2]"},
        ),
    )
    revised, missing, _, _ = verify_and_filter_phases(
        phases,
        repo_path=code,
        exploration=exploration,
        paper_id="becbo",
    )
    ids = [phase.phase_id for phase in revised]
    assert "smoke" in ids
    assert not any(phase_id.startswith("ablation_") for phase_id in ids)
    summarize = next(phase for phase in revised if phase.phase_id == "summarize")
    assert "ablation_" not in " ".join(summarize.depends_on)
    assert any("collapsed" in item and "ablation" in item for item in missing)
    assert all(
        phase.matrix or phase.phase_id in {"setup", "summarize"} for phase in revised
    )


def test_native_repair_writes_driver_stub(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "CMakeLists.txt").write_text("project(stag)\n", encoding="utf-8")
    (code / "INSTALL").write_text("Depends on Eigen and Spectra.\n", encoding="utf-8")
    (code / "README.md").write_text("# STAG\n", encoding="utf-8")
    (code / "stagtools").mkdir()
    (code / "stagtools" / "sbm.cpp").write_text("int main(){}\n", encoding="utf-8")
    (code / "test").mkdir()
    (code / "test" / "graph_test.cpp").write_text("int main(){}\n", encoding="utf-8")
    exploration = explore_repository(code)
    phases = build_plan_phases(
        paper_id="stag_sparse",
        build_system="cmake -S . -B build && cmake --build build",
        exploration=exploration,
        analyst=SectionExtraction(
            research_question="Q?",
            methodology="M",
            datasets_or_benchmarks=["SBM"],
            evaluation_metrics=["ARI"],
            hyperparameters={},
        ),
    )
    revised, missing, warnings, _ = verify_and_filter_phases(
        phases,
        repo_path=code,
        exploration=exploration,
        paper_id="stag_sparse",
    )
    by_id = {phase.phase_id: phase for phase in revised}
    assert "deps_check" in by_id
    assert "generate_inputs" in by_id
    assert "reproduce_similar" in by_id
    assert by_id["reproduce_similar"].matrix
    assert "run_stag_cluster_driver.py" in by_id["reproduce_similar"].matrix[0].run_command
    assert (tmp_path / "planner_stubs" / "run_stag_cluster_driver.py").is_file()
    assert (tmp_path / "planner_stubs" / "stag_spectral_cluster_driver.cpp").is_file()
    assert any("driver" in item.lower() or "stub" in item.lower() for item in missing + warnings)


def test_repair_cleared_phases_library_demo_stub(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "setup.py").write_text("from setuptools import setup\nsetup(name='x')\n", encoding="utf-8")
    (code / "README.md").write_text("# lib\n", encoding="utf-8")
    (code / "demo.ipynb").write_text(
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
        encoding="utf-8",
    )
    from src.state import PlanPhase

    phases = [
        PlanPhase(phase_id="setup", title="Setup"),
        PlanPhase(
            phase_id="reproduce_similar",
            title="Reproduce",
            depends_on=["setup"],
            axes={"notebook": ["demo.ipynb"]},
            matrix=[],
        ),
        PlanPhase(phase_id="summarize", title="Sum", depends_on=["reproduce_similar"]),
    ]
    repaired, notes, warnings, stubs = repair_cleared_phases(
        phases,
        repo_path=code,
        exploration={"execution_surface": "library", "notebooks": ["demo.ipynb"]},
        paper_id="lib_paper",
        analyst_metrics=["Simple Regret", "Validation Error Rate (%)"],
    )
    by_id = {phase.phase_id: phase for phase in repaired}
    assert by_id["reproduce_similar"].matrix
    row = by_id["reproduce_similar"].matrix[0]
    assert "port_demo_metrics.py" in row.run_command
    assert "--repo-root" in row.run_command
    assert "--out results/lib_paper/reproduce_similar/demo" in row.run_command
    assert row.variables == {}
    assert "stub" not in row.variables
    assert "notebook" not in by_id["reproduce_similar"].variables
    assert by_id["reproduce_similar"].axes == {}
    assert row.metrics == ["Simple Regret", "Validation Error Rate (%)"]
    assert "not paper" in by_id["reproduce_similar"].planned_actions.lower() or (
        "not full paper" in by_id["reproduce_similar"].goal.lower()
    )
    assert "hand-code" in by_id["reproduce_similar"].planned_actions.lower()
    assert "Simple Regret" in by_id["reproduce_similar"].planned_actions
    assert any("data/papers/lib_paper" in ref for ref in row.code_refs)
    stub_path = Path(stubs[0])
    stub_src = stub_path.read_text(encoding="utf-8")
    assert "metric_name" in stub_src and "source" in stub_src and "notes" in stub_src
    assert "--repo-root" in stub_src
    assert stubs
    assert notes
    assert any("demo-port gate" in item or "nbconvert" in item for item in notes)
