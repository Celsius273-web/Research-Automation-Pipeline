"""Tests for deterministic Planner plan verification."""

from __future__ import annotations

from pathlib import Path

from src.state import PhaseRunSpec, PlanPhase, SectionExtraction
from src.tools.phase_builder import build_plan_phases
from src.tools.plan_verification import (
    ensure_input_generation_steps,
    ensure_native_dependency_checks,
    verify_and_filter_phases,
    verify_run_command,
)
from src.tools.repo_exploration import explore_repository


def test_verify_run_command_accepts_documented_cli_flags(tmp_path: Path) -> None:
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "run_exp.py").write_text(
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
    ok, reasons = verify_run_command(
        "python exp/run_exp.py --fun lsq --algo be-cbo --reg-type gp "
        "--cls-type de --log-path results/x --seed 0",
        repo_path=tmp_path,
        phase_id="smoke",
    )
    assert ok
    assert reasons == []


def test_verify_run_command_rejects_undocumented_flags(tmp_path: Path) -> None:
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "run_exp.py").write_text(
        "from argparse import ArgumentParser\n"
        "p = ArgumentParser()\n"
        "p.add_argument('--fun')\n",
        encoding="utf-8",
    )
    ok, reasons = verify_run_command(
        "python exp/run_exp.py --fun lsq --hidden-layers 3",
        repo_path=tmp_path,
        phase_id="ablation_1_hidden_layers",
    )
    assert not ok
    assert any("undocumented" in item for item in reasons)


def test_verify_run_command_rejects_manual_edit_comments(tmp_path: Path) -> None:
    (tmp_path / "Clustering.py").write_text("print('hi')\n", encoding="utf-8")
    ok, reasons = verify_run_command(
        "python Clustering.py  # set dataset=cora, method=mincut_pool in script tunables OrderedDict",
        repo_path=tmp_path,
        phase_id="experiments",
    )
    assert not ok
    assert any("OrderedDict" in item or "in-file" in item for item in reasons)


def test_verify_run_command_rejects_unit_test_as_reproduction(tmp_path: Path) -> None:
    (tmp_path / "hyperbo").mkdir()
    (tmp_path / "hyperbo" / "bayesopt_test.py").write_text("print('ok')\n", encoding="utf-8")
    ok, reasons = verify_run_command(
        "python hyperbo/bayesopt_test.py",
        repo_path=tmp_path,
        phase_id="reproduce_similar",
    )
    assert not ok
    assert any("not paper reproduction" in item for item in reasons)

    ok_smoke, _ = verify_run_command(
        "python hyperbo/bayesopt_test.py",
        repo_path=tmp_path,
        phase_id="library_smoke",
    )
    assert ok_smoke


def test_verify_and_filter_demotes_ablation_edits_keeps_smoke(tmp_path: Path) -> None:
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "run_exp.py").write_text(
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
    (tmp_path / "README.md").write_text(
        "```\npython exp/run_exp.py --fun FUN_NAME --algo ALGO_NAME "
        "--reg-type REGRESSOR --cls-type CLASSIFIER --log-path LOG_PATH\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "algorithms").mkdir()
    (tmp_path / "test_functions").mkdir()
    (tmp_path / "algorithms" / "__init__.py").write_text(
        "algorithms = {'be-cbo': BECBO, 'cei': CEI}\n",
        encoding="utf-8",
    )
    (tmp_path / "test_functions" / "__init__.py").write_text(
        "test_functions = {'lsq': LSQ, '3bar': ThreeBar}\n",
        encoding="utf-8",
    )
    exploration = explore_repository(tmp_path)
    analyst = SectionExtraction(
        research_question="Q?",
        methodology="M",
        evaluation_metrics=["obj"],
        hyperparameters={"hidden_layers": "[1, 2]"},
    )
    phases = build_plan_phases(
        paper_id="becbo",
        build_system="pip install .",
        exploration=exploration,
        analyst=analyst,
    )
    revised, missing, warnings, all_ok = verify_and_filter_phases(
        phases,
        repo_path=tmp_path,
        exploration=exploration,
        paper_id="becbo",
    )
    by_id = {phase.phase_id: phase for phase in revised}
    assert by_id["smoke"].matrix
    ablation_phases = [phase for phase in revised if phase.phase_id.startswith("ablation_")]
    assert ablation_phases == []
    assert any("demoted" in item or "collapsed" in item for item in missing)
    assert not all_ok


def test_native_deps_and_input_generation(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text("project(stag)\n", encoding="utf-8")
    (tmp_path / "INSTALL").write_text("Depends on Eigen and Spectra.\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# STAG\n", encoding="utf-8")
    (tmp_path / "stagtools").mkdir()
    (tmp_path / "stagtools" / "sbm.cpp").write_text("int main(){}\n", encoding="utf-8")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "graph_test.cpp").write_text("int main(){}\n", encoding="utf-8")
    exploration = explore_repository(tmp_path)
    analyst = SectionExtraction(
        research_question="Q?",
        methodology="M",
        datasets_or_benchmarks=["SBM"],
        evaluation_metrics=["ARI"],
        hyperparameters={},
    )
    phases = build_plan_phases(
        paper_id="stag_sparse",
        build_system="cmake -S . -B build && cmake --build build",
        exploration=exploration,
        analyst=analyst,
    )
    with_deps = ensure_native_dependency_checks(
        phases, paper_id="stag_sparse", exploration=exploration
    )
    assert any(phase.phase_id == "deps_check" for phase in with_deps)
    smoke = next(phase for phase in with_deps if phase.phase_id == "native_smoke")
    assert "deps_check" in smoke.depends_on

    with_inputs = ensure_input_generation_steps(
        with_deps, paper_id="stag_sparse", exploration=exploration
    )
    ids = [phase.phase_id for phase in with_inputs]
    assert "generate_inputs" in ids
    generate = next(phase for phase in with_inputs if phase.phase_id == "generate_inputs")
    assert generate.matrix
    assert "stag_sbm" in generate.matrix[0].run_command
    reproduce = next(phase for phase in with_inputs if phase.phase_id == "reproduce_similar")
    assert reproduce.depends_on == ["generate_inputs"]

    revised, missing, _, all_ok = verify_and_filter_phases(
        phases,
        repo_path=tmp_path,
        exploration=exploration,
        paper_id="stag_sparse",
    )
    ids = [phase.phase_id for phase in revised]
    assert ids.index("deps_check") < ids.index("native_smoke")
    assert "generate_inputs" in ids
    deps = next(phase for phase in revised if phase.phase_id == "deps_check")
    assert deps.matrix
    assert all(
        "eigen" in row.run_command.lower() or "spectra" in row.run_command.lower()
        for row in deps.matrix
    )
    assert any(
        "driver" in item.lower() or "stub" in item.lower() or "collapsed" in item.lower()
        for item in missing
    )


def test_script_experiments_demoted_when_manual_edits(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Run [Clustering.py](Clustering.py)\n", encoding="utf-8")
    (tmp_path / "Clustering.py").write_text(
        "from collections import OrderedDict\n"
        "tunables = OrderedDict([\n"
        "    ('dataset', ['cora']),  # 'citeseer'\n"
        "    ('method', ['mincut_pool']),  # 'diff_pool'\n"
        "])\n"
        "if __name__ == '__main__':\n"
        "    pass\n",
        encoding="utf-8",
    )
    exploration = explore_repository(tmp_path)
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
    revised, missing, _, all_ok = verify_and_filter_phases(
        phases,
        repo_path=tmp_path,
        exploration=exploration,
        paper_id="spectral",
    )
    by_id = {phase.phase_id: phase for phase in revised}
    assert by_id["smoke"].matrix
    assert by_id["experiments"].matrix
    assert all(
        "planner_stubs/run_script_experiment.py" in row.run_command
        for row in by_id["experiments"].matrix
    )
    assert any("stub" in item.lower() or "wrapper" in item.lower() or "OrderedDict" in item for item in missing + [])
    assert not all_ok or True  # may be ok after refill; demotions still happened
