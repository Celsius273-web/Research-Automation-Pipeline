from __future__ import annotations

from pathlib import Path

from src.state import SectionExtraction
from src.tools.phase_builder import (
    annotate_ungrounded_factor,
    build_plan_phases,
    command_grounds_factor,
    extract_ablation_axes,
    parse_analyst_list_values,
    selectable_benchmark_labels,
    split_benchmarks,
)
from src.tools.repo_exploration import explore_repository


def test_split_benchmarks_synthetic_vs_real() -> None:
    synthetic, real = split_benchmarks(
        ["lsq", "tow-shift", "3bar", "beam", "sim"]
    )
    assert synthetic == ["lsq", "tow-shift", "sim"]
    assert real == ["3bar", "beam"]


def test_parse_analyst_list_and_ablations() -> None:
    assert parse_analyst_list_values("[1, 2, 3, 4]") == [1, 2, 3, 4]
    axes = extract_ablation_axes(
        {
            "hidden_layers": "[1, 2, 3, 4]",
            "neurons_factor_c": "[16, 32, 64]",
            "learning_rate": "3e-4",
        }
    )
    names = [name for name, _ in axes]
    assert "hidden_layers" in names
    assert "neuron_scale_C" in names


def test_selectable_benchmark_labels_drops_junk_prose() -> None:
    labels = selectable_benchmark_labels(
        [
            "ImageNet ResNet50 1024",
            "HPO-B (Hyperparameter Optimization Benchmark) dataset",
            "PD1 online tuning tasks",
            "Real-world problems including hyperparameter tuning",
            "Multi-task hyperparameter tuning benchmark",
            "Generic cases with non-matching input locations",
            "Neo4j Movies database",
            "Stochastic Block Model graphs of varying sizes",
            "EdgeList format datasets",
            "wiki-topcats dataset (1,791,489 vertices)",
        ]
    )
    assert "imagenet" in labels
    assert "hpo_b" in labels
    assert "pd1" in labels
    assert "sbm" in labels
    assert "edgelist" in labels
    assert "wiki-topcats" in labels
    assert "Real-world" not in labels
    assert "Generic" not in labels
    assert "Neo4j" not in labels
    assert not any(label.lower() in {"real-world", "multi-task", "generic"} for label in labels)


def test_command_grounds_factor_ignores_results_path_embeddings() -> None:
    cmd = (
        "python exp/run_exp.py --fun lsq --algo be-cbo --reg-type gp --cls-type de "
        "--log-path results/paper/ablation_1_hidden_layers/lsq/hidden_layers=1/0 --seed 0"
    )
    assert not command_grounds_factor(cmd, "hidden_layers", 1)
    annotated = annotate_ungrounded_factor(cmd, axis_name="hidden_layers", value=1)
    assert "# set hidden_layers=1 in code/config" in annotated
    flagged = cmd.replace("--seed 0", "--hidden-layers 1 --seed 0")
    assert command_grounds_factor(flagged, "hidden_layers", 1)
    assert annotate_ungrounded_factor(flagged, axis_name="hidden_layers", value=1) == flagged


def test_build_plan_phases_for_becbo_like_repo(tmp_path: Path) -> None:
    (tmp_path / "algorithms").mkdir()
    (tmp_path / "test_functions").mkdir()
    (tmp_path / "exp").mkdir()
    (tmp_path / "README.md").write_text(
        "# Demo\n\n```\npython exp/run_exp.py --fun FUN_NAME --algo ALGO_NAME "
        "--reg-type REGRESSOR --cls-type CLASSIFIER --log-path LOG_PATH\n```\n"
        "For classifier, we support `gp` and `de`.\n",
        encoding="utf-8",
    )
    (tmp_path / "exp" / "run_exp.py").write_text(
        "from argparse import ArgumentParser\n"
        "parser = ArgumentParser()\n"
        "parser.add_argument('--fun')\n"
        "parser.add_argument('--algo')\n"
        "parser.add_argument('--reg-type')\n"
        "parser.add_argument('--cls-type')\n"
        "parser.add_argument('--seed')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_functions" / "__init__.py").write_text(
        "test_functions = {\n"
        "  'lsq': LSQ, 'tow': Townsend, 'sim': Simionescu,\n"
        "  '3bar': ThreeBar, 'beam': Beam, 'ten': Ten,\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "algorithms" / "__init__.py").write_text(
        "algorithms = {'be-cbo': BECBO, 'cei': CEI, 'scbo-t-re': SCBO}\n",
        encoding="utf-8",
    )

    exploration = explore_repository(tmp_path)
    analyst = SectionExtraction(
        research_question="How to optimize under unknown constraints?",
        methodology="BE-CBO with deep ensembles",
        evaluation_metrics=["Best objective function value found ($f(x^*)$)"],
        hyperparameters={
            "hidden_layers": "[1, 2, 3, 4]",
            "neurons_factor_c": "[16, 32, 64, 128]",
            "learning_rates": "[3e-3, 1e-3, 3e-4]",
        },
    )
    phases = build_plan_phases(
        paper_id="boundary_exploration_bo",
        build_system="pip install numpy torch",
        exploration=exploration,
        analyst=analyst,
    )
    ids = [phase.phase_id for phase in phases]
    assert ids[0] == "setup"
    assert "smoke" in ids
    assert "synthetic" in ids
    assert "real_world" in ids
    assert any(phase_id.startswith("ablation_") for phase_id in ids)
    assert ids[-1] == "summarize"

    synthetic = next(phase for phase in phases if phase.phase_id == "synthetic")
    assert synthetic.variables == ["benchmark", "algorithm", "seed"]
    assert set(synthetic.axes["benchmark"]) <= {"lsq", "tow", "sim"}
    assert len(synthetic.matrix) >= 1
    row = synthetic.matrix[0]
    assert "python exp/run_exp.py" in row.run_command
    assert row.code_refs
    assert row.verify
    assert row.variables.get("benchmark")
    assert row.variables.get("algorithm")

    by_id = {phase.phase_id: phase for phase in phases}
    assert by_id["smoke"].depends_on == ["setup"]
    assert "smoke" in by_id["synthetic"].depends_on

    ablation = next(phase for phase in phases if phase.phase_id.startswith("ablation_"))
    assert ablation.matrix
    assert all(
        "# set " in row.run_command and "in code/config" in row.run_command
        for row in ablation.matrix
    )
    assert all(
        not command_grounds_factor(
            row.run_command.split("  #", 1)[0],
            ablation.variables[0],
            row.variables[ablation.variables[0]],
        )
        for row in ablation.matrix
    )


def test_build_library_phases_for_notebook_repo(tmp_path: Path) -> None:
    (tmp_path / "hyperbo").mkdir()
    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup(name='hyperbo')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# HyperBO\n\nFollow the Jupyter Notebook.\n\n```\npip install git+https://example.com/hyperbo.git\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "hyperbo" / "bayesopt_test.py").write_text(
        "if __name__ == '__main__':\n    print('ok')\n",
        encoding="utf-8",
    )
    (tmp_path / "hyperbo" / "gp_test.py").write_text(
        "if __name__ == '__main__':\n    print('ok')\n",
        encoding="utf-8",
    )
    (tmp_path / "hyperbo" / "hyperbo_demo.ipynb").write_text(
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
        encoding="utf-8",
    )

    exploration = explore_repository(tmp_path)
    assert exploration["execution_surface"] == "library"
    assert exploration["test_files"]
    assert exploration["notebooks"]

    analyst = SectionExtraction(
        research_question="Can pre-trained GPs improve BO?",
        methodology="Pre-train GP priors then run BO",
        datasets_or_benchmarks=[
            "HPO-B",
            "PD1",
            "Real-world problems including hyperparameter tuning",
            "Generic cases with non-matching input locations",
        ],
        evaluation_metrics=["Regret compared to random search and other BO baselines"],
        hyperparameters={},
    )
    phases = build_plan_phases(
        paper_id="pretrained_gp_bo",
        build_system='pip install "git+https://example.com/hyperbo.git"',
        exploration=exploration,
        analyst=analyst,
    )
    ids = [phase.phase_id for phase in phases]
    assert ids[0] == "setup"
    assert "library_smoke" in ids
    assert "reproduce_similar" in ids
    assert ids[-1] == "summarize"
    smoke = next(phase for phase in phases if phase.phase_id == "library_smoke")
    assert smoke.matrix
    assert all(row.run_command.startswith("python ") for row in smoke.matrix)
    assert all(row.verify for row in smoke.matrix)
    assert all(row.metrics for row in smoke.matrix)
    assert "Regret" in smoke.matrix[0].metrics[0] or "paper targets" in smoke.planned_actions
    assert "does NOT produce" in smoke.planned_actions or "API validation only" in smoke.planned_actions
    reproduce = next(phase for phase in phases if phase.phase_id == "reproduce_similar")
    assert "target_benchmark" not in reproduce.variables
    assert "target_benchmark" not in reproduce.axes
    assert all(set(row.variables) <= {"test_module", "notebook"} for row in reproduce.matrix)
    assert all(row.metrics for row in reproduce.matrix)
    assert not any("metrics.csv" in item for row in reproduce.matrix for item in row.verify)
    assert not any("summary.json" in item for row in reproduce.matrix for item in row.verify)
    assert "Analyst" in reproduce.planned_actions or "reported" in reproduce.planned_actions.lower()
    cmds = {row.run_command for row in reproduce.matrix}
    # Each distinct command appears once — no decorative benchmark cartesian.
    assert len(cmds) == len(reproduce.matrix)


def test_build_script_phases_with_tunables(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "Run [Clustering.py](Clustering.py)\n",
        encoding="utf-8",
    )
    (tmp_path / "Clustering.py").write_text(
        "from collections import OrderedDict\n"
        "tunables = OrderedDict([\n"
        "    ('dataset', ['cora']),  # 'cora', 'citeseer'\n"
        "    ('method', ['mincut_pool']),  # 'mincut_pool', 'diff_pool'\n"
        "])\n"
        "if __name__ == '__main__':\n"
        "    pass\n",
        encoding="utf-8",
    )
    exploration = explore_repository(tmp_path)
    analyst = SectionExtraction(
        research_question="Spectral clustering with GNNs?",
        methodology="MinCut pooling",
        evaluation_metrics=["NMI"],
        hyperparameters={},
    )
    phases = build_plan_phases(
        paper_id="spectral_clustering_gnn",
        build_system="pip install -r requirements.txt",
        exploration=exploration,
        analyst=analyst,
    )
    ids = [phase.phase_id for phase in phases]
    assert ids == ["setup", "smoke", "experiments", "summarize"]
    experiments = next(phase for phase in phases if phase.phase_id == "experiments")
    assert "benchmark" in experiments.axes or "script" in experiments.axes
    assert experiments.matrix
    assert any("Clustering.py" in row.run_command for row in experiments.matrix)


def test_build_native_phases_with_cmake_smoke(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text("project(stag)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# STAG\n", encoding="utf-8")
    (tmp_path / "stagtools").mkdir()
    (tmp_path / "stagtools" / "sbm.cpp").write_text("int main(){}\n", encoding="utf-8")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "graph_test.cpp").write_text("int main(){}\n", encoding="utf-8")
    exploration = explore_repository(tmp_path)
    analyst = SectionExtraction(
        research_question="Sparse spectral clustering?",
        methodology="STAG C++ library",
        datasets_or_benchmarks=[
            "SBM",
            "EdgeList format datasets",
            "Neo4j Movies database",
            "wiki-topcats dataset (1,791,489 vertices)",
        ],
        evaluation_metrics=["ARI"],
        hyperparameters={},
    )
    phases = build_plan_phases(
        paper_id="stag_sparse",
        build_system="cmake -S . -B build && cmake --build build",
        exploration=exploration,
        analyst=analyst,
    )
    ids = [phase.phase_id for phase in phases]
    assert ids[0] == "setup"
    assert "native_smoke" in ids
    assert "reproduce_similar" in ids
    assert ids[-1] == "summarize"
    smoke = next(phase for phase in phases if phase.phase_id == "native_smoke")
    assert smoke.matrix
    assert any("ctest" in row.run_command or "make" in row.run_command for row in smoke.matrix)
    reproduce = next(phase for phase in phases if phase.phase_id == "reproduce_similar")
    assert all(
        row.run_command.strip() and not row.run_command.startswith("#")
        for row in reproduce.matrix
    )
    assert reproduce.axes.get("benchmark") == ["sbm"]
    assert all(row.variables.get("benchmark") == "sbm" for row in reproduce.matrix)
    assert "wiki-topcats" in reproduce.planned_actions or "edgelist" in reproduce.planned_actions.lower()


def test_build_config_phases_matrix(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.yaml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "configs" / "b.yaml").write_text("x: 2\n", encoding="utf-8")
    (tmp_path / "train.py").write_text("if __name__ == '__main__':\n  pass\n", encoding="utf-8")
    exploration = {
        "execution_surface": "config",
        "config_files": ["configs/a.yaml", "configs/b.yaml"],
        "script_entrypoints": ["train.py"],
        "example_commands": ["python train.py"],
        "readme_file": "README.md",
    }
    analyst = SectionExtraction(
        research_question="Q?",
        methodology="M",
        evaluation_metrics=["acc"],
        hyperparameters={},
    )
    phases = build_plan_phases(
        paper_id="cfg_paper",
        build_system="pip install .",
        exploration=exploration,
        analyst=analyst,
    )
    ids = [phase.phase_id for phase in phases]
    assert "smoke" in ids
    assert "experiments" in ids
    experiments = next(phase for phase in phases if phase.phase_id == "experiments")
    assert "config" in experiments.axes
    assert len(experiments.matrix) >= 1


def test_unknown_surface_never_returns_empty_phases() -> None:
    analyst = SectionExtraction(
        research_question="Q?",
        methodology="M",
        evaluation_metrics=[],
        hyperparameters={},
    )
    phases = build_plan_phases(
        paper_id="empty_ish",
        build_system="unknown",
        exploration={"execution_surface": "unknown", "available": True},
        analyst=analyst,
    )
    assert phases
    ids = [phase.phase_id for phase in phases]
    assert ids[0] == "setup"
    assert "missing_context" in ids
