from __future__ import annotations

from pathlib import Path

from src.tools.repo_context import extract_entrypoint_hints, extract_example_commands
from src.tools.repo_exploration import explore_repository


def test_explore_repository_reads_readme_tree_and_registries(tmp_path: Path) -> None:
    (tmp_path / "algorithms").mkdir()
    (tmp_path / "test_functions").mkdir()
    (tmp_path / "exp").mkdir()
    (tmp_path / "README.md").write_text(
        "# Demo\n\nRun experiments with:\n\n"
        "```\npython exp/run_exp.py --fun FUN_NAME --algo ALGO_NAME --reg-type REGRESSOR --cls-type CLASSIFIER --log-path LOG_PATH\n```\n"
        "For regressor, we support `gp` only. For classifier, we support `gp` and `de`.\n",
        encoding="utf-8",
    )
    (tmp_path / "exp" / "run_exp.py").write_text(
        "from argparse import ArgumentParser\n"
        "parser = ArgumentParser()\n"
        "parser.add_argument('--fun', type=str)\n"
        "parser.add_argument('--algo', type=str)\n"
        "parser.add_argument('--reg-type', type=str)\n"
        "parser.add_argument('--cls-type', type=str)\n"
        "print('run')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_functions" / "__init__.py").write_text(
        "test_functions = {\n    'tow': Townsend,\n    'sim': Simionescu,\n    'lsq': LSQ,\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "algorithms" / "__init__.py").write_text(
        "algorithms = {\n    'be-cbo': BECBO,\n    'cei': CEI,\n    'random-sobol': RandomSobol,\n}\n",
        encoding="utf-8",
    )

    exploration = explore_repository(tmp_path)

    assert exploration["available"] is True
    assert "python exp/run_exp.py --fun FUN_NAME --algo ALGO_NAME" in exploration["example_commands"][0]
    assert "tow" in exploration["registry_ids"]["functions_or_benchmarks"]
    assert "be-cbo" in exploration["registry_ids"]["algorithms_or_methods"]
    assert any(path.endswith("/") for path in exploration["file_tree_deep"])
    assert any(item["path"] == "exp/run_exp.py" for item in exploration["source_excerpts"])
    assert "Run experiments with" in exploration["readme_full"]
    assert "reg_type" in exploration["cli_flags"]
    candidates = exploration["experiment_candidates"]
    assert len(candidates) >= 6
    assert candidates[0]["method"] == "be-cbo"
    assert "--fun tow" in candidates[0]["run_command"] or "--fun lsq" in candidates[0]["run_command"] or "--fun sim" in candidates[0]["run_command"]
    assert "--algo be-cbo" in candidates[0]["run_command"]
    assert candidates[0]["hyperparameters"]["cls_type"] == "de"


def test_extract_example_commands_from_markdown_run_links(tmp_path: Path) -> None:
    (tmp_path / "Clustering.py").write_text("print('cluster')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "## Clustering\n\n"
        "Run [Clustering.py](https://example.com/Clustering.py) to cluster citation networks.\n",
        encoding="utf-8",
    )

    assert extract_example_commands(tmp_path) == ["python Clustering.py"]
    assert "Clustering.py" in extract_entrypoint_hints(tmp_path)


def test_explore_repository_marks_library_surface_without_cli(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup(name='pkg')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Lib\n\nSee notebook.\n", encoding="utf-8")
    (tmp_path / "pkg" / "gp_test.py").write_text("print('t')\n", encoding="utf-8")
    exploration = explore_repository(tmp_path)
    assert exploration["execution_surface"] == "library"
    assert "pkg/gp_test.py" in exploration["test_files"]
    assert exploration["library_verification_commands"]


def test_explore_spectral_like_script_surface(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "## Clustering\n\nRun [Clustering.py](Clustering.py) and [Segmentation.py](Segmentation.py).\n",
        encoding="utf-8",
    )
    (tmp_path / "Clustering.py").write_text(
        "from collections import OrderedDict\n"
        "tunables = OrderedDict([\n"
        "    ('dataset', ['cora']),  # 'cora', 'citeseer', 'pubmed', 'cloud', or 'synth'\n"
        "    ('method', ['mincut_pool']),  # 'mincut_pool', 'diff_pool'\n"
        "])\n"
        "if __name__ == '__main__':\n"
        "    print('cluster')\n",
        encoding="utf-8",
    )
    (tmp_path / "Segmentation.py").write_text(
        "if __name__ == '__main__':\n    print('seg')\n",
        encoding="utf-8",
    )

    exploration = explore_repository(tmp_path)
    assert exploration["execution_surface"] == "script"
    assert "Clustering.py" in exploration["script_entrypoints"]
    assert any("Clustering.py" in cmd for cmd in exploration["example_commands"])
    tunables = exploration["script_tunables"]
    assert "cora" in tunables.get("dataset", [])
    assert "citeseer" in tunables.get("dataset", [])
    assert "mincut_pool" in tunables.get("method", [])
    assert "diff_pool" in tunables.get("method", [])


def test_explore_stag_like_native_surface(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)\n", encoding="utf-8")
    (tmp_path / "INSTALL").write_text("mkdir build && cd build && cmake ..\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# STAG\n\nC++ spectral graph library.\n", encoding="utf-8")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "graph_test.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    exploration = explore_repository(tmp_path)
    assert exploration["execution_surface"] == "native"
    assert exploration["native_build"]["available"] is True
    assert "CMakeLists.txt" in exploration["native_build"]["files"]
    assert any("cmake" in cmd for cmd in exploration["native_build"]["commands"])
    assert "test/graph_test.cpp" in exploration["native_tests"]


def test_explore_config_and_makefile_surfaces(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train.yaml").write_text("lr: 0.1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Train\n\n```\npython train.py --config configs/train.yaml\n```\n", encoding="utf-8")
    (tmp_path / "train.py").write_text(
        "if __name__ == '__main__':\n    print('train')\n",
        encoding="utf-8",
    )
    exploration = explore_repository(tmp_path)
    # Script/README commands without registries → script (higher priority than config).
    assert exploration["execution_surface"] in {"script", "config"}
    assert "configs/train.yaml" in exploration["config_files"]

    make_root = tmp_path / "make_only"
    make_root.mkdir()
    (make_root / "Makefile").write_text(
        ".PHONY: test train\ntest:\n\t./run_tests\ntrain:\n\t./train\n",
        encoding="utf-8",
    )
    (make_root / "README.md").write_text("# Make repo\n", encoding="utf-8")
    make_exploration = explore_repository(make_root)
    assert make_exploration["execution_surface"] == "native"
    assert "test" in make_exploration["make_targets"]
