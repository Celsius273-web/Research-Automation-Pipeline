"""Unit tests for persistent paper-repo venv command helpers."""

from __future__ import annotations

from pathlib import Path

from src.tools.paper_venv import (
    build_persistent_setup_command,
    prefer_requirements_install,
    rewrite_command_for_paper_venv,
)


def test_prefer_requirements_install_when_file_exists(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    out = prefer_requirements_install("pip install torch", tmp_path)
    assert out == "pip install -r requirements.txt"


def test_prefer_requirements_keeps_planned_when_missing(tmp_path: Path) -> None:
    out = prefer_requirements_install("pip install torch", tmp_path)
    assert out == "pip install torch"


def test_build_persistent_setup_prefers_requirements_txt(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    out = build_persistent_setup_command("pip install torch", repo_path=tmp_path)
    assert out == (
        "python -m venv --clear .venv && "
        ".venv/bin/pip install -U pip && "
        ".venv/bin/pip install -r requirements.txt"
    )


def test_build_persistent_setup_uses_planned_pip_without_requirements(tmp_path: Path) -> None:
    out = build_persistent_setup_command(
        "pip install numpy torch gpytorch==1.7.0",
        repo_path=tmp_path,
    )
    assert ".venv/bin/pip install numpy torch gpytorch==1.7.0" in out
    assert out.startswith("python -m venv --clear .venv &&")


def test_build_persistent_setup_appends_matplotlib_from_imports(tmp_path: Path) -> None:
    (tmp_path / "plot").mkdir()
    (tmp_path / "plot" / "plot.py").write_text(
        "import matplotlib.pyplot as plt\n",
        encoding="utf-8",
    )
    out = build_persistent_setup_command(
        "pip install numpy torch",
        repo_path=tmp_path,
    )
    assert "matplotlib" in out
    assert ".venv/bin/pip install numpy torch matplotlib" in out


def test_rewrite_command_for_paper_venv_rewrites_python() -> None:
    out = rewrite_command_for_paper_venv(
        "python exp/run_exp.py --fun lsq && python -m pytest -q"
    )
    assert out == (
        ".venv/bin/python exp/run_exp.py --fun lsq && .venv/bin/python -m pytest -q"
    )


def test_rewrite_command_leaves_native_build_alone() -> None:
    assert rewrite_command_for_paper_venv("cmake -S . -B build") == "cmake -S . -B build"
