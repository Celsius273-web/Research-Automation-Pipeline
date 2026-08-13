from __future__ import annotations

from pathlib import Path

from src.tools.language_detect import detect_language


def test_detect_language_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    out = detect_language(str(tmp_path))
    assert out.language == "python"
    assert out.build_system == "pyproject"


def test_detect_language_cpp(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n", encoding="utf-8")
    out = detect_language(str(tmp_path))
    assert out.language == "cpp"
    assert out.build_system == "cmake"


def test_detect_language_rust(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\nversion='0.1.0'\n", encoding="utf-8")
    out = detect_language(str(tmp_path))
    assert out.language == "rust"
    assert out.build_system == "cargo"


def test_detect_language_unknown(tmp_path: Path) -> None:
    out = detect_language(str(tmp_path))
    assert out.language == "unknown"


def test_detect_language_python_from_source_files(tmp_path: Path) -> None:
    (tmp_path / "benchmark.py").write_text("print('ok')\n", encoding="utf-8")
    out = detect_language(str(tmp_path))
    assert out.language == "python"
    assert out.has_code is True
