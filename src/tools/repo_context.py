"""Deterministic repository summaries for Planner input."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import PLANNER_MAX_EXAMPLE_COMMANDS

README_NAMES = ("README.md", "README.rst", "README.txt", "README")

_RUN_COMMAND_PREFIXES = (
    "python ",
    "python3 ",
    "python -m ",
    "python3 -m ",
    "./",
    "bash ",
    "sh ",
    "cargo ",
    "make ",
    "cmake ",
)


def _read_readme(repo_path: Path) -> str | None:
    readme = next((repo_path / name for name in README_NAMES if (repo_path / name).is_file()), None)
    if readme is None:
        return None
    try:
        return readme.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def summarize_repo_tree(repo_path: Path) -> str:
    """Return a stable one-line summary of top-level repository entries."""
    if not repo_path.is_dir():
        return ""
    names = sorted(
        entry.name + ("/" if entry.is_dir() else "")
        for entry in repo_path.iterdir()
        if not entry.name.startswith(".")
    )
    return ", ".join(names)


def summarize_readme(repo_path: Path) -> str:
    """Return the first two or three prose sentences from a repository README."""
    content = _read_readme(repo_path)
    if content is None:
        return "No README"

    text = re.sub(r"```.*?```", " ", content, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)|\[[^\]]+\]\([^)]*\)", " ", text)
    text = re.sub(r"(?m)^[#>*+\-\s]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentence for sentence in sentences[:3] if sentence)
    return summary[:1000] or "No README"


def extract_example_commands(repo_path: Path) -> list[str]:
    """Pull concrete run commands from README code fences and indented blocks."""
    content = _read_readme(repo_path)
    if not content:
        return []

    candidates: list[str] = []
    for fence in re.findall(
        r"```(?:bash|sh|shell|console|zsh)?\n(.*?)```",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        candidates.extend(fence.splitlines())
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("    ", "\t")):
            candidates.append(stripped)

    commands: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        command = raw.strip().lstrip("$").strip()
        if not command or command.startswith("#"):
            continue
        if command.startswith(("pip install ", "python -m pip install ")):
            continue
        if not command.startswith(_RUN_COMMAND_PREFIXES):
            continue
        if command in seen:
            continue
        seen.add(command)
        commands.append(command)
        if len(commands) >= PLANNER_MAX_EXAMPLE_COMMANDS:
            break
    return commands


def infer_build_command(repo_path: Path, detected_build_system: str) -> str:
    """Convert repository markers into an executable setup command."""
    if (repo_path / "requirements.txt").is_file():
        return "pip install -r requirements.txt"
    if (repo_path / "pyproject.toml").is_file() or (repo_path / "setup.py").is_file():
        return "pip install ."
    if (repo_path / "CMakeLists.txt").is_file():
        return "cmake -S . -B build && cmake --build build"
    if (repo_path / "Makefile").is_file() or (repo_path / "makefile").is_file():
        return "make"
    if (repo_path / "Cargo.toml").is_file():
        return "cargo build"
    content = _read_readme(repo_path)
    if content is not None:
        for line in content.splitlines():
            command = line.strip()
            if command.startswith(("pip install ", "python -m pip install ")):
                return command
    return detected_build_system
