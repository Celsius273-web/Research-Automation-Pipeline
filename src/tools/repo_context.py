"""Deterministic repository summaries for Planner input."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import PLANNER_MAX_ENTRYPOINT_HINTS, PLANNER_MAX_EXAMPLE_COMMANDS

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
    "make\t",
    "cmake ",
    "ctest ",
    "docker ",
    "docker-compose ",
)
_NON_EXPERIMENT_COMMAND_PARTS = (
    " -m venv ",
    " pytest",
    " unittest",
    "pip install",
)
# Library/native smoke may still use python path/to/*_test.py via verification lists.
_ENTRYPOINT_SUFFIXES = (".py", ".ipynb", ".sh")
_ENTRYPOINT_NAME_PARTS = (
    "demo",
    "example",
    "experiment",
    "benchmark",
    "run",
    "cluster",
    "segment",
    "train",
    "eval",
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
    """Pull concrete run commands from README code fences, indented blocks, and script links."""
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

    # README prose often says "Run Clustering.py" via markdown links rather than fences.
    for match in re.findall(
        r"(?:Run|run)\s+[`'\"]([A-Za-z0-9_./-]+\.py)[`'\"]",
        content,
    ):
        candidates.append(f"python {match}")
    for match in re.findall(
        r"\[([^\]]*?\.py)\]\([^)]+\)",
        content,
    ):
        script = match.strip().split("/")[-1]
        candidates.append(f"python {script}")
    for match in re.findall(
        r"\((?:https?://[^)\s]+/)?([A-Za-z0-9_.-]+\.py)\)",
        content,
    ):
        candidates.append(f"python {match}")

    commands: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        command = raw.strip().lstrip("$").strip()
        if not command or command.startswith("#"):
            continue
        if command.startswith("python ") or command.startswith("python3 "):
            script = command.split()[1] if len(command.split()) > 1 else ""
            script = script.strip("`'")
            if script.endswith(".py") and not (repo_path / script).is_file():
                # Allow nested paths referenced from README when they exist.
                nested = next(
                    (
                        path.relative_to(repo_path).as_posix()
                        for path in repo_path.rglob(Path(script).name)
                        if path.is_file()
                    ),
                    None,
                )
                if nested:
                    command = f"{command.split()[0]} {nested}"
                elif "/" not in script and not (repo_path / script).is_file():
                    continue
        if not is_experiment_command(command):
            continue
        if command in seen:
            continue
        seen.add(command)
        commands.append(command)
        if len(commands) >= PLANNER_MAX_EXAMPLE_COMMANDS:
            break
    return commands


def is_experiment_command(command: str) -> bool:
    """Return whether a command runs experiments rather than setup/venv."""
    normalized = f" {command.strip().lower()} "
    text = command.strip()
    if not text:
        return False
    if not text.startswith(_RUN_COMMAND_PREFIXES) and not text.startswith("make"):
        return False
    if any(part in normalized for part in _NON_EXPERIMENT_COMMAND_PARTS):
        return False
    # Pure install/build-only make install is setup, not an experiment matrix row.
    if re.match(r"^make(\s+install)?$", text.lower()):
        return False
    return True


def is_verification_command(command: str) -> bool:
    """Return whether a command is a grounded smoke/verify step (tests, build, scripts)."""
    text = command.strip()
    if not text or text.startswith("#"):
        return False
    normalized = f" {text.lower()} "
    if " -m venv " in normalized:
        return False
    if text.startswith(_RUN_COMMAND_PREFIXES) or text.startswith("make"):
        return True
    if text.endswith(".py") or text.endswith(".sh"):
        return True
    return False


def extract_entrypoint_hints(repo_path: Path) -> list[str]:
    """Find README-referenced or conventionally named runnable files."""
    if not repo_path.is_dir():
        return []

    candidates: list[str] = []
    readme = _read_readme(repo_path)
    if readme:
        referenced_paths = re.findall(
            r"(?:\(|\s|`)([A-Za-z0-9_./-]+\.(?:py|ipynb|sh))(?:\)|\s|`)",
            readme,
        )
        candidates.extend(referenced_paths)

    for path in sorted(repo_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _ENTRYPOINT_SUFFIXES:
            continue
        relative = path.relative_to(repo_path)
        lowered_parts = {part.lower() for part in relative.parts}
        if "test" in lowered_parts or "tests" in lowered_parts:
            continue
        if any(part in path.stem.lower() for part in _ENTRYPOINT_NAME_PARTS):
            candidates.append(relative.as_posix())

    hints: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip().lstrip("./")
        if not normalized or normalized in seen:
            continue
        if not (repo_path / normalized).is_file():
            continue
        seen.add(normalized)
        hints.append(normalized)
        if len(hints) >= PLANNER_MAX_ENTRYPOINT_HINTS:
            break
    return hints


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
