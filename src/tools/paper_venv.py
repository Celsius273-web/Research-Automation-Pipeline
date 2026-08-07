"""Persistent per-paper venv helpers for Docker-mounted experiment repos."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import PAPER_VENV_DIRNAME

# Top-level imports that pip must satisfy but READMEs often omit.
_IMPORT_TO_PIP = {
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "yaml": "pyyaml",
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "scipy": "scipy",
}


def prefer_requirements_install(install_command: str, repo_path: Path | None) -> str:
    """Prefer pip install -r requirements.txt when that file exists in the paper repo."""
    if repo_path is not None and (repo_path / "requirements.txt").is_file():
        return "pip install -r requirements.txt"
    return install_command.strip()


def discover_missing_pip_packages(repo_path: Path, install_command: str) -> list[str]:
    """Find third-party imports used by the repo that are absent from the install command."""
    if not repo_path.exists():
        return []
    install_lower = install_command.lower()
    found: set[str] = set()
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(" + "|".join(re.escape(k) for k in _IMPORT_TO_PIP) + r")\b",
        re.MULTILINE,
    )
    for path in repo_path.rglob("*.py"):
        if ".venv" in path.parts or "site-packages" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in pattern.finditer(text):
            found.add(match.group(1))

    missing: list[str] = []
    for import_name in sorted(found):
        package = _IMPORT_TO_PIP[import_name]
        if package.lower() not in install_lower:
            missing.append(package)
    return list(dict.fromkeys(missing))


def append_packages_to_pip_install(install_command: str, packages: list[str]) -> str:
    """Append extra packages to a pip install command."""
    if not packages:
        return install_command
    command = install_command.strip()
    if not is_python_install_command(command):
        return command
    if command.rstrip().endswith("requirements.txt") or " -r " in command:
        extras = " ".join(packages)
        return f"{command} && pip install {extras}"
    return f"{command} {' '.join(packages)}"


def _to_venv_pip_command(install_command: str, venv_dirname: str) -> str:
    command = install_command.strip()
    if not command:
        return f"{venv_dirname}/bin/pip install -r requirements.txt"

    venv_pip = f"{venv_dirname}/bin/pip"
    # Rewrite each pip invocation in a chained command.
    parts = re.split(r"\s+&&\s+", command)
    rewritten_parts: list[str] = []
    for part in parts:
        piece = part.strip()
        if piece.startswith(f"{venv_pip} "):
            rewritten_parts.append(piece)
        elif piece.startswith("python -m pip "):
            rewritten_parts.append(piece.replace("python -m pip ", f"{venv_pip} ", 1))
        elif piece.startswith("python3 -m pip "):
            rewritten_parts.append(piece.replace("python3 -m pip ", f"{venv_pip} ", 1))
        elif piece.startswith("pip "):
            rewritten_parts.append(piece.replace("pip ", f"{venv_pip} ", 1))
        else:
            rewritten_parts.append(piece)
    return " && ".join(rewritten_parts)


def is_python_install_command(command: str) -> bool:
    lowered = command.strip().lower()
    return (
        lowered.startswith("pip ")
        or lowered.startswith("python -m pip ")
        or lowered.startswith("python3 -m pip ")
        or f"/{PAPER_VENV_DIRNAME}/bin/pip " in lowered
        or lowered.startswith(f"{PAPER_VENV_DIRNAME}/bin/pip ")
    )


def build_persistent_setup_command(
    install_command: str,
    *,
    repo_path: Path | None = None,
    venv_dirname: str = PAPER_VENV_DIRNAME,
) -> str:
    """Create repo-local venv and install deps so later containers still see packages."""
    install = prefer_requirements_install(install_command, repo_path)
    if not is_python_install_command(install) and install not in {"", "unknown"}:
        # Native build systems are not wrapped in a Python venv.
        return install

    if install in {"", "unknown"}:
        install = "pip install -r requirements.txt"
        if repo_path is not None and not (repo_path / "requirements.txt").is_file():
            if (repo_path / "pyproject.toml").is_file() or (repo_path / "setup.py").is_file():
                install = "pip install ."

    if repo_path is not None:
        missing = discover_missing_pip_packages(repo_path, install)
        install = append_packages_to_pip_install(install, missing)

    pip_command = _to_venv_pip_command(install, venv_dirname)
    # --clear recreates the venv when the Docker Python version changes (e.g. 3.11 -> 3.8).
    return (
        f"python -m venv --clear {venv_dirname} && "
        f"{venv_dirname}/bin/pip install -U pip && "
        f"{pip_command}"
    )


def rewrite_command_for_paper_venv(
    command: str,
    *,
    venv_dirname: str = PAPER_VENV_DIRNAME,
) -> str:
    """Point python/pytest invocations at the paper repo venv when present in the command path."""
    text = command.strip()
    if not text:
        return command
    if f"{venv_dirname}/bin/" in text:
        return text
    if is_python_install_command(text):
        return build_persistent_setup_command(text, venv_dirname=venv_dirname)

    # Rewrite leading python/python3 and after && / ; separators.
    pattern = re.compile(r"(^|[;&]\s*|\|\|\s*|&&\s*)(python3?)\b")
    rewritten = pattern.sub(rf"\1{venv_dirname}/bin/python", text)
    rewritten = re.sub(
        r"(^|[;&]\s*|\|\|\s*|&&\s*)pytest\b",
        rf"\1{venv_dirname}/bin/pytest",
        rewritten,
    )
    return rewritten
