"""Language/build-system detection for cloned experiment repositories."""

from __future__ import annotations

from pathlib import Path

from src.state import RepoContext


def detect_language(repo_path: str, repo_url: str = "") -> RepoContext:
    root = Path(repo_path).resolve()
    if not root.exists() or not root.is_dir():
        return RepoContext(
            repo_url=repo_url,
            repo_path=str(root),
            language="unknown",
            build_system="unknown",
            notes="Repository path does not exist or is not a directory.",
        )

    python_markers = ("pyproject.toml", "requirements.txt", "setup.py")
    cpp_markers = ("CMakeLists.txt", "Makefile", "makefile")
    rust_markers = ("Cargo.toml",)

    if any((root / marker).exists() for marker in python_markers):
        build_system = "setuptools"
        if (root / "pyproject.toml").exists():
            build_system = "pyproject"
        return RepoContext(
            repo_url=repo_url,
            repo_path=str(root),
            language="python",
            build_system=build_system,
            notes="Detected from Python marker files.",
        )

    if any((root / marker).exists() for marker in cpp_markers):
        build_system = "cmake" if (root / "CMakeLists.txt").exists() else "make"
        return RepoContext(
            repo_url=repo_url,
            repo_path=str(root),
            language="cpp",
            build_system=build_system,
            notes="Detected from C/C++ marker files.",
        )

    if any((root / marker).exists() for marker in rust_markers):
        return RepoContext(
            repo_url=repo_url,
            repo_path=str(root),
            language="rust",
            build_system="cargo",
            notes="Detected from Rust marker files.",
        )

    return RepoContext(
        repo_url=repo_url,
        repo_path=str(root),
        language="unknown",
        build_system="unknown",
        notes="No known language marker files found.",
    )
