"""Deterministic deep repository exploration for Planner context.

Walks README first, then the file tree, then key source files (entrypoints and
registry modules) so the Planner can hand the Engineer concrete run details
without inventing paths.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.config import (
    PLANNER_CONFIG_FILES_MAX,
    PLANNER_LIBRARY_TEST_MAX,
    PLANNER_MAKE_TARGETS_MAX,
    PLANNER_MATRIX_CANDIDATE_MAX,
    PLANNER_MATRIX_MAX_ALGORITHMS,
    PLANNER_MATRIX_MAX_FUNCTIONS,
    PLANNER_NATIVE_TESTS_MAX,
    PLANNER_REPO_EXPLORATION_CHARS,
    PLANNER_REPO_MAX_SOURCE_FILES,
    PLANNER_REPO_README_CHARS,
    PLANNER_REPO_SOURCE_FILE_CHARS,
    PLANNER_REPO_TREE_DEPTH,
    PLANNER_REPO_TREE_MAX_ENTRIES,
    PLANNER_SCRIPT_ENTRYPOINTS_MAX,
)
from src.tools.repo_context import (
    README_NAMES,
    _read_readme,
    extract_entrypoint_hints,
    extract_example_commands,
    is_experiment_command,
)

_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
    ".eggs",
}
_REGISTRY_FILE_NAMES = ("__init__.py",)
_REGISTRY_PARENT_HINTS = (
    "algorithm",
    "algorithms",
    "test_function",
    "test_functions",
    "benchmark",
    "benchmarks",
    "experiment",
    "experiments",
    "exp",
    "demo",
    "example",
    "examples",
)
_DICT_ENTRY_RE = re.compile(
    r"""['"]([A-Za-z0-9_./+-]+)['"]\s*:\s*[A-Za-z_][A-Za-z0-9_]*""",
)
_CLI_FLAG_RE = re.compile(
    r"""add_argument\(\s*['"]--([A-Za-z0-9-]+)['"]""",
)
_PLACEHOLDER_FUN = ("FUN_NAME1", "FUN_NAME2", "FUN_NAME")
_PLACEHOLDER_ALGO = ("ALGO_NAME1", "ALGO_NAME2", "ALGO_NAME")
_PLACEHOLDER_REG = ("REGRESSOR1", "REGRESSOR2", "REGRESSOR")
_PLACEHOLDER_CLS = ("CLASSIFIER1", "CLASSIFIER2", "CLASSIFIER")
_PLACEHOLDER_LOG = ("LOG_PATH1", "LOG_PATH2", "LOG_PATH")
_LIBRARY_TEST_PRIORITY = (
    "bayesopt",
    "gp_test",
    "objectives",
    "kernel",
    "acfun",
    "mean",
    "data_test",
)


def build_repo_tree(repo_path: Path, max_depth: int = PLANNER_REPO_TREE_DEPTH) -> list[str]:
    """Return a depth-limited relative path listing for the repository."""
    if not repo_path.is_dir():
        return []

    entries: list[str] = []

    def walk(current: Path, depth: int) -> None:
        if len(entries) >= PLANNER_REPO_TREE_MAX_ENTRIES or depth > max_depth:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for child in children:
            if child.name.startswith(".") or child.name in _SKIP_DIR_NAMES:
                continue
            relative = child.relative_to(repo_path).as_posix()
            if child.is_dir():
                entries.append(f"{relative}/")
                walk(child, depth + 1)
            else:
                entries.append(relative)
            if len(entries) >= PLANNER_REPO_TREE_MAX_ENTRIES:
                return

    walk(repo_path, 1)
    return entries


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n...[truncated]..."


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _extract_registry_ids(source: str) -> list[str]:
    """Best-effort extraction of string keys from Python registry dicts."""
    ids: list[str] = []
    seen: set[str] = set()
    for match in _DICT_ENTRY_RE.finditer(source):
        key = match.group(1)
        if key in seen:
            continue
        seen.add(key)
        ids.append(key)
    return ids


def _priority_source_files(repo_path: Path) -> list[Path]:
    """Prefer README-linked entrypoints and registry modules over arbitrary sources."""
    candidates: list[Path] = []
    for hint in extract_entrypoint_hints(repo_path):
        path = repo_path / hint
        if path.is_file():
            candidates.append(path)

    for path in sorted(repo_path.rglob("*.py")):
        if any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in path.parts):
            continue
        relative_parts = {part.lower() for part in path.relative_to(repo_path).parts}
        if path.name in _REGISTRY_FILE_NAMES and relative_parts & set(_REGISTRY_PARENT_HINTS):
            candidates.append(path)
            continue
        stem = path.stem.lower()
        if any(token in stem for token in ("run", "exp", "demo", "example", "benchmark", "main")):
            candidates.append(path)

    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        ordered.append(path)
        if len(ordered) >= PLANNER_REPO_MAX_SOURCE_FILES:
            break
    return ordered


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _extract_cli_flags(source: str) -> list[str]:
    flags: list[str] = []
    seen: set[str] = set()
    for match in _CLI_FLAG_RE.finditer(source):
        # argparse uses dashes; Planner hyperparameter keys often use underscores.
        key = match.group(1).replace("-", "_")
        if key in seen:
            continue
        seen.add(key)
        flags.append(key)
    return flags


def _infer_model_type_options(text: str) -> tuple[list[str], list[str]]:
    """Infer regressor/classifier CLI options mentioned in README or sources."""
    lowered = text.lower()
    reg_types = ["gp"]
    cls_types = ["gp"]
    if re.search(r"classifier[s]?.*\bgp\b.*\bde\b|\bcls-type\b.*\bgp\b.*\bde\b", lowered):
        cls_types = ["gp", "de"]
    elif re.search(r"\bsupport `gp` and `de`\b", lowered):
        cls_types = ["gp", "de"]
    elif "`de`" in lowered or " cls_type" in lowered or "deep ensemble" in lowered:
        if "de" not in cls_types:
            cls_types.append("de")
    return reg_types, cls_types


def _pick_run_command_template(commands: list[str]) -> str | None:
    """Prefer a single-run experiment script over plot or parallel batch templates."""
    ranked: list[tuple[int, str]] = []
    for command in commands:
        if not is_experiment_command(command):
            continue
        lowered = command.lower()
        if "plot" in lowered:
            continue
        score = 1
        if "parallel" in lowered:
            score = 2
        if re.search(r"run_exp\.py\b", lowered) and "parallel" not in lowered:
            score = 4
        elif re.search(r"\brun[_-]?exp\b", lowered) and "parallel" not in lowered:
            score = 3
        ranked.append((score, command))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _replace_placeholders(template: str, mapping: dict[str, str]) -> str:
    command = template
    # Longer keys first so FUN_NAME1 wins over FUN_NAME.
    for key in sorted(mapping, key=len, reverse=True):
        command = command.replace(key, mapping[key])
    # Drop leftover ellipsis tokens from README multi-arg examples.
    command = re.sub(r"\s+\.\.\.", "", command)
    return " ".join(command.split())


def build_experiment_candidates(
    *,
    functions: list[str],
    algorithms: list[str],
    example_commands: list[str],
    readme: str = "",
    source_text: str = "",
    max_candidates: int = PLANNER_MATRIX_CANDIDATE_MAX,
    max_functions: int = PLANNER_MATRIX_MAX_FUNCTIONS,
    max_algorithms: int = PLANNER_MATRIX_MAX_ALGORITHMS,
) -> list[dict[str, object]]:
    """Build a representative (not exhaustive) set of filled runnable experiments."""
    template = _pick_run_command_template(example_commands)
    if not template or not functions or not algorithms:
        return []

    selected_functions = _unique(functions)[:max_functions]
    # Prefer the paper method (often last / namespaced) by putting non-random first.
    ranked_algos = sorted(
        _unique(algorithms),
        key=lambda name: (
            0 if any(token in name.lower() for token in ("be-cbo", "becbo", "proposed")) else 1,
            0 if not name.lower().startswith("random") else 2,
            name,
        ),
    )
    selected_algorithms = ranked_algos[:max_algorithms]
    reg_types, cls_types = _infer_model_type_options(f"{readme}\n{source_text}")
    default_reg = reg_types[0]
    # Use the richest classifier option for the first algorithm (paper method),
    # and the default for baselines so the matrix stays a decent chunk, not a full grid.
    primary_cls = cls_types[-1] if cls_types else "gp"
    baseline_cls = cls_types[0] if cls_types else "gp"

    candidates: list[dict[str, object]] = []
    for fun_name in selected_functions:
        for index, algo_name in enumerate(selected_algorithms):
            cls_type = primary_cls if index == 0 else baseline_cls
            log_path = f"LOG_PATH/{fun_name}/{default_reg}_{cls_type}/{algo_name}/0"
            run_command = _replace_placeholders(
                template,
                {
                    **{key: fun_name for key in _PLACEHOLDER_FUN},
                    **{key: algo_name for key in _PLACEHOLDER_ALGO},
                    **{key: default_reg for key in _PLACEHOLDER_REG},
                    **{key: cls_type for key in _PLACEHOLDER_CLS},
                    **{key: log_path for key in _PLACEHOLDER_LOG},
                    "LOG_DIR": f"LOG_PATH/{fun_name}",
                    "N_SEED": "1",
                    "NUM_PROC": "1",
                },
            )
            candidates.append(
                {
                    "name": f"{fun_name}__{algo_name}__{default_reg}_{cls_type}",
                    "benchmark": fun_name,
                    "method": algo_name,
                    "run_command": run_command,
                    "hyperparameters": {
                        "reg_type": default_reg,
                        "cls_type": cls_type,
                    },
                    "execution_pattern": run_command,
                }
            )
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def discover_test_files(repo_path: Path, max_files: int = PLANNER_LIBRARY_TEST_MAX) -> list[str]:
    """Find unit/integration test modules, preferring BO/GP-related names."""
    if not repo_path.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(repo_path.rglob("*.py")):
        if any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in path.parts):
            continue
        name = path.name.lower()
        if name.endswith("_test.py") or name.startswith("test_"):
            found.append(path)

    def rank(path: Path) -> tuple[int, str]:
        stem = path.stem.lower()
        for index, token in enumerate(_LIBRARY_TEST_PRIORITY):
            if token in stem:
                return (index, path.as_posix())
        return (len(_LIBRARY_TEST_PRIORITY), path.as_posix())

    ordered = sorted(found, key=rank)
    return [path.relative_to(repo_path).as_posix() for path in ordered[:max_files]]


def discover_notebooks(repo_path: Path, max_files: int = 5) -> list[str]:
    if not repo_path.is_dir():
        return []
    notebooks: list[str] = []
    for path in sorted(repo_path.rglob("*.ipynb")):
        if any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in path.parts):
            continue
        notebooks.append(path.relative_to(repo_path).as_posix())
        if len(notebooks) >= max_files:
            break
    return notebooks


def discover_script_entrypoints(
    repo_path: Path,
    *,
    example_commands: list[str],
    max_files: int = PLANNER_SCRIPT_ENTRYPOINTS_MAX,
) -> list[str]:
    """Find runnable Python scripts from README commands and conventional names."""
    if not repo_path.is_dir():
        return []
    found: list[str] = []
    seen: set[str] = set()

    def add(relative: str) -> None:
        text = relative.strip().lstrip("./")
        if not text or text in seen:
            return
        path = repo_path / text
        if not path.is_file() or path.suffix.lower() != ".py":
            return
        # Prefer non-test application scripts.
        if path.name.endswith("_test.py") or path.name.startswith("test_"):
            return
        seen.add(text)
        found.append(text)

    for command in example_commands:
        tokens = command.split()
        for token in tokens:
            if token.endswith(".py"):
                add(token.strip("`'\""))
    for hint in extract_entrypoint_hints(repo_path):
        add(hint)

    # Top-level and scripts/ conventional entrypoints.
    search_roots = [repo_path]
    scripts_dir = repo_path / "scripts"
    if scripts_dir.is_dir():
        search_roots.append(scripts_dir)
    for root in search_roots:
        try:
            children = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for path in children:
            if not path.is_file() or path.suffix.lower() != ".py":
                continue
            if path.name in {"setup.py", "setup.cfg", "conftest.py", "conf.py"}:
                continue
            stem = path.stem.lower()
            relative = path.relative_to(repo_path).as_posix()
            if any(part in stem for part in _ENTRYPOINT_NAME_HINTS):
                add(relative)
                continue
            content = _read_text_file(path) or ""
            if 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content:
                add(relative)
            if len(found) >= max_files:
                return found[:max_files]
    return found[:max_files]


_ENTRYPOINT_NAME_HINTS = (
    "demo",
    "example",
    "experiment",
    "benchmark",
    "run",
    "cluster",
    "segment",
    "train",
    "eval",
    "classification",
    "autoencoder",
)


def discover_native_build(repo_path: Path) -> dict[str, object]:
    """Detect CMake/Make/INSTALL and propose grounded build/test commands."""
    if not repo_path.is_dir():
        return {"available": False, "commands": [], "files": []}
    files: list[str] = []
    commands: list[str] = []
    if (repo_path / "CMakeLists.txt").is_file():
        files.append("CMakeLists.txt")
        commands.append("cmake -S . -B build && cmake --build build")
        commands.append("ctest --test-dir build --output-on-failure")
    if (repo_path / "INSTALL").is_file() or (repo_path / "INSTALL.md").is_file():
        files.append("INSTALL" if (repo_path / "INSTALL").is_file() else "INSTALL.md")
    makefile = next(
        (name for name in ("Makefile", "makefile", "GNUmakefile") if (repo_path / name).is_file()),
        None,
    )
    if makefile:
        files.append(makefile)
        if "make" not in " ".join(commands):
            commands.append("make")
        commands.append("make test")
    return {
        "available": bool(files),
        "files": files,
        "commands": _unique(commands),
    }


def discover_native_tests(
    repo_path: Path, max_files: int = PLANNER_NATIVE_TESTS_MAX
) -> list[str]:
    if not repo_path.is_dir():
        return []
    found: list[str] = []
    for path in sorted(repo_path.rglob("*.cpp")):
        if any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in path.parts):
            continue
        name = path.name.lower()
        relative = path.relative_to(repo_path).as_posix()
        if name.endswith("_test.cpp") or "/test/" in f"/{relative.lower()}":
            found.append(relative)
        if len(found) >= max_files:
            break
    return found


def discover_config_files(
    repo_path: Path, max_files: int = PLANNER_CONFIG_FILES_MAX
) -> list[str]:
    if not repo_path.is_dir():
        return []
    found: list[str] = []
    for folder_name in ("configs", "config", "conf", "experiments"):
        folder = repo_path / folder_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".yaml", ".yml", ".json", ".toml"}:
                continue
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            found.append(path.relative_to(repo_path).as_posix())
            if len(found) >= max_files:
                return found
    return found


def discover_container_files(repo_path: Path) -> list[str]:
    if not repo_path.is_dir():
        return []
    found: list[str] = []
    for name in ("Dockerfile", "dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml"):
        if (repo_path / name).is_file():
            found.append(name)
    for path in sorted(repo_path.glob("docker-compose*.yml")):
        relative = path.relative_to(repo_path).as_posix()
        if relative not in found:
            found.append(relative)
    return found


def discover_make_targets(
    repo_path: Path, max_targets: int = PLANNER_MAKE_TARGETS_MAX
) -> list[str]:
    makefile = next(
        (repo_path / name for name in ("Makefile", "makefile", "GNUmakefile") if (repo_path / name).is_file()),
        None,
    )
    if makefile is None:
        return []
    content = _read_text_file(makefile) or ""
    targets: list[str] = []
    for match in re.findall(r"^\.PHONY:\s*(.+)$", content, flags=re.MULTILINE):
        for part in match.replace(",", " ").split():
            if part and part not in targets:
                targets.append(part)
    interesting = ("train", "test", "experiment", "eval", "run", "benchmark", "all")
    for match in re.findall(r"^([A-Za-z0-9_./-]+)\s*:", content, flags=re.MULTILINE):
        if match.startswith(".") or match in targets:
            continue
        if any(token in match.lower() for token in interesting):
            targets.append(match)
        if len(targets) >= max_targets:
            break
    return targets[:max_targets]


def discover_artifact_dirs(repo_path: Path) -> list[str]:
    if not repo_path.is_dir():
        return []
    names = ("results", "figures", "figs", "outputs", "checkpoints")
    found: list[str] = []
    for name in names:
        path = repo_path / name
        if path.is_dir():
            found.append(f"{name}/")
    return found


_TUNABLE_LINE_RE = re.compile(
    r"""['"](dataset|method|benchmark|model|algo|algorithm)['"]\s*,\s*\[([^\]]*)\]"""
    r"""\s*\)?\s*,?\s*(?:#\s*([^\n]*))?""",
    re.IGNORECASE,
)


def scrape_script_tunables(repo_path: Path, scripts: list[str]) -> dict[str, list[str]]:
    """Best-effort extraction of dataset/method lists from script sources."""
    axes: dict[str, list[str]] = {}
    for relative in scripts[:8]:
        content = _read_text_file(repo_path / relative)
        if not content:
            continue
        for match in _TUNABLE_LINE_RE.finditer(content):
            key = match.group(1).lower()
            listed = re.findall(r"""['"]([^'"]+)['"]""", match.group(2) or "")
            commented = re.findall(r"""['`]([A-Za-z0-9_.-]+)['`]""", match.group(3) or "")
            values: list[str] = []
            for part in listed + commented:
                if part and part not in values and part.lower() not in {"none", "true", "false"}:
                    values.append(part)
            if not values:
                continue
            bucket = axes.setdefault(key, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
    # Normalize keys used by phase builder.
    if "dataset" in axes:
        axes["benchmark"] = list(axes["dataset"])
    if "method" in axes and "algorithm" not in axes:
        axes["algorithm"] = list(axes["method"])
    return axes


def infer_execution_surface(
    *,
    example_commands: list[str],
    test_files: list[str],
    notebooks: list[str],
    has_setup_py_or_pyproject: bool,
    script_entrypoints: list[str] | None = None,
    native_build: dict[str, object] | None = None,
    native_tests: list[str] | None = None,
    config_files: list[str] | None = None,
    container_files: list[str] | None = None,
    make_targets: list[str] | None = None,
    artifact_dirs: list[str] | None = None,
    registry_functions: list[str] | None = None,
    registry_algorithms: list[str] | None = None,
) -> str:
    """Classify how the Engineer should drive the repository (priority order)."""
    scripts = script_entrypoints or []
    native = native_build or {}
    configs = config_files or []
    containers = container_files or []
    artifacts = artifact_dirs or []
    functions = registry_functions or []
    algorithms = registry_algorithms or []

    has_cli_commands = any(is_experiment_command(command) for command in example_commands)
    if has_cli_commands and functions and algorithms:
        return "cli"
    if has_cli_commands or scripts:
        # Scripts/README python entrypoints without registries (Spectral).
        if not (functions and algorithms):
            return "script"
        return "cli"
    if test_files or notebooks or has_setup_py_or_pyproject:
        return "library"
    if native.get("available") or (native_tests or []):
        return "native"
    if configs and (has_cli_commands or scripts or make_targets):
        return "config"
    if configs:
        return "config"
    if containers:
        return "container"
    if artifacts and not has_cli_commands and not scripts:
        return "artifact"
    if make_targets:
        return "native"
    return "unknown"


def explore_repository(repo_path: Path | str | None) -> dict[str, object]:
    """Build a Planner-facing deep dive of a cloned repository."""
    if repo_path is None:
        return {
            "available": False,
            "reason": "No local repository path was provided.",
        }

    root = Path(repo_path)
    if not root.is_dir():
        return {
            "available": False,
            "reason": f"Repository path does not exist: {root}",
        }

    readme = _read_readme(root) or ""
    readme_name = next((name for name in README_NAMES if (root / name).is_file()), None)
    tree = build_repo_tree(root)
    source_excerpts: list[dict[str, object]] = []
    function_ids: list[str] = []
    algorithm_ids: list[str] = []
    cli_flags: list[str] = []
    source_blob_parts: list[str] = []

    for path in _priority_source_files(root):
        content = _read_text_file(path)
        if content is None:
            continue
        relative = path.relative_to(root).as_posix()
        excerpt = _clip(content, PLANNER_REPO_SOURCE_FILE_CHARS)
        source_excerpts.append(
            {
                "path": relative,
                "chars": len(content),
                "excerpt": excerpt,
            }
        )
        source_blob_parts.append(content)
        cli_flags.extend(_extract_cli_flags(content))
        lowered = relative.lower()
        ids = _extract_registry_ids(content)
        if "test_function" in lowered or "benchmark" in lowered:
            function_ids.extend(ids)
        if "algorithm" in lowered or lowered.endswith("algorithms/__init__.py"):
            algorithm_ids.extend(ids)

    functions = _unique(function_ids)
    algorithms = _unique(algorithm_ids)
    commands = extract_example_commands(root)
    source_text = "\n".join(source_blob_parts)
    candidates = build_experiment_candidates(
        functions=functions,
        algorithms=algorithms,
        example_commands=commands,
        readme=readme,
        source_text=source_text,
    )
    test_files = discover_test_files(root)
    notebooks = discover_notebooks(root)
    script_entrypoints = discover_script_entrypoints(root, example_commands=commands)
    native_build = discover_native_build(root)
    native_tests = discover_native_tests(root)
    config_files = discover_config_files(root)
    container_files = discover_container_files(root)
    make_targets = discover_make_targets(root)
    artifact_dirs = discover_artifact_dirs(root)
    script_tunables = scrape_script_tunables(root, script_entrypoints)
    has_package = (root / "setup.py").is_file() or (root / "pyproject.toml").is_file()
    execution_surface = infer_execution_surface(
        example_commands=commands,
        test_files=test_files,
        notebooks=notebooks,
        has_setup_py_or_pyproject=has_package,
        script_entrypoints=script_entrypoints,
        native_build=native_build,
        native_tests=native_tests,
        config_files=config_files,
        container_files=container_files,
        make_targets=make_targets,
        artifact_dirs=artifact_dirs,
        registry_functions=functions,
        registry_algorithms=algorithms,
    )
    library_commands = [f"python {path}" for path in test_files]
    verification_commands = _unique(
        [
            *commands,
            *[f"python {path}" for path in script_entrypoints],
            *library_commands,
            *[str(item) for item in (native_build.get("commands") or [])],
            *[f"make {target}" for target in make_targets[:5]],
        ]
    )

    exploration: dict[str, object] = {
        "available": True,
        "repo_path": str(root),
        "readme_file": readme_name or "",
        "readme_full": _clip(readme, PLANNER_REPO_README_CHARS),
        "file_tree_deep": tree,
        "example_commands": commands,
        "entrypoint_hints": extract_entrypoint_hints(root),
        "registry_ids": {
            "functions_or_benchmarks": functions,
            "algorithms_or_methods": algorithms,
        },
        "cli_flags": _unique(cli_flags),
        "experiment_candidates": candidates,
        "test_files": test_files,
        "notebooks": notebooks,
        "script_entrypoints": script_entrypoints,
        "script_tunables": script_tunables,
        "native_build": native_build,
        "native_tests": native_tests,
        "config_files": config_files,
        "container_files": container_files,
        "make_targets": make_targets,
        "artifact_dirs": artifact_dirs,
        "library_verification_commands": library_commands,
        "verification_commands": verification_commands,
        "execution_surface": execution_surface,
        "source_excerpts": source_excerpts,
        "exploration_notes": [
            "Start from README, then file_tree_deep, then source_excerpts.",
            f"execution_surface={execution_surface}; use matching evidence lists for phases.",
            "Prefer compact axes + a few example matrix rows (run_command/code_refs/verify).",
            "Do not invent files or CLI IDs outside this exploration.",
        ],
    }

    encoded = str(exploration)
    if len(encoded) > PLANNER_REPO_EXPLORATION_CHARS and source_excerpts:
        trimmed = list(source_excerpts)
        while trimmed and len(str({**exploration, "source_excerpts": trimmed})) > PLANNER_REPO_EXPLORATION_CHARS:
            trimmed.pop()
        exploration["source_excerpts"] = trimmed
        exploration["exploration_notes"].append(
            "Some source excerpts were truncated to fit the Planner context budget."
        )
    return exploration
