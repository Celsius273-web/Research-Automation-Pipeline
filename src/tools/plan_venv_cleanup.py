"""Strip Docker-persistence venv mechanics from Planner plan text.

Plans should describe bare ``pip`` / ``python`` commands. Engineer creates a
workspace-mounted ``.venv`` at runtime so installs survive container teardown.
Embedding ``python -m venv`` in the plan confuses that split of responsibility.
"""

from __future__ import annotations

import re

VENV_PLAN_REWRITE_WARNING = (
    "setup phase contained python -m venv / .venv/bin patterns; rewritten to bare pip/python"
)
VENV_PLAN_REWRITE_NOTE = (
    "setup phase rewritten to avoid venv-in-plan pattern; uses bare pip/python. "
    "Engineer creates a workspace-mounted .venv at runtime so installs persist across Docker containers."
)

_VENV_CREATE_RE = re.compile(
    r"python3?\s+-m\s+venv(?:\s+--clear)?\s+\S+\s*&&\s*",
    re.IGNORECASE,
)
_VENV_PIP_UPGRADE_RE = re.compile(
    r"(?:\.venv/bin/)?pip\s+install\s+-U\s+pip\s*&&\s*",
    re.IGNORECASE,
)


def command_mentions_venv(command: str) -> bool:
    lowered = command.lower()
    return "python -m venv" in lowered or "python3 -m venv" in lowered or ".venv/bin/" in lowered


def strip_venv_from_command(command: str) -> str:
    """Remove venv create/upgrade wrappers and ``.venv/bin/`` prefixes."""
    text = command.strip()
    if not text:
        return text
    rewritten = _VENV_CREATE_RE.sub("", text)
    rewritten = _VENV_PIP_UPGRADE_RE.sub("", rewritten)
    rewritten = rewritten.replace(".venv/bin/python", "python")
    rewritten = rewritten.replace(".venv/bin/pip", "pip")
    rewritten = rewritten.replace(".venv/bin/", "")
    parts = [part.strip() for part in re.split(r"\s+&&\s+", rewritten) if part.strip()]
    return " && ".join(parts)


def cleanup_venv_patterns_in_phases(
    phases: list[dict[str, object]],
) -> tuple[list[dict[str, object]], bool]:
    """Rewrite phase commands in-place-copy; return (phases, changed)."""
    changed = False
    cleaned: list[dict[str, object]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            cleaned.append(phase)
            continue
        phase_copy = dict(phase)
        run_template = str(phase_copy.get("run_template", "") or "")
        if command_mentions_venv(run_template):
            phase_copy["run_template"] = strip_venv_from_command(run_template)
            changed = True
        matrix = phase_copy.get("matrix")
        if isinstance(matrix, list):
            new_matrix: list[object] = []
            for row in matrix:
                if not isinstance(row, dict):
                    new_matrix.append(row)
                    continue
                row_copy = dict(row)
                run_command = str(row_copy.get("run_command", "") or "")
                if command_mentions_venv(run_command):
                    row_copy["run_command"] = strip_venv_from_command(run_command)
                    changed = True
                new_matrix.append(row_copy)
            phase_copy["matrix"] = new_matrix
        cleaned.append(phase_copy)
    return cleaned, changed
