"""Allocate sequential Engineer run directory names (R1, R2, …)."""

from __future__ import annotations

import re
from pathlib import Path

_RUN_ID_RE = re.compile(r"^R(\d+)$", re.IGNORECASE)


def next_run_id(runs_dir: Path) -> str:
    """Return the next ``R{n}`` id under ``runs_dir`` (R1 if empty)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    highest = 0
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = _RUN_ID_RE.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"R{highest + 1}"


def is_run_id(name: str) -> bool:
    return bool(_RUN_ID_RE.match((name or "").strip()))
