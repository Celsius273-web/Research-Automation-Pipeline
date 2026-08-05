"""Deterministic experiment results path helpers.

Code owns the results layout. Planner and Engineer may only use paths under
results/{paper_id}/; measured values are written by execution, never by an LLM.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.config import (
    RESULTS_DIR,
    RESULTS_LOGS_FILENAME,
    RESULTS_METRICS_FILENAME,
    RESULTS_SUMMARY_FILENAME,
)


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_RUN_DIR_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}__[a-z0-9]+(?:-[a-z0-9]+)*"
    r"__[a-z0-9]+(?:-[a-z0-9]+)*__seed-\d{2,}$"
)


def slugify_label(value: str) -> str:
    """Convert a human label into a filesystem-safe lowercase slug."""
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "item"


def paper_results_dir(paper_id: str) -> Path:
    return RESULTS_DIR / paper_id


def results_summary_path(paper_id: str) -> Path:
    return paper_results_dir(paper_id) / RESULTS_SUMMARY_FILENAME


def results_summary_relpath(paper_id: str) -> str:
    """Repo-relative summary path used in plans and prompts."""
    return f"results/{paper_id}/{RESULTS_SUMMARY_FILENAME}"


def build_run_dir_name(
    benchmark: str,
    method: str,
    seed: int,
    when: datetime | None = None,
) -> str:
    """Build a human-readable timestamped run directory name."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return (
        f"{stamp}__{slugify_label(benchmark)}__{slugify_label(method)}__seed-{seed:02d}"
    )


def run_dir_path(paper_id: str, run_dir_name: str) -> Path:
    return paper_results_dir(paper_id) / run_dir_name


def run_metrics_path(paper_id: str, run_dir_name: str) -> Path:
    return run_dir_path(paper_id, run_dir_name) / RESULTS_METRICS_FILENAME


def run_logs_path(paper_id: str, run_dir_name: str) -> Path:
    return run_dir_path(paper_id, run_dir_name) / RESULTS_LOGS_FILENAME


def is_valid_run_dir_name(name: str) -> bool:
    return bool(_RUN_DIR_RE.match(name.strip()))


def is_under_paper_results(path_str: str, paper_id: str) -> bool:
    """True when path_str is the summary or a file under results/{paper_id}/."""
    normalized = path_str.strip().replace("\\", "/")
    root = f"results/{paper_id}"
    if normalized == root or normalized == f"{root}/":
        return True
    if normalized == results_summary_relpath(paper_id):
        return True
    prefix = f"{root}/"
    if not normalized.startswith(prefix):
        return False
    remainder = normalized[len(prefix) :]
    if not remainder or remainder.startswith("/") or ".." in remainder.split("/"):
        return False
    return True


def results_contract_for_prompt(paper_id: str) -> dict[str, str]:
    """Facts injected into the Planner prompt; never invented by the LLM."""
    return {
        "results_root": f"results/{paper_id}",
        "summary_path": results_summary_relpath(paper_id),
        "run_dir_pattern": (
            "YYYY-MM-DD_HH-MM-SS__{benchmark_slug}__{method_slug}__seed-{seed}"
        ),
        "metrics_filename": RESULTS_METRICS_FILENAME,
        "logs_filename": RESULTS_LOGS_FILENAME,
    }
