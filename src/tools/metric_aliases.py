"""Canonicalize paper-reported benchmarks/metrics for Reviewer matching.

Analyst often keeps paper wording ("Three-bar truss design problem", "f(x*)").
Engineer captures registry IDs and derived names ("3bar", "best_objective").
This module maps both sides onto shared keys without inventing numeric values.
"""

from __future__ import annotations

import re

from src.tools.result_comparator import normalize_metric_name

# Paper phrasing / long names -> short registry-style IDs used in plans and captures.
BENCHMARK_ALIASES: dict[str, str] = {
    "lsq": "lsq",
    "lsq function": "lsq",
    "lsq-shift": "lsq-shift",
    "sim": "sim",
    "3bar": "3bar",
    "three-bar truss": "3bar",
    "three-bar truss design": "3bar",
    "three-bar truss design problem": "3bar",
    "tension/compression spring design": "spring",
    "tension compression spring design": "spring",
    "welded beam design": "welded-beam",
    "gas transmission compressor design": "gas",
    "pressure vessel design": "pressure-vessel",
    "speed reducer design": "speed-reducer",
}

# Paper metric labels -> Engineer capture names (BE-CBO pickle Y trajectory).
METRIC_ALIASES: dict[str, str] = {
    "fx": "best_objective",
    "f(x*)": "best_objective",
    "best objective": "best_objective",
    "best_objective": "best_objective",
    "final objective": "final_objective",
    "final_objective": "final_objective",
    "objective value": "best_objective",
    "objective function value": "best_objective",
    "global optimum": "best_objective",
    "n_evaluations": "n_evaluations",
    "n evaluations": "n_evaluations",
}

# Prefer these algorithms when a reported row has no algorithm but several captures exist.
PREFERRED_ALGORITHMS: tuple[str, ...] = (
    "be-cbo",
    "becbo",
    "proposed",
    "ours",
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def canonicalize_benchmark(benchmark: str) -> str:
    text = _compact(benchmark)
    if not text:
        return ""
    if text in BENCHMARK_ALIASES:
        return BENCHMARK_ALIASES[text]
    # Prefix / contains match for verbose paper names.
    for alias, canonical in BENCHMARK_ALIASES.items():
        if alias in text or text in alias:
            return canonical
    return text


def canonicalize_metric(metric_name: str) -> str:
    text = _compact(metric_name)
    if not text:
        return ""
    if text in METRIC_ALIASES:
        return METRIC_ALIASES[text]
    normalized = normalize_metric_name(metric_name)
    for alias, canonical in METRIC_ALIASES.items():
        if normalize_metric_name(alias) == normalized:
            return canonical
    return normalized


def canonicalize_algorithm(algorithm: str) -> str:
    return _compact(algorithm).replace("_", "-")
