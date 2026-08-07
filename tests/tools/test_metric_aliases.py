"""Tests for metric/benchmark alias canonicalization."""

from __future__ import annotations

from src.tools.metric_aliases import canonicalize_algorithm, canonicalize_benchmark, canonicalize_metric


def test_canonicalize_benchmark_truss() -> None:
    assert canonicalize_benchmark("Three-bar truss design problem") == "3bar"
    assert canonicalize_benchmark("3bar") == "3bar"


def test_canonicalize_metric_fx() -> None:
    assert canonicalize_metric("f(x*)") == "best_objective"
    assert canonicalize_metric("best_objective") == "best_objective"


def test_canonicalize_algorithm() -> None:
    assert canonicalize_algorithm("BE-CBO") == "be-cbo"
