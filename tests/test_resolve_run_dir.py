"""Tests for explicit Engineer run_id resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.bundle as bundle_mod
from src.persistence import resolve_run_dir


def test_resolve_run_id_requires_existing_timestamp(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bundle_mod, "PAPER_BUNDLES_DIR", tmp_path / "papers")
    bundle = bundle_mod.PaperBundle("demo")
    bundle.create_bundle_dir()
    run = bundle.runs_dir / "R1"
    run.mkdir()

    assert resolve_run_dir("demo", "R1") == run.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_run_dir("demo", "R99")
    with pytest.raises(ValueError):
        resolve_run_dir("demo", "../escape")
