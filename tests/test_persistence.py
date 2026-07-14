"""Tests for persistence.py — specifically the extraction bundle functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.persistence import _format_bundle_as_text, persist_extraction_bundle
from src.state import (
    ExtractionBundle,
    PaperMetadata,
    ReportedResult,
    ReviewRecord,
    SectionExtraction,
)


def _make_paper() -> PaperMetadata:
    return PaperMetadata(
        paper_id="test_paper",
        title="A Test Paper on Bayesian Optimization",
        pdf_path="/tmp/test.pdf",
    )


def _make_bundle() -> ExtractionBundle:
    abstract_ext = SectionExtraction(
        research_question="How does X affect Y?",
        methodology="GP surrogate with EI",
        datasets_or_benchmarks=["BBOB"],
        variables=["lengthscale"],
        hyperparameters={},
        evaluation_metrics=["simple regret"],
        reported_results=[
            ReportedResult(benchmark="BBOB", metric_name="simple regret", value="0.12", source="abstract")
        ],
        notes="Abstract only.",
    )
    method_ext = SectionExtraction(
        research_question="",
        methodology="Matern52 kernel with ARD",
        hyperparameters={"learning_rate": "1e-3", "batch_size": "32"},
        evaluation_metrics=["NLL"],
        notes="",
    )
    from src.agents.analyst import merge_section_extractions
    by_section = {"abstract": abstract_ext, "method": method_ext}
    merged = merge_section_extractions(
        {
            "abstract": abstract_ext,
            "method": method_ext,
            "experiments": SectionExtraction(),
            "hyperparameters": SectionExtraction(),
            "appendix": SectionExtraction(),
        }
    )
    return ExtractionBundle(by_section=by_section, merged=merged)


# ---------------------------------------------------------------------------
# _format_bundle_as_text
# ---------------------------------------------------------------------------


def test_format_bundle_includes_paper_title() -> None:
    paper = _make_paper()
    bundle = _make_bundle()
    text = _format_bundle_as_text(paper, bundle)
    assert paper.title in text


def test_format_bundle_includes_section_headers() -> None:
    paper = _make_paper()
    bundle = _make_bundle()
    text = _format_bundle_as_text(paper, bundle)
    assert "Section: Abstract" in text
    assert "Section: Method" in text


def test_format_bundle_includes_merged_section() -> None:
    text = _format_bundle_as_text(_make_paper(), _make_bundle())
    assert "=== Merged ===" in text


def test_format_bundle_json_is_parseable() -> None:
    """The merged block must contain valid JSON with the expected schema fields."""
    text = _format_bundle_as_text(_make_paper(), _make_bundle())
    merged_idx = text.index("=== Merged ===")
    # The JSON object starts at the first '{' after the merged header
    json_start = text.index("{", merged_idx)
    parsed = json.loads(text[json_start:])
    assert "research_question" in parsed


# ---------------------------------------------------------------------------
# persist_extraction_bundle
# ---------------------------------------------------------------------------


def test_persist_extraction_bundle_writes_json_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.persistence.EXTRACTIONS_DIR", tmp_path)
    paper = _make_paper()
    bundle = _make_bundle()
    review = ReviewRecord(status="approved")

    result_path = persist_extraction_bundle(paper, bundle, review)

    assert result_path.exists()
    assert result_path.suffix == ".json"


def test_persist_extraction_bundle_writes_txt_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.persistence.EXTRACTIONS_DIR", tmp_path)
    paper = _make_paper()
    bundle = _make_bundle()
    review = ReviewRecord(status="approved")

    persist_extraction_bundle(paper, bundle, review)

    txt_path = tmp_path / f"{paper.paper_id}_sections.txt"
    assert txt_path.exists()
    assert "Section: Abstract" in txt_path.read_text(encoding="utf-8")


def test_persist_extraction_bundle_json_contains_by_section_and_merged(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("src.persistence.EXTRACTIONS_DIR", tmp_path)
    paper = _make_paper()
    bundle = _make_bundle()
    review = ReviewRecord(status="approved")

    json_path = persist_extraction_bundle(paper, bundle, review)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "by_section" in payload
    assert "merged" in payload
    assert "abstract" in payload["by_section"]
    assert payload["merged"]["evaluation_metrics"]


def test_persist_extraction_bundle_review_status_saved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.persistence.EXTRACTIONS_DIR", tmp_path)
    paper = _make_paper()
    bundle = _make_bundle()
    review = ReviewRecord(status="approved", notes="looks good")

    json_path = persist_extraction_bundle(paper, bundle, review)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["review"]["status"] == "approved"
    assert payload["review"]["notes"] == "looks good"
