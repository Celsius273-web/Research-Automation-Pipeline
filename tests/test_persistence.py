"""Tests for persistence.py — specifically the extraction bundle functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bundle import PaperBundle
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
    # Mock bundle directory
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", tmp_path)
    monkeypatch.setattr("src.persistence.EXTRACTIONS_DIR", tmp_path)
    paper = _make_paper()
    bundle = _make_bundle()
    review = ReviewRecord(status="approved")

    persist_extraction_bundle(paper, bundle, review)

    # Check bundle structure
    bundle_dir = tmp_path / paper.paper_id
    txt_path = bundle_dir / "extraction_sections.txt"
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


# ---------------------------------------------------------------------------
# PaperBundle management tests
# ---------------------------------------------------------------------------


def test_paper_bundle_creation(tmp_path, monkeypatch):
    """Test basic bundle creation and directory structure."""
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", tmp_path)
    
    bundle = PaperBundle("test_paper")
    assert not bundle.exists()
    
    bundle.create_bundle_dir()
    assert bundle.exists()
    assert bundle.bundle_dir.exists()
    assert bundle.runs_dir.exists()


def test_paper_bundle_extraction_persistence(tmp_path, monkeypatch):
    """Test extraction save/load in bundle format."""
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", tmp_path)
    
    bundle = PaperBundle("test_paper")
    bundle.create_bundle_dir()
    
    # Create test data
    paper = PaperMetadata(paper_id="test_paper", title="Test Paper", pdf_path="test.pdf")
    extraction = SectionExtraction(
        research_question="Test question",
        methodology="Test methodology",
        datasets_or_benchmarks=["Dataset1"],
        evaluation_metrics=["accuracy"],
    )
    extraction_bundle = ExtractionBundle(by_section={}, merged=extraction)
    review = ReviewRecord(status="approved", notes="Test review")
    
    # Save extraction
    bundle.save_extraction(extraction_bundle, review, paper)
    assert bundle.extraction_path.exists()
    
    # Load extraction
    loaded_bundle = bundle.get_extraction()
    assert loaded_bundle is not None
    assert loaded_bundle.merged.research_question == "Test question"
    assert loaded_bundle.merged.methodology == "Test methodology"


def test_paper_bundle_no_code_scenario(tmp_path, monkeypatch):
    """Test bundle behavior when no code repository is available."""
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", tmp_path)
    
    bundle = PaperBundle("test_paper")
    bundle.create_bundle_dir()
    
    # Test no code detection
    assert not bundle.has_code()
    
    # Test repo info for no code
    repo_info = bundle.get_repo_info()
    assert repo_info.language == "unknown"
    assert "No code repository available" in repo_info.notes
    
    # Test setup guide for no code
    setup_guide = bundle.get_setup_guide()
    assert setup_guide == ""


def test_paper_bundle_with_code(tmp_path, monkeypatch):
    """Test bundle behavior when code repository is available."""
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", tmp_path)
    
    bundle = PaperBundle("test_paper")
    bundle.create_bundle_dir()
    
    # Create mock code directory with files
    bundle.code_dir.mkdir()
    (bundle.code_dir / "main.py").write_text("print('hello')")
    (bundle.code_dir / "requirements.txt").write_text("numpy==1.21.0")
    
    # Create README with setup instructions
    readme_content = """# Test Project

## Installation

Run the following commands:

```bash
pip install -r requirements.txt
python main.py
```

## Usage

This is a test project.
"""
    (bundle.code_dir / "README.md").write_text(readme_content)
    
    # Test code detection
    assert bundle.has_code()
    
    # Test setup guide extraction
    setup_guide = bundle.get_setup_guide()
    assert "Installation" in setup_guide
    assert "pip install -r requirements.txt" in setup_guide


def test_paper_bundle_hyperparameter_reference(tmp_path, monkeypatch):
    """Test hyperparameter reference extraction from extraction."""
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", tmp_path)
    
    bundle = PaperBundle("test_paper")
    bundle.create_bundle_dir()
    
    # Create extraction with hyperparameters
    paper = PaperMetadata(paper_id="test_paper", title="Test Paper", pdf_path="test.pdf")
    extraction = SectionExtraction(
        research_question="Test question",
        hyperparameters={"lr": "0.001", "batch_size": "32", "epochs": "100"}
    )
    extraction_bundle = ExtractionBundle(by_section={}, merged=extraction)
    review = ReviewRecord(status="approved")
    
    bundle.save_extraction(extraction_bundle, review, paper)
    
    # Test hyperparameter reference
    ref = bundle.get_hyperparameter_reference()
    assert "lr: 0.001" in ref
    assert "batch_size: 32" in ref
    assert "epochs: 100" in ref


def test_paper_bundle_metadata(tmp_path, monkeypatch):
    """Test metadata save/load functionality."""
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", tmp_path)
    
    bundle = PaperBundle("test_paper")
    
    # Test no metadata initially
    assert bundle.get_metadata() is None
    
    # Save metadata
    metadata = {
        "paper_id": "test_paper",
        "title": "Test Paper",
        "arxiv_id": "1234.5678",
        "repo_url": "https://github.com/test/repo"
    }
    bundle.save_metadata(metadata)
    assert bundle.metadata_path.exists()
    
    # Load metadata
    loaded = bundle.get_metadata()
    assert loaded is not None
    assert loaded["paper_id"] == "test_paper"
    assert loaded["title"] == "Test Paper"
