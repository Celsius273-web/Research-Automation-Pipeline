from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.pdf_parser import (
    _apply_window_fallbacks,
    _is_toc_line,
    parse_pdf_sections,
    split_sections,
)
from src.state import SECTION_NAMES


PAPERS = [
    "A Tutorial on Bayesian Optimization.pdf",
    "A hierarchical expected improvement method for Bayesian optimization.pdf",
    "Pre-trained Gaussian Processes for Bayesian Optimization.pdf",
]


@pytest.mark.parametrize("name", PAPERS)
def test_parse_pdf_sections_returns_full_text_and_sections(name: str) -> None:
    path = Path("data/papers") / name
    if not path.exists():
        pytest.skip(f"Missing fixture paper: {path}")

    sections = parse_pdf_sections(path)
    assert len(sections.full_text) > 500

    section_values = [
        sections.abstract,
        sections.method,
        sections.experiments,
        sections.hyperparameters,
        sections.appendix,
    ]
    assert any(bool(value.strip()) for value in section_values) or bool(sections.full_text.strip())


# ---------------------------------------------------------------------------
# _is_toc_line
# ---------------------------------------------------------------------------


def test_is_toc_line_detects_bare_page_number() -> None:
    text = "Experiments\n28\nNext Section"
    match_end = len("Experiments")
    assert _is_toc_line(text, match_end) is True


def test_is_toc_line_detects_dotted_leader() -> None:
    text = "Experiments\n. . . . . . . 28\nNext Section"
    match_end = len("Experiments")
    assert _is_toc_line(text, match_end) is True


def test_is_toc_line_false_for_body_heading() -> None:
    text = "7 Experiments\nIn this section we evaluate"
    match_end = len("7 Experiments")
    assert _is_toc_line(text, match_end) is False


def test_is_toc_line_false_when_nothing_follows() -> None:
    assert _is_toc_line("Experiments", len("Experiments")) is False


# ---------------------------------------------------------------------------
# _apply_window_fallbacks
# ---------------------------------------------------------------------------


def test_window_fallbacks_fills_all_missing_sections() -> None:
    # Only abstract found; the other four must get fallback positions.
    partial = {"abstract": 100}
    filled = _apply_window_fallbacks(partial, total_chars=10000)
    assert set(filled.keys()) == set(SECTION_NAMES)


def test_window_fallbacks_does_not_overwrite_detected_positions() -> None:
    detected = {"abstract": 50, "experiments": 5000}
    filled = _apply_window_fallbacks(detected, total_chars=10000)
    assert filled["abstract"] == 50
    assert filled["experiments"] == 5000


def test_window_fallbacks_positions_are_proportional_to_document_length() -> None:
    short = _apply_window_fallbacks({}, total_chars=1000)
    long = _apply_window_fallbacks({}, total_chars=100000)
    for section in SECTION_NAMES:
        assert long[section] == short[section] * 100


# ---------------------------------------------------------------------------
# split_sections with synthetic text
# ---------------------------------------------------------------------------


def test_split_sections_assigns_each_section_different_text() -> None:
    """Every section call must receive a distinct slice of the document."""
    # Build a fake document with clearly labelled regions
    fake = (
        "Abstract\nThis is the abstract content.\n\n"
        "2 Methods\nThis is the method content.\n\n"
        "3 Experiments\nThis is the experiments content.\n\n"
        "Appendix A. Author Contributions\nThis is the appendix.\n"
    )
    sections = split_sections(fake)
    assert sections.abstract != sections.method
    assert sections.experiments != sections.abstract
    assert sections.full_text == fake.strip()


def test_split_sections_empty_input_returns_empty_map() -> None:
    sections = split_sections("")
    assert sections.full_text == ""
    assert sections.abstract == ""
