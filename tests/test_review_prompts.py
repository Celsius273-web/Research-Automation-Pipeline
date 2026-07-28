"""Tests for interactive review prompt helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.review_prompts import (
    ReviewCancelledError,
    is_quit_choice,
    prompt_input,
    run_cli_review,
)
from src.state import SectionExtraction


def test_is_quit_choice_recognizes_common_aliases() -> None:
    assert is_quit_choice("q")
    assert is_quit_choice("QUIT")
    assert is_quit_choice(" exit ")
    assert not is_quit_choice("quick")


def test_prompt_input_optional_allows_empty_value() -> None:
    with patch("builtins.input", return_value=""):
        assert prompt_input("Optional review notes", optional=True) == ""


def test_prompt_input_raises_when_user_quits() -> None:
    with patch("builtins.input", return_value="q"):
        with pytest.raises(ReviewCancelledError):
            prompt_input("Optional review notes", optional=True)


def test_prompt_input_raises_on_eof() -> None:
    with patch("builtins.input", side_effect=EOFError):
        with pytest.raises(ReviewCancelledError):
            prompt_input("Optional review notes", optional=True)


def test_run_cli_review_allows_skipping_optional_notes() -> None:
    extraction = SectionExtraction(research_question="What is X?")
    approvals = [""] * 8 + ["", ""]
    with patch("builtins.input", side_effect=approvals):
        approved, review = run_cli_review(extraction)
    assert approved.research_question == "What is X?"
    assert review.status == "approved"
    assert review.notes == ""


def test_run_cli_review_quits_from_optional_notes() -> None:
    extraction = SectionExtraction(research_question="What is X?")
    approvals = [""] * 8 + ["", "q"]
    with patch("builtins.input", side_effect=approvals):
        with pytest.raises(ReviewCancelledError):
            run_cli_review(extraction)
