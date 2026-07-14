"""Acceptance tests for the Paper Analyst agent.

Each test names a concrete behavior and would catch a real regression if it failed.
Tests against parsing and validation logic use fixed sample outputs; no live model call is made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.agents.analyst import (
    PaperAnalyst,
    _clean_json_response,
    _dedupe_keep_order,
    merge_section_extractions,
)
from src.state import ReportedResult, SectionExtraction, SectionTextMap


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_clean_json_strips_json_fenced_block() -> None:
    raw = '```json\n{"key": "value"}\n```'
    assert _clean_json_response(raw) == '{"key": "value"}'


def test_clean_json_strips_plain_fenced_block() -> None:
    raw = '```\n{"key": "value"}\n```'
    assert _clean_json_response(raw) == '{"key": "value"}'


def test_clean_json_passes_through_bare_json() -> None:
    raw = '{"key": "value"}'
    assert _clean_json_response(raw) == '{"key": "value"}'


def test_dedupe_keep_order_removes_case_insensitive_duplicates() -> None:
    result = _dedupe_keep_order(["BBOB", "COCO", "bbob", "COCO"])
    assert result == ["BBOB", "COCO"]


def test_dedupe_keep_order_strips_surrounding_whitespace() -> None:
    result = _dedupe_keep_order(["  BBOB  ", "BBOB"])
    assert result == ["BBOB"]


def test_dedupe_keep_order_drops_empty_and_whitespace_only_strings() -> None:
    result = _dedupe_keep_order(["", "BBOB", "   "])
    assert result == ["BBOB"]


# ---------------------------------------------------------------------------
# merge_section_extractions
# ---------------------------------------------------------------------------


def _ext(**kwargs: object) -> SectionExtraction:
    return SectionExtraction(**kwargs)


def test_merge_uses_first_nonempty_research_question() -> None:
    extractions = {
        "abstract": _ext(research_question="RQ from abstract"),
        "method": _ext(research_question="RQ from method"),
        "experiments": _ext(),
        "hyperparameters": _ext(),
        "appendix": _ext(),
    }
    merged = merge_section_extractions(extractions)
    assert merged.research_question == "RQ from abstract"


def test_merge_deduplicates_datasets_across_sections() -> None:
    extractions = {
        "abstract": _ext(datasets_or_benchmarks=["BBOB", "COCO"]),
        "method": _ext(datasets_or_benchmarks=["BBOB"]),
        "experiments": _ext(datasets_or_benchmarks=["HPO-B"]),
        "hyperparameters": _ext(),
        "appendix": _ext(),
    }
    merged = merge_section_extractions(extractions)
    assert merged.datasets_or_benchmarks == ["BBOB", "COCO", "HPO-B"]


def test_merge_experiments_section_overwrites_earlier_reported_results_for_same_key() -> None:
    """The experiments section is the authoritative source for numeric results."""
    early = ReportedResult(benchmark="BBOB", metric_name="regret", value="0.50", source="abstract")
    definitive = ReportedResult(benchmark="BBOB", metric_name="regret", value="0.12", source="Table 1")
    extractions = {
        "abstract": _ext(reported_results=[early]),
        "method": _ext(),
        "experiments": _ext(reported_results=[definitive]),
        "hyperparameters": _ext(),
        "appendix": _ext(),
    }
    merged = merge_section_extractions(extractions)
    assert len(merged.reported_results) == 1
    assert merged.reported_results[0].value == "0.12"


def test_merge_hyperparameters_keeps_first_seen_value_for_duplicate_keys() -> None:
    extractions = {
        "abstract": _ext(hyperparameters={"lr": "1e-3"}),
        "method": _ext(hyperparameters={"lr": "5e-4", "batch_size": "32"}),
        "experiments": _ext(),
        "hyperparameters": _ext(),
        "appendix": _ext(),
    }
    merged = merge_section_extractions(extractions)
    assert merged.hyperparameters["lr"] == "1e-3"
    assert merged.hyperparameters["batch_size"] == "32"


def test_merge_notes_are_concatenated_across_sections() -> None:
    extractions = {
        "abstract": _ext(notes="CPU-only mentioned"),
        "method": _ext(notes="Theoretical bound included"),
        "experiments": _ext(),
        "hyperparameters": _ext(),
        "appendix": _ext(),
    }
    merged = merge_section_extractions(extractions)
    assert "CPU-only mentioned" in merged.notes
    assert "Theoretical bound included" in merged.notes


# ---------------------------------------------------------------------------
# PaperAnalyst._call_ollama_json — parsing and retry logic
# ---------------------------------------------------------------------------


def _flat_payload() -> dict:
    return {
        "research_question": "How does X affect Y?",
        "methodology": "GP surrogate with EI acquisition",
        "datasets_or_benchmarks": ["BBOB"],
        "variables": ["acquisition temperature", "lengthscale"],
        "hyperparameters": {"seed": "42", "budget": "200"},
        "evaluation_metrics": ["simple regret", "log regret"],
        "reported_results": [
            {
                "benchmark": "BBOB",
                "metric_name": "simple regret",
                "value": "0.12",
                "source": "Table 1",
            }
        ],
        "notes": "CPU-only run",
    }


def _urlopen_returning(payload: dict) -> MagicMock:
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"message": {"content": json.dumps(payload)}}).encode()

    return MagicMock(return_value=FakeResp())


def test_call_ollama_json_parses_all_fields_from_valid_response(monkeypatch) -> None:
    analyst = PaperAnalyst(max_parse_retries=0)
    monkeypatch.setattr("src.agents.analyst.request.urlopen", _urlopen_returning(_flat_payload()))
    result = analyst._call_ollama_json("method", "some method text")
    assert result.research_question == "How does X affect Y?"
    assert result.evaluation_metrics == ["simple regret", "log regret"]
    assert result.hyperparameters == {"seed": "42", "budget": "200"}
    assert result.reported_results[0].metric_name == "simple regret"


def test_call_ollama_json_retries_exactly_max_retries_plus_one_times(monkeypatch) -> None:
    """max_parse_retries=1 means 2 total HTTP calls before the RuntimeError is raised."""
    analyst = PaperAnalyst(max_parse_retries=1)
    call_count = 0

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            nonlocal call_count
            call_count += 1
            return json.dumps({"message": {"content": json.dumps({"not_a_valid": "schema"})}}).encode()

    monkeypatch.setattr("src.agents.analyst.request.urlopen", MagicMock(return_value=FakeResp()))
    monkeypatch.setattr("src.agents.analyst.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError):
        analyst._call_ollama_json("abstract", "some text")

    assert call_count == 2


def test_call_ollama_json_error_message_includes_raw_response(monkeypatch) -> None:
    """Raw response must be visible in the error so failures are debuggable."""
    analyst = PaperAnalyst(max_parse_retries=0)
    garbage = "THIS IS NOT JSON AT ALL"

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"message": {"content": garbage}}).encode()

    monkeypatch.setattr("src.agents.analyst.request.urlopen", MagicMock(return_value=FakeResp()))
    monkeypatch.setattr("src.agents.analyst.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError) as exc_info:
        analyst._call_ollama_json("abstract", "some text")

    assert garbage in str(exc_info.value)


# ---------------------------------------------------------------------------
# PaperAnalyst.extract — pipeline-level behavior
# ---------------------------------------------------------------------------


def _analyst_with_fixed_response() -> PaperAnalyst:
    """Returns a PaperAnalyst whose _call_ollama_json always returns a fixed extraction."""
    flat = _flat_payload()
    analyst = PaperAnalyst()
    analyst._call_ollama_json = lambda section, text: SectionExtraction.model_validate(flat)
    return analyst


def test_extract_produces_all_five_section_keys() -> None:
    analyst = _analyst_with_fixed_response()
    bundle = analyst.extract(
        SectionTextMap(
            abstract="Abstract.",
            method="Method.",
            experiments="Experiments.",
            hyperparameters="Hyperparameters.",
            appendix="Appendix.",
        )
    )
    assert set(bundle.by_section.keys()) == {"abstract", "method", "experiments", "hyperparameters", "appendix"}


def test_extract_merged_result_contains_nonempty_core_fields() -> None:
    analyst = _analyst_with_fixed_response()
    bundle = analyst.extract(
        SectionTextMap(
            abstract="Abstract.",
            method="Method.",
            experiments="Experiments.",
            hyperparameters="Hyperparameters.",
            appendix="Appendix.",
        )
    )
    assert bundle.merged.research_question
    assert bundle.merged.evaluation_metrics
    assert bundle.merged.hyperparameters


def test_extract_passes_full_text_to_empty_sections() -> None:
    """Empty named sections should receive full_text, not an empty string."""
    received: dict[str, str] = {}

    def tracking_call(section: str, text: str) -> SectionExtraction:
        received[section] = text
        return SectionExtraction.model_validate(_flat_payload())

    analyst = PaperAnalyst()
    analyst._call_ollama_json = tracking_call

    analyst.extract(SectionTextMap(abstract="Has content", full_text="FALLBACK"))

    assert received["abstract"] == "Has content"
    for section in ("method", "experiments", "hyperparameters", "appendix"):
        assert received[section] == "FALLBACK", f"section '{section}' should receive full_text"


def test_extract_raises_when_no_text_available_for_any_section() -> None:
    analyst = PaperAnalyst()
    with pytest.raises(RuntimeError, match="No usable text"):
        analyst.extract(SectionTextMap())


# ---------------------------------------------------------------------------
# _build_retry_reminder — targeted error feedback to the model
# ---------------------------------------------------------------------------


def test_build_retry_reminder_names_forbidden_fields_from_validation_error() -> None:
    """Forbidden field names from the ValidationError must appear in the retry prompt."""
    from src.agents.analyst import _build_retry_reminder
    from pydantic import ValidationError as PydanticError

    try:
        SectionExtraction.model_validate({"title": "foo", "authors": ["bar"]})
    except PydanticError as exc:
        reminder = _build_retry_reminder(exc)

    assert "title" in reminder
    assert "authors" in reminder


def test_build_retry_reminder_includes_valid_schema_example() -> None:
    from src.agents.analyst import _build_retry_reminder, _SCHEMA_REMINDER

    reminder = _build_retry_reminder(None)
    assert _SCHEMA_REMINDER in reminder


# ---------------------------------------------------------------------------
# extract — graceful per-section failure handling
# ---------------------------------------------------------------------------


def test_extract_continues_when_one_section_fails_after_retries() -> None:
    """A RuntimeError from a single section must not abort the whole extraction."""

    def failing_on_abstract(section: str, text: str) -> SectionExtraction:
        if section == "abstract":
            raise RuntimeError("Simulated model failure for abstract")
        return SectionExtraction.model_validate(_flat_payload())

    analyst = PaperAnalyst()
    analyst._call_ollama_json = failing_on_abstract

    bundle = analyst.extract(
        SectionTextMap(
            abstract="A", method="M", experiments="E",
            hyperparameters="H", appendix="P",
        )
    )

    assert bundle.by_section["abstract"] == SectionExtraction()
    assert bundle.by_section["method"].research_question == "How does X affect Y?"
    assert set(bundle.by_section.keys()) == {"abstract", "method", "experiments", "hyperparameters", "appendix"}
