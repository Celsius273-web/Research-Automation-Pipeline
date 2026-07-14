from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agents.analyst import PaperAnalyst
from src.state import SectionExtraction, SectionTextMap


def test_analyst_extract_with_mocked_ollama_call(monkeypatch) -> None:
    analyst = PaperAnalyst()

    def fake_call(section: str, section_text: str) -> SectionExtraction:
        return SectionExtraction(
            research_question=f"rq-{section}",
            methodology=f"m-{section}",
            datasets_or_benchmarks=[section],
            variables=[f"v-{section}"],
            hyperparameters={f"hp_{section}": "1"},
            evaluation_metrics=[f"metric-{section}"],
            notes=f"n-{section}",
        )

    monkeypatch.setattr(analyst, "_call_ollama_json", fake_call)

    sections = SectionTextMap(
        abstract="Abstract text",
        method="Method text",
        experiments="Experiments text",
        hyperparameters="Hyperparameters text",
        appendix="Appendix text",
        full_text="Fallback full text",
    )
    out = analyst.extract(sections)

    assert set(out.by_section.keys()) == {
        "abstract",
        "method",
        "experiments",
        "hyperparameters",
        "appendix",
    }
    assert out.merged.datasets_or_benchmarks
    assert out.merged.hyperparameters


def test_analyst_parses_flat_section_extraction(monkeypatch) -> None:
    """Model JSON matching SectionExtraction validates directly."""
    analyst = PaperAnalyst(max_parse_retries=0)
    flat_payload = {
        "research_question": "How does X affect Y?",
        "methodology": "Experimental study with a GP surrogate",
        "datasets_or_benchmarks": ["BBOB"],
        "variables": ["acquisition temperature"],
        "hyperparameters": {"seed": "0", "budget": "100"},
        "evaluation_metrics": ["simple regret"],
        "reported_results": [
            {
                "benchmark": "BBOB",
                "metric_name": "simple regret",
                "value": "0.12",
                "source": "Table 1",
            }
        ],
        "notes": "CPU-only mention in text",
    }

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"message": {"content": json.dumps(flat_payload)}}
            ).encode("utf-8")

    monkeypatch.setattr(
        "src.agents.analyst.request.urlopen",
        MagicMock(return_value=FakeResp()),
    )

    out = analyst._call_ollama_json("method", "some method text")
    assert out.research_question == "How does X affect Y?"
    assert out.datasets_or_benchmarks == ["BBOB"]
    assert out.hyperparameters == {"seed": "0", "budget": "100"}
    assert out.reported_results[0].metric_name == "simple regret"


def test_analyst_rejects_envelope_shaped_payload(monkeypatch) -> None:
    """Envelope-shaped JSON must not be accepted as SectionExtraction."""
    analyst = PaperAnalyst(max_parse_retries=0)
    envelope_payload = {
        "schema_version": "2.0",
        "agent": "analyst",
        "status": "ok",
        "unknowns": [],
        "warnings": [],
        "payload": {
            "core": {
                "research_question": "How does X affect Y?",
                "methodology": "Experimental",
                "datasets": ["d1"],
                "variables": [],
                "evaluation_metrics": [],
            }
        },
    }

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"message": {"content": json.dumps(envelope_payload)}}
            ).encode("utf-8")

    monkeypatch.setattr(
        "src.agents.analyst.request.urlopen",
        MagicMock(return_value=FakeResp()),
    )
    monkeypatch.setattr("src.agents.analyst.time.sleep", lambda *_: None)

    try:
        analyst._call_ollama_json("method", "some method text")
        raised = False
    except RuntimeError:
        raised = True
    assert raised
