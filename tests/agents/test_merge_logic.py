from __future__ import annotations

from src.agents.analyst import merge_section_extractions
from src.state import SectionExtraction


def test_merge_section_extractions_dedupes_and_preserves_first_scalar() -> None:
    by_section = {
        "abstract": SectionExtraction(
            research_question="RQ",
            methodology="Method-A",
            datasets_or_benchmarks=["PD1", "HPO-B"],
            variables=["x", "y"],
            hyperparameters={"lr": "0.01"},
            evaluation_metrics=["regret"],
        ),
        "method": SectionExtraction(
            research_question="",
            methodology="Method-B",
            datasets_or_benchmarks=["HPO-B", "Synthetic"],
            variables=["y", "z"],
            hyperparameters={"batch_size": "32", "lr": "0.02"},
            evaluation_metrics=["regret", "nll"],
        ),
    }

    merged = merge_section_extractions(by_section)
    assert merged.research_question == "RQ"
    assert merged.methodology == "Method-A"
    assert merged.datasets_or_benchmarks == ["PD1", "HPO-B", "Synthetic"]
    assert merged.variables == ["x", "y", "z"]
    assert merged.hyperparameters["lr"] == "0.01"
    assert merged.hyperparameters["batch_size"] == "32"
    assert merged.evaluation_metrics == ["regret", "nll"]
