from __future__ import annotations

from pathlib import Path

from src.planner_input import (
    build_unified_planner_input,
    derive_planner_flags,
    load_unified_planner_input,
)
from src.state import (
    PaperMetadata,
    PlannerAnalystOutput,
    PlannerInputContext,
    PlannerRepoContext,
    SectionExtraction,
)
from src.tools.repo_context import infer_build_command, summarize_readme, summarize_repo_tree


def test_boundary_example_validates_against_unified_contract() -> None:
    fixture = Path(__file__).parents[1] / "planner_input_example_boundary_exploration_bo.json"
    planner_input = load_unified_planner_input(fixture)

    assert planner_input.paper_context.paper_id == "boundary_exploration_bo"
    assert planner_input.flags.paper_type == "methods"
    assert "Townsend Function (2D)" in planner_input.analyst_output.datasets_or_benchmarks
    assert planner_input.analyst_output.hyperparameters["hidden_layers"] == [1, 2, 3, 4]
    assert "unknown physical limits" in planner_input.analyst_output.paper_overview.lower()


def test_build_unified_input_derives_flags_without_unknown_sentinels() -> None:
    context = PlannerInputContext(
        paper=PaperMetadata(paper_id="p1", title="Method", pdf_path="paper.pdf"),
        approved_extraction=SectionExtraction(
            methodology="Propose a constrained optimization method.",
            datasets_or_benchmarks=["Townsend"],
        ),
        repo_context={"language": "python", "has_code": True},
    )

    planner_input = build_unified_planner_input(context)

    assert planner_input.analyst_output.research_question == ""
    assert planner_input.flags.has_research_question is False
    assert planner_input.flags.has_methodology is True
    assert planner_input.flags.has_code_repo is True
    assert planner_input.flags.has_datasets is True
    assert planner_input.flags.paper_type == "methods"


def test_derive_flags_classifies_toolkit() -> None:
    analyst = PlannerAnalystOutput(
        research_question="How can graph algorithms run efficiently?",
        methodology="An open-source C++ and Python library implementing spectral algorithms.",
        datasets_or_benchmarks=[],
        variables=[],
        hyperparameters={},
        evaluation_metrics=[],
        reported_results=[],
        notes="",
    )
    repo = PlannerRepoContext(
        url="",
        language="python",
        build_system="pip install .",
        has_code=True,
        setup_time_minutes=5,
        file_tree="src/",
        readme_summary="Toolkit",
        example_commands=["python examples/run.py"],
    )

    flags = derive_planner_flags(analyst, repo)
    assert flags.paper_type == "toolkit"
    assert flags.has_methodology is True


def test_repo_context_helpers_extract_concrete_setup(tmp_path: Path) -> None:
    (tmp_path / "exp").mkdir()
    (tmp_path / "run.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Demo\n\nOfficial implementation. It runs benchmarks.\n\n"
        "## Installation\n\n```\npip install numpy torch\n```\n\n"
        "## Run\n\n```\npython exp/run_exp.py --fun townsend --algo becbo\n```\n",
        encoding="utf-8",
    )

    from src.tools.repo_context import extract_example_commands

    assert summarize_repo_tree(tmp_path) == "README.md, exp/, run.py"
    assert summarize_readme(tmp_path).startswith("Demo Official implementation.")
    assert infer_build_command(tmp_path, "unknown") == "pip install numpy torch"
    assert extract_example_commands(tmp_path) == [
        "python exp/run_exp.py --fun townsend --algo becbo"
    ]
