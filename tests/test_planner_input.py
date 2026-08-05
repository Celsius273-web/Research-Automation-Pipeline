from __future__ import annotations

from pathlib import Path

from src.planner_input import (
    build_planner_prompt_context,
    build_unified_planner_input,
    categorize_planner_notes,
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
from src.tools.repo_context import (
    extract_entrypoint_hints,
    infer_build_command,
    summarize_readme,
    summarize_repo_tree,
)


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


def test_planner_prompt_context_cleans_lists_and_categorizes_notes() -> None:
    context = PlannerInputContext(
        paper=PaperMetadata(paper_id="p1", title="Method", pdf_path="paper.pdf"),
        approved_extraction=SectionExtraction(
            research_question="How does transfer improve BO?",
            methodology="Pre-train a GP prior.",
            datasets_or_benchmarks=["PD1", "PD1", "unknown: benchmark"],
            variables=[
                "Acquisition function",
                "key_findings_100_words_max_no_tables_allowed_in_output_field",
            ],
            hyperparameters={"optimizer": "Adam"},
            evaluation_metrics=["Regret", "Regret"],
            notes=(
                "- Adam optimizer was used for training.\n"
                "- ImageNet has only 100 points due to resource constraints.\n"
                "- Carefully pin dependency versions.\n"
                "- The method transfers knowledge across related tasks."
            ),
        ),
        repo_context={
            "language": "python",
            "has_code": True,
            "build_system": "pip install .",
            "example_commands": [
                "python3 -m venv env-pd",
                "python scripts/run_pd1.py --task TASK",
            ],
        },
    )

    prompt_context = build_planner_prompt_context(build_unified_planner_input(context))
    analyst = prompt_context["analyst_output"]
    repo = prompt_context["repo_context"]

    assert analyst["datasets_or_benchmarks"] == ["PD1"]
    assert analyst["variables"] == ["Acquisition function"]
    assert analyst["evaluation_metrics"] == ["Regret"]
    assert analyst["notes"]["implementation_details"] == [
        "Adam optimizer was used for training."
    ]
    assert analyst["notes"]["issues"] == [
        "ImageNet has only 100 points due to resource constraints."
    ]
    assert analyst["notes"]["warnings"] == ["Carefully pin dependency versions."]
    assert analyst["notes"]["what_to_know"] == [
        "The method transfers knowledge across related tasks."
    ]
    assert repo["example_commands"] == ["python scripts/run_pd1.py --task TASK"]
    assert repo["has_runnable_experiment_command"] is True


def test_repo_context_excludes_setup_commands_and_finds_entrypoint_hints(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "hyperbo_demo.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "Use [the demo](hyperbo_demo.ipynb).\n"
        "```\npython3 -m venv env-pd\npython -m pytest\n```\n",
        encoding="utf-8",
    )

    from src.tools.repo_context import extract_example_commands

    assert extract_example_commands(tmp_path) == []
    assert extract_entrypoint_hints(tmp_path) == ["hyperbo_demo.ipynb"]


def test_planner_prompt_context_includes_repo_exploration(tmp_path: Path) -> None:
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "run_exp.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Demo\n\n```\npython exp/run_exp.py --fun townsend\n```\n",
        encoding="utf-8",
    )
    context = PlannerInputContext(
        paper=PaperMetadata(paper_id="p1", title="Method", pdf_path="paper.pdf"),
        approved_extraction=SectionExtraction(
            research_question="How does transfer improve BO?",
            methodology="Pre-train a GP prior.",
            datasets_or_benchmarks=["PD1"],
            variables=["Acquisition function"],
            hyperparameters={"optimizer": "Adam"},
            evaluation_metrics=["Regret"],
            notes="- Use Adam optimizer.",
        ),
        repo_context={
            "language": "python",
            "has_code": True,
            "build_system": "pip install .",
            "example_commands": ["python exp/run_exp.py --fun townsend"],
            "repo_path": str(tmp_path),
        },
    )

    prompt_context = build_planner_prompt_context(
        build_unified_planner_input(context),
        repo_path=tmp_path,
    )

    assert prompt_context["repo_exploration"]["available"] is True
    assert "readme_full" in prompt_context["repo_exploration"]
    assert prompt_context["repo_context"]["has_runnable_experiment_command"] is True
