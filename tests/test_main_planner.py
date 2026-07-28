from __future__ import annotations

import json
from pathlib import Path

import src.main as main_module
import src.persistence as persistence_module
from src.bundle import PaperBundle
from src.config import PAPER_BUNDLES_DIR
from src.state import (
    AgentEnvelope,
    ExecutionPlan,
    PaperMetadata,
    PlannerPayload,
    PlanReviewRecord,
    ReviewRecord,
    SectionExtraction,
)


def test_persist_plan_writes_expected_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(persistence_module, "PLANS_DIR", tmp_path / "plans")
    paper = PaperMetadata(paper_id="p1", title="Paper 1", pdf_path="paper.pdf")
    plan = ExecutionPlan(
        plan_summary="Test plan",
        steps=[{"step_id": "s1", "title": "Step one"}],
    )
    review = PlanReviewRecord(status="approved", notes="looks good")

    saved = persistence_module.persist_plan(
        paper=paper,
        plan=plan,
        plan_review=review,
        source_extraction_path="data/extractions/p1.json",
    )
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["paper"]["paper_id"] == "p1"
    assert payload["plan_review"]["status"] == "approved"
    assert payload["execution_plan"]["steps"][0]["step_id"] == "s1"


def test_run_plan_with_paper_id_uses_extraction_artifact(tmp_path, monkeypatch) -> None:
    # Set up bundle directory structure
    bundles_dir = tmp_path / "papers"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.config.PAPER_BUNDLES_DIR", bundles_dir)
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", bundles_dir)
    
    # Create paper bundle with extraction
    bundle = PaperBundle("paper_x")
    bundle.create_bundle_dir()
    
    # Create extraction in bundle format
    extraction_payload = {
        "paper": {"paper_id": "paper_x", "title": "Paper X", "pdf_path": "paper_x.pdf"},
        "review": ReviewRecord(status="approved", notes="ok").model_dump(),
        "by_section": {},
        "merged": SectionExtraction(
            research_question="RQ",
            methodology="BO",
        ).model_dump(),
    }
    bundle.extraction_path.write_text(json.dumps(extraction_payload), encoding="utf-8")

    def fake_make_planner_node(non_interactive: bool):
        assert non_interactive is True

        def planner_node(state):
            state["planner_output"] = ExecutionPlan(
                plan_summary="Planner output",
                steps=[{"step_id": "s1", "title": "Run"}],
            )
            state["planner_output_json"] = state["planner_output"].model_dump()
            state["plan_review"] = PlanReviewRecord(status="approved", notes="auto")
            return state

        return planner_node

    monkeypatch.setattr(main_module, "make_planner_node", fake_make_planner_node)
    code = main_module.run_plan(extraction_path=None, paper_id="paper_x", non_interactive=True)
    assert code == 0
    # Check that plan was saved to bundle
    assert bundle.plan_path.exists()


def test_run_plan_accepts_unified_input_json(tmp_path, monkeypatch) -> None:
    bundles_dir = tmp_path / "papers"
    monkeypatch.setattr("src.bundle.PAPER_BUNDLES_DIR", bundles_dir)
    fixture = Path(__file__).parents[1] / "planner_input_example_boundary_exploration_bo.json"

    def fake_make_planner_node(non_interactive: bool, unified_input=None):
        assert non_interactive is True
        assert unified_input.paper_context.paper_id == "boundary_exploration_bo"

        def planner_node(state):
            state["planner_output"] = AgentEnvelope[PlannerPayload].model_validate(
                {
                    "schema_version": "2.0",
                    "agent": "planner",
                    "status": "ok",
                    "unknowns": [],
                    "warnings": [],
                    "payload": {
                        "core": {
                            "plan_summary": "Run BE-CBO on Townsend using the official repository.",
                            "domain": "bayesian_optimization",
                            "objective": "Evaluate BE-CBO at unknown feasibility boundaries.",
                            "steps": [
                                {
                                    "step_id": "run_townsend",
                                    "title": "Run Townsend",
                                    "goal": "Evaluate BE-CBO on Townsend Function (2D).",
                                    "run_command": "python exp/run_exp.py --fun townsend --algo becbo",
                                    "depends_on": [],
                                    "results_path": "results/townsend.json",
                                }
                            ],
                        },
                        "extensions": {},
                    },
                }
            )
            state["plan_review"] = PlanReviewRecord(status="approved", notes="auto")
            return state

        return planner_node

    monkeypatch.setattr(main_module, "make_planner_node", fake_make_planner_node)
    code = main_module.run_plan(
        extraction_path=None,
        paper_id=None,
        non_interactive=True,
        input_json=str(fixture),
    )

    assert code == 0
    saved = PaperBundle("boundary_exploration_bo").plan_path
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert "BE-CBO" in payload["plan_envelope"]["payload"]["core"]["plan_summary"]
