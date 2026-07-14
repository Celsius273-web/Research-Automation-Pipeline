from __future__ import annotations

import json

import src.main as main_module
import src.persistence as persistence_module
from src.state import ExecutionPlan, PaperMetadata, PlanReviewRecord, ReviewRecord, SectionExtraction


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
    extraction_dir = tmp_path / "extractions"
    plan_dir = tmp_path / "plans"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(persistence_module, "EXTRACTIONS_DIR", extraction_dir)
    monkeypatch.setattr(persistence_module, "PLANS_DIR", plan_dir)

    extraction_payload = {
        "paper": {"paper_id": "paper_x", "title": "Paper X", "pdf_path": "paper_x.pdf"},
        "review": ReviewRecord(status="approved", notes="ok").model_dump(),
        "approved_extraction": SectionExtraction(
            research_question="RQ",
            methodology="BO",
        ).model_dump(),
    }
    extraction_path = extraction_dir / "paper_x.json"
    extraction_path.write_text(json.dumps(extraction_payload), encoding="utf-8")

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
    output_path = plan_dir / "paper_x.json"
    assert output_path.exists()
