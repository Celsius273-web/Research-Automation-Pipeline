from __future__ import annotations

from src.graphs.research_graph import build_phase2_graph
from src.state import (
    AgentEnvelope,
    PaperMetadata,
    PlannerPayload,
    PlanReviewRecord,
    PlanPhase,
    ReviewRecord,
    SectionExtraction,
    make_initial_state,
)


def test_phase2_graph_calls_planner_when_review_approved() -> None:
    state = make_initial_state(
        PaperMetadata(paper_id="paper_a", title="Paper A", pdf_path="data/papers/paper_a.pdf")
    )
    calls = {"planner": 0}

    def parse_node(s):
        return s

    def analyst_node(s):
        s["extraction"].merged = SectionExtraction(research_question="rq")
        return s

    def review_node(s):
        s["approved_extraction"] = SectionExtraction(research_question="rq")
        s["review"] = ReviewRecord(status="approved", notes="")
        return s

    def planner_node(s):
        calls["planner"] += 1
        s["planner_output"] = AgentEnvelope[PlannerPayload](
            schema_version="2.0",
            agent="planner",
            status="ok",
            unknowns=[],
            warnings=[],
            payload=PlannerPayload(
                plan_summary="Plan ready",
                phases=[PlanPhase(phase_id="step_1", title="Run experiment")],
            ),
        )
        s["plan_review"] = PlanReviewRecord(status="approved")
        return s

    graph = build_phase2_graph(parse_node, analyst_node, review_node, planner_node)
    out = graph.invoke(state)

    assert calls["planner"] == 1
    assert out["planner_output"].payload.plan_summary == "Plan ready"


def test_phase2_graph_skips_planner_when_review_rejected() -> None:
    state = make_initial_state(
        PaperMetadata(paper_id="paper_b", title="Paper B", pdf_path="data/papers/paper_b.pdf")
    )
    calls = {"planner": 0}

    def parse_node(s):
        return s

    def analyst_node(s):
        return s

    def review_node(s):
        s["review"] = ReviewRecord(status="rejected", notes="missing data")
        return s

    def planner_node(s):
        calls["planner"] += 1
        return s

    graph = build_phase2_graph(parse_node, analyst_node, review_node, planner_node)
    graph.invoke(state)
    assert calls["planner"] == 0
