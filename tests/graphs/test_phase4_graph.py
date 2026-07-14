from __future__ import annotations

from src.graphs.research_graph import build_phase4_graph
from src.state import (
    EngineerReviewRecord,
    ExecutorResult,
    PaperMetadata,
    ReviewRecord,
    RunAttempt,
    make_initial_state,
)


def _base_state():
    state = make_initial_state(PaperMetadata(paper_id="paper_1", title="Paper", pdf_path="paper.pdf"))
    state["review"] = ReviewRecord(status="approved")
    return state


def test_phase4_routes_to_reviewer_on_success() -> None:
    state = _base_state()
    calls = {"reviewer": 0}

    def passthrough(s):
        return s

    def engineer_node(s):
        return s

    def engineer_review_node(s):
        s["engineer_review"] = EngineerReviewRecord(status="approved")
        return s

    def executor_node(s):
        s["executor_result"] = ExecutorResult(
            attempts=[RunAttempt(attempt_number=1, step_id="s1", command="pytest", exit_code=0, success=True, failure_type="none")],
            final_status="success",
            total_attempts=1,
        )
        return s

    def reviewer_node(s):
        calls["reviewer"] += 1
        return s

    graph = build_phase4_graph(
        parse_node=passthrough,
        analyst_node=passthrough,
        review_node=passthrough,
        planner_node=passthrough,
        engineer_node=engineer_node,
        engineer_review_node=engineer_review_node,
        executor_node=executor_node,
        reviewer_node=reviewer_node,
        max_retries=2,
    )
    graph.invoke(state)
    assert calls["reviewer"] == 1


def test_phase4_routes_to_reviewer_when_engineer_rejected() -> None:
    state = _base_state()
    calls = {"reviewer": 0, "executor": 0}

    def passthrough(s):
        return s

    def engineer_node(s):
        return s

    def engineer_review_node(s):
        s["engineer_review"] = EngineerReviewRecord(status="rejected")
        return s

    def executor_node(s):
        calls["executor"] += 1
        return s

    def reviewer_node(s):
        calls["reviewer"] += 1
        return s

    graph = build_phase4_graph(
        parse_node=passthrough,
        analyst_node=passthrough,
        review_node=passthrough,
        planner_node=passthrough,
        engineer_node=engineer_node,
        engineer_review_node=engineer_review_node,
        executor_node=executor_node,
        reviewer_node=reviewer_node,
        max_retries=2,
    )
    graph.invoke(state)
    assert calls["executor"] == 0
    assert calls["reviewer"] == 1


def test_phase4_routes_to_reviewer_on_exhausted_retries() -> None:
    state = _base_state()
    calls = {"reviewer": 0}

    def passthrough(s):
        return s

    def engineer_node(s):
        return s

    def engineer_review_node(s):
        s["engineer_review"] = EngineerReviewRecord(status="approved")
        return s

    def executor_node(s):
        s["retry_count"] = 2
        s["executor_result"] = ExecutorResult(
            attempts=[RunAttempt(attempt_number=2, step_id="s1", command="pytest", exit_code=1, success=False)],
            final_status="exhausted_retries",
            total_attempts=2,
        )
        return s

    def reviewer_node(s):
        calls["reviewer"] += 1
        return s

    graph = build_phase4_graph(
        parse_node=passthrough,
        analyst_node=passthrough,
        review_node=passthrough,
        planner_node=passthrough,
        engineer_node=engineer_node,
        engineer_review_node=engineer_review_node,
        executor_node=executor_node,
        reviewer_node=reviewer_node,
        max_retries=2,
    )
    graph.invoke(state)
    assert calls["reviewer"] == 1
