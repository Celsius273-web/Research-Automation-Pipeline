from __future__ import annotations

from src.graphs.research_graph import build_phase3_graph
from src.state import (
    EngineerOutput,
    EngineerReviewRecord,
    ExecutorResult,
    PaperMetadata,
    ReviewRecord,
    RunAttempt,
    make_initial_state,
)


def _base_state():
    state = make_initial_state(PaperMetadata(paper_id="p1", title="Paper", pdf_path="paper.pdf"))
    state["review"] = ReviewRecord(status="approved")
    return state


def test_phase3_graph_success_exits() -> None:
    state = _base_state()
    calls = {"engineer": 0, "executor": 0}

    def passthrough(s):
        return s

    def engineer_node(s):
        calls["engineer"] += 1
        s["engineer_output"] = EngineerOutput(step_id="s1")
        return s

    def engineer_review_node(s):
        s["engineer_review"] = EngineerReviewRecord(status="approved")
        return s

    def executor_node(s):
        calls["executor"] += 1
        s["executor_result"] = ExecutorResult(
            attempts=[RunAttempt(attempt_number=1, step_id="s1", command="pytest -q", exit_code=0, success=True, failure_type="none")],
            final_status="success",
            total_attempts=1,
        )
        return s

    graph = build_phase3_graph(
        parse_node=passthrough,
        analyst_node=passthrough,
        review_node=passthrough,
        planner_node=passthrough,
        engineer_node=engineer_node,
        engineer_review_node=engineer_review_node,
        executor_node=executor_node,
        max_retries=5,
    )
    out = graph.invoke(state)
    assert calls["engineer"] == 1
    assert calls["executor"] == 1
    assert out["executor_result"].final_status == "success"


def test_phase3_graph_retry_is_bounded() -> None:
    state = _base_state()
    calls = {"engineer": 0}

    def passthrough(s):
        return s

    def engineer_node(s):
        calls["engineer"] += 1
        s["engineer_output"] = EngineerOutput(step_id="s1")
        return s

    def engineer_review_node(s):
        s["engineer_review"] = EngineerReviewRecord(status="approved")
        return s

    def executor_node(s):
        retry_count = int(s.get("retry_count", 0)) + 1
        s["retry_count"] = retry_count
        s["executor_result"] = ExecutorResult(
            attempts=[RunAttempt(attempt_number=retry_count, step_id="s1", command="pytest -q", exit_code=1, success=False)],
            final_status="failed",
            total_attempts=retry_count,
        )
        return s

    graph = build_phase3_graph(
        parse_node=passthrough,
        analyst_node=passthrough,
        review_node=passthrough,
        planner_node=passthrough,
        engineer_node=engineer_node,
        engineer_review_node=engineer_review_node,
        executor_node=executor_node,
        max_retries=2,
    )
    out = graph.invoke(state)
    assert calls["engineer"] == 2
    assert out["retry_count"] == 2


def test_phase3_graph_rejected_engineer_review_skips_executor() -> None:
    state = _base_state()
    calls = {"executor": 0}

    def passthrough(s):
        return s

    def engineer_node(s):
        s["engineer_output"] = EngineerOutput(step_id="s1")
        return s

    def engineer_review_node(s):
        s["engineer_review"] = EngineerReviewRecord(status="rejected")
        return s

    def executor_node(s):
        calls["executor"] += 1
        return s

    graph = build_phase3_graph(
        parse_node=passthrough,
        analyst_node=passthrough,
        review_node=passthrough,
        planner_node=passthrough,
        engineer_node=engineer_node,
        engineer_review_node=engineer_review_node,
        executor_node=executor_node,
        max_retries=2,
    )
    graph.invoke(state)
    assert calls["executor"] == 0
