"""Phase-wise LangGraph wiring."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from src.config import MAX_RETRY_ATTEMPTS
from src.state import ResearchState


def build_phase1_graph(
    parse_node: Callable[[ResearchState], ResearchState],
    analyst_node: Callable[[ResearchState], ResearchState],
    review_node: Callable[[ResearchState], ResearchState],
):
    graph = StateGraph(ResearchState)

    graph.add_node("parse_pdf", parse_node)
    graph.add_node("analyst_extract", analyst_node)
    graph.add_node("review_checkpoint", review_node)
    graph.add_edge(START, "parse_pdf")
    graph.add_edge("parse_pdf", "analyst_extract")
    graph.add_edge("analyst_extract", "review_checkpoint")
    graph.add_edge("review_checkpoint", END)
    return graph.compile()


def build_phase2_graph(
    parse_node: Callable[[ResearchState], ResearchState],
    analyst_node: Callable[[ResearchState], ResearchState],
    review_node: Callable[[ResearchState], ResearchState],
    planner_node: Callable[[ResearchState], ResearchState],
):
    graph = StateGraph(ResearchState)

    graph.add_node("parse_pdf", parse_node)
    graph.add_node("analyst_extract", analyst_node)
    graph.add_node("review_checkpoint", review_node)
    graph.add_node("planner_plan", planner_node)
    graph.add_edge(START, "parse_pdf")
    graph.add_edge("parse_pdf", "analyst_extract")
    graph.add_edge("analyst_extract", "review_checkpoint")

    def _route_after_review(state: ResearchState) -> str:
        review = state.get("review")
        status = getattr(review, "status", "pending")
        if status == "approved":
            return "planner"
        return "end"

    graph.add_conditional_edges(
        "review_checkpoint",
        _route_after_review,
        {"planner": "planner_plan", "end": END},
    )
    graph.add_edge("planner_plan", END)
    return graph.compile()


def build_phase3_graph(
    parse_node: Callable[[ResearchState], ResearchState],
    analyst_node: Callable[[ResearchState], ResearchState],
    review_node: Callable[[ResearchState], ResearchState],
    planner_node: Callable[[ResearchState], ResearchState],
    engineer_node: Callable[[ResearchState], ResearchState],
    engineer_review_node: Callable[[ResearchState], ResearchState],
    executor_node: Callable[[ResearchState], ResearchState],
    max_retries: int = MAX_RETRY_ATTEMPTS,
):
    graph = StateGraph(ResearchState)

    graph.add_node("parse_pdf", parse_node)
    graph.add_node("analyst_extract", analyst_node)
    graph.add_node("review_checkpoint", review_node)
    graph.add_node("planner_plan", planner_node)
    graph.add_node("engineer_patch", engineer_node)
    graph.add_node("engineer_review_checkpoint", engineer_review_node)
    graph.add_node("executor_run", executor_node)

    graph.add_edge(START, "parse_pdf")
    graph.add_edge("parse_pdf", "analyst_extract")
    graph.add_edge("analyst_extract", "review_checkpoint")

    def _route_after_review(state: ResearchState) -> str:
        review = state.get("review")
        status = getattr(review, "status", "pending")
        if status == "approved":
            return "planner"
        return "end"

    graph.add_conditional_edges(
        "review_checkpoint",
        _route_after_review,
        {"planner": "planner_plan", "end": END},
    )
    graph.add_edge("planner_plan", "engineer_patch")
    graph.add_edge("engineer_patch", "engineer_review_checkpoint")

    def _route_after_engineer_review(state: ResearchState) -> str:
        review = state.get("engineer_review")
        status = getattr(review, "status", "pending")
        if status == "approved":
            return "executor"
        return "end"

    graph.add_conditional_edges(
        "engineer_review_checkpoint",
        _route_after_engineer_review,
        {"executor": "executor_run", "end": END},
    )

    def _route_after_executor(state: ResearchState) -> str:
        result = state.get("executor_result")
        status = getattr(result, "final_status", "failed")
        retries = int(state.get("retry_count", 0))
        if status == "success":
            return "end"
        if retries < max_retries:
            return "engineer"
        return "end"

    graph.add_conditional_edges(
        "executor_run",
        _route_after_executor,
        {"engineer": "engineer_patch", "end": END},
    )
    return graph.compile()


def build_phase4_graph(
    parse_node: Callable[[ResearchState], ResearchState],
    analyst_node: Callable[[ResearchState], ResearchState],
    review_node: Callable[[ResearchState], ResearchState],
    planner_node: Callable[[ResearchState], ResearchState],
    engineer_node: Callable[[ResearchState], ResearchState],
    engineer_review_node: Callable[[ResearchState], ResearchState],
    executor_node: Callable[[ResearchState], ResearchState],
    reviewer_node: Callable[[ResearchState], ResearchState],
    max_retries: int = MAX_RETRY_ATTEMPTS,
):
    graph = StateGraph(ResearchState)

    graph.add_node("parse_pdf", parse_node)
    graph.add_node("analyst_extract", analyst_node)
    graph.add_node("review_checkpoint", review_node)
    graph.add_node("planner_plan", planner_node)
    graph.add_node("engineer_patch", engineer_node)
    graph.add_node("engineer_review_checkpoint", engineer_review_node)
    graph.add_node("executor_run", executor_node)
    graph.add_node("reviewer_generate", reviewer_node)

    graph.add_edge(START, "parse_pdf")
    graph.add_edge("parse_pdf", "analyst_extract")
    graph.add_edge("analyst_extract", "review_checkpoint")

    def _route_after_review(state: ResearchState) -> str:
        review = state.get("review")
        status = getattr(review, "status", "pending")
        if status == "approved":
            return "planner"
        return "reviewer"

    graph.add_conditional_edges(
        "review_checkpoint",
        _route_after_review,
        {"planner": "planner_plan", "reviewer": "reviewer_generate"},
    )

    graph.add_edge("planner_plan", "engineer_patch")
    graph.add_edge("engineer_patch", "engineer_review_checkpoint")

    def _route_after_engineer_review(state: ResearchState) -> str:
        review = state.get("engineer_review")
        status = getattr(review, "status", "pending")
        if status == "approved":
            return "executor"
        return "reviewer"

    graph.add_conditional_edges(
        "engineer_review_checkpoint",
        _route_after_engineer_review,
        {"executor": "executor_run", "reviewer": "reviewer_generate"},
    )

    def _route_after_executor(state: ResearchState) -> str:
        result = state.get("executor_result")
        status = getattr(result, "final_status", "failed")
        retries = int(state.get("retry_count", 0))
        if status == "success":
            return "reviewer"
        if retries < max_retries:
            return "engineer"
        return "reviewer"

    graph.add_conditional_edges(
        "executor_run",
        _route_after_executor,
        {"engineer": "engineer_patch", "reviewer": "reviewer_generate"},
    )
    graph.add_edge("reviewer_generate", END)
    return graph.compile()
