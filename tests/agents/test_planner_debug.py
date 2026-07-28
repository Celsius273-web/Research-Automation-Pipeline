"""Tests for planner debug trace recording and log conversion."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.planner_debug import (
    PlannerDebugTrace,
    compare_planner_output_to_saved_plan,
    parse_planner_log_text,
    render_planner_debug_markdown,
    write_planner_debug_files,
)


def test_write_planner_debug_files_includes_input_and_output(tmp_path: Path) -> None:
    trace = PlannerDebugTrace(
        paper_id="demo",
        model="test-model",
        received_context={
            "analyst_output": {
                "research_question": "How to optimize?",
                "methodology": "BO",
                "datasets_or_benchmarks": ["BBOB"],
                "variables": ["acq"],
                "hyperparameters": {"batch_size": "16"},
                "evaluation_metrics": ["regret"],
                "reported_results": [{"metric_name": "regret", "value": "0.1"}],
                "notes": "cpu",
            },
            "extraction_sections": {},
            "repo_context": {"language": "python"},
        },
        system_prompt="system",
    )
    attempt = trace.add_attempt(reminder="none", user_prompt="user", system_prompt="system")
    attempt.raw_response = '{"schema_version":"2.0"}'
    attempt.parsed = {
        "schema_version": "2.0",
        "agent": "planner",
        "status": "ok",
        "unknowns": [],
        "warnings": [],
        "payload": {
            "core": {
                "plan_summary": "Run BO",
                "domain": "bo",
                "objective": "How to optimize?",
                "steps": [{"step_id": "s1", "title": "run"}],
            },
            "extensions": {"experiment_matrix": []},
        },
    }
    attempt.outcome = "accepted"
    trace.final_output = attempt.parsed

    saved_plan = {
        "execution_plan": {
            "plan_summary": "Run BO",
            "objective": "How to optimize?",
            "steps": [{"step_id": "s1", "title": "run"}],
            "experiment_matrix": [],
        }
    }
    json_path, md_path = write_planner_debug_files(trace, tmp_path, saved_plan=saved_plan)
    assert json_path.exists()
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "Input the planner received" in md
    assert "How to optimize?" in md
    assert "Attempt 1" in md
    assert "Comparison to saved plan JSON" in md
    assert "objective matches saved plan" in md


def test_compare_flags_rq_marked_unknown_when_present() -> None:
    notes = compare_planner_output_to_saved_plan(
        final_output={
            "unknowns": [{"field": "research_question", "reason": "missing"}],
            "payload": {
                "core": {"plan_summary": "x", "objective": "", "steps": []},
                "extensions": {"experiment_matrix": []},
            },
        },
        saved_plan={"execution_plan": {"plan_summary": "", "objective": "", "steps": []}},
        received_context={
            "analyst_output": {
                "research_question": "Real question",
                "datasets_or_benchmarks": ["Cora", "QM9"],
            }
        },
    )
    assert any("research_question" in note and "ISSUE" in note for note in notes)
    assert any("saved plan is empty" in note for note in notes)


def test_parse_planner_log_text_extracts_payload_and_response() -> None:
    req = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": (
                    "Create a plan.\n\nContext JSON:\n"
                    + json.dumps(
                        {
                            "analyst_output": {
                                "research_question": "Q?",
                                "datasets_or_benchmarks": ["BBOB"],
                            }
                        }
                    )
                ),
            },
        ],
    }
    resp = {
        "schema_version": "2.0",
        "agent": "planner",
        "status": "ok",
        "payload": {"core": {"plan_summary": "s", "objective": "Q?", "steps": []}},
    }
    log_text = (
        f"INFO: Planner prompt payload: {json.dumps(req)}\n"
        f"INFO: Planner raw response: {json.dumps(resp)}\n"
    )
    trace = parse_planner_log_text(log_text, paper_id="from_log")
    assert trace.model == "qwen"
    assert trace.received_context["analyst_output"]["research_question"] == "Q?"
    assert len(trace.attempts) == 1
    assert trace.attempts[0].parsed["agent"] == "planner"
    assert trace.final_output is not None
    md = render_planner_debug_markdown(trace)
    assert "from_log" in md
    assert "Q?" in md
