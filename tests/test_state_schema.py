from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.state import (
    AgentEnvelope,
    AnalystPayload,
    AnalystPayloadCore,
    AnalystPayloadExtensions,
    PlannerPayload,
    PlanStep,
    EngineerPayload,
    EngineerPayloadCore,
    EngineerPayloadExtensions,
    ReviewerPayload,
    ReviewerPayloadCore,
    ReviewerPayloadExtensions,
    ComparisonRow,
    UnknownItem,
    PaperMetadata,
    make_initial_state,
)


def test_make_initial_state_contains_phase_placeholders() -> None:
    paper = PaperMetadata(
        paper_id="sample_paper",
        title="Sample Paper",
        pdf_path="data/papers/sample.pdf",
    )
    state = make_initial_state(paper)

    assert state["paper"].paper_id == "sample_paper"
    assert state["review"].status == "pending"
    assert state["planner_output"].schema_version == "2.0"
    assert state["planner_output"].agent == "planner"
    assert state["planner_output"].payload.plan_summary == ""
    assert state["planner_output_json"] == {}
    assert state["plan_review"].status == "pending"
    assert state["engineer_output"].step_id == ""
    assert state["engineer_review"].status == "pending"
    assert state["executor_result"].final_status == "pending"
    assert state["retry_count"] == 0
    assert state["reviewer_report"].verdict == "inconclusive"


def test_agent_envelope_analyst_full_payload() -> None:
    """Test successful full analyst payload with envelope."""
    payload = AnalystPayload(
        core=AnalystPayloadCore(
            research_question="How does X affect Y?",
            methodology="Experimental study",
            datasets=["dataset1", "dataset2"],
            variables=["var1", "var2"],
            evaluation_metrics=["metric1", "metric2"],
        ),
        extensions=AnalystPayloadExtensions(
            hyperparameters={"learning_rate": "0.01"},
            reported_results=[],
            assumptions=["assumption1"],
            notes="Some notes",
        ),
    )
    envelope = AgentEnvelope[AnalystPayload](
        schema_version="2.0",
        agent="analyst",
        status="ok",
        unknowns=[],
        warnings=[],
        payload=payload,
    )
    assert envelope.schema_version == "2.0"
    assert envelope.agent == "analyst"
    assert envelope.status == "ok"
    assert envelope.payload.core.research_question == "How does X affect Y?"


def test_agent_envelope_analyst_partial_with_unknowns() -> None:
    """Test partial analyst payload with unknowns."""
    payload = AnalystPayload(
        core=AnalystPayloadCore(
            research_question="How does X affect Y?",
            methodology="Experimental study",
            datasets=["dataset1"],
            variables=[],
            evaluation_metrics=["metric1"],
        ),
        extensions=AnalystPayloadExtensions(),
    )
    envelope = AgentEnvelope[AnalystPayload](
        schema_version="2.0",
        agent="analyst",
        status="partial",
        unknowns=[
            UnknownItem(
                field="payload.core.variables",
                reason="No variables mentioned in section",
                severity="medium",
            )
        ],
        warnings=["Limited information available"],
        payload=payload,
    )
    assert envelope.status == "partial"
    assert len(envelope.unknowns) == 1
    assert envelope.unknowns[0].field == "payload.core.variables"
    assert envelope.unknowns[0].severity == "medium"


def test_agent_envelope_planner_flat_payload() -> None:
    """Test planner payload with phase DAG fields."""
    from src.state import PlanPhase, PhaseRunSpec

    payload = PlannerPayload(
        plan_summary="Execute benchmark experiments",
        domain="bayesian_optimization",
        objective="Reproduce paper results",
        phases=[
            PlanPhase(
                phase_id="step_1",
                title="Run baseline",
                goal="Execute baseline experiment",
                run_template="python run.py --config baseline",
                matrix=[
                    PhaseRunSpec(
                        name="baseline",
                        run_command="python run.py --config baseline",
                        code_refs=["run.py"],
                        verify=["exists:results/sample/summary.json"],
                        results_path="results/sample/summary.json",
                    )
                ],
                results_path="results/sample/summary.json",
            )
        ],
        assumptions=["CPU-only execution"],
        constraints=["No external APIs"],
        missing_context=[],
        verification_checks=["Exit code 0"],
        risks=["May timeout"],
        organization=["Setup then matrix"],
        execution=["Reuse run.py"],
        repo_usage=["pip install -r requirements.txt"],
        engineer_notes=["Pin torch versions"],
        results_summary_path="results/sample/summary.json",
    )
    envelope = AgentEnvelope[PlannerPayload](
        schema_version="2.0",
        agent="planner",
        status="ok",
        unknowns=[],
        warnings=[],
        payload=payload,
    )
    assert len(envelope.payload.phases) == 1
    assert envelope.payload.phases[0].phase_id == "step_1"
    assert envelope.payload.phases[0].matrix[0].run_command == "python run.py --config baseline"
    assert envelope.payload.results_summary_path == "results/sample/summary.json"


def test_agent_envelope_engineer_with_patches() -> None:
    """Test engineer payload with patches."""
    payload = EngineerPayload(
        core=EngineerPayloadCore(
            step_id="step_1",
            detected_language="python",
            patches=[],
            verification_commands=["python -m pytest"],
        ),
        extensions=EngineerPayloadExtensions(
            rationale="Detected python from requirements.txt",
            missing_context=[],
            risk_analysis=["Changes assume no external deps"],
        ),
    )
    envelope = AgentEnvelope[EngineerPayload](
        schema_version="2.0",
        agent="engineer",
        status="ok",
        unknowns=[],
        warnings=[],
        payload=payload,
    )
    assert envelope.payload.core.step_id == "step_1"
    assert envelope.payload.core.detected_language == "python"


def test_agent_envelope_reviewer_with_comparison() -> None:
    """Test reviewer payload with comparison rows."""
    payload = ReviewerPayload(
        core=ReviewerPayloadCore(
            summary="Most metrics reproduced",
            verdict="partially_reproduced",
            comparison_rows=[
                ComparisonRow(
                    metric_name="accuracy",
                    benchmark="test_set",
                    reported_value="0.95",
                    reproduced_value="0.93",
                    absolute_difference=-0.02,
                    relative_difference_pct=-2.1,
                    match_status="close",
                )
            ],
        ),
        extensions=ReviewerPayloadExtensions(
            reproduction_rate=0.8,
            risks=["Different random seed"],
            notes=["Close tolerance"],
            artifacts=[],
            run_summary={},
            deep_diagnostics=[],
        ),
    )
    envelope = AgentEnvelope[ReviewerPayload](
        schema_version="2.0",
        agent="reviewer",
        status="ok",
        unknowns=[],
        warnings=[],
        payload=payload,
    )
    assert envelope.payload.core.verdict == "partially_reproduced"
    assert len(envelope.payload.core.comparison_rows) == 1
    assert envelope.payload.core.comparison_rows[0].match_status == "close"


def test_agent_envelope_requires_core_fields() -> None:
    """Test that core fields are required and validated."""
    with pytest.raises(ValidationError):
        # Missing required payload field should fail
        AgentEnvelope(
            schema_version="2.0",
            agent="analyst",
            status="ok",
            unknowns=[],
            warnings=[],
            # payload is missing entirely
        )


def test_agent_envelope_extensions_are_optional() -> None:
    """Test that extensions can be omitted without failure."""
    payload = AnalystPayload(
        core=AnalystPayloadCore(
            research_question="How does X affect Y?",
            methodology="Experimental study",
            datasets=["dataset1"],
            variables=["var1"],
            evaluation_metrics=["metric1"],
        ),
        # extensions omitted, should use default
    )
    envelope = AgentEnvelope[AnalystPayload](
        schema_version="2.0",
        agent="analyst",
        status="ok",
        unknowns=[],
        warnings=[],
        payload=payload,
    )
    assert envelope.payload.extensions is not None
    assert envelope.payload.extensions.hyperparameters == {}
    assert envelope.payload.extensions.notes == ""


def test_unknown_item_severity_levels() -> None:
    """Test UnknownItem with different severity levels."""
    low = UnknownItem(field="test", reason="Missing optional field", severity="low")
    medium = UnknownItem(field="test", reason="Missing recommended field", severity="medium")
    high = UnknownItem(field="test", reason="Missing critical field", severity="high")
    
    assert low.severity == "low"
    assert medium.severity == "medium"
    assert high.severity == "high"
