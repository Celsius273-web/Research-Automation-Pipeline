from __future__ import annotations

import json
from urllib import error

import src.agents.planner as planner_module
from src.agents.planner import PaperPlanner, _convert_envelope_to_legacy_plan, _normalize_legacy_plan_payload
from src.state import (
    AgentEnvelope,
    ExecutionPlan,
    PaperMetadata,
    PlannerInputContext,
    PlannerPayload,
    PlannerPayloadCore,
    PlannerPayloadExtensions,
    PlanStepCore,
    SectionExtraction,
    UnknownItem,
)


def _sample_context() -> PlannerInputContext:
    return PlannerInputContext(
        paper=PaperMetadata(
            paper_id="paper_1",
            title="Example Paper",
            pdf_path="data/papers/example.pdf",
        ),
        approved_extraction=SectionExtraction(
            research_question="How to optimize black-box functions?",
            methodology="Bayesian optimization",
            datasets_or_benchmarks=["BBOB"],
            variables=["acquisition function"],
            hyperparameters={"batch_size": "16"},
            evaluation_metrics=["best regret"],
            notes="CPU only experiments.",
        ),
        runtime_constraints={"hardware": "cpu_only"},
        repo_context={"repo_url": "https://example.com/repo"},
        repo_setup_guide="pip install -r requirements.txt",
        hyperparameter_reference="Table 1: batch_size=16",
    )


def test_planner_build_plan_with_mocked_ollama_call(monkeypatch) -> None:
    planner = PaperPlanner()
    context = _sample_context()

    def fake_call(ctx: PlannerInputContext) -> ExecutionPlan:
        assert ctx.paper.paper_id == "paper_1"
        assert ctx.repo_setup_guide == "pip install -r requirements.txt"
        assert ctx.hyperparameter_reference == "Table 1: batch_size=16"
        return ExecutionPlan(
            plan_summary="Run baseline then BO variants.",
            domain="optimization",
            objective="Reproduce reported regret curves",
            steps=[{"step_id": "setup", "title": "Set up environment"}],
        )

    monkeypatch.setattr(planner, "_call_ollama_json", fake_call)
    out = planner.build_plan(context)
    assert out.plan_summary
    assert out.steps[0].step_id == "setup"


def test_call_ollama_json_retries_after_urlerror(monkeypatch) -> None:
    planner = PaperPlanner()
    context = _sample_context()
    attempts = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            payload = {
                "message": {
                    "content": json.dumps(
                        {
                            "schema_version": "1.0",
                            "plan_summary": "Recovered plan",
                            "domain": "optimization",
                            "objective": "Reproduce paper",
                            "steps": [{"step_id": "s1", "title": "Do task"}],
                        }
                    )
                }
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(_req, timeout=180):
        _ = timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise error.URLError("temporary network issue")
        return _Response()

    monkeypatch.setattr(planner_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)

    out = planner._call_ollama_json(context)
    assert out.plan_summary == "Recovered plan"
    assert out.steps[0].step_id == "s1"
    assert attempts["count"] == 2


def test_planner_parse_envelope_format_new() -> None:
    """Test parsing new envelope format for planner."""
    envelope = AgentEnvelope[PlannerPayload](
        schema_version="2.0",
        agent="planner",
        status="ok",
        unknowns=[],
        warnings=[],
        payload=PlannerPayload(
            core=PlannerPayloadCore(
                plan_summary="Execute benchmark experiments",
                domain="bayesian_optimization",
                objective="Reproduce results",
                steps=[
                    PlanStepCore(
                        step_id="step_1",
                        title="Run baseline",
                        goal="Execute baseline experiment",
                        run_command="python run.py",
                        depends_on=[],
                        results_path="outputs/baseline.json",
                    )
                ],
            ),
            extensions=PlannerPayloadExtensions(
                assumptions=["CPU-only"],
                constraints=["No external APIs"],
            ),
        ),
    )
    
    legacy = _convert_envelope_to_legacy_plan(envelope)
    assert legacy.plan_summary == "Execute benchmark experiments"
    assert legacy.domain == "bayesian_optimization"
    assert len(legacy.steps) == 1
    assert legacy.steps[0].step_id == "step_1"
    assert legacy.assumptions == ["CPU-only"]


def test_planner_parse_envelope_with_unknowns() -> None:
    """Test parsing envelope with unknowns (partial status)."""
    envelope = AgentEnvelope[PlannerPayload](
        schema_version="2.0",
        agent="planner",
        status="partial",
        unknowns=[
            UnknownItem(
                field="payload.extensions.missing_context",
                reason="exact random seed not specified",
                severity="low",
            )
        ],
        warnings=[],
        payload=PlannerPayload(
            core=PlannerPayloadCore(
                plan_summary="Execute experiments",
                domain="optimization",
                objective="Reproduce results",
                steps=[],
            ),
        ),
    )
    
    legacy = _convert_envelope_to_legacy_plan(envelope)
    assert legacy.plan_summary == "Execute experiments"
    assert len(legacy.steps) == 0


def test_planner_normalize_legacy_payload() -> None:
    """Test legacy planner payload normalization."""
    payload = {
        "schema_version": "1.0",
        "plan_summary": "Test plan",
        "domain": "optimization",
        "objective": "Reproduce results",
        "assumptions": "CPU-only",  # Should convert to list
        "steps": [
            {
                "step_id": "step_1",
                "title": "Run experiment",
                "goal": "Execute code",
            }
        ],
    }
    
    normalized = _normalize_legacy_plan_payload(payload)
    assert isinstance(normalized["assumptions"], list)
    assert len(normalized["steps"]) == 1
