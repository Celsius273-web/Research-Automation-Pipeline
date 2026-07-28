from __future__ import annotations

import json
from pathlib import Path
from urllib import error

import src.agents.planner as planner_module
from src.agents.planner import (
    PaperPlanner,
    _convert_envelope_to_legacy_plan,
    _extraction_to_analyst_dict,
    _normalize_legacy_plan_payload,
    _plan_misses_aim,
)
from src.state import (
    AgentEnvelope,
    ExecutionPlan,
    PaperMetadata,
    PlannerInputContext,
    PlannerPayload,
    PlannerPayloadCore,
    PlannerPayloadExtensions,
    PlanStepCore,
    ReportedResult,
    SectionExtraction,
    UnknownItem,
)


def _sample_context(tmp_path: Path | None = None) -> PlannerInputContext:
    bundle_path = str(tmp_path) if tmp_path is not None else None
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
            reported_results=[
                ReportedResult(
                    benchmark="BBOB",
                    metric_name="best regret",
                    value="0.12",
                    source="Table 1",
                )
            ],
            notes="CPU only experiments.",
        ),
        runtime_constraints={"hardware": "cpu_only"},
        repo_context={"repo_url": "https://example.com/repo"},
        repo_setup_guide="pip install -r requirements.txt",
        hyperparameter_reference="Table 1: batch_size=16",
        paper_bundle_path=bundle_path,
    )


def _ok_envelope_content() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "agent": "planner",
        "status": "ok",
        "unknowns": [],
        "warnings": [],
        "payload": {
            "core": {
                "plan_summary": "Reproduce BO benchmarks from the paper aim.",
                "domain": "optimization",
                "objective": "How to optimize black-box functions under a limited budget.",
                "steps": [
                    {
                        "step_id": "s1",
                        "title": "Run baseline",
                        "goal": "Execute baseline",
                        "run_command": "python run.py",
                        "depends_on": [],
                        "results_path": "outputs/out.json",
                    }
                ],
            },
            "extensions": {
                "assumptions": [],
                "constraints": [],
                "missing_context": [],
                "experiment_matrix": [
                    {
                        "name": "bbob",
                        "target": "best regret",
                        "variables": ["acquisition function"],
                        "hyperparameters": {"batch_size": "16"},
                        "metrics": ["best regret"],
                    }
                ],
                "verification_checks": [],
                "risks": [],
            },
        },
    }


def _bad_aim_envelope_content() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "agent": "planner",
        "status": "partial",
        "unknowns": [
            {
                "field": "research_question",
                "reason": "Analyst output explicitly states missing research question",
                "severity": "high",
            }
        ],
        "warnings": [],
        "payload": {
            "core": {
                "plan_summary": "",
                "domain": "optimization",
                "objective": "",
                "steps": [],
            },
            "extensions": {},
        },
    }


def test_extraction_to_analyst_dict_uses_full_schema() -> None:
    extraction = _sample_context().approved_extraction
    payload = _extraction_to_analyst_dict(extraction)
    assert "datasets_or_benchmarks" in payload
    assert payload["datasets_or_benchmarks"] == ["BBOB"]
    assert isinstance(payload["hyperparameters"], dict)
    assert payload["hyperparameters"] == {"batch_size": "16"}
    assert payload["reported_results"][0]["metric_name"] == "best regret"
    assert payload["notes"] == "CPU only experiments."
    assert "datasets" not in payload


def test_planner_ollama_payload_includes_full_analyst_output(monkeypatch, tmp_path: Path) -> None:
    planner = PaperPlanner()
    context = _sample_context(tmp_path)
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": json.dumps(_ok_envelope_content())}}).encode(
                "utf-8"
            )

    def fake_urlopen(req, timeout=180):
        _ = timeout
        body = json.loads(req.data.decode("utf-8"))
        user_content = body["messages"][1]["content"]
        # Context JSON is after the header text.
        context_json = user_content.split("Context JSON:\n", 1)[1]
        captured["context"] = json.loads(context_json)
        return _Response()

    monkeypatch.setattr(planner_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)

    out = planner._call_ollama_json(context)
    assert isinstance(out, AgentEnvelope)
    planner_input = captured["context"]
    assert set(planner_input) == {
        "analyst_output",
        "repo_context",
        "paper_context",
        "flags",
    }
    analyst = planner_input["analyst_output"]
    assert "datasets_or_benchmarks" in analyst
    assert isinstance(analyst["hyperparameters"], dict)
    assert "reported_results" in analyst
    assert analyst["notes"] == "CPU only experiments."
    assert planner_input["flags"] == {
        "has_research_question": True,
        "has_methodology": True,
        "has_code_repo": False,
        "has_datasets": True,
        "paper_type": "methods",
    }
    assert "example_commands" in planner_input["repo_context"]
    assert (tmp_path / "planner_debug.md").exists()


def test_planner_soft_retries_when_aim_marked_unknown(monkeypatch, tmp_path: Path) -> None:
    planner = PaperPlanner()
    context = _sample_context(tmp_path)
    attempts = {"count": 0}
    user_prompts: list[str] = []

    class _Response:
        def __init__(self, content: dict[str, object]):
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": json.dumps(self._content)}}).encode("utf-8")

    def fake_urlopen(req, timeout=180):
        _ = timeout
        attempts["count"] += 1
        body = json.loads(req.data.decode("utf-8"))
        user_prompts.append(body["messages"][1]["content"])
        if attempts["count"] == 1:
            return _Response(_bad_aim_envelope_content())
        return _Response(_ok_envelope_content())

    monkeypatch.setattr(planner_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)

    out = planner._call_ollama_json(context)
    assert attempts["count"] == 2
    assert isinstance(out, AgentEnvelope)
    assert out.payload.core.objective
    assert "do not mark it unknown" in user_prompts[1]
    assert not _plan_misses_aim(
        context.approved_extraction.research_question,
        context.approved_extraction.methodology,
        out,
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


def test_call_ollama_json_retries_after_urlerror(monkeypatch, tmp_path: Path) -> None:
    planner = PaperPlanner()
    context = _sample_context(tmp_path)
    attempts = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            payload = {
                "message": {
                    "content": json.dumps(_ok_envelope_content())
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
    assert out.payload.core.plan_summary == "Reproduce BO benchmarks from the paper aim."
    assert out.payload.core.steps[0].step_id == "s1"
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


def test_plan_misses_aim_detects_unknown_research_question() -> None:
    extraction = _sample_context().approved_extraction
    bad = AgentEnvelope[PlannerPayload].model_validate(_bad_aim_envelope_content())
    good = AgentEnvelope[PlannerPayload].model_validate(_ok_envelope_content())
    assert _plan_misses_aim(extraction.research_question, extraction.methodology, bad) is True
    assert _plan_misses_aim(extraction.research_question, extraction.methodology, good) is False


def test_soften_blocked_when_methodology_present() -> None:
    from src.agents.planner import _should_soften_blocked, _soften_blocked_envelope

    blocked = AgentEnvelope[PlannerPayload].model_validate(
        {
            "schema_version": "2.0",
            "agent": "planner",
            "status": "blocked",
            "unknowns": [
                {
                    "field": "research_question",
                    "reason": "missing",
                    "severity": "high",
                }
            ],
            "warnings": [],
            "payload": {
                "core": {
                    "plan_summary": "Run toolkit examples",
                    "domain": "graphs",
                    "objective": "Exercise STAG local clustering",
                    "steps": [],
                },
                "extensions": {},
            },
        }
    )
    assert _should_soften_blocked("", "STAG is an open-source library.", blocked) is True
    softened = _soften_blocked_envelope(blocked)
    assert softened.status == "partial"
