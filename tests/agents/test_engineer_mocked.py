from __future__ import annotations

import json
from urllib import error

import src.agents.engineer as engineer_module
from src.agents.engineer import PaperEngineer
from src.state import (
    AgentEnvelope,
    EngineerInputContext,
    PaperMetadata,
    PlannerPayload,
    PlanPhase,
    PlanStep,
    RepoContext,
    project_phases_to_steps,
)


def _sample_context() -> EngineerInputContext:
    plan = AgentEnvelope[PlannerPayload](
        schema_version="2.0",
        agent="planner",
        status="ok",
        unknowns=[],
        warnings=[],
        payload=PlannerPayload(
            domain="optimization",
            objective="reproduce benchmark",
            phases=[
                PlanPhase(
                    phase_id="s1",
                    title="Setup",
                    run_template="pytest -q",
                    matrix=[],
                )
            ],
        ),
    )
    step = project_phases_to_steps(plan.payload)[0]
    return EngineerInputContext(
        paper=PaperMetadata(
            paper_id="paper_1",
            title="Example Paper",
            pdf_path="data/papers/example.pdf",
        ),
        execution_plan=plan,
        plan_step=step,
        repo_context=RepoContext(
            repo_path="/tmp/repo",
            language="python",
            build_system="pyproject",
        ),
        runtime_constraints={"max_retry_attempts": "5"},
    )


def test_engineer_propose_patch_with_mocked_call(monkeypatch) -> None:
    agent = PaperEngineer()
    context = _sample_context()

    def fake_call(_: EngineerInputContext):
        return engineer_module.EngineerOutput(
            step_id="s1",
            patches=[
                {
                    "file_path": "src/app.py",
                    "action": "modify",
                    "content": "print('ok')\n",
                    "rationale": "Fix command path.",
                }
            ],
            verification_commands=["pytest -q"],
            rationale="Minimal fix",
        )

    monkeypatch.setattr(agent, "_call_ollama_json", fake_call)
    out = agent.propose_patch(context)
    assert out.step_id == "s1"
    assert out.patches[0].action == "modify"


def test_engineer_call_retries_after_urlerror(monkeypatch) -> None:
    agent = PaperEngineer()
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
                            "step_id": "s1",
                            "patches": [
                                {
                                    "file_path": "main.py",
                                    "action": "modify",
                                    "content": "print('run')\n",
                                    "rationale": "update",
                                }
                            ],
                            "verification_commands": ["pytest -q"],
                            "rationale": "done",
                            "missing_context": [],
                        }
                    )
                }
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(_req, timeout=180):
        _ = timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise error.URLError("temporary issue")
        return _Response()

    monkeypatch.setattr(engineer_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(engineer_module.time, "sleep", lambda _seconds: None)
    out = agent._call_ollama_json(context)
    assert out.step_id == "s1"
    assert attempts["count"] == 2
