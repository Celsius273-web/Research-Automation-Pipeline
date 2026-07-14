from __future__ import annotations

import json
from urllib import error

import src.agents.engineer as engineer_module
from src.agents.engineer import PaperEngineer
from src.state import (
    EngineerInputContext,
    ExecutionPlan,
    PaperMetadata,
    PlanStep,
    RepoContext,
)


def _sample_context() -> EngineerInputContext:
    return EngineerInputContext(
        paper=PaperMetadata(
            paper_id="paper_1",
            title="Example Paper",
            pdf_path="data/papers/example.pdf",
        ),
        execution_plan=ExecutionPlan(
            domain="optimization",
            objective="reproduce benchmark",
            steps=[PlanStep(step_id="s1", title="Setup", verification=["pytest -q"])],
        ),
        plan_step=PlanStep(step_id="s1", title="Setup", verification=["pytest -q"]),
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
