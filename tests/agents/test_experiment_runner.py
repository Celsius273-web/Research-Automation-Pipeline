"""Unit tests for plan-driven ExperimentRunner / Executor integration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agents.experiment_runner import ExperimentRunner
from src.agents.executor import ExecutorAgent
from src.state import RepoContext
from src.tools.docker_executor import DockerExecutor


class _FakeContainer:
    def __init__(self, status_code: int = 0, stdout: str = "ok", stderr: str = ""):
        self._status_code = status_code
        self._stdout = stdout
        self._stderr = stderr
        self.removed = False

    def wait(self, timeout=600):
        _ = timeout
        return {"StatusCode": self._status_code}

    def logs(self, stdout=True, stderr=False):
        if stdout:
            return self._stdout.encode("utf-8")
        if stderr:
            return self._stderr.encode("utf-8")
        return b""

    def kill(self):
        self._status_code = 124

    def remove(self, force=True):
        _ = force
        self.removed = True


class _FakeDockerClient:
    def __init__(self, status_code: int = 0):
        self.images = SimpleNamespace(build=self._build)
        self.containers = SimpleNamespace(run=self._run)
        self.status_code = status_code
        self.commands: list[str] = []

    def ping(self) -> bool:
        return True

    def _build(self, **_kwargs):
        return None, []

    def _run(self, image, command=None, **_kwargs):
        _ = image
        if command and len(command) >= 3:
            self.commands.append(command[2])
        container = _FakeContainer(status_code=self.status_code)
        self.last_container = container
        return container


def _monkeypatch_docker(monkeypatch, fake_client: _FakeDockerClient) -> None:
    fake_module = SimpleNamespace(
        from_env=lambda **_kwargs: fake_client,
        DockerClient=lambda **_kwargs: fake_client,
        errors=SimpleNamespace(APIError=Exception),
    )
    monkeypatch.setitem(__import__("sys").modules, "docker", fake_module)


def _write_minimal_plan(plan_path: Path, paper_id: str) -> None:
    plan = {
        "paper": {"paper_id": paper_id, "title": "Test", "pdf_path": "x.pdf"},
        "plan_envelope": {
            "schema_version": "2.0",
            "agent": "planner",
            "status": "ok",
            "unknowns": [],
            "warnings": [],
            "payload": {
                "plan_summary": "smoke",
                "domain": "bo",
                "objective": "run",
                "results_summary_path": "results/summary.json",
                "phases": [
                    {
                        "phase_id": "setup",
                        "title": "Setup",
                        "goal": "install",
                        "depends_on": [],
                        "run_template": "pip install -e .",
                        "matrix": [],
                    },
                    {
                        "phase_id": "smoke",
                        "title": "Smoke",
                        "goal": "one run",
                        "depends_on": ["setup"],
                        "matrix": [
                            {
                                "name": "smoke__lsq",
                                "variables": {"benchmark": "lsq"},
                                "run_command": "python run.py --fun lsq",
                                "results_path": "results/row.json",
                            }
                        ],
                    },
                ],
            },
        },
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")


def test_executor_run_container_command(monkeypatch, tmp_path: Path) -> None:
    fake_client = _FakeDockerClient(status_code=0)
    _monkeypatch_docker(monkeypatch, fake_client)
    docker = DockerExecutor(project_root=tmp_path)
    dockerfile = tmp_path / "docker" / "python.Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM python:3.11-slim\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    agent = ExecutorAgent(project_root=tmp_path, runs_dir=tmp_path / "runs", docker_executor=docker)
    result = agent.run_container_command(
        repo_context=RepoContext(repo_path=str(repo), language="python"),
        command="pytest -q",
    )
    assert result.exit_code == 0
    assert fake_client.commands == ["pytest -q"]


def test_experiment_runner_executes_phases_and_captures_metrics(monkeypatch, tmp_path: Path) -> None:
    fake_client = _FakeDockerClient(status_code=0)
    _monkeypatch_docker(monkeypatch, fake_client)

    paper_id = "demo_paper"
    papers_dir = tmp_path / "papers" / paper_id
    code_dir = papers_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    results_dir = code_dir / "results"
    results_dir.mkdir()
    (results_dir / "row.json").write_text(
        json.dumps([{"benchmark": "lsq", "metric_name": "regret", "value": 0.04}]),
        encoding="utf-8",
    )
    (results_dir / "summary.json").write_text(
        json.dumps([{"benchmark": "lsq", "metric_name": "regret", "value": 0.04}]),
        encoding="utf-8",
    )
    _write_minimal_plan(papers_dir / f"{paper_id}_plan.json", paper_id)

    import src.bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "PAPER_BUNDLES_DIR", tmp_path / "papers")
    monkeypatch.setattr("src.agents.experiment_runner.available_memory_gb", lambda: 8.0)
    monkeypatch.setattr("src.agents.experiment_runner.ROOT_DIR", tmp_path)

    docker = DockerExecutor(project_root=tmp_path)
    dockerfile = tmp_path / "docker" / "python.Dockerfile"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("FROM python:3.11-slim\n", encoding="utf-8")

    runner = ExperimentRunner(docker_executor=docker, max_attempts=2)
    metrics_doc, run_dir = runner.execute_paper(paper_id=paper_id)

    assert metrics_doc.run_status == "SUCCESS"
    assert metrics_doc.phases_completed == ["setup", "smoke"]
    assert metrics_doc.phases_failed == []
    assert any(m.metric_name == "regret" for m in metrics_doc.metrics)
    assert run_dir.name == "R1"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "engineer.log").exists()
    assert len(fake_client.commands) == 2


def test_experiment_runner_retries_then_fails(monkeypatch, tmp_path: Path) -> None:
    fake_client = _FakeDockerClient(status_code=1)
    _monkeypatch_docker(monkeypatch, fake_client)

    paper_id = "fail_paper"
    papers_dir = tmp_path / "papers" / paper_id
    code_dir = papers_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "setup.py").write_text("from setuptools import setup\nsetup(name='x')\n", encoding="utf-8")
    _write_minimal_plan(papers_dir / f"{paper_id}_plan.json", paper_id)

    import src.bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "PAPER_BUNDLES_DIR", tmp_path / "papers")
    monkeypatch.setattr("src.agents.experiment_runner.available_memory_gb", lambda: 8.0)
    monkeypatch.setattr("src.agents.experiment_runner.ROOT_DIR", tmp_path)

    docker = DockerExecutor(project_root=tmp_path)
    dockerfile = tmp_path / "docker" / "python.Dockerfile"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("FROM python:3.11-slim\n", encoding="utf-8")

    runner = ExperimentRunner(docker_executor=docker, max_attempts=3)
    metrics_doc, run_dir = runner.execute_paper(paper_id=paper_id)

    assert metrics_doc.run_status == "FAILED"
    assert "setup" in metrics_doc.phases_failed
    assert "smoke" in metrics_doc.phases_failed
    assert metrics_doc.attempts == 3
    assert (run_dir / "engineer_attempt_1.log").exists()
    assert (run_dir / "engineer_attempt_3.log").exists()
    # setup retries 3 times; smoke is skipped via depends_on (no extra docker calls)
    assert len(fake_client.commands) == 3


def test_experiment_runner_continues_after_row_failure(monkeypatch, tmp_path: Path) -> None:
    class _SelectiveClient(_FakeDockerClient):
        def _run(self, image, command=None, **_kwargs):
            _ = image
            cmd = command[2] if command and len(command) >= 3 else ""
            self.commands.append(cmd)
            status = 1 if "failme" in cmd else 0
            container = _FakeContainer(status_code=status)
            self.last_container = container
            return container

    fake_client = _SelectiveClient()
    _monkeypatch_docker(monkeypatch, fake_client)

    paper_id = "partial_paper"
    papers_dir = tmp_path / "papers" / paper_id
    code_dir = papers_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    results_dir = code_dir / "results"
    results_dir.mkdir()
    (results_dir / "ok.json").write_text(
        json.dumps([{"benchmark": "lsq", "metric_name": "regret", "value": 0.01}]),
        encoding="utf-8",
    )
    plan = {
        "paper": {"paper_id": paper_id, "title": "Test", "pdf_path": "x.pdf"},
        "plan_envelope": {
            "schema_version": "2.0",
            "agent": "planner",
            "status": "ok",
            "unknowns": [],
            "warnings": [],
            "payload": {
                "plan_summary": "two rows",
                "phases": [
                    {
                        "phase_id": "suite",
                        "title": "Suite",
                        "depends_on": [],
                        "matrix": [
                            {
                                "name": "bad",
                                "variables": {"benchmark": "lsq"},
                                "run_command": "python failme.py",
                                "results_path": "results/missing.json",
                            },
                            {
                                "name": "good",
                                "variables": {"benchmark": "lsq"},
                                "run_command": "python ok.py",
                                "results_path": "results/ok.json",
                            },
                        ],
                    }
                ],
            },
        },
    }
    (papers_dir / f"{paper_id}_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    import src.bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "PAPER_BUNDLES_DIR", tmp_path / "papers")
    monkeypatch.setattr("src.agents.experiment_runner.available_memory_gb", lambda: 8.0)
    monkeypatch.setattr("src.agents.experiment_runner.ROOT_DIR", tmp_path)

    docker = DockerExecutor(project_root=tmp_path)
    dockerfile = tmp_path / "docker" / "python.Dockerfile"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("FROM python:3.11-slim\n", encoding="utf-8")

    runner = ExperimentRunner(docker_executor=docker, max_attempts=1)
    metrics_doc, run_dir = runner.execute_paper(paper_id=paper_id)

    assert any("failme.py" in cmd for cmd in fake_client.commands)
    assert any("ok.py" in cmd for cmd in fake_client.commands)
    assert metrics_doc.run_status == "FAILED"
    assert any(row.status == "failed" for row in metrics_doc.experiment_matrix)
    assert any(row.status == "completed" for row in metrics_doc.experiment_matrix)
    assert any(m.metric_name == "regret" for m in metrics_doc.metrics)
    log_text = (run_dir / "engineer.log").read_text(encoding="utf-8")
    assert "continuing to next row" in log_text


def test_experiment_runner_skips_on_low_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.agents.experiment_runner.available_memory_gb", lambda: 0.5)
    docker = DockerExecutor.__new__(DockerExecutor)
    runner = ExperimentRunner(docker_executor=docker, min_free_memory_gb=2.0)
    try:
        runner.execute_paper("any")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Insufficient free memory" in str(exc)
