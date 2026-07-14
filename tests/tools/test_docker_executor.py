from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    def __init__(self):
        self.images = SimpleNamespace(build=self._build)
        self.containers = SimpleNamespace(run=self._run)
        self.last_container = _FakeContainer()

    def _build(self, **_kwargs):
        return None, []

    def _run(self, *_args, **_kwargs):
        return self.last_container


def _monkeypatch_docker(monkeypatch, fake_client: _FakeDockerClient) -> None:
    fake_module = SimpleNamespace(from_env=lambda: fake_client)
    monkeypatch.setitem(__import__("sys").modules, "docker", fake_module)


def test_build_image_uses_cache(monkeypatch, tmp_path: Path) -> None:
    fake_client = _FakeDockerClient()
    _monkeypatch_docker(monkeypatch, fake_client)
    tool = DockerExecutor(project_root=tmp_path)
    dockerfile = tmp_path / "docker" / "python.Dockerfile"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("FROM python:3.11-slim\n", encoding="utf-8")
    first = tool.build_image("python", str(dockerfile))
    second = tool.build_image("python", str(dockerfile))
    assert first == second


def test_run_container_returns_structured_result(monkeypatch, tmp_path: Path) -> None:
    fake_client = _FakeDockerClient()
    _monkeypatch_docker(monkeypatch, fake_client)
    tool = DockerExecutor(project_root=tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    out = tool.run_container(
        image_tag="research-assistant-python:latest",
        repo_path=str(repo_path),
        command="pytest -q",
        timeout_seconds=60,
    )
    assert out.exit_code == 0
    assert "ok" in out.stdout
    assert fake_client.last_container.removed is True
