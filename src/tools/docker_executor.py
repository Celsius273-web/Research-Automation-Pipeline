"""Low-level Docker wrapper for deterministic experiment execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import time

import requests

from src.config import EXECUTOR_LOG_MAX_CHARS, EXECUTOR_TIMEOUT_SECONDS


def _trim_text(text: str, max_chars: int = EXECUTOR_LOG_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def resolve_docker_base_url() -> str | None:
    """Prefer DOCKER_HOST, then Docker Desktop's user socket, then /var/run/docker.sock."""
    env_host = os.getenv("DOCKER_HOST", "").strip()
    if env_host:
        return env_host

    # Docker Desktop on macOS exposes the engine here; /var/run/docker.sock is often absent.
    candidates = [
        Path.home() / ".docker" / "run" / "docker.sock",
        Path("/var/run/docker.sock"),
    ]
    for socket_path in candidates:
        if socket_path.exists():
            return f"unix://{socket_path}"
    return None


def create_docker_client(docker_module: object) -> object:
    """Create a Docker client with a clear error when the daemon socket is missing."""
    base_url = resolve_docker_base_url()
    try:
        if base_url is not None:
            return docker_module.DockerClient(base_url=base_url)  # type: ignore[attr-defined]
        return docker_module.from_env()  # type: ignore[attr-defined]
    except Exception as exc:
        desktop_sock = Path.home() / ".docker" / "run" / "docker.sock"
        raise RuntimeError(
            "Cannot connect to the Docker daemon. "
            "Start Docker Desktop, then retry. On macOS the Python SDK often needs:\n"
            f"  export DOCKER_HOST=unix://{desktop_sock}\n"
            f"Underlying error: {exc}"
        ) from exc


@dataclass
class ContainerRunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    runtime_seconds: float = 0.0


@dataclass
class DockerExecutor:
    project_root: Path
    image_cache: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            import docker  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Docker SDK not installed. Add 'docker' to requirements.") from exc

        self._docker = docker
        self.client = create_docker_client(docker)

    def build_image(self, language: str, dockerfile_path: str) -> str:
        if language in self.image_cache:
            return self.image_cache[language]

        tag = f"research-assistant-{language}:latest"
        dockerfile = Path(dockerfile_path).resolve()
        context_path = dockerfile.parent
        try:
            self.client.images.build(
                path=str(context_path),
                dockerfile=dockerfile.name,
                tag=tag,
                rm=True,
                forcerm=True,
                platform="linux/arm64",
            )
        except Exception as exc:  # pragma: no cover - depends on Docker runtime
            raise RuntimeError(f"Failed to build image '{tag}': {exc}") from exc

        self.image_cache[language] = tag
        return tag

    def run_container(
        self,
        image_tag: str,
        repo_path: str,
        command: str,
        timeout_seconds: int = EXECUTOR_TIMEOUT_SECONDS,
    ) -> ContainerRunResult:
        repo = Path(repo_path).resolve()
        if not repo.exists():
            raise RuntimeError(f"Repository path does not exist: {repo}")

        started = time.time()
        container = None
        timed_out = False
        stderr = ""
        try:
            container = self.client.containers.run(
                image_tag,
                command=["/bin/sh", "-lc", command],
                detach=True,
                working_dir="/workspace",
                volumes={str(repo): {"bind": "/workspace", "mode": "rw"}},
            )

            try:
                wait_result = container.wait(timeout=timeout_seconds)
            except requests.exceptions.ReadTimeout:
                timed_out = True
                container.kill()
                wait_result = {"StatusCode": 124}

            exit_code = int(wait_result.get("StatusCode", 1))
            logs_bytes = container.logs(stdout=True, stderr=False)
            err_bytes = container.logs(stdout=False, stderr=True)
            stdout = _trim_text(logs_bytes.decode("utf-8", errors="replace") if logs_bytes else "")
            stderr = _trim_text(
                err_bytes.decode("utf-8", errors="replace") if err_bytes else stderr
            )
            return ContainerRunResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                runtime_seconds=max(time.time() - started, 0.0),
            )
        except Exception as exc:  # pragma: no cover - depends on Docker runtime
            raise RuntimeError(f"Container run failed for command '{command}': {exc}") from exc
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except self._docker.errors.APIError:
                    # Best-effort cleanup: container may already be gone or the
                    # daemon may be mid-teardown. Nothing actionable to do here.
                    pass
