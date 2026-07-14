"""Low-level Docker wrapper for deterministic experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import requests

from src.config import EXECUTOR_TIMEOUT_SECONDS


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
        self.client = docker.from_env()

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
            stdout = logs_bytes.decode("utf-8", errors="replace") if logs_bytes else ""
            stderr = err_bytes.decode("utf-8", errors="replace") if err_bytes else stderr
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
