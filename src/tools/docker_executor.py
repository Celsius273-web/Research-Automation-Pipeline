"""Low-level Docker wrapper for deterministic experiment execution."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import time

import requests

from src.config import EXECUTOR_LOG_MAX_CHARS, EXECUTOR_TIMEOUT_SECONDS


logger = logging.getLogger(__name__)


def _trim_text(text: str, max_chars: int = EXECUTOR_LOG_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def resolve_docker_base_url() -> str | None:
    """Prefer DOCKER_HOST, then Docker Desktop's user socket, then /var/run/docker.sock."""
    env_host = os.getenv("DOCKER_HOST", "").strip()
    if env_host:
        logger.info("Using DOCKER_HOST=%s", env_host)
        return env_host

    candidates = [
        Path.home() / ".docker" / "run" / "docker.sock",
        Path("/var/run/docker.sock"),
    ]

    for socket_path in candidates:
        logger.debug("Checking Docker socket: %s", socket_path)
        if socket_path.exists():
            logger.info("Using Docker socket: %s", socket_path)
            return f"unix://{socket_path}"

    logger.error("No Docker socket found. Checked: %s", candidates)
    return None


def create_docker_client(docker_module: object) -> object:
    """Create a Docker client with a clear error when the daemon socket is missing."""
    base_url = resolve_docker_base_url()

    try:
        logger.info(
            "Creating Docker client. base_url=%s",
            base_url or "from environment",
        )

        if base_url is not None:
            client = docker_module.DockerClient(  # type: ignore[attr-defined]
                base_url=base_url
            )
        else:
            client = docker_module.from_env()  # type: ignore[attr-defined]

        logger.info("Docker client created successfully")

        try:
            client.ping()  # type: ignore[attr-defined]
            logger.info("Docker daemon ping successful")
        except Exception:
            logger.exception("Docker client created, but daemon ping failed")
            raise

        return client

    except Exception as exc:
        logger.exception("Failed to create or connect to Docker client")
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
        logger.info("Initializing DockerExecutor")
        logger.info("Project root: %s", self.project_root)

        try:
            import docker  # type: ignore
        except ImportError as exc:
            logger.exception("Docker SDK is not installed")
            raise RuntimeError(
                "Docker SDK not installed. Add 'docker' to requirements."
            ) from exc

        self._docker = docker
        self.client = create_docker_client(docker)
        logger.info("DockerExecutor initialized successfully")

    def build_image(self, language: str, dockerfile_path: str) -> str:
        if language in self.image_cache:
            logger.info(
                "Using cached Docker image for language=%s: %s",
                language,
                self.image_cache[language],
            )
            return self.image_cache[language]

        tag = f"research-assistant-{language}:latest"
        dockerfile = Path(dockerfile_path).resolve()
        context_path = dockerfile.parent

        logger.info("Starting Docker image build")
        logger.info("Image tag: %s", tag)
        logger.info("Dockerfile: %s", dockerfile)
        logger.info("Build context: %s", context_path)
        logger.info("Platform: linux/arm64")

        try:
            build_result = self.client.images.build(
                path=str(context_path),
                dockerfile=dockerfile.name,
                tag=tag,
                rm=True,
                forcerm=True,
                platform="linux/arm64",
            )

            logger.info("Docker image build completed: %s", tag)

            if isinstance(build_result, tuple) and len(build_result) >= 1:
                image = build_result[0]
                logger.debug("Built image object: %s", image)

        except Exception as exc:  # pragma: no cover - depends on Docker runtime
            logger.exception(
                "Docker image build failed. tag=%s dockerfile=%s",
                tag,
                dockerfile,
            )
            raise RuntimeError(
                f"Failed to build image '{tag}': {exc}"
            ) from exc

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
            logger.error("Repository path does not exist: %s", repo)
            raise RuntimeError(f"Repository path does not exist: {repo}")

        started = time.time()
        container = None
        timed_out = False
        stderr = ""

        logger.info("Starting Docker container")
        logger.info("Image: %s", image_tag)
        logger.info("Repository: %s", repo)
        logger.info("Working directory: /workspace")
        logger.info("Timeout: %s seconds", timeout_seconds)
        logger.info("Command: %s", command)

        try:
            container = self.client.containers.run(
                image_tag,
                command=["/bin/sh", "-lc", command],
                detach=True,
                working_dir="/workspace",
                volumes={
                    str(repo): {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
            )

            logger.info(
                "Container started successfully. container_id=%s",
                getattr(container, "short_id", getattr(container, "id", "unknown")),
            )

            try:
                logger.debug(
                    "Waiting for container completion. timeout=%s",
                    timeout_seconds,
                )
                wait_result = container.wait(timeout=timeout_seconds)
                logger.info("Container wait completed: %s", wait_result)

            except requests.exceptions.ReadTimeout:
                timed_out = True
                logger.warning(
                    "Container timed out after %s seconds. Killing container.",
                    timeout_seconds,
                )

                try:
                    container.kill()
                    logger.info("Timed-out container killed successfully")
                except Exception:
                    logger.exception("Failed to kill timed-out container")

                wait_result = {"StatusCode": 124}

            exit_code = int(wait_result.get("StatusCode", 1))

            logger.info("Container exit code: %s", exit_code)

            logger.debug("Collecting container stdout")
            logs_bytes = container.logs(stdout=True, stderr=False)

            logger.debug("Collecting container stderr")
            err_bytes = container.logs(stdout=False, stderr=True)

            stdout = _trim_text(
                logs_bytes.decode("utf-8", errors="replace")
                if logs_bytes
                else ""
            )

            stderr = _trim_text(
                err_bytes.decode("utf-8", errors="replace")
                if err_bytes
                else stderr
            )

            logger.info(
                "Container completed in %.2f seconds. exit_code=%s timed_out=%s",
                max(time.time() - started, 0.0),
                exit_code,
                timed_out,
            )

            if stdout:
                logger.info(
                    "DOCKER STDOUT:\n%s",
                    stdout,
                )
            else:
                logger.info("DOCKER STDOUT: <empty>")

            if stderr:
                logger.warning(
                    "DOCKER STDERR:\n%s",
                    stderr,
                )
            else:
                logger.info("DOCKER STDERR: <empty>")

            return ContainerRunResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                runtime_seconds=max(time.time() - started, 0.0),
            )

        except Exception as exc:  # pragma: no cover - depends on Docker runtime
            runtime = max(time.time() - started, 0.0)

            logger.exception(
                "Container execution failed after %.2f seconds",
                runtime,
            )
            logger.error("Image: %s", image_tag)
            logger.error("Repository: %s", repo)
            logger.error("Command: %s", command)
            logger.error("Container object: %s", container)

            if container is not None:
                try:
                    logger.error(
                        "Attempting to collect logs after container failure"
                    )

                    failure_stdout = container.logs(
                        stdout=True,
                        stderr=False,
                    )
                    failure_stderr = container.logs(
                        stdout=False,
                        stderr=True,
                    )

                    if failure_stdout:
                        logger.error(
                            "FAILURE DOCKER STDOUT:\n%s",
                            _trim_text(
                                failure_stdout.decode(
                                    "utf-8",
                                    errors="replace",
                                )
                            ),
                        )

                    if failure_stderr:
                        logger.error(
                            "FAILURE DOCKER STDERR:\n%s",
                            _trim_text(
                                failure_stderr.decode(
                                    "utf-8",
                                    errors="replace",
                                )
                            ),
                        )

                except Exception:
                    logger.exception(
                        "Could not retrieve Docker logs after failure"
                    )

            raise RuntimeError(
                f"Container run failed for command '{command}': {exc}"
            ) from exc

        finally:
            if container is not None:
                try:
                    logger.debug(
                        "Removing container: %s",
                        getattr(
                            container,
                            "short_id",
                            getattr(container, "id", "unknown"),
                        ),
                    )
                    container.remove(force=True)
                    logger.debug("Container removed successfully")
                except self._docker.errors.APIError:
                    logger.warning(
                        "Container cleanup failed with Docker API error",
                        exc_info=True,
                    )
                except Exception:
                    logger.exception("Unexpected container cleanup failure")