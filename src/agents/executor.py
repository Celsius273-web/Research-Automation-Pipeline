"""Executor agent: deterministic patch application and containerized execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import EXECUTOR_TIMEOUT_SECONDS
from src.state import EngineerOutput, ExecutorResult, MetricResult, RepoContext, RunAttempt
from src.tools.docker_executor import ContainerRunResult, DockerExecutor


def _trim_log(text: str, max_lines: int = 50) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _is_build_command(command: str) -> bool:
    lowered = command.lower()
    build_tokens = ("make", "cmake", "cargo build", "python -m build", "pip install")
    return any(token in lowered for token in build_tokens)


def _load_captured_metrics(repo_root: Path, results_path: str) -> tuple[list[MetricResult], str | None]:
    if not results_path.strip():
        return [], "No results_path configured for this plan step."
    target = (repo_root / results_path).resolve()
    if not target.exists():
        return [], f"Step succeeded but no results file found at {results_path}."

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], f"Results file at {results_path} could not be parsed as JSON."

    normalized: list[MetricResult] = []
    if isinstance(payload, dict):
        for metric_name, value in payload.items():
            name = str(metric_name).strip()
            if not name:
                continue
            normalized.append(
                MetricResult(
                    metric_name=name,
                    value=str(value).strip(),
                    source_path=results_path,
                )
            )
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            metric_name = str(item.get("metric_name", "")).strip()
            if not metric_name:
                continue
            normalized.append(
                MetricResult(
                    benchmark=str(item.get("benchmark", "")).strip(),
                    metric_name=metric_name,
                    value=str(item.get("value", "")).strip(),
                    source_path=results_path,
                )
            )
    else:
        return [], f"Results JSON at {results_path} must be an object or list."

    if not normalized:
        return [], f"Results file at {results_path} contained no metric rows."
    return normalized, None


def _dockerfile_for_language(project_root: Path, language: str) -> Path | None:
    dockerfile_map = {
        "python": project_root / "docker" / "python.Dockerfile",
        "cpp": project_root / "docker" / "cpp.Dockerfile",
        "rust": project_root / "docker" / "rust.Dockerfile",
    }
    return dockerfile_map.get(language)


@dataclass
class ExecutorAgent:
    project_root: Path
    runs_dir: Path
    docker_executor: DockerExecutor

    def apply_patches(self, repo_context: RepoContext, output: EngineerOutput) -> None:
        repo_root = Path(repo_context.repo_path).resolve()
        for patch in output.patches:
            target = (repo_root / patch.file_path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            if patch.action == "delete":
                if target.exists():
                    target.unlink()
                continue
            target.write_text(patch.content, encoding="utf-8")

    def run_container_command(
        self,
        repo_context: RepoContext,
        command: str,
        timeout_seconds: int = EXECUTOR_TIMEOUT_SECONDS,
    ) -> ContainerRunResult:
        """Run one shell command in a CPU-only container and return the raw result."""
        language = repo_context.language or "unknown"
        dockerfile = _dockerfile_for_language(self.project_root, language)
        if dockerfile is None:
            return ContainerRunResult(
                stdout="",
                stderr=f"Unsupported language '{language}'",
                exit_code=1,
                timed_out=False,
                runtime_seconds=0.0,
            )
        image_tag = self.docker_executor.build_image(language=language, dockerfile_path=str(dockerfile))
        return self.docker_executor.run_container(
            image_tag=image_tag,
            repo_path=repo_context.repo_path,
            command=command,
            timeout_seconds=timeout_seconds,
        )

    def execute_step(
        self,
        paper_id: str,
        step_id: str,
        repo_context: RepoContext,
        verification_commands: list[str],
        current_attempt: int,
        results_path: str = "",
        timeout_seconds: int = EXECUTOR_TIMEOUT_SECONDS,
    ) -> ExecutorResult:
        language = repo_context.language or "unknown"
        dockerfile = _dockerfile_for_language(self.project_root, language)
        if dockerfile is None:
            result = ExecutorResult(final_status="failed", total_attempts=current_attempt)
            result.attempts.append(
                RunAttempt(
                    attempt_number=current_attempt,
                    step_id=step_id,
                    stage="runtime",
                    command="",
                    exit_code=1,
                    success=False,
                    stderr_excerpt=f"Unsupported language '{language}'",
                    failure_type="runtime_error",
                )
            )
            return result

        image_tag = self.docker_executor.build_image(language=language, dockerfile_path=str(dockerfile))
        attempts: list[RunAttempt] = []
        run_root = self.runs_dir / paper_id
        run_root.mkdir(parents=True, exist_ok=True)

        for command in verification_commands:
            run = self.docker_executor.run_container(
                image_tag=image_tag,
                repo_path=repo_context.repo_path,
                command=command,
                timeout_seconds=timeout_seconds,
            )
            stage = "build" if _is_build_command(command) else "runtime"
            failure_type = "none"
            if run.timed_out:
                stage = "timeout"
                failure_type = "timeout"
            elif run.exit_code != 0:
                failure_type = "build_error" if stage == "build" else "runtime_error"

            log_file = run_root / f"{step_id}_attempt_{current_attempt}.log"
            log_file.write_text(
                f"$ {command}\n\nSTDOUT:\n{run.stdout}\n\nSTDERR:\n{run.stderr}\n",
                encoding="utf-8",
            )

            attempt = RunAttempt(
                attempt_number=current_attempt,
                step_id=step_id,
                stage=stage,
                command=command,
                exit_code=run.exit_code,
                success=(run.exit_code == 0 and not run.timed_out),
                stdout_excerpt=_trim_log(run.stdout),
                stderr_excerpt=_trim_log(run.stderr),
                logs_path=str(log_file),
                failure_type=failure_type,
            )
            attempts.append(attempt)

            if not attempt.success:
                return ExecutorResult(
                    attempts=attempts,
                    final_status="failed",
                    total_attempts=current_attempt,
                )

        result = ExecutorResult(
            attempts=attempts,
            final_status="success",
            total_attempts=current_attempt,
        )
        metrics, warning = _load_captured_metrics(Path(repo_context.repo_path).resolve(), results_path)
        result.captured_metrics = metrics
        if warning:
            # Keep warning attached to the run output for upstream reporting.
            if result.attempts:
                result.attempts[-1].stderr_excerpt = (
                    (result.attempts[-1].stderr_excerpt + "\n" + warning).strip()
                )
        return result
