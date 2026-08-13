"""Plan-driven Engineer: execute Planner phases via Docker without modifying repo code."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.agents.executor import ExecutorAgent
from src.bundle import PaperBundle
from src.config import (
    ENGINEER_LOG_FILENAME,
    ENGINEER_MAX_ATTEMPTS,
    ENGINEER_METRICS_FILENAME,
    EXECUTOR_TIMEOUT_SECONDS,
    MIN_FREE_MEMORY_GB,
    ROOT_DIR,
)
from src.persistence import load_planner_envelope, resolve_plan_path
from src.state import (
    AgentEnvelope,
    ExperimentMatrixRow,
    MetricsDocument,
    PhaseRunSpec,
    PlannerPayload,
    PlanPhase,
    RepoContext,
)
from src.tools.command_template import phase_commands
from src.tools.docker_executor import ContainerRunResult, DockerExecutor
from src.tools.language_detect import detect_language
from src.tools.memory import available_memory_gb
from src.tools.metrics_capture import load_metrics_from_path, merge_unique_metrics
from src.tools.paper_venv import build_persistent_setup_command, rewrite_command_for_paper_venv
from src.tools.run_ids import next_run_id

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class ExperimentRunner:
    """Execute a Planner payload phase-by-phase inside Docker (CPU only)."""

    docker_executor: DockerExecutor
    max_attempts: int = ENGINEER_MAX_ATTEMPTS
    timeout_seconds: int = EXECUTOR_TIMEOUT_SECONDS
    min_free_memory_gb: float = MIN_FREE_MEMORY_GB

    def execute_paper(
        self,
        paper_id: str,
        repo_path: str | None = None,
        plan_path: str | Path | None = None,
    ) -> tuple[MetricsDocument, Path]:
        """Run Engineer for one paper. Returns metrics document and run directory."""
        free_gb = available_memory_gb()
        if free_gb < self.min_free_memory_gb:
            raise RuntimeError(
                f"Insufficient free memory ({free_gb:.2f} GB < {self.min_free_memory_gb} GB); "
                f"skipping paper '{paper_id}'."
            )

        resolved_plan = resolve_plan_path(
            plan_path=str(plan_path) if plan_path is not None else None,
            paper_id=paper_id,
        )
        if not resolved_plan.exists():
            raise FileNotFoundError(f"Plan file does not exist: {resolved_plan}")

        payload = json.loads(resolved_plan.read_text(encoding="utf-8"))
        envelope = load_planner_envelope(payload)
        if not envelope.payload.phases:
            raise ValueError(f"Plan for '{paper_id}' has no phases to execute.")

        bundle = PaperBundle(paper_id)
        bundle.create_bundle_dir()
        resolved_repo = repo_path or str(bundle.code_dir)
        if not Path(resolved_repo).exists():
            raise FileNotFoundError(f"Repository path does not exist: {resolved_repo}")

        repo_context = detect_language(repo_path=resolved_repo)
        run_id = next_run_id(bundle.runs_dir)
        run_dir = bundle.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / ENGINEER_LOG_FILENAME
        metrics_path = run_dir / ENGINEER_METRICS_FILENAME

        metrics_doc = MetricsDocument(timestamp=_iso_now(), attempts=0, logs_captured=True)
        self._log(log_path, f"Engineer start paper_id={paper_id} run_id={run_id} free_memory_gb={free_gb:.2f}")
        self._log(log_path, f"Plan: {resolved_plan}")
        self._log(log_path, f"Repo: {resolved_repo} language={repo_context.language}")

        try:
            self._run_phases(
                envelope=envelope,
                repo_context=repo_context,
                paper_id=paper_id,
                run_dir=run_dir,
                log_path=log_path,
                metrics_path=metrics_path,
                metrics_doc=metrics_doc,
            )
        except Exception as exc:
            metrics_doc.errors.append(str(exc))
            metrics_doc.run_status = "FAILED"
            metrics_doc.exit_code = 1
            self._log(log_path, f"FATAL: {exc}")
            self._persist_metrics(metrics_path, metrics_doc)
            raise

        self._finalize_status(metrics_doc, envelope.payload)
        self._persist_metrics(metrics_path, metrics_doc)
        self._log(
            log_path,
            f"Engineer finished status={metrics_doc.run_status} "
            f"completed={metrics_doc.phases_completed} failed={metrics_doc.phases_failed}",
        )
        return metrics_doc, run_dir

    def _run_phases(
        self,
        envelope: AgentEnvelope[PlannerPayload],
        repo_context: RepoContext,
        paper_id: str,
        run_dir: Path,
        log_path: Path,
        metrics_path: Path,
        metrics_doc: MetricsDocument,
    ) -> None:
        executor = ExecutorAgent(
            project_root=ROOT_DIR,
            runs_dir=run_dir,
            docker_executor=self.docker_executor,
        )
        completed: set[str] = set()
        failed: set[str] = set()
        results_summary = envelope.payload.results_summary_path.strip()

        for phase in envelope.payload.phases:
            self._log(
                log_path,
                f"Phase start id={phase.phase_id} title={phase.title!r} "
                f"goal={phase.goal!r} depends_on={phase.depends_on}",
            )
            blocked_by = [dep for dep in phase.depends_on if dep in failed]
            if blocked_by:
                msg = f"Skipping phase '{phase.phase_id}' because depends_on failed: {blocked_by}"
                self._log(log_path, msg)
                metrics_doc.errors.append(msg)
                failed.add(phase.phase_id)
                metrics_doc.phases_failed = sorted(failed)
                self._persist_metrics(metrics_path, metrics_doc)
                continue

            commands = phase_commands(phase, paper_id=paper_id)
            if phase.axes and len(commands) > len(phase.matrix):
                self._log(
                    log_path,
                    f"Expanded phase '{phase.phase_id}' axes to {len(commands)} run(s) "
                    f"(plan matrix had {len(phase.matrix)} example row(s))",
                )
            if not commands:
                self._log(log_path, f"Phase '{phase.phase_id}' has no commands; marking completed.")
                completed.add(phase.phase_id)
                metrics_doc.phases_completed = sorted(completed)
                self._persist_metrics(metrics_path, metrics_doc)
                continue

            phase_ok = True
            repo_root = Path(repo_context.repo_path)
            for command, row in commands:
                prepared = self._prepare_command(
                    phase_id=phase.phase_id,
                    command=command,
                    repo_root=repo_root,
                )
                if prepared != command:
                    self._log(log_path, f"Rewrote command for paper venv: {prepared!r}")
                row_name = row.name if row is not None else phase.phase_id
                success = self._run_command_with_retries(
                    executor=executor,
                    repo_context=repo_context,
                    command=prepared,
                    step_id=f"{phase.phase_id}__{row_name}",
                    run_dir=run_dir,
                    log_path=log_path,
                    metrics_doc=metrics_doc,
                    metrics_path=metrics_path,
                )
                if not success:
                    phase_ok = False
                    self._record_matrix_row(
                        metrics_doc,
                        phase=phase,
                        row=row,
                        status="failed",
                    )
                    self._log(
                        log_path,
                        f"Command failed for {row_name}; capturing stderr and continuing to next row.",
                    )
                    self._persist_metrics(metrics_path, metrics_doc)
                    continue

                capture_paths = self._results_paths_for_row(phase, row, results_summary)
                algorithm = self._algorithm_from_row(row)
                benchmark = self._benchmark_from_row(row)
                for path in capture_paths:
                    self._capture_metrics(
                        repo_root=repo_root,
                        results_relpath=path,
                        default_benchmark=benchmark,
                        default_algorithm=algorithm,
                        metrics_doc=metrics_doc,
                        log_path=log_path,
                    )
                self._record_matrix_row(
                    metrics_doc,
                    phase=phase,
                    row=row,
                    status="completed",
                    results_path=capture_paths[0] if capture_paths else "",
                )
                self._persist_metrics(metrics_path, metrics_doc)

            if phase_ok:
                completed.add(phase.phase_id)
                metrics_doc.phases_completed = sorted(completed)
                # Only capture summary.json when it exists; avoid noisy missing-path errors.
                if results_summary and (repo_root / results_summary).exists():
                    self._capture_metrics(
                        repo_root=repo_root,
                        results_relpath=results_summary,
                        default_benchmark="",
                        default_algorithm="",
                        metrics_doc=metrics_doc,
                        log_path=log_path,
                    )
            else:
                failed.add(phase.phase_id)
                metrics_doc.phases_failed = sorted(failed)

            self._persist_metrics(metrics_path, metrics_doc)

    @staticmethod
    def _prepare_command(phase_id: str, command: str, repo_root: Path) -> str:
        """Ensure setup installs into a persistent venv and later phases use it."""
        if phase_id == "setup":
            return build_persistent_setup_command(command, repo_path=repo_root)
        return rewrite_command_for_paper_venv(command)

    def _run_command_with_retries(
        self,
        executor: ExecutorAgent,
        repo_context: RepoContext,
        command: str,
        step_id: str,
        run_dir: Path,
        log_path: Path,
        metrics_doc: MetricsDocument,
        metrics_path: Path,
    ) -> bool:
        last_result: ContainerRunResult | None = None
        for attempt in range(1, self.max_attempts + 1):
            metrics_doc.attempts = max(metrics_doc.attempts, attempt)
            attempt_log = run_dir / f"engineer_attempt_{attempt}.log"
            self._log(log_path, f"Attempt {attempt}/{self.max_attempts} step={step_id} cmd={command!r}")
            result = executor.run_container_command(
                repo_context=repo_context,
                command=command,
                timeout_seconds=self.timeout_seconds,
            )
            last_result = result
            excerpt = (
                f"$ {command}\nexit_code={result.exit_code} timed_out={result.timed_out} "
                f"runtime_seconds={result.runtime_seconds:.2f}\n\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n"
            )
            attempt_log.write_text(excerpt, encoding="utf-8")
            self._log(
                log_path,
                f"Attempt {attempt} exit_code={result.exit_code} timed_out={result.timed_out} "
                f"runtime={result.runtime_seconds:.2f}s",
            )
            if result.exit_code != 0:
                logger.warning(
                    "FAILED ATTEMPT %s/%s: %s\nSTDOUT:\n%s\nSTDERR:\n%s",
                    attempt,
                    self.max_attempts,
                    step_id,
                    result.stdout,
                    result.stderr,
                )
            metrics_doc.exit_code = result.exit_code if not result.timed_out else 124
            self._persist_metrics(metrics_path, metrics_doc)
            if result.exit_code == 0 and not result.timed_out:
                return True
            reason = "timeout" if result.timed_out else f"exit_code={result.exit_code}"
            metrics_doc.errors.append(f"{step_id} attempt {attempt} failed ({reason})")
            if attempt < self.max_attempts:
                self._log(log_path, f"Retrying same command after failure ({reason})")

        self._log(log_path, f"Exhausted retries for step={step_id}")
        if last_result is not None:
            metrics_doc.exit_code = 124 if last_result.timed_out else last_result.exit_code
        else:
            metrics_doc.exit_code = 1
        return False

    def _capture_metrics(
        self,
        repo_root: Path,
        results_relpath: str,
        default_benchmark: str,
        default_algorithm: str,
        metrics_doc: MetricsDocument,
        log_path: Path,
    ) -> None:
        if not results_relpath.strip():
            return
        target = (repo_root / results_relpath).resolve()
        captured, error = load_metrics_from_path(
            target,
            default_benchmark=default_benchmark,
            default_algorithm=default_algorithm,
        )
        if error:
            warning = (
                f"Results file not found at {results_relpath}; continuing to next row. "
                f"Detail: {error}"
            )
            self._log(log_path, warning)
            metrics_doc.errors.append(warning)
            return
        metrics_doc.metrics = merge_unique_metrics(metrics_doc.metrics, captured)
        self._log(
            log_path,
            f"Captured {len(captured)} metric(s) from {results_relpath}"
            + (f" algorithm={default_algorithm}" if default_algorithm else ""),
        )

    @staticmethod
    def _record_matrix_row(
        metrics_doc: MetricsDocument,
        *,
        phase: PlanPhase,
        row: PhaseRunSpec | None,
        status: str,
        results_path: str = "",
    ) -> None:
        if row is None and phase.phase_id == "setup":
            return
        name = row.name if row is not None else phase.phase_id
        variables = row.variables if row is not None else {}
        path = results_path or (row.results_path if row is not None else phase.results_path)
        metrics_doc.experiment_matrix.append(
            ExperimentMatrixRow(
                phase_id=phase.phase_id,
                name=name,
                benchmark=str(
                    variables.get("benchmark")
                    or variables.get("fun")
                    or variables.get("function")
                    or variables.get("dataset")
                    or ""
                ),
                algorithm=str(
                    variables.get("algorithm") or variables.get("algo") or variables.get("method") or ""
                ),
                seed=str(variables.get("seed", "")),
                results_path=path.strip(),
                status=status,  # type: ignore[arg-type]
            )
        )

    @staticmethod
    def _results_paths_for_row(
        phase: PlanPhase,
        row: PhaseRunSpec | None,
        results_summary: str,
    ) -> list[str]:
        paths: list[str] = []
        if row is not None and row.results_path.strip():
            paths.append(row.results_path.strip())
        elif phase.results_path.strip() and phase.phase_id != "setup":
            paths.append(phase.results_path.strip())
        _ = results_summary
        return paths

    @staticmethod
    def _benchmark_from_row(row: PhaseRunSpec | None) -> str:
        if row is None:
            return ""
        for key in ("benchmark", "fun", "function", "dataset"):
            if key in row.variables:
                return str(row.variables[key])
        return ""

    @staticmethod
    def _algorithm_from_row(row: PhaseRunSpec | None) -> str:
        if row is None:
            return ""
        for key in ("algorithm", "algo", "method"):
            if key in row.variables:
                return str(row.variables[key])
        return ""

    @staticmethod
    def _finalize_status(metrics_doc: MetricsDocument, payload: PlannerPayload) -> None:
        total = len(payload.phases)
        completed = len(metrics_doc.phases_completed)
        if metrics_doc.phases_failed:
            metrics_doc.run_status = "FAILED" if completed == 0 else "PARTIAL"
            if metrics_doc.exit_code == 0:
                metrics_doc.exit_code = 1
            return
        if completed == total and total > 0:
            metrics_doc.run_status = "SUCCESS"
            metrics_doc.exit_code = 0
            return
        if completed > 0:
            metrics_doc.run_status = "PARTIAL"
            metrics_doc.exit_code = 0
            return
        metrics_doc.run_status = "FAILED"
        metrics_doc.exit_code = 1

    @staticmethod
    def _persist_metrics(path: Path, metrics_doc: MetricsDocument) -> None:
        _write_json(path, metrics_doc.model_dump())

    @staticmethod
    def _log(log_path: Path, message: str) -> None:
        line = f"{_iso_now()} {message}\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        logger.info(message)
