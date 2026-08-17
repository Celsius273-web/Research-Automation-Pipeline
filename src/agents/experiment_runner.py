"""Plan-driven Engineer: execute Planner phases via Docker without modifying repo code."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.agents.engineer import PaperEngineer
from src.agents.executor import ExecutorAgent
from src.bundle import PaperBundle
from src.config import (
    ENGINEER_LOG_FILENAME,
    ENGINEER_MAX_ATTEMPTS,
    ENGINEER_METRICS_FILENAME,
    FATAL_COMMAND_MARKERS,
    EXECUTOR_LOG_MAX_CHARS,
    EXECUTOR_TIMEOUT_SECONDS,
    MIN_FREE_MEMORY_GB,
    ROOT_DIR,
)
from src.persistence import load_planner_envelope, resolve_plan_path
from src.state import (
    AgentEnvelope,
    EngineerInputContext,
    EngineerOutput,
    FailureContext,
    ExperimentMatrixRow,
    MetricsDocument,
    PaperMetadata,
    PhaseRunSpec,
    PlanStep,
    PlannerPayload,
    PlanPhase,
    RepoContext,
)
from src.tools.command_template import phase_commands
from src.tools.docker_executor import ContainerRunResult, DockerExecutor
from src.tools.language_detect import detect_language
from src.tools.memory import available_memory_gb
from src.tools.metrics_capture import load_metrics_from_path, load_metrics_from_text, merge_unique_metrics
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
    engineer: PaperEngineer = field(default_factory=PaperEngineer)

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
        run_id = next_run_id(bundle.runs_dir)
        run_dir = bundle.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        source_repo = Path(repo_path).resolve() if repo_path else bundle.code_dir
        workspace = self._prepare_workspace(source_repo, run_dir, envelope.payload)
        repo_context = detect_language(repo_path=str(workspace))
        if repo_context.language == "unknown" and any(
            path.endswith(".py")
            for phase in envelope.payload.phases
            for path in phase.required_artifacts
        ):
            repo_context.language = "python"
        log_path = run_dir / ENGINEER_LOG_FILENAME
        metrics_path = run_dir / ENGINEER_METRICS_FILENAME

        metrics_doc = MetricsDocument(timestamp=_iso_now(), attempts=0, logs_captured=True)
        self._log(log_path, f"Engineer start paper_id={paper_id} run_id={run_id} free_memory_gb={free_gb:.2f}")
        self._log(log_path, f"Plan: {resolved_plan}")
        self._log(log_path, f"Workspace: {workspace} language={repo_context.language}")

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
        if workspace != source_repo:
            shutil.rmtree(workspace, ignore_errors=True)
        self._log(
            log_path,
            f"Engineer finished status={metrics_doc.run_status} "
            f"completed={metrics_doc.phases_completed} failed={metrics_doc.phases_failed}",
        )
        return metrics_doc, run_dir

    @staticmethod
    def _prepare_workspace(source_repo: Path, run_dir: Path, payload: PlannerPayload) -> Path:
        """Stage only declared construction inputs; execution-only plans use their repository."""
        construct_phases = [phase for phase in payload.phases if phase.kind == "construct"]
        if not construct_phases:
            if not source_repo.exists():
                raise FileNotFoundError(f"Repository path does not exist: {source_repo}")
            return source_repo

        workspace = run_dir / "_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        input_paths = {
            path
            for phase in construct_phases
            for path in phase.input_paths
            if path.strip()
        }
        for relative in sorted(input_paths):
            relpath = Path(relative)
            if relpath.is_absolute() or ".." in relpath.parts:
                raise ValueError(f"Construction input must be a safe relative path: {relative}")
            source = (source_repo / relpath).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Construction input does not exist: {source}")
            target = workspace / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return workspace

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

            if phase.kind == "construct":
                phase_ok = self._run_construct_phase(
                    phase=phase,
                    envelope=envelope,
                    repo_context=repo_context,
                    paper_id=paper_id,
                    executor=executor,
                    run_dir=run_dir,
                    log_path=log_path,
                    metrics_path=metrics_path,
                    metrics_doc=metrics_doc,
                )
                if phase_ok:
                    completed.add(phase.phase_id)
                    metrics_doc.phases_completed = sorted(completed)
                else:
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
                if phase.kind == "summary":
                    self._log(log_path, f"Summary phase '{phase.phase_id}' completed.")
                    completed.add(phase.phase_id)
                    metrics_doc.phases_completed = sorted(completed)
                else:
                    message = f"Phase '{phase.phase_id}' has no work configured."
                    self._log(log_path, message)
                    metrics_doc.errors.append(message)
                    failed.add(phase.phase_id)
                    metrics_doc.phases_failed = sorted(failed)
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
                success, is_fatal, _excerpt, stdout = self._run_command_with_retries(
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
                    if is_fatal:
                        self._log(
                            log_path,
                            f"Fatal script error for {row_name}; skipping remaining rows.",
                        )
                        self._persist_metrics(metrics_path, metrics_doc)
                        break
                    self._log(
                        log_path,
                        f"Command failed for {row_name}; capturing stderr and continuing to next row.",
                    )
                    self._persist_metrics(metrics_path, metrics_doc)
                    continue

                capture_paths = self._results_paths_for_row(phase, row, results_summary)
                algorithm = self._algorithm_from_row(row)
                benchmark = self._benchmark_from_row(row)
                seed = str(row.variables.get("seed", "")) if row is not None else ""
                captured, stdout_error = load_metrics_from_text(
                    stdout,
                    default_benchmark=benchmark,
                    default_algorithm=algorithm,
                    default_seed=seed,
                )
                if captured:
                    metrics_doc.metrics = merge_unique_metrics(metrics_doc.metrics, captured)
                    self._log(log_path, f"Captured {len(captured)} metric(s) from stdout")
                elif capture_paths:
                    for path in capture_paths:
                        self._capture_metrics(
                            repo_root=repo_root,
                            results_relpath=path,
                            default_benchmark=benchmark,
                            default_algorithm=algorithm,
                            default_seed=seed,
                            metrics_doc=metrics_doc,
                            log_path=log_path,
                        )
                elif stdout_error:
                    self._log(log_path, f"No metrics captured: {stdout_error}")
                    metrics_doc.errors.append(stdout_error)
                self._record_matrix_row(
                    metrics_doc,
                    phase=phase,
                    row=row,
                    status="completed",
                    results_path="stdout" if captured else (capture_paths[0] if capture_paths else ""),
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
                        default_seed="",
                        metrics_doc=metrics_doc,
                        log_path=log_path,
                    )
            else:
                failed.add(phase.phase_id)
                metrics_doc.phases_failed = sorted(failed)

            self._persist_metrics(metrics_path, metrics_doc)

    def _run_construct_phase(
        self,
        *,
        phase: PlanPhase,
        envelope: AgentEnvelope[PlannerPayload],
        repo_context: RepoContext,
        paper_id: str,
        executor: ExecutorAgent,
        run_dir: Path,
        log_path: Path,
        metrics_path: Path,
        metrics_doc: MetricsDocument,
    ) -> bool:
        if self._load_reference_artifacts(phase, repo_context, metrics_doc, log_path):
            accepted, excerpt = self._run_acceptance_commands(
                phase, executor, repo_context, run_dir, log_path, metrics_path, metrics_doc
            )
            if accepted:
                return True
            metrics_doc.errors.append(f"Construction phase '{phase.phase_id}' failed: {excerpt}")
            metrics_doc.exit_code = 1
            return False

        failure: FailureContext | None = None
        last_error = "Construction failed"
        for llm_attempt in range(1, self.max_attempts + 1):
            context = self._construct_context(phase, envelope, repo_context, paper_id, failure)
            try:
                output = self.engineer.propose_patch(context)
                self._apply_construct_output(
                    phase=phase,
                    output=output,
                    executor=executor,
                    repo_context=repo_context,
                    metrics_doc=metrics_doc,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = str(exc)
                self._log(log_path, f"Construction attempt {llm_attempt} failed: {exc}")
                failure = FailureContext(stage="runtime", log_excerpt=last_error)
                continue

            metrics_doc.attempts = max(metrics_doc.attempts, llm_attempt)
            self._persist_metrics(metrics_path, metrics_doc)
            accepted, excerpt = self._run_acceptance_commands(
                phase, executor, repo_context, run_dir, log_path, metrics_path, metrics_doc
            )
            if accepted:
                return True
            last_error = excerpt
            failure = FailureContext(
                stage="runtime",
                log_excerpt=excerpt[:EXECUTOR_LOG_MAX_CHARS],
                prior_patch_summary=f"acceptance failed on construct attempt {llm_attempt}",
            )
            self._log(
                log_path,
                f"Acceptance failed; requesting engineer patch ({llm_attempt}/{self.max_attempts})",
            )
        metrics_doc.errors.append(f"Construction phase '{phase.phase_id}' failed: {last_error}")
        metrics_doc.exit_code = 1
        return False

    @staticmethod
    def _load_reference_artifacts(
        phase: PlanPhase,
        repo_context: RepoContext,
        metrics_doc: MetricsDocument,
        log_path: Path,
    ) -> bool:
        """Use staged reference files when every required artifact is already present."""
        required = [Path(path).as_posix() for path in phase.required_artifacts if path.strip()]
        if not required:
            return False

        repo_root = Path(repo_context.repo_path)
        for artifact in required:
            if not (repo_root / artifact).is_file():
                return False

        for artifact in required:
            metrics_doc.generated_code[artifact] = (repo_root / artifact).read_text(encoding="utf-8")

        ExperimentRunner._log(
            log_path,
            f"Using reference artifacts for phase={phase.phase_id}: {', '.join(required)}",
        )
        return True

    def _construct_context(
        self,
        phase: PlanPhase,
        envelope: AgentEnvelope[PlannerPayload],
        repo_context: RepoContext,
        paper_id: str,
        failure: FailureContext | None,
    ) -> EngineerInputContext:
        slim_envelope = envelope.model_copy(deep=True)
        slim_envelope.payload.phases = [phase]
        slim_envelope.payload.engineer_notes = [
            note for note in envelope.payload.engineer_notes if note.strip()
        ]
        return EngineerInputContext(
            paper=PaperMetadata(paper_id=paper_id, title=paper_id, pdf_path=""),
            execution_plan=slim_envelope,
            plan_step=PlanStep(
                step_id=phase.phase_id,
                title=phase.title,
                goal=phase.goal,
                depends_on=phase.depends_on,
                results_path=phase.results_path,
            ),
            repo_context=repo_context,
            failure_context=failure,
        )

    def _run_acceptance_commands(
        self,
        phase: PlanPhase,
        executor: ExecutorAgent,
        repo_context: RepoContext,
        run_dir: Path,
        log_path: Path,
        metrics_path: Path,
        metrics_doc: MetricsDocument,
    ) -> tuple[bool, str]:
        last_excerpt = ""
        for command in phase.acceptance_commands:
            success, _is_fatal, excerpt, stdout = self._run_command_with_retries(
                executor=executor,
                repo_context=repo_context,
                command=self._prepare_command(phase.phase_id, command, Path(repo_context.repo_path)),
                step_id=f"{phase.phase_id}__acceptance",
                run_dir=run_dir,
                log_path=log_path,
                metrics_doc=metrics_doc,
                metrics_path=metrics_path,
            )
            last_excerpt = excerpt
            if not success:
                return False, excerpt
            captured, _error = load_metrics_from_text(stdout)
            if captured:
                metrics_doc.metrics = merge_unique_metrics(metrics_doc.metrics, captured)
                self._persist_metrics(metrics_path, metrics_doc)
        return True, last_excerpt

    @staticmethod
    def _apply_construct_output(
        *,
        phase: PlanPhase,
        output: EngineerOutput,
        executor: ExecutorAgent,
        repo_context: RepoContext,
        metrics_doc: MetricsDocument,
    ) -> None:
        required = {Path(path).as_posix() for path in phase.required_artifacts if path.strip()}
        if not required:
            raise ValueError("required_artifacts is empty")
        if not output.patches:
            raise ValueError("Engineer returned no code")

        for patch in output.patches:
            path = Path(patch.file_path)
            normalized = path.as_posix()
            if path.is_absolute() or ".." in path.parts or normalized not in required:
                raise ValueError(f"Engineer returned undeclared artifact: {patch.file_path}")
            if patch.action == "delete":
                raise ValueError(f"Engineer attempted to delete required artifact: {patch.file_path}")

        executor.apply_patches(repo_context, output)
        repo_root = Path(repo_context.repo_path)
        for artifact in sorted(required):
            target = repo_root / artifact
            if not target.is_file():
                raise ValueError(f"Required artifact was not generated: {artifact}")
            metrics_doc.generated_code[artifact] = target.read_text(encoding="utf-8")

    @staticmethod
    def _prepare_command(phase_id: str, command: str, repo_root: Path) -> str:
        """Ensure setup installs into a persistent venv and later phases use it."""
        if phase_id == "setup":
            return build_persistent_setup_command(command, repo_path=repo_root)
        return rewrite_command_for_paper_venv(command)

    @staticmethod
    def _is_fatal_command_error(result: ContainerRunResult) -> bool:
        blob = f"{result.stdout}\n{result.stderr}"
        return any(marker in blob for marker in FATAL_COMMAND_MARKERS)

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
    ) -> tuple[bool, bool, str, str]:
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
                return True, False, excerpt, result.stdout
            reason = "timeout" if result.timed_out else f"exit_code={result.exit_code}"
            metrics_doc.errors.append(f"{step_id} attempt {attempt} failed ({reason})")
            if self._is_fatal_command_error(result):
                self._log(log_path, f"Fatal script error for step={step_id}; not retrying.")
                return False, True, excerpt, result.stdout
            if attempt < self.max_attempts:
                self._log(log_path, f"Retrying same command after failure ({reason})")

        self._log(log_path, f"Exhausted retries for step={step_id}")
        if last_result is not None:
            metrics_doc.exit_code = 124 if last_result.timed_out else last_result.exit_code
            return False, False, excerpt, last_result.stdout
        metrics_doc.exit_code = 1
        return False, False, "", ""

    def _capture_metrics(
        self,
        repo_root: Path,
        results_relpath: str,
        default_benchmark: str,
        default_algorithm: str,
        default_seed: str,
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
            default_seed=default_seed,
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
                    variables.get("algorithm")
                    or variables.get("algo")
                    or variables.get("method")
                    or variables.get("optimizer")
                    or ""
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
        for key in ("algorithm", "algo", "method", "optimizer"):
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
