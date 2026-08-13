"""Run a saved Planner envelope without paper ingestion (Analyst/Planner skipped)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.agents.experiment_runner import ExperimentRunner
from src.agents.reviewer import PaperReviewer
from src.bundle import PaperBundle
from src.config import ENGINEER_METRICS_FILENAME, REVIEWER_REPORT_FILENAME, ROOT_DIR
from src.persistence import (
    load_metrics_document,
    load_planner_envelope,
    load_reported_results,
    persist_reviewer_run_report,
)
from src.state import (
    AgentEnvelope,
    ExtractionBundle,
    MetricsDocument,
    PaperMetadata,
    PlannerPayload,
    ReviewRecord,
    SectionExtraction,
)
from src.tools.docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


def _load_plan_payload(plan_path: Path) -> dict:
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Plan file does not exist: {plan_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plan JSON could not be parsed: {plan_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Plan JSON must be an object: {plan_path}")
    return payload


def paper_metadata_from_plan(payload: dict, paper_id: str) -> PaperMetadata:
    """Build PaperMetadata from a plan artifact, or a minimal stub when absent."""
    raw = payload.get("paper") or payload.get("paper_context") or {}
    if not isinstance(raw, dict):
        raw = {}
    title = str(raw.get("title") or "").strip() or paper_id
    pdf_path = str(raw.get("pdf_path") or "").strip()
    arxiv_id = raw.get("arxiv_id")
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        pdf_path=pdf_path,
        arxiv_id=str(arxiv_id) if arxiv_id else None,
    )


def _ensure_bundle(paper: PaperMetadata, envelope: AgentEnvelope[PlannerPayload], plan_path: Path) -> PaperBundle:
    bundle = PaperBundle(paper.paper_id)
    bundle.create_bundle_dir()
    if not bundle.plan_path.exists():
        bundle.save_plan(
            envelope,
            ReviewRecord(status="approved", notes="Loaded via run-plan (Analyst/Planner skipped)."),
            paper,
            str(plan_path),
        )
    if not bundle.extraction_path.exists():
        bundle.save_extraction(
            ExtractionBundle(
                merged=SectionExtraction(notes="Stub extraction for run-plan (Analyst skipped).")
            ),
            ReviewRecord(status="approved", notes="Stub; Analyst skipped."),
            paper,
        )
    return bundle


def _load_or_empty_metrics(run_dir: Path) -> MetricsDocument:
    metrics_path = run_dir / ENGINEER_METRICS_FILENAME
    if not metrics_path.exists():
        logger.warning("metrics.json not written at %s; reviewer will treat captures as missing.", metrics_path)
        return MetricsDocument(run_status="FAILED", exit_code=1, logs_captured=False)
    return load_metrics_document(run_dir)


def _write_reviewer_report(
    paper_id: str,
    bundle: PaperBundle,
    run_dir: Path,
    metrics_doc: MetricsDocument,
) -> Path:
    reported = load_reported_results(bundle.extraction_path)
    report = PaperReviewer.compare_reported_to_captured(
        reported,
        metrics_doc.metrics,
        paper_id=paper_id,
        metrics_doc=metrics_doc,
    )
    return persist_reviewer_run_report(run_dir, report, paper_id=paper_id)


def run_saved_plan(
    *,
    plan_path: str,
    paper_id: str,
    repo_path: str,
    non_interactive: bool = True,
) -> int:
    """Skip Analyst/Planner; run Engineer/Executor then Reviewer from a saved plan."""
    _ = non_interactive
    source = Path(plan_path).expanduser().resolve()
    repo = Path(repo_path).expanduser().resolve()
    if not source.exists():
        logger.error("Plan file does not exist: %s", source)
        return 1
    if not repo.exists():
        logger.error("Repository path does not exist: %s", repo)
        return 1

    try:
        payload = _load_plan_payload(source)
        envelope = load_planner_envelope(payload)
        paper = paper_metadata_from_plan(payload, paper_id=paper_id)
        bundle = _ensure_bundle(paper, envelope, source)
        runner = ExperimentRunner(docker_executor=DockerExecutor(project_root=ROOT_DIR))
        metrics_doc, run_dir = runner.execute_paper(
            paper_id=paper_id,
            repo_path=str(repo),
            plan_path=source,
        )
    except (FileNotFoundError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.error("run-plan failed: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    metrics_doc = _load_or_empty_metrics(run_dir)
    report_path = _write_reviewer_report(paper_id, bundle, run_dir, metrics_doc)
    logger.info("Run directory: %s", run_dir)
    logger.info("Wrote %s, %s, %s", ENGINEER_METRICS_FILENAME, "engineer.log", REVIEWER_REPORT_FILENAME)
    logger.info("Reviewer report: %s", report_path)
    return 0 if metrics_doc.run_status in {"SUCCESS", "PARTIAL"} else 1
