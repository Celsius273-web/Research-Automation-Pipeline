"""Reviewer agent: deterministic comparison plus narrative report text."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from urllib import error, request

from pydantic import BaseModel, Field, ValidationError

from src.agents.prompts.reviewer_prompt import build_reviewer_system_prompt
from src.config import MODEL_NUM_PREDICT, MODEL_TEMPERATURE, OLLAMA_HOST, REVIEWER_MODEL
from src.state import (
    AgentEnvelope,
    ComparisonRow,
    MetricResult,
    ReportedResult,
    ReviewerPayload,
    ReviewerPayloadCore,
    ReviewerPayloadExtensions,
    ReviewerReport,
)
from src.tools.result_comparator import compare_results, verdict_from_rate

logger = logging.getLogger(__name__)

STRICT_RETRY_REMINDER = (
    "Your previous response was empty or invalid. Return only the JSON object matching the schema above,"
    " with every field populated."
)


class ReviewerNarrativeOutput(BaseModel):
    summary: str = ""
    comparison_rows: list[dict[str, str]] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _clean_json_response(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = raw.rstrip("`").strip()
    return raw


def _ensure_list_str(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_legacy_narrative_payload(payload: dict[str, object]) -> dict[str, object]:
    """Normalize old-style reviewer payload to legacy ReviewerNarrativeOutput format."""
    data = dict(payload)
    data["summary"] = str(data.get("summary", "")).strip()
    rows = data.get("comparison_rows", [])
    if not isinstance(rows, list):
        rows = []
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized_rows.append(
            {
                "metric_name": str(row.get("metric_name", "")).strip(),
                "paper_reported_value": str(row.get("paper_reported_value", "")).strip(),
                "reproduced_value": str(row.get("reproduced_value", "")).strip(),
                "delta": str(row.get("delta", "")).strip(),
                "match_status": str(row.get("match_status", "")).strip(),
            }
        )
    data["comparison_rows"] = normalized_rows
    data["risks"] = _ensure_list_str(data.get("risks"))
    data["notes"] = _ensure_list_str(data.get("notes"))
    return data


def _convert_envelope_to_narrative(envelope: AgentEnvelope[ReviewerPayload]) -> ReviewerNarrativeOutput:
    """Convert new envelope structure to ReviewerNarrativeOutput format."""
    core = envelope.payload.core
    ext = envelope.payload.extensions
    
    # Convert comparison rows to legacy format for the narrative output
    comparison_rows_dicts = [
        {
            "metric_name": row.metric_name,
            "paper_reported_value": row.reported_value,
            "reproduced_value": row.reproduced_value,
            "delta": str(row.absolute_difference) if row.absolute_difference is not None else "",
            "match_status": row.match_status,
        }
        for row in core.comparison_rows
    ]
    
    return ReviewerNarrativeOutput(
        summary=core.summary,
        comparison_rows=comparison_rows_dicts,
        risks=ext.risks,
        notes=ext.notes,
    )


@dataclass
class PaperReviewer:
    model: str = REVIEWER_MODEL
    max_parse_retries: int = 1

    def _call_narrative_json(self, prompt_payload: dict[str, object]) -> ReviewerNarrativeOutput:
        system_prompt = build_reviewer_system_prompt(
            domain_vocabulary_block=(
                "- bayesian optimization\n"
                "- reproduced metrics\n"
                "- reported metrics\n"
                "- delta\n"
                "- tolerance"
            )
        )
        prompt = f"""
Write a concise review narrative from this deterministic comparison context:
{json.dumps(prompt_payload, indent=2)}
""".strip()

        last_error: Exception | None = None
        raw_response: str = ""
        for attempt in range(self.max_parse_retries + 1):
            attempt_prompt = prompt
            if attempt > 0:
                attempt_prompt = f"{prompt}\n\n{STRICT_RETRY_REMINDER}"
            req_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": attempt_prompt},
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "temperature": MODEL_TEMPERATURE,
                    "num_predict": MODEL_NUM_PREDICT,
                },
            }
            logger.info("Reviewer prompt payload: %s", json.dumps(req_payload, indent=2))
            try:
                req = request.Request(
                    f"{OLLAMA_HOST}/api/chat",
                    method="POST",
                    data=json.dumps(req_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with request.urlopen(req, timeout=180) as response:
                    body = json.loads(response.read().decode("utf-8"))
                raw_response = str(body.get("message", {}).get("content", ""))
                logger.info("Reviewer raw response: %s", raw_response)
                if not raw_response.strip():
                    raise json.JSONDecodeError("Empty model response", "", 0)
                parsed = json.loads(_clean_json_response(raw_response))
                
                # Try parsing as new envelope format first
                if "schema_version" in parsed and parsed.get("schema_version") == "2.0":
                    envelope = AgentEnvelope[ReviewerPayload].model_validate(parsed)
                    
                    # Check status and log warnings/unknowns
                    if envelope.status == "blocked":
                        raise RuntimeError(
                            f"Reviewer blocked. Unknowns: {envelope.unknowns}, Warnings: {envelope.warnings}"
                        )
                    if envelope.unknowns:
                        logger.warning("Reviewer has unknowns: %s", envelope.unknowns)
                    if envelope.warnings:
                        logger.warning("Reviewer warnings: %s", envelope.warnings)
                    
                    # Core fields are validated by Pydantic; extensions are optional
                    # Convert to narrative output format
                    return _convert_envelope_to_narrative(envelope)
                else:
                    # Fall back to legacy format
                    logger.info("Parsing as legacy reviewer format")
                    normalized = _normalize_legacy_narrative_payload(parsed)
                    return ReviewerNarrativeOutput.model_validate(normalized)
            except (
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
            except (error.URLError, error.HTTPError, TimeoutError) as exc:
                raise RuntimeError(f"Reviewer request failed: {exc}") from exc
        raise RuntimeError(
            "Failed to build reviewer narrative JSON after retry. "
            f"last_error={last_error}. raw_response={raw_response!r}"
        )

    def generate_report(
        self,
        paper_id: str,
        domain: str,
        reported_results: list[ReportedResult],
        captured_metrics: list[MetricResult],
        run_summary: dict[str, str],
    ) -> ReviewerReport:
        rows, reproduction_rate = compare_results(reported_results, captured_metrics)
        verdict = verdict_from_rate(reproduction_rate, has_comparable_rows=any(row.absolute_difference is not None for row in rows))
        narrative = self._call_narrative_json(
            {
                "paper_id": paper_id,
                "domain": domain,
                "verdict": verdict,
                "reproduction_rate": reproduction_rate,
                "run_summary": run_summary,
                "comparison_table": [row.model_dump() for row in rows],
            }
        )

        return ReviewerReport(
            paper_id=paper_id,
            domain=domain,
            summary=narrative.summary,
            verdict=verdict,
            reproduction_rate=reproduction_rate,
            comparison_table=rows,
            risks=narrative.risks,
            notes=narrative.notes,
            run_summary=run_summary,
        )
