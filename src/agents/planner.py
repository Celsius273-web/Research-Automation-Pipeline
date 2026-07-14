"""Planner agent: generate a structured execution plan from approved extraction."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from urllib import error, request

from pydantic import ValidationError

from src.agents.prompts.planner_prompt import build_planner_system_prompt
from src.config import MODEL_NUM_PREDICT, MODEL_TEMPERATURE, OLLAMA_HOST, PLANNER_MODEL
from src.state import (
    AgentEnvelope,
    ExecutionPlan,
    PlannerInputContext,
    PlannerPayload,
    PlannerPayloadCore,
    PlannerPayloadExtensions,
    PlanStep,
    ExperimentSpec,
)

logger = logging.getLogger(__name__)

STRICT_RETRY_REMINDER = (
    "Your previous response was empty or invalid. Return only the JSON object matching the schema above,"
    " with every field populated."
)


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


def _normalize_legacy_plan_payload(payload: dict[str, object]) -> dict[str, object]:
    """Normalize old-style planner payload to legacy ExecutionPlan format."""
    data = dict(payload)
    for field in (
        "assumptions",
        "constraints",
        "missing_context",
        "verification_checks",
        "risks",
    ):
        data[field] = _ensure_list_str(data.get(field))

    steps_raw = data.get("steps", [])
    if not isinstance(steps_raw, list):
        steps_raw = []
    normalized_steps: list[dict[str, object]] = []
    for idx, step in enumerate(steps_raw, start=1):
        if not isinstance(step, dict):
            continue
        normalized_steps.append(
            {
                "step_id": str(step.get("step_id", f"step_{idx}")).strip() or f"step_{idx}",
                "title": str(step.get("title", f"Step {idx}")).strip() or f"Step {idx}",
                "goal": str(step.get("goal", "")).strip(),
                "actions": _ensure_list_str(step.get("actions")),
                "inputs": _ensure_list_str(step.get("inputs")),
                "outputs": _ensure_list_str(step.get("outputs")),
                "verification": _ensure_list_str(step.get("verification")),
                "results_path": str(step.get("results_path", "")).strip(),
                "depends_on": _ensure_list_str(step.get("depends_on")),
                "failure_modes": _ensure_list_str(step.get("failure_modes")),
            }
        )
    data["steps"] = normalized_steps

    experiments_raw = data.get("experiment_matrix", [])
    if not isinstance(experiments_raw, list):
        experiments_raw = []
    normalized_experiments: list[dict[str, object]] = []
    for idx, item in enumerate(experiments_raw, start=1):
        if not isinstance(item, dict):
            continue
        hp = item.get("hyperparameters", {})
        normalized_hp: dict[str, str] = {}
        if isinstance(hp, dict):
            for key, value in hp.items():
                normalized_hp[str(key)] = str(value)
        normalized_experiments.append(
            {
                "name": str(item.get("name", f"experiment_{idx}")).strip() or f"experiment_{idx}",
                "target": str(item.get("target", "")).strip(),
                "variables": _ensure_list_str(item.get("variables")),
                "hyperparameters": normalized_hp,
                "metrics": _ensure_list_str(item.get("metrics")),
            }
        )
    data["experiment_matrix"] = normalized_experiments
    return data


def _convert_envelope_to_legacy_plan(envelope: AgentEnvelope[PlannerPayload]) -> ExecutionPlan:
    """Convert new envelope structure to legacy ExecutionPlan format."""
    core = envelope.payload.core
    ext = envelope.payload.extensions
    
    # Convert core steps to legacy format
    legacy_steps = [
        PlanStep(
            step_id=step.step_id,
            title=step.title,
            goal=step.goal,
            actions=[],
            inputs=[],
            outputs=[],
            verification=[],
            results_path=step.results_path,
            depends_on=step.depends_on,
            failure_modes=[],
        )
        for step in core.steps
    ]
    
    return ExecutionPlan(
        schema_version="1.0",
        plan_summary=core.plan_summary,
        domain=core.domain,
        objective=core.objective,
        assumptions=ext.assumptions,
        constraints=ext.constraints,
        missing_context=ext.missing_context,
        steps=legacy_steps,
        experiment_matrix=ext.experiment_matrix,
        verification_checks=ext.verification_checks,
        risks=ext.risks,
    )


@dataclass
class PaperPlanner:
    model: str = PLANNER_MODEL
    max_parse_retries: int = 1

    def _call_ollama_json(self, context: PlannerInputContext) -> ExecutionPlan:
        extraction = context.approved_extraction
        system_prompt = build_planner_system_prompt(
            domain_vocabulary_block=(
                "- bayesian optimization\n"
                "- gaussian process surrogate\n"
                "- acquisition function\n"
                "- evaluation budget\n"
                "- regret"
            )
        )
        analyst_structured = {
            "research_question": extraction.research_question or "unknown: missing research question",
            "methodology": extraction.methodology or "unknown: missing methodology",
            "datasets": extraction.datasets_or_benchmarks
            or ["unknown: no datasets identified"],
            "variables": extraction.variables or ["unknown: no variables identified"],
            "hyperparameters": [
                {"name": key, "value": value}
                for key, value in extraction.hyperparameters.items()
            ]
            or [{"name": "unknown", "value": "unknown: no hyperparameters identified"}],
            "evaluation_metrics": extraction.evaluation_metrics
            or ["unknown: no evaluation metrics identified"],
        }
        payload = {
            "paper": context.paper.model_dump(),
            "analyst_output": analyst_structured,
            "runtime_constraints": context.runtime_constraints,
            "repo_context": context.repo_context,
        }
        prompt = f"""
Create a structured execution plan from the approved extraction context.
Prioritize reproducibility and deterministic validation steps.
Keep each action concrete and implementation-oriented.

Context JSON:
{json.dumps(payload, indent=2)}
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
            logger.info("Planner prompt payload: %s", json.dumps(req_payload, indent=2))
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
                logger.info("Planner raw response: %s", raw_response)
                if not raw_response.strip():
                    raise json.JSONDecodeError("Empty model response", "", 0)
                parsed = json.loads(_clean_json_response(raw_response))
                
                # Try parsing as new envelope format first
                if "schema_version" in parsed and parsed.get("schema_version") == "2.0":
                    envelope = AgentEnvelope[PlannerPayload].model_validate(parsed)
                    
                    # Check status and log warnings/unknowns
                    if envelope.status == "blocked":
                        raise RuntimeError(
                            f"Planner blocked. Unknowns: {envelope.unknowns}, Warnings: {envelope.warnings}"
                        )
                    if envelope.unknowns:
                        logger.warning("Planner has unknowns: %s", envelope.unknowns)
                    if envelope.warnings:
                        logger.warning("Planner warnings: %s", envelope.warnings)
                    
                    # Core fields are validated by Pydantic; extensions are optional
                    # Convert to legacy format for backward compatibility
                    return _convert_envelope_to_legacy_plan(envelope)
                else:
                    # Fall back to legacy format
                    logger.info("Parsing as legacy plan format")
                    normalized = _normalize_legacy_plan_payload(parsed)
                    return ExecutionPlan.model_validate(normalized)
            except (
                json.JSONDecodeError,
                error.URLError,
                error.HTTPError,
                TimeoutError,
                ValidationError,
            ) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            "Failed to build planner JSON after retry. "
            f"last_error={last_error}. raw_response={raw_response!r}"
        )

    def build_plan(self, context: PlannerInputContext) -> ExecutionPlan:
        return self._call_ollama_json(context)
