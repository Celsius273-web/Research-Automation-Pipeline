"""Planner agent: generate a structured execution plan from approved extraction."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

from pydantic import ValidationError

from src.agents.planner_debug import PlannerDebugTrace, write_planner_debug_files
from src.agents.prompts.planner_prompt import build_planner_system_prompt
from src.config import MODEL_NUM_PREDICT, MODEL_TEMPERATURE, OLLAMA_HOST, PLANNER_MODEL, PLANNER_MAX_RETRIES
from src.planner_input import build_unified_planner_input
from src.state import (
    AgentEnvelope,
    ExecutionPlan,
    PlannerInputContext,
    PlannerPayload,
    PlanStep,
    SectionExtraction,
    UnifiedPlannerInput,
)

logger = logging.getLogger(__name__)

STRICT_RETRY_REMINDER = (
    "Your previous response was empty or invalid. Return only the JSON object matching the schema above,"
    " with every field populated."
)

AIM_GROUNDING_RETRY_REMINDER = (
    "Research question was provided in analyst_output; do not mark it unknown. "
    "Restate the paper aim in objective, then plan repo steps for the Engineer."
)

METHODOLOGY_AIM_RETRY_REMINDER = (
    "flags.has_research_question is false but methodology is present. "
    "Do not block. Synthesize objective from methodology, leave unknown entrypoints as empty "
    "run_command with unknowns, and use status partial or ok."
)

BLOCKED_SOFTEN_RETRY_REMINDER = (
    "Do not use status blocked when methodology or research_question is present. "
    "Use status partial, keep unknowns, and continue planning from available context."
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


def _extraction_to_analyst_dict(extraction: SectionExtraction) -> dict[str, object]:
    """Serialize SectionExtraction into the analyst_output shape sent to the LLM."""
    return {
        "research_question": extraction.research_question,
        "methodology": extraction.methodology,
        "datasets_or_benchmarks": extraction.datasets_or_benchmarks,
        "variables": extraction.variables,
        "hyperparameters": dict(extraction.hyperparameters),
        "evaluation_metrics": extraction.evaluation_metrics,
        "reported_results": [item.model_dump() for item in extraction.reported_results],
        "notes": extraction.notes,
    }


def _is_present_research_question(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and not text.startswith("unknown:")


def _unknown_mentions_aim(unknowns: list[object]) -> bool:
    """True if unknowns claim research_question / aim is missing."""
    for item in unknowns:
        field = ""
        if hasattr(item, "field"):
            field = str(getattr(item, "field", "")).lower()
        elif isinstance(item, dict):
            field = str(item.get("field", "")).lower()
        if "research_question" in field or field.endswith(".aim") or field == "aim":
            return True
    return False


def _plan_misses_aim(
    research_question: str,
    methodology: str,
    plan: AgentEnvelope[PlannerPayload] | ExecutionPlan,
) -> bool:
    """Soft gate: present aim sources must yield a non-empty objective/summary."""
    has_rq = _is_present_research_question(research_question)
    has_method = bool((methodology or "").strip()) and not (methodology or "").strip().lower().startswith(
        "unknown:"
    )
    if not has_rq and not has_method:
        return False

    if isinstance(plan, AgentEnvelope):
        if has_rq and _unknown_mentions_aim(list(plan.unknowns)):
            return True
        core = plan.payload.core
        return not (core.objective or "").strip() or not (core.plan_summary or "").strip()

    return not (plan.objective or "").strip() or not (plan.plan_summary or "").strip()


def _should_soften_blocked(
    research_question: str,
    methodology: str,
    plan: AgentEnvelope[PlannerPayload],
) -> bool:
    """Blocked is only valid when both RQ and methodology are absent."""
    if plan.status != "blocked":
        return False
    has_rq = _is_present_research_question(research_question)
    has_method = bool((methodology or "").strip()) and not (methodology or "").strip().lower().startswith(
        "unknown:"
    )
    return has_rq or has_method


def _soften_blocked_envelope(plan: AgentEnvelope[PlannerPayload]) -> AgentEnvelope[PlannerPayload]:
    """Downgrade an over-aggressive blocked response to partial."""
    warnings = list(plan.warnings)
    note = "status downgraded from blocked to partial because methodology or research_question is present"
    if note not in warnings:
        warnings.append(note)
    return plan.model_copy(update={"status": "partial", "warnings": warnings})


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


def _resolve_debug_dir(context: PlannerInputContext | UnifiedPlannerInput) -> Path:
    if isinstance(context, PlannerInputContext) and context.paper_bundle_path:
        return Path(context.paper_bundle_path)
    paper = context.paper if isinstance(context, PlannerInputContext) else context.paper_context
    return Path("data") / "papers" / paper.paper_id


def _write_debug_trace(
    context: PlannerInputContext | UnifiedPlannerInput,
    trace: PlannerDebugTrace,
) -> None:
    output_dir = _resolve_debug_dir(context)
    json_path, md_path = write_planner_debug_files(trace, output_dir, saved_plan=None)
    logger.info("Planner debug written to %s (and %s)", md_path, json_path)


@dataclass
class PaperPlanner:
    model: str = PLANNER_MODEL
    max_parse_retries: int = PLANNER_MAX_RETRIES

    def _call_ollama_json(
        self,
        context: PlannerInputContext | UnifiedPlannerInput,
    ) -> AgentEnvelope[PlannerPayload]:
        unified_input = (
            context
            if isinstance(context, UnifiedPlannerInput)
            else build_unified_planner_input(context)
        )
        research_question = unified_input.analyst_output.research_question
        methodology = unified_input.analyst_output.methodology
        system_prompt = build_planner_system_prompt()
        payload = unified_input.model_dump()

        trace = PlannerDebugTrace(
            paper_id=unified_input.paper_context.paper_id,
            model=self.model,
            received_context=payload,
            system_prompt=system_prompt,
        )

        prompt = f"""
        Create an Engineer-ready execution plan from this unified Planner input.
        Use analyst_output as ground truth and honor repo_context and flags.

        Context JSON:
        {json.dumps(payload, indent=2)}
        """.strip()

        last_error: Exception | None = None
        raw_response: str = ""
        parse_failures = 0
        aim_retry_used = False
        blocked_retry_used = False
        use_aim_reminder = False
        use_blocked_reminder = False
        use_method_aim_reminder = False

        try:
            while parse_failures <= self.max_parse_retries:
                reminder = "none"
                attempt_prompt = prompt
                if use_aim_reminder:
                    attempt_prompt = f"{prompt}\n\n{AIM_GROUNDING_RETRY_REMINDER}"
                    reminder = "aim"
                    use_aim_reminder = False
                elif use_method_aim_reminder:
                    attempt_prompt = f"{prompt}\n\n{METHODOLOGY_AIM_RETRY_REMINDER}"
                    reminder = "method_aim"
                    use_method_aim_reminder = False
                elif use_blocked_reminder:
                    attempt_prompt = f"{prompt}\n\n{BLOCKED_SOFTEN_RETRY_REMINDER}"
                    reminder = "blocked"
                    use_blocked_reminder = False
                elif parse_failures > 0:
                    attempt_prompt = f"{prompt}\n\n{STRICT_RETRY_REMINDER}"
                    reminder = "strict"

                attempt = trace.add_attempt(
                    reminder=reminder,
                    user_prompt=attempt_prompt,
                    system_prompt=system_prompt,
                )

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
                    attempt.raw_response = raw_response

                    if not raw_response.strip():
                        raise json.JSONDecodeError("Empty model response", "", 0)
                    parsed = json.loads(_clean_json_response(raw_response))
                    attempt.parsed = parsed

                    envelope = AgentEnvelope[PlannerPayload].model_validate(parsed)
                    if envelope.schema_version != "2.0" or envelope.agent != "planner":
                        raise ValueError("Planner response must use schema_version 2.0 and agent planner")
                    if envelope.status not in {"ok", "partial", "blocked"}:
                        raise ValueError("Planner status must be ok, partial, or blocked")

                    if _should_soften_blocked(research_question, methodology, envelope):
                        if not blocked_retry_used:
                            attempt.outcome = "blocked_retry"
                            blocked_retry_used = True
                            use_blocked_reminder = True
                            continue
                        envelope = _soften_blocked_envelope(envelope)

                    if _plan_misses_aim(research_question, methodology, envelope) and not aim_retry_used:
                        attempt.outcome = "aim_retry"
                        aim_retry_used = True
                        if _is_present_research_question(research_question):
                            use_aim_reminder = True
                        else:
                            use_method_aim_reminder = True
                        continue
                    attempt.outcome = "accepted"
                    trace.final_output = envelope.model_dump()
                    return envelope
                except (
                    json.JSONDecodeError,
                    error.URLError,
                    error.HTTPError,
                    TimeoutError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    last_error = exc
                    attempt.error = str(exc)
                    attempt.outcome = "error"
                    parse_failures += 1
                    time.sleep(1.5 * parse_failures)

            trace.final_error = (
                f"Failed to build planner JSON after retry. last_error={last_error}. "
                f"raw_response={raw_response!r}"
            )
            raise RuntimeError(trace.final_error)
        finally:
            _write_debug_trace(context, trace)

    def build_plan(
        self,
        context: PlannerInputContext | UnifiedPlannerInput,
    ) -> AgentEnvelope[PlannerPayload]:
        return self._call_ollama_json(context)
