"""Engineer agent: propose code patches for one execution-plan step."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from urllib import error, request

from pydantic import ValidationError

from src.agents.prompts.engineer_prompt import build_engineer_system_prompt
from src.config import ENGINEER_MODEL, MODEL_NUM_PREDICT, MODEL_TEMPERATURE, OLLAMA_HOST
from src.state import (
    AgentEnvelope,
    EngineerInputContext,
    EngineerOutput,
    EngineerPayload,
    EngineerPayloadCore,
    EngineerPayloadExtensions,
    PatchProposal,
)

logger = logging.getLogger(__name__)

STRICT_RETRY_REMINDER = (
    "Your previous response was empty or invalid. Return only the JSON object matching the schema above,"
    " with every field populated."
)

VALID_ACTIONS = {"create", "modify", "delete"}


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


def _normalize_legacy_engineer_payload(payload: dict[str, object]) -> dict[str, object]:
    """Normalize old-style engineer payload to legacy EngineerOutput format."""
    data = dict(payload)
    data["step_id"] = str(data.get("step_id", "")).strip()
    data["rationale"] = str(data.get("rationale", "")).strip()
    data["verification_commands"] = _ensure_list_str(data.get("verification_commands"))
    data["missing_context"] = _ensure_list_str(data.get("missing_context"))

    patches_raw = data.get("patches", [])
    if not isinstance(patches_raw, list):
        patches_raw = []
    normalized_patches: list[dict[str, str]] = []
    for patch in patches_raw:
        if not isinstance(patch, dict):
            continue
        action = str(patch.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            continue
        file_path = str(patch.get("file_path", "")).strip()
        if not file_path:
            continue
        content = str(patch.get("content", ""))
        rationale = str(patch.get("rationale", "")).strip()
        if action == "delete":
            content = ""
        normalized_patches.append(
            {
                "file_path": file_path,
                "action": action,
                "content": content,
                "rationale": rationale,
            }
        )
    data["patches"] = normalized_patches
    return data


def _convert_envelope_to_legacy_engineer(envelope: AgentEnvelope[EngineerPayload]) -> EngineerOutput:
    """Convert new envelope structure to legacy EngineerOutput format."""
    core = envelope.payload.core
    ext = envelope.payload.extensions
    
    return EngineerOutput(
        step_id=core.step_id,
        patches=core.patches,
        verification_commands=core.verification_commands,
        rationale=ext.rationale,
        missing_context=ext.missing_context,
    )


@dataclass
class PaperEngineer:
    model: str = ENGINEER_MODEL
    max_parse_retries: int = 1

    def _call_ollama_json(self, context: EngineerInputContext) -> EngineerOutput:
        system_prompt = build_engineer_system_prompt(
            domain_vocabulary_block=(
                "- bayesian optimization\n"
                "- objective value\n"
                "- regret\n"
                "- benchmark\n"
                "- deterministic seed"
            )
        )
        payload = context.model_dump()
        prompt = f"""
Generate a patch proposal for this single plan step.
Keep changes minimal and directly tied to the goal.
If failure_context is present, address that failure explicitly.

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
            logger.info("Engineer prompt payload: %s", json.dumps(req_payload, indent=2))
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
                logger.info("Engineer raw response: %s", raw_response)
                if not raw_response.strip():
                    raise json.JSONDecodeError("Empty model response", "", 0)
                parsed = json.loads(_clean_json_response(raw_response))
                
                # Try parsing as new envelope format first
                if "schema_version" in parsed and parsed.get("schema_version") == "2.0":
                    envelope = AgentEnvelope[EngineerPayload].model_validate(parsed)
                    
                    # Check status and log warnings/unknowns
                    if envelope.status == "blocked":
                        raise RuntimeError(
                            f"Engineer blocked. Unknowns: {envelope.unknowns}, Warnings: {envelope.warnings}"
                        )
                    if envelope.unknowns:
                        logger.warning("Engineer has unknowns: %s", envelope.unknowns)
                    if envelope.warnings:
                        logger.warning("Engineer warnings: %s", envelope.warnings)
                    
                    # Core fields are validated by Pydantic; extensions are optional
                    # Convert to legacy format for backward compatibility
                    return _convert_envelope_to_legacy_engineer(envelope)
                else:
                    # Fall back to legacy format
                    logger.info("Parsing as legacy engineer format")
                    normalized = _normalize_legacy_engineer_payload(parsed)
                    return EngineerOutput.model_validate(normalized)
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
            "Failed to build engineer JSON after retry. "
            f"last_error={last_error}. raw_response={raw_response!r}"
        )

    def propose_patch(self, context: EngineerInputContext) -> EngineerOutput:
        return self._call_ollama_json(context)
