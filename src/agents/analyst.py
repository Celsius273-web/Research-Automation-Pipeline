from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from urllib import error, request

from pydantic import ValidationError

from src.agents.prompts.analyst_prompt import build_analyst_system_prompt
from src.config import ANALYST_MODEL, ANALYST_SECTION_CHARS, MODEL_NUM_PREDICT, MODEL_TEMPERATURE, OLLAMA_HOST
from src.state import (
    ExtractionBundle,
    ReportedResult,
    SECTION_NAMES,
    SectionExtraction,
    SectionTextMap,
)

logger = logging.getLogger(__name__)

_VALID_SECTION_FIELDS = frozenset({
    "research_question", "methodology", "datasets_or_benchmarks",
    "variables", "hyperparameters", "evaluation_metrics", "reported_results", "notes",
})

# JSON Schema passed to Ollama's structured-output feature so the model is
# constrained to our exact field set rather than inventing its own schema.
_SECTION_EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "research_question": {"type": "string"},
        "methodology": {"type": "string"},
        "datasets_or_benchmarks": {"type": "array", "items": {"type": "string"}},
        "variables": {"type": "array", "items": {"type": "string"}},
        "hyperparameters": {"type": "object", "additionalProperties": {"type": "string"}},
        "evaluation_metrics": {"type": "array", "items": {"type": "string"}},
        "reported_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "benchmark": {"type": "string"},
                    "metric_name": {"type": "string"},
                    "value": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["benchmark", "metric_name", "value", "source"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": [
        "research_question", "methodology", "datasets_or_benchmarks",
        "variables", "hyperparameters", "evaluation_metrics",
        "reported_results", "notes",
    ],
}

_SCHEMA_REMINDER = (
    '{"research_question":"","methodology":"","datasets_or_benchmarks":[],'
    '"variables":[],"hyperparameters":{},"evaluation_metrics":[],'
    '"reported_results":[],"notes":""}'
)


def _build_retry_reminder(last_error: Exception | None) -> str:
    """Build a targeted retry prompt that names the exact fields to remove."""
    lines = [
        "Your previous response did not match the required schema.",
        "Return ONLY a JSON object with these exact fields — no others are permitted:",
        _SCHEMA_REMINDER,
    ]
    if isinstance(last_error, ValidationError):
        forbidden = [
            str(err["loc"][0])
            for err in last_error.errors()
            if err["type"] == "extra_forbidden"
        ]
        if forbidden:
            lines.append(
                f"Remove these forbidden fields you used: {', '.join(forbidden)}"
            )
    return "\n".join(lines)


def _clean_json_response(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = raw.rstrip("`").strip()
    return raw


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def merge_section_extractions(
    extractions: dict[str, SectionExtraction],
) -> SectionExtraction:
    merged = SectionExtraction()
    merged_reported: dict[tuple[str, str], ReportedResult] = {}
    for section in SECTION_NAMES:
        ext = extractions.get(section)
        if not ext:
            continue
        if not merged.research_question and ext.research_question:
            merged.research_question = ext.research_question
        if not merged.methodology and ext.methodology:
            merged.methodology = ext.methodology

        merged.datasets_or_benchmarks = _dedupe_keep_order(
            merged.datasets_or_benchmarks + ext.datasets_or_benchmarks
        )
        merged.variables = _dedupe_keep_order(merged.variables + ext.variables)
        merged.evaluation_metrics = _dedupe_keep_order(
            merged.evaluation_metrics + ext.evaluation_metrics
        )
        for item in ext.reported_results:
            key = (item.benchmark.strip().lower(), item.metric_name.strip().lower())
            if key not in merged_reported or section == "experiments":
                merged_reported[key] = item

        for k, v in ext.hyperparameters.items():
            if k not in merged.hyperparameters and str(v).strip():
                merged.hyperparameters[k] = str(v).strip()
        if ext.notes:
            merged.notes = (merged.notes + "\n" + ext.notes).strip()
    merged.reported_results = list(merged_reported.values())
    return merged


_DEFAULT_DOMAIN_VOCAB = (
    "- bayesian optimization\n"
    "- surrogate model\n"
    "- acquisition function\n"
    "- black-box objective\n"
    "- sample efficiency"
)


@dataclass
class PaperAnalyst:
    model: str = ANALYST_MODEL
    max_parse_retries: int = 1
    domain_vocabulary: str = _DEFAULT_DOMAIN_VOCAB

    def _call_ollama_json(self, section: str, section_text: str) -> SectionExtraction:
        system_prompt = build_analyst_system_prompt(
            domain_vocabulary_block=self.domain_vocabulary
        )
        prompt = f"""
Section name: {section}

Extract fields from this section only.

Text:
{section_text[:ANALYST_SECTION_CHARS]}
""".strip()

        last_error: Exception | None = None
        raw_response: str = ""
        for attempt in range(self.max_parse_retries + 1):
            attempt_prompt = prompt
            if attempt > 0:
                attempt_prompt = f"{prompt}\n\n{_build_retry_reminder(last_error)}"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": attempt_prompt},
                ],
                "stream": False,
                "think": False,
                "format": _SECTION_EXTRACTION_JSON_SCHEMA,
                "options": {
                    "temperature": MODEL_TEMPERATURE,
                    "num_predict": MODEL_NUM_PREDICT,
                },
            }
            logger.info("Analyst prompt (section=%s): %s", section, attempt_prompt)
            try:
                req = request.Request(
                    f"{OLLAMA_HOST}/api/chat",
                    method="POST",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with request.urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                raw_response = str(body.get("message", {}).get("content", ""))
                logger.info("Analyst raw response (section=%s): %s", section, raw_response)
                if not raw_response.strip():
                    raise json.JSONDecodeError("Empty model response", "", 0)
                cleaned = _clean_json_response(raw_response)
                parsed = json.loads(cleaned)
                return SectionExtraction.model_validate(parsed)
            except (
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
            except (error.URLError, error.HTTPError, TimeoutError) as exc:
                raise RuntimeError(
                    f"Analyst request failed for section '{section}': {exc}"
                ) from exc

        raise RuntimeError(
            "Failed to parse analyst JSON extraction for section "
            f"'{section}' after retry. last_error={last_error}. raw_response={raw_response!r}"
        )

    def extract(self, sections: SectionTextMap) -> ExtractionBundle:
        by_section: dict[str, SectionExtraction] = {}
        fallback_text = sections.full_text[:18000]

        for name in SECTION_NAMES:
            text = getattr(sections, name, "").strip()
            if not text:
                text = fallback_text
            if not text:
                raise RuntimeError(f"No usable text available for section '{name}'.")
            try:
                by_section[name] = self._call_ollama_json(name, text)
            except RuntimeError as exc:
                # Partial extraction is better than total failure. Log the
                # section error and continue so the merged result and file
                # output are still produced for the sections that did succeed.
                logger.warning("Section '%s' extraction failed after retries: %s", name, exc)
                by_section[name] = SectionExtraction()

        merged = merge_section_extractions(by_section)
        return ExtractionBundle(by_section=by_section, merged=merged)
