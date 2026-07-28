from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from urllib import error, request

from pydantic import ValidationError

from src.agents.prompts.analyst_prompt import build_analyst_system_prompt
from src.config import (
    ANALYST_MAX_RESULT_CHUNKS,
    ANALYST_MODEL,
    ANALYST_NUM_PREDICT,
    ANALYST_RESULT_CHUNK_OVERLAP,
    ANALYST_RESULT_SECTION_CHARS,
    ANALYST_SECTION_CHARS,
    MODEL_TEMPERATURE,
    OLLAMA_HOST,
)
from src.state import (
    ExtractionBundle,
    ReportedResult,
    SECTION_NAMES,
    SectionExtraction,
    SectionTextMap,
)

logger = logging.getLogger(__name__)

_VALID_SECTION_FIELDS = frozenset({
    "research_question", "paper_overview", "methodology", "datasets_or_benchmarks",
    "variables", "hyperparameters", "evaluation_metrics", "reported_results", "notes",
})

# JSON Schema passed to Ollama's structured-output feature so the model is
# constrained to our exact field set rather than inventing its own schema.
_SECTION_EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "research_question": {"type": "string"},
        "paper_overview": {"type": "string"},
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
        "research_question", "paper_overview", "methodology", "datasets_or_benchmarks",
        "variables", "hyperparameters", "evaluation_metrics",
        "reported_results", "notes",
    ],
}

_SCHEMA_REMINDER = (
    '{"research_question":"","paper_overview":"","methodology":"",'
    '"datasets_or_benchmarks":[],"variables":[],"hyperparameters":{},'
    '"evaluation_metrics":[],"reported_results":[],"notes":""}'
)

_RESULT_FOCUS_SECTIONS = frozenset({"experiments", "appendix"})

_WEAK_RESULT_VALUE_PREFIXES = (
    "see table",
    "see figure",
    "refer to table",
    "refer to figure",
    "as shown in table",
    "as shown in figure",
    "reported in table",
    "reported in tab",
)

_TOOLKIT_TERMS = (
    "library",
    "toolkit",
    "software package",
    "framework implementing",
    "open-source",
)

_VAGUE_DATASET_PHRASES = (
    "synthetic",
    "real-world",
    "various domains",
    "various datasets",
    "benchmark problems",
    "standard benchmarks",
    "unknown",
)

_JUNK_DATASET_MARKERS = (
    "lemma",
    "theorem",
    "corollary",
    "et al",
    "implementation details",
)

# Map alternate hyperparameter keys onto a canonical name (lowercase).
_HPARAM_KEY_ALIASES: dict[str, str] = {
    "neurons_per_layer_formula": "hidden_layer_size_formula",
    "neuron_scaling_factor_c": "constant_factor_c_tested",
    "constant_factor_c": "constant_factor_c_tested",
    "optimizer_nll": "optimizer",
    "learning_rate_nll": "learning_rate",
    "gp_ucb_coefficient": "ucb_coefficient",
    "ucb_coefficient_stboh": "ucb_coefficient",
    "mean_network_architecture_h-nll": "mean_network_architecture",
    "kernel_type_h-nll_fsbo": "kernel_type",
    "deep ensembles": "deep_ensembles",
    "acquisition function": "acquisition_function",
    "constraint_handling_method": "constraint_handling",
}

_DATASET_DIM_SUFFIX_RE = re.compile(r"\s*\((\d+)\s*[dD]\)\s*$")
_DATASET_VARIANT_RE = re.compile(
    r"^(?P<dataset>.+?)\s+(?P<arch>[A-Za-z][\w+.-]*)\s+(?P<size>\d+)$"
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


def _is_present_text(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and not text.lower().startswith("unknown:")


def _looks_like_toolkit(methodology: str, notes: str = "") -> bool:
    description = f"{methodology} {notes}".lower()
    return any(term in description for term in _TOOLKIT_TERMS)


def _is_junk_dataset(name: str) -> bool:
    text = name.strip().lower()
    if not text:
        return True
    if any(marker in text for marker in _JUNK_DATASET_MARKERS):
        return True
    if any(phrase in text for phrase in _VAGUE_DATASET_PHRASES) and len(text.split()) <= 5:
        return True
    return False


def _clean_datasets(values: list[str]) -> list[str]:
    return _dedupe_keep_order([item for item in values if not _is_junk_dataset(item)])


def _normalize_dataset_key(name: str) -> str:
    text = re.sub(r"\s+", " ", (name or "").strip().lower())
    text = _DATASET_DIM_SUFFIX_RE.sub("", text).strip()
    return text


def _dataset_specificity(name: str) -> tuple[int, int]:
    """Prefer dimensioned / longer names when collapsing duplicates."""
    has_dim = 1 if _DATASET_DIM_SUFFIX_RE.search(name) else 0
    return (has_dim, len(name))


def _collapse_dataset_aliases(values: list[str]) -> list[str]:
    """Keep one entry per base name; prefer the more specific spelling."""
    best: dict[str, str] = {}
    order: list[str] = []
    for item in values:
        key = _normalize_dataset_key(item)
        if not key:
            continue
        if key not in best:
            best[key] = item
            order.append(key)
            continue
        if _dataset_specificity(item) > _dataset_specificity(best[key]):
            best[key] = item
    return [best[key] for key in order]


def _matrix_collapse_dataset_variants(values: list[str]) -> list[str]:
    """Collapse 'Dataset Arch Batch' triples into matrix notation when possible."""
    groups: dict[str, dict[str, set[str] | list[str]]] = {}
    leftovers: list[str] = []
    group_order: list[str] = []

    for item in values:
        match = _DATASET_VARIANT_RE.match(item.strip())
        if not match:
            leftovers.append(item)
            continue
        dataset = match.group("dataset").strip()
        arch = match.group("arch").strip()
        size = match.group("size").strip()
        key = dataset.lower()
        if key not in groups:
            groups[key] = {"dataset": dataset, "arches": set(), "sizes": set(), "raw": []}
            group_order.append(key)
        groups[key]["arches"].add(arch)  # type: ignore[union-attr]
        groups[key]["sizes"].add(size)  # type: ignore[union-attr]
        groups[key]["raw"].append(item)  # type: ignore[union-attr]

    collapsed: list[str] = []
    for key in group_order:
        group = groups[key]
        raw = list(group["raw"])  # type: ignore[arg-type]
        arches = sorted(group["arches"])  # type: ignore[arg-type]
        sizes = sorted(group["sizes"], key=lambda value: int(value))  # type: ignore[arg-type]
        if len(raw) >= 2 and (len(arches) > 1 or len(sizes) > 1):
            dataset = str(group["dataset"])
            collapsed.append(
                f"{dataset} × {{{', '.join(arches)}}} × batch {{{', '.join(sizes)}}}"
            )
        else:
            collapsed.extend(raw)
    return _dedupe_keep_order(collapsed + leftovers)


def _organize_datasets(values: list[str]) -> list[str]:
    cleaned = _clean_datasets(values)
    collapsed = _collapse_dataset_aliases(cleaned)
    return _matrix_collapse_dataset_variants(collapsed)


def _normalize_hparam_key(key: str) -> str:
    text = re.sub(r"[\s/]+", "_", (key or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return _HPARAM_KEY_ALIASES.get(text, text)


def _values_equivalent(left: str, right: str) -> bool:
    def _norm(value: str) -> str:
        text = re.sub(r"\s+", " ", (value or "").strip().lower())
        text = text.replace("$", "").replace("\\times", "x").replace("×", "x")
        return text

    return _norm(left) == _norm(right)


def _merge_hyperparameter_aliases(params: dict[str, str]) -> dict[str, str]:
    """Collapse alias keys onto canonical names when values agree or only one exists."""
    merged: dict[str, str] = {}
    for raw_key, raw_value in params.items():
        value = str(raw_value).strip()
        if not value:
            continue
        key = _normalize_hparam_key(raw_key)
        if key not in merged:
            merged[key] = value
            continue
        if _values_equivalent(merged[key], value):
            # Prefer the longer/more specific wording when equal after normalize.
            if len(value) > len(merged[key]):
                merged[key] = value
            continue
        # Conflicting values: keep the original distinct key spelling.
        conflict_key = re.sub(r"[\s/]+", "_", raw_key.strip().lower())
        if conflict_key not in merged:
            merged[conflict_key] = value
    return merged


def _pick_preferred_text(*candidates: str) -> str:
    for value in candidates:
        if _is_present_text(value):
            return value.strip()
    return ""


def _has_numeric_value(value: str) -> bool:
    """True when value contains a digit suitable for reproduction comparison."""
    text = (value or "").strip()
    if not text or text in {"}", "{", "}, {", "], ["}:
        return False
    return bool(re.search(r"\d", text))


def _clean_reported_results(results: list[ReportedResult]) -> list[ReportedResult]:
    cleaned: list[ReportedResult] = []
    for item in results:
        metric = (item.metric_name or "").strip()
        value = (item.value or "").strip()
        if not metric or not value:
            continue
        lowered = value.lower()
        if any(lowered.startswith(prefix) for prefix in _WEAK_RESULT_VALUE_PREFIXES):
            continue
        if lowered in {"n/a", "na", "none", "-", "null"}:
            continue
        if not _has_numeric_value(value):
            continue
        cleaned.append(item)
    return cleaned


def chunk_section_text(
    text: str,
    chunk_size: int = ANALYST_RESULT_SECTION_CHARS,
    overlap: int = ANALYST_RESULT_CHUNK_OVERLAP,
    max_chunks: int = ANALYST_MAX_RESULT_CHUNKS,
) -> list[str]:
    """Split long result-heavy sections into overlapping windows."""
    content = (text or "").strip()
    if not content:
        return []
    if len(content) <= chunk_size:
        return [content]

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(content) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(content))
        chunks.append(content[start:end])
        if end >= len(content):
            break
        start += step
    return chunks


def merge_chunk_extractions(chunks: list[SectionExtraction]) -> SectionExtraction:
    """Merge multiple extractions from the same section's text windows."""
    if not chunks:
        return SectionExtraction()
    if len(chunks) == 1:
        return chunks[0]

    merged = SectionExtraction()
    merged_reported: dict[tuple[str, str, str], ReportedResult] = {}
    for chunk in chunks:
        if not merged.research_question and chunk.research_question:
            merged.research_question = chunk.research_question
        if not merged.paper_overview and chunk.paper_overview:
            merged.paper_overview = chunk.paper_overview
        if not merged.methodology and chunk.methodology:
            merged.methodology = chunk.methodology
        merged.datasets_or_benchmarks = _dedupe_keep_order(
            merged.datasets_or_benchmarks + chunk.datasets_or_benchmarks
        )
        merged.variables = _dedupe_keep_order(merged.variables + chunk.variables)
        merged.evaluation_metrics = _dedupe_keep_order(
            merged.evaluation_metrics + chunk.evaluation_metrics
        )
        for item in chunk.reported_results:
            key = (
                item.benchmark.strip().lower(),
                item.metric_name.strip().lower(),
                item.value.strip().lower(),
            )
            merged_reported[key] = item
        for key, value in chunk.hyperparameters.items():
            if key not in merged.hyperparameters and str(value).strip():
                merged.hyperparameters[key] = str(value).strip()
        if chunk.notes:
            merged.notes = (merged.notes + "\n" + chunk.notes).strip()
    merged.reported_results = list(merged_reported.values())
    return merged


def merge_section_extractions(
    extractions: dict[str, SectionExtraction],
) -> SectionExtraction:
    merged = SectionExtraction()
    merged_reported: dict[tuple[str, str], ReportedResult] = {}

    abstract = extractions.get("abstract")
    if abstract and _is_present_text(abstract.research_question):
        merged.research_question = abstract.research_question.strip()

    overview_candidates = [
        (extractions.get(name).paper_overview if extractions.get(name) else "")
        for name in ("abstract", "method", "experiments", "hyperparameters", "appendix")
    ]
    merged.paper_overview = _pick_preferred_text(*overview_candidates)

    for section in SECTION_NAMES:
        ext = extractions.get(section)
        if not ext:
            continue
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


def soft_fill_research_question(
    extraction: SectionExtraction,
    paper_title: str = "",
) -> SectionExtraction:
    """Fill empty RQ from toolkit/overview/methodology/title and mark inference in notes."""
    if _is_present_text(extraction.research_question):
        return extraction

    methodology = (extraction.methodology or "").strip()
    overview = (extraction.paper_overview or "").strip()
    title = (paper_title or "").strip()
    notes = extraction.notes or ""

    if _looks_like_toolkit(methodology, notes) or _looks_like_toolkit(overview, notes):
        purpose = methodology or overview or title or "an open-source research toolkit"
        filled = f"Toolkit paper: {purpose}"
        inference_note = (
            "[inferred] research_question: toolkit paper purpose synthesized from methodology/title"
        )
    elif overview:
        first = re.split(r"(?<=[.!?])\s+", overview, maxsplit=1)[0].strip()
        filled = first if first.endswith("?") else f"How can we address: {first}"
        inference_note = "[inferred] research_question synthesized from paper_overview"
    elif methodology:
        filled = f"How can we implement and evaluate the following method: {methodology}"
        inference_note = "[inferred] research_question synthesized from methodology"
    elif title:
        filled = f"What scientific contribution does '{title}' make, and how can it be reproduced?"
        inference_note = "[inferred] research_question synthesized from paper title"
    else:
        return extraction

    if len(filled) > 500:
        filled = filled[:497].rstrip() + "..."

    updated_notes = (notes + "\n" + inference_note).strip()
    return extraction.model_copy(
        update={
            "research_question": filled,
            "notes": updated_notes,
        }
    )


def finalize_merged_extraction(
    extraction: SectionExtraction,
    paper_title: str = "",
) -> SectionExtraction:
    """Deterministic cleanup + soft-fill after section merge."""
    cleaned = extraction.model_copy(
        update={
            "paper_overview": (extraction.paper_overview or "").strip(),
            "datasets_or_benchmarks": _organize_datasets(extraction.datasets_or_benchmarks),
            "hyperparameters": _merge_hyperparameter_aliases(dict(extraction.hyperparameters)),
            "reported_results": _clean_reported_results(extraction.reported_results),
        }
    )
    return soft_fill_research_question(cleaned, paper_title=paper_title)


_DEFAULT_DOMAIN_VOCAB = (
    "- named datasets and benchmarks\n"
    "- experimental hyperparameters and numeric settings\n"
    "- evaluation metrics\n"
    "- algorithmic methods and toolkit purpose statements"
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
        # Caller already sized the window; keep a hard cap as a safety net.
        clipped = section_text[: max(ANALYST_SECTION_CHARS, ANALYST_RESULT_SECTION_CHARS)]
        prompt = f"""
Section name: {section}

Extract fields from this section only. For experiments/appendix chunks, prioritize
copying quantitative table and figure results into reported_results with non-empty values.

Text:
{clipped}
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
                    "num_predict": ANALYST_NUM_PREDICT,
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

    def _extract_section(self, section: str, text: str) -> SectionExtraction:
        """Extract one section, chunking experiments/appendix so result tables are covered."""
        if section not in _RESULT_FOCUS_SECTIONS:
            return self._call_ollama_json(section, text[:ANALYST_SECTION_CHARS])

        windows = chunk_section_text(text)
        if not windows:
            return SectionExtraction()
        if len(windows) == 1:
            return self._call_ollama_json(section, windows[0])

        chunk_extractions: list[SectionExtraction] = []
        for idx, window in enumerate(windows, start=1):
            label = f"{section} (chunk {idx}/{len(windows)})"
            logger.info(
                "Analyst chunking section=%s chunk=%s/%s chars=%s",
                section,
                idx,
                len(windows),
                len(window),
            )
            try:
                chunk_extractions.append(self._call_ollama_json(label, window))
            except RuntimeError as exc:
                logger.warning("Section '%s' chunk %s failed: %s", section, idx, exc)
        if not chunk_extractions:
            raise RuntimeError(f"All chunks failed for section '{section}'.")
        return merge_chunk_extractions(chunk_extractions)

    def extract(
        self,
        sections: SectionTextMap,
        paper_title: str = "",
    ) -> ExtractionBundle:
        by_section: dict[str, SectionExtraction] = {}
        has_named_section_text = False

        for name in SECTION_NAMES:
            text = getattr(sections, name, "").strip()
            if not text:
                by_section[name] = SectionExtraction()
                continue
            has_named_section_text = True
            try:
                by_section[name] = self._extract_section(name, text)
            except RuntimeError as exc:
                # Partial extraction is better than total failure. Log the
                # section error and continue so the merged result and file
                # output are still produced for the sections that did succeed.
                logger.warning("Section '%s' extraction failed after retries: %s", name, exc)
                by_section[name] = SectionExtraction()

        if not has_named_section_text:
            fallback_text = (sections.full_text or "").strip()
            if not fallback_text:
                raise RuntimeError("No usable text available for any section.")
            try:
                by_section["abstract"] = self._call_ollama_json(
                    "abstract",
                    fallback_text[:ANALYST_SECTION_CHARS],
                )
            except RuntimeError as exc:
                logger.warning("Fallback abstract extraction failed: %s", exc)
                raise RuntimeError("No usable text available for any section.") from exc

        merged = finalize_merged_extraction(
            merge_section_extractions(by_section),
            paper_title=paper_title,
        )
        return ExtractionBundle(by_section=by_section, merged=merged)
