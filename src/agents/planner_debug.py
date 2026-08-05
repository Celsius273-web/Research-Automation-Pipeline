"""Structured planner debug traces: record I/O, write readable reports, parse logs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PlannerAttemptRecord:
    attempt_index: int
    reminder: str
    user_prompt: str
    system_prompt_char_count: int
    raw_response: str = ""
    parsed: dict[str, Any] | None = None
    error: str | None = None
    outcome: str = "pending"


@dataclass
class PlannerDebugTrace:
    paper_id: str
    model: str
    received_context: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    attempts: list[PlannerAttemptRecord] = field(default_factory=list)
    final_output: dict[str, Any] | None = None
    final_error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_attempt(
        self,
        *,
        reminder: str,
        user_prompt: str,
        system_prompt: str,
    ) -> PlannerAttemptRecord:
        record = PlannerAttemptRecord(
            attempt_index=len(self.attempts) + 1,
            reminder=reminder,
            user_prompt=user_prompt,
            system_prompt_char_count=len(system_prompt),
        )
        if not self.system_prompt:
            self.system_prompt = system_prompt
        self.attempts.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "paper_id": self.paper_id,
            "model": self.model,
            "received_context": self.received_context,
            "system_prompt": self.system_prompt,
            "attempts": [
                {
                    "attempt_index": a.attempt_index,
                    "reminder": a.reminder,
                    "user_prompt": a.user_prompt,
                    "system_prompt_char_count": a.system_prompt_char_count,
                    "raw_response": a.raw_response,
                    "parsed": a.parsed,
                    "error": a.error,
                    "outcome": a.outcome,
                }
                for a in self.attempts
            ],
            "final_output": self.final_output,
            "final_error": self.final_error,
        }


def extract_plan_body(saved_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize saved plan JSON to a comparable plan body."""
    if "plan_envelope" in saved_plan and isinstance(saved_plan["plan_envelope"], dict):
        return saved_plan["plan_envelope"]
    if "execution_plan" in saved_plan and isinstance(saved_plan["execution_plan"], dict):
        return saved_plan["execution_plan"]
    if saved_plan.get("agent") == "planner" or "payload" in saved_plan:
        return saved_plan
    return saved_plan


def _core_fields(plan: dict[str, Any]) -> tuple[str, str, list[Any], list[Any]]:
    if "payload" in plan and isinstance(plan["payload"], dict):
        payload = plan["payload"]
        if "plan_summary" in payload or "phases" in payload or "steps" in payload:
            phases = list(payload.get("phases") or [])
            if not phases and (payload.get("steps") or payload.get("experiment_matrix")):
                phases = list(payload.get("steps") or [])
            matrix_rows: list[Any] = []
            for phase in phases:
                if isinstance(phase, dict):
                    matrix_rows.extend(list(phase.get("matrix") or []))
            if not matrix_rows:
                matrix_rows = list(payload.get("experiment_matrix") or [])
            return (
                str(payload.get("plan_summary") or ""),
                str(payload.get("objective") or ""),
                phases,
                matrix_rows,
            )
        core = payload.get("core") or {}
        ext = payload.get("extensions") or {}
        return (
            str(core.get("plan_summary") or ""),
            str(core.get("objective") or ""),
            list(core.get("phases") or core.get("steps") or []),
            list(ext.get("experiment_matrix") or []),
        )
    return (
        str(plan.get("plan_summary") or ""),
        str(plan.get("objective") or ""),
        list(plan.get("phases") or plan.get("steps") or []),
        list(plan.get("experiment_matrix") or []),
    )


def compare_planner_output_to_saved_plan(
    final_output: dict[str, Any] | None,
    saved_plan: dict[str, Any] | None,
    received_context: dict[str, Any] | None = None,
) -> list[str]:
    """Human-readable comparison notes between LLM final output and saved plan JSON."""
    notes: list[str] = []
    if final_output is None:
        notes.append("No final planner LLM output was recorded.")
        return notes
    if not saved_plan:
        notes.append("No saved plan JSON provided for comparison.")
        return notes

    saved_body = extract_plan_body(saved_plan)
    out_summary, out_objective, out_steps, out_exps = _core_fields(final_output)
    save_summary, save_objective, save_steps, save_exps = _core_fields(saved_body)

    if out_summary != save_summary:
        notes.append(
            f"plan_summary differs: llm={out_summary!r} vs saved={save_summary!r}"
        )
    else:
        notes.append("plan_summary matches saved plan.")

    if out_objective != save_objective:
        notes.append(
            f"objective differs: llm={out_objective!r} vs saved={save_objective!r}"
        )
    else:
        notes.append("objective matches saved plan.")

    notes.append(f"phases: llm={len(out_steps)} saved={len(save_steps)}")
    notes.append(f"matrix_rows: llm={len(out_exps)} saved={len(save_exps)}")

    if not save_summary and not save_objective and not save_steps:
        notes.append("WARNING: saved plan is empty (no summary, objective, or phases).")

    analyst = (received_context or {}).get("analyst_output") or {}
    rq = str(analyst.get("research_question") or "")
    if rq and not rq.startswith("unknown:"):
        unknowns = final_output.get("unknowns") or []
        for item in unknowns:
            field_name = str(item.get("field", "")).lower() if isinstance(item, dict) else ""
            if "research_question" in field_name or field_name in {"aim", "objective"}:
                notes.append(
                    "ISSUE: analyst_output had a research_question but the plan marked "
                    f"aim/RQ unknown ({field_name})."
                )
                break
        if not out_objective.strip():
            notes.append("ISSUE: analyst_output had research_question but objective is empty.")

    datasets = analyst.get("datasets_or_benchmarks") or analyst.get("datasets") or []
    if isinstance(datasets, list) and datasets and not any(
        str(d).startswith("unknown:") for d in datasets
    ):
        plan_text = json.dumps(final_output).lower()
        missing = [
            str(d)
            for d in datasets
            if str(d).split()[0].lower().strip("(),") not in plan_text
            and str(d).lower() not in plan_text
        ]
        if missing:
            notes.append(
                "ISSUE: analyst datasets mostly absent from plan output "
                f"(sample missing: {missing[:5]})"
            )

    return notes


def _md_fence(language: str, body: str) -> str:
    return f"```{language}\n{body.rstrip()}\n```\n"


def _research_question_is_missing(analyst: dict[str, Any]) -> bool:
    rq = str(analyst.get("research_question") or "").strip()
    return not rq or rq.startswith("unknown:")


def _summarize_analyst(analyst: dict[str, Any]) -> list[str]:
    return [
        f"- research_question empty/unknown: {_research_question_is_missing(analyst)}",
        f"- methodology chars: {len(str(analyst.get('methodology') or ''))}",
        f"- datasets_or_benchmarks count: {len(analyst.get('datasets_or_benchmarks') or analyst.get('datasets') or [])}",
        f"- variables count: {len(analyst.get('variables') or [])}",
        f"- hyperparameters count: {len(analyst.get('hyperparameters') or {})}",
        f"- evaluation_metrics count: {len(analyst.get('evaluation_metrics') or [])}",
        f"- reported_results count: {len(analyst.get('reported_results') or [])}",
        f"- notes chars: {len(str(analyst.get('notes') or ''))}",
        f"- keys: {sorted(analyst.keys())}",
    ]
    #needs to be updated to handle the new analyst output structure + a way to check if the analyst output is correct

def render_planner_debug_markdown(
    trace: PlannerDebugTrace,
    saved_plan: dict[str, Any] | None = None,
) -> str:
    """Render a readable markdown report for one planner run."""
    lines: list[str] = [
        f"# Planner debug: {trace.paper_id}",
        "",
        f"- created_at: `{trace.created_at}`",
        f"- model: `{trace.model}`",
        f"- attempts: {len(trace.attempts)}",
        f"- final_error: `{trace.final_error}`" if trace.final_error else "- final_error: none",
        "",
        "## Input the planner received",
        "",
    ]

    analyst = trace.received_context.get("analyst_output") or {}
    lines.append("### analyst_output summary")
    lines.extend(_summarize_analyst(analyst if isinstance(analyst, dict) else {}))
    lines.append("")
    lines.append("### analyst_output (full)")
    lines.append(_md_fence("json", json.dumps(analyst, indent=2)))

    for key in ("repo_context", "paper_context", "flags"):
        lines.append(f"### {key}")
        value = trace.received_context.get(key)
        if isinstance(value, (dict, list)):
            lines.append(_md_fence("json", json.dumps(value, indent=2)))
        else:
            lines.append(_md_fence("text", str(value or "")))
        if key == "repo_context" and isinstance(value, dict):
            commands = value.get("example_commands") or []
            lines.append(
                f"- example_commands count: {len(commands) if isinstance(commands, list) else 0}"
            )

    lines.append("## System prompt")
    lines.append(f"- char_count: {len(trace.system_prompt)}")
    lines.append(_md_fence("text", trace.system_prompt or "(empty)"))

    lines.append("## LLM attempts")
    lines.append("")
    for attempt in trace.attempts:
        lines.append(f"### Attempt {attempt.attempt_index}")
        lines.append(f"- reminder: `{attempt.reminder}`")
        lines.append(f"- outcome: `{attempt.outcome}`")
        lines.append(f"- error: `{attempt.error}`" if attempt.error else "- error: none")
        lines.append(f"- user_prompt chars: {len(attempt.user_prompt)}")
        lines.append("")
        lines.append("#### User prompt (what the model saw this call)")
        lines.append(_md_fence("text", attempt.user_prompt or "(empty)"))
        lines.append("#### Raw model response")
        lines.append(_md_fence("text", attempt.raw_response or "(empty)"))
        lines.append("#### Parsed JSON")
        if attempt.parsed is None:
            lines.append("_not parsed_")
            lines.append("")
        else:
            lines.append(_md_fence("json", json.dumps(attempt.parsed, indent=2)))

    lines.append("## Final accepted planner output")
    if trace.final_output is None:
        lines.append("_none_")
        lines.append("")
    else:
        lines.append(_md_fence("json", json.dumps(trace.final_output, indent=2)))

    lines.append("## Comparison to saved plan JSON")
    notes = compare_planner_output_to_saved_plan(
        trace.final_output,
        saved_plan,
        received_context=trace.received_context,
    )
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    if saved_plan is not None:
        lines.append("### Saved plan JSON (normalized body)")
        lines.append(_md_fence("json", json.dumps(extract_plan_body(saved_plan), indent=2)))

    return "\n".join(lines).rstrip() + "\n"


def write_planner_debug_files(
    trace: PlannerDebugTrace,
    output_dir: Path,
    saved_plan: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write planner_debug.json and planner_debug.md under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "planner_debug.json"
    md_path = output_dir / "planner_debug.md"
    payload = trace.to_dict()
    if saved_plan is not None:
        payload["saved_plan_comparison_notes"] = compare_planner_output_to_saved_plan(
            trace.final_output,
            saved_plan,
            received_context=trace.received_context,
        )
        payload["saved_plan"] = saved_plan
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        render_planner_debug_markdown(trace, saved_plan=saved_plan),
        encoding="utf-8",
    )
    return json_path, md_path


def _extract_balanced_json(text: str, start: int) -> tuple[dict[str, Any] | None, int]:
    """Parse a JSON object starting at text[start] (must be '{')."""
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : idx + 1]
                try:
                    return json.loads(chunk), idx + 1
                except json.JSONDecodeError:
                    return None, idx + 1
    return None, start


def parse_planner_log_text(log_text: str, paper_id: str = "unknown") -> PlannerDebugTrace:
    """Convert raw terminal/log text from a planner run into a structured trace."""
    trace = PlannerDebugTrace(paper_id=paper_id, model="unknown")
    current: PlannerAttemptRecord | None = None

    payload_marker = "Planner prompt payload: "
    raw_marker = "Planner raw response: "

    idx = 0
    while idx < len(log_text):
        payload_pos = log_text.find(payload_marker, idx)
        raw_pos = log_text.find(raw_marker, idx)
        candidates = [p for p in (payload_pos, raw_pos) if p >= 0]
        if not candidates:
            break
        next_pos = min(candidates)

        if next_pos == payload_pos:
            start = payload_pos + len(payload_marker)
            parsed, end = _extract_balanced_json(log_text, start)
            reminder = "none"
            user_prompt = ""
            system_prompt = ""
            model = trace.model
            if isinstance(parsed, dict):
                model = str(parsed.get("model") or model)
                messages = parsed.get("messages") or []
                if isinstance(messages, list):
                    for message in messages:
                        if not isinstance(message, dict):
                            continue
                        if message.get("role") == "system":
                            system_prompt = str(message.get("content") or "")
                        elif message.get("role") == "user":
                            user_prompt = str(message.get("content") or "")
                if "AIM_GROUNDING" in user_prompt or "do not mark it unknown" in user_prompt:
                    reminder = "aim"
                elif "previous response was empty or invalid" in user_prompt:
                    reminder = "strict"
                context_match = re.search(
                    r"Context JSON:\s*(\{.*)\Z",
                    user_prompt,
                    flags=re.DOTALL,
                )
                if context_match and not trace.received_context:
                    try:
                        trace.received_context = json.loads(context_match.group(1))
                    except json.JSONDecodeError:
                        pass
            trace.model = model
            current = trace.add_attempt(
                reminder=reminder,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
            current.outcome = "requested"
            idx = end
            continue

        # raw response branch
        start = raw_pos + len(raw_marker)
        end_candidates = [
            p
            for p in (
                log_text.find(payload_marker, start),
                log_text.find("PLANNER_", start),
                log_text.find("\nINFO", start),
                log_text.find("\nWARNING", start),
            )
            if p >= 0
        ]
        end = min(end_candidates) if end_candidates else len(log_text)
        raw_response = log_text[start:end].strip()
        if current is None:
            current = trace.add_attempt(
                reminder="none",
                user_prompt="",
                system_prompt=trace.system_prompt,
            )
        current.raw_response = raw_response
        try:
            current.parsed = json.loads(_strip_fences(raw_response))
            current.outcome = "parsed"
            trace.final_output = current.parsed
        except json.JSONDecodeError as exc:
            current.error = str(exc)
            current.outcome = "parse_error"
        idx = end

    return trace


def _strip_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = raw.rstrip("`").strip()
    return raw


def load_saved_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trace_from_debug_json(payload: dict[str, Any]) -> PlannerDebugTrace:
    """Rebuild a PlannerDebugTrace from planner_debug.json."""
    trace = PlannerDebugTrace(
        paper_id=str(payload.get("paper_id") or "unknown"),
        model=str(payload.get("model") or "unknown"),
        received_context=dict(payload.get("received_context") or {}),
        system_prompt=str(payload.get("system_prompt") or ""),
        final_output=payload.get("final_output"),
        final_error=payload.get("final_error"),
        created_at=str(payload.get("created_at") or datetime.now(timezone.utc).isoformat()),
    )
    for item in payload.get("attempts") or []:
        if not isinstance(item, dict):
            continue
        trace.attempts.append(
            PlannerAttemptRecord(
                attempt_index=int(item.get("attempt_index") or len(trace.attempts) + 1),
                reminder=str(item.get("reminder") or "none"),
                user_prompt=str(item.get("user_prompt") or ""),
                system_prompt_char_count=int(item.get("system_prompt_char_count") or 0),
                raw_response=str(item.get("raw_response") or ""),
                parsed=item.get("parsed"),
                error=item.get("error"),
                outcome=str(item.get("outcome") or "pending"),
            )
        )
    return trace


def refresh_planner_debug_with_saved_plan(
    output_dir: Path,
    saved_plan_path: Path | None = None,
) -> tuple[Path, Path] | None:
    """Re-render planner_debug.md/.json using the saved plan for comparison."""
    debug_json = output_dir / "planner_debug.json"
    if not debug_json.exists():
        return None
    payload = json.loads(debug_json.read_text(encoding="utf-8"))
    trace = trace_from_debug_json(payload)
    plan_path = saved_plan_path or (output_dir / f"{trace.paper_id}_plan.json")
    saved_plan = load_saved_plan(plan_path) if plan_path.exists() else None
    return write_planner_debug_files(trace, output_dir, saved_plan=saved_plan)
