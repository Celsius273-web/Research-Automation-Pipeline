"""Interactive CLI checkpoints for human review of agent outputs."""

from __future__ import annotations

from src.state import (
    EngineerOutput,
    EngineerReviewRecord,
    ExecutionPlan,
    PlanReviewRecord,
    ReviewRecord,
    SectionExtraction,
)


class ReviewCancelledError(Exception):
    """Raised when the user aborts an interactive review checkpoint."""


_QUIT_CHOICES = frozenset({"q", "quit", "exit"})


def is_quit_choice(raw: str) -> bool:
    return raw.strip().lower() in _QUIT_CHOICES


def prompt_input(label: str, *, optional: bool = False) -> str:
    hints: list[str] = []
    if optional:
        hints.append("Enter to skip")
    hints.append("q to quit")
    hint = f" ({', '.join(hints)})"
    try:
        raw = input(f"{label}{hint}: ")
    except EOFError:
        raise ReviewCancelledError("Review cancelled.") from None
    if is_quit_choice(raw):
        raise ReviewCancelledError("Review cancelled.")
    return raw.strip()


def parse_list_input(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_dict_input(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if "=" in item:
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def run_cli_review(extraction: SectionExtraction) -> tuple[SectionExtraction, ReviewRecord]:
    print("\n--- Review checkpoint ---")
    print("Approve/edit each field. Enter to accept current value. Type q at any prompt to quit.")

    edited = extraction.model_copy(deep=True)
    fields = [
        "research_question",
        "methodology",
        "datasets_or_benchmarks",
        "variables",
        "hyperparameters",
        "evaluation_metrics",
        "notes",
    ]

    for field in fields:
        current = getattr(edited, field)
        print(f"\nField: {field}")
        print(f"Current: {current}")
        choice = prompt_input("[a]pprove / [e]dit / [r]eject field (default a)").lower()
        if choice in ("", "a"):
            continue
        if choice == "r":
            if isinstance(current, list):
                setattr(edited, field, [])
            elif isinstance(current, dict):
                setattr(edited, field, {})
            else:
                setattr(edited, field, "")
            continue
        if choice == "e":
            raw = prompt_input("New value")
            if field in ("datasets_or_benchmarks", "variables", "evaluation_metrics"):
                setattr(edited, field, parse_list_input(raw))
            elif field == "hyperparameters":
                setattr(edited, field, parse_dict_input(raw))
            else:
                setattr(edited, field, raw)

    print("\nReported results (read-only in v1):")
    for item in edited.reported_results:
        source = f" [{item.source}]" if item.source else ""
        benchmark = f"{item.benchmark} - " if item.benchmark else ""
        print(f"- {benchmark}{item.metric_name}: {item.value}{source}")

    finalize = prompt_input("Finalize extraction as approved? [Y/n/q]").lower()
    if finalize in ("", "y", "yes"):
        notes = prompt_input("Optional review notes", optional=True)
        return edited, ReviewRecord(status="approved", notes=notes)
    notes = prompt_input("Rejection notes", optional=True)
    return extraction, ReviewRecord(status="rejected", notes=notes)


def run_cli_plan_review(plan: ExecutionPlan) -> tuple[ExecutionPlan, PlanReviewRecord]:
    print("\n--- Planner review checkpoint ---")
    print(f"Summary: {plan.plan_summary or '(empty)'}")
    print(f"Objective: {plan.objective or '(empty)'}")
    print("Steps:")
    for idx, step in enumerate(plan.steps, start=1):
        print(f"  {idx}. {step.step_id} - {step.title}")

    choice = prompt_input("[a]pprove / [e]dit / [r]eject plan (default a)").lower()
    if choice in ("", "a"):
        notes = prompt_input("Optional review notes", optional=True)
        return plan, PlanReviewRecord(status="approved", notes=notes)
    if choice == "r":
        notes = prompt_input("Rejection notes", optional=True)
        return plan, PlanReviewRecord(status="rejected", notes=notes)

    edited = plan.model_copy(deep=True)
    edited_fields: list[str] = []
    summary = prompt_input("New plan_summary (enter to keep)", optional=True)
    if summary:
        edited.plan_summary = summary
        edited_fields.append("plan_summary")
    missing_context = prompt_input(
        "Add missing_context items (comma-separated, enter to keep)",
        optional=True,
    )
    if missing_context:
        edited.missing_context = parse_list_input(missing_context)
        edited_fields.append("missing_context")
    notes = prompt_input("Review notes", optional=True)
    return (
        edited,
        PlanReviewRecord(status="approved", notes=notes, edited_fields=edited_fields),
    )


def run_cli_engineer_review(output: EngineerOutput) -> EngineerReviewRecord:
    print("\n--- Engineer review checkpoint ---")
    print(f"Step id: {output.step_id or '(empty)'}")
    print(f"Patch count: {len(output.patches)}")
    print(f"Verification commands: {output.verification_commands}")
    choice = prompt_input("[a]pprove / [r]eject (default a)").lower()
    if choice in ("", "a"):
        notes = prompt_input("Optional review notes", optional=True)
        return EngineerReviewRecord(status="approved", notes=notes)
    notes = prompt_input("Rejection notes", optional=True)
    return EngineerReviewRecord(status="rejected", notes=notes)
