"""Convert planner run logs / debug JSON into a readable report, optionally vs saved plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.planner import _extraction_to_analyst_dict
from src.agents.planner_debug import (
    PlannerDebugTrace,
    load_saved_plan,
    parse_planner_log_text,
    refresh_planner_debug_with_saved_plan,
    trace_from_debug_json,
    write_planner_debug_files,
)
from src.bundle import PaperBundle
from src.config import PAPER_BUNDLES_DIR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Organize planner debug output into a readable markdown/JSON report. "
            "Use after a plan run, or to convert an old terminal log dump."
        )
    )
    parser.add_argument(
        "--paper-id",
        help="Paper bundle id under data/papers/ (uses existing planner_debug.json if present)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Raw terminal/log file from a planner run to convert",
    )
    parser.add_argument(
        "--debug-json",
        type=Path,
        help="Existing planner_debug.json to re-render",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        help="Saved plan JSON to compare against (defaults to bundle plan if --paper-id set)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for planner_debug.md / planner_debug.json (defaults to paper bundle or cwd)",
    )
    parser.add_argument(
        "--from-extraction",
        action="store_true",
        help=(
            "Build a diagnostic report from the saved extraction + plan without an LLM run "
            "(shows what the planner would receive vs what is in the plan JSON)."
        ),
    )
    return parser


def _diagnose_from_extraction(paper_id: str, output_dir: Path, plan_path: Path) -> tuple[Path, Path]:
    bundle = PaperBundle(paper_id)
    extraction = bundle.get_extraction()
    if extraction is None:
        raise SystemExit(f"No extraction found for {paper_id}")

    sections = {
        name: _extraction_to_analyst_dict(section)
        for name, section in extraction.by_section.items()
    }
    received = {
        "paper": bundle.get_metadata() or {"paper_id": paper_id},
        "analyst_output": _extraction_to_analyst_dict(extraction.merged),
        "extraction_sections": sections,
        "repo_context": bundle.get_repo_info().model_dump(),
        "repo_setup_guide": bundle.get_setup_guide(),
        "hyperparameter_reference": bundle.get_hyperparameter_reference(),
    }
    saved_plan = load_saved_plan(plan_path) if plan_path.exists() else None
    final_output = None
    if saved_plan:
        if "plan_envelope" in saved_plan:
            final_output = saved_plan["plan_envelope"]
        elif "execution_plan" in saved_plan:
            final_output = saved_plan["execution_plan"]

    trace = PlannerDebugTrace(
        paper_id=paper_id,
        model="(no LLM run — diagnostic from extraction + saved plan)",
        received_context=received,
        system_prompt="(not captured — re-run plan to record live prompts)",
        final_output=final_output,
    )
    return write_planner_debug_files(trace, output_dir, saved_plan=saved_plan)


def main() -> int:
    args = _build_parser().parse_args()
    if not args.paper_id and not args.log_file and not args.debug_json:
        _build_parser().error("Provide --paper-id, --log-file, and/or --debug-json")

    output_dir = args.output_dir
    saved_plan = None
    if args.plan_json:
        saved_plan = load_saved_plan(args.plan_json)

    if args.paper_id and args.from_extraction:
        bundle_dir = PAPER_BUNDLES_DIR / args.paper_id
        output_dir = output_dir or bundle_dir
        plan_path = args.plan_json or (bundle_dir / f"{args.paper_id}_plan.json")
        json_path, md_path = _diagnose_from_extraction(args.paper_id, output_dir, plan_path)
        print(f"Wrote: {md_path}")
        print(f"Wrote: {json_path}")
        return 0

    if args.paper_id:
        bundle_dir = PAPER_BUNDLES_DIR / args.paper_id
        output_dir = output_dir or bundle_dir
        plan_path = args.plan_json or (bundle_dir / f"{args.paper_id}_plan.json")
        if plan_path.exists() and saved_plan is None:
            saved_plan = load_saved_plan(plan_path)
        debug_json = bundle_dir / "planner_debug.json"
        if args.log_file is None and args.debug_json is None and debug_json.exists():
            paths = refresh_planner_debug_with_saved_plan(bundle_dir, plan_path)
            if paths:
                print(f"Updated: {paths[1]}")
                print(f"Updated: {paths[0]}")
                return 0

    if args.log_file:
        log_text = args.log_file.read_text(encoding="utf-8")
        paper_id = args.paper_id or args.log_file.stem
        trace = parse_planner_log_text(log_text, paper_id=paper_id)
        output_dir = output_dir or Path.cwd()
        json_path, md_path = write_planner_debug_files(trace, output_dir, saved_plan=saved_plan)
        print(f"Wrote: {md_path}")
        print(f"Wrote: {json_path}")
        return 0

    if args.debug_json:
        payload = load_saved_plan(args.debug_json)
        trace = trace_from_debug_json(payload)
        output_dir = output_dir or args.debug_json.parent
        json_path, md_path = write_planner_debug_files(trace, output_dir, saved_plan=saved_plan)
        print(f"Wrote: {md_path}")
        print(f"Wrote: {json_path}")
        return 0

    print(
        "No planner_debug.json found to refresh. "
        "Run plan first, or pass --log-file / --from-extraction."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
