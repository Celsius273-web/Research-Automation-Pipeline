"""Tests for stripping venv patterns from Planner plans."""

from __future__ import annotations

from pathlib import Path

from src.agents.planner import (
    _annotate_results_summary_feasibility,
    _normalize_envelope_dict,
    _normalize_planner_payload,
)
from src.state import AgentEnvelope, PlannerPayload, PlanPhase
from src.tools.plan_venv_cleanup import (
    VENV_PLAN_REWRITE_NOTE,
    VENV_PLAN_REWRITE_WARNING,
    strip_venv_from_command,
)


def test_strip_venv_from_setup_chain() -> None:
    raw = (
        "python -m venv --clear .venv && .venv/bin/pip install -U pip && "
        ".venv/bin/pip install numpy torch"
    )
    assert strip_venv_from_command(raw) == "pip install numpy torch"


def test_strip_venv_bin_python() -> None:
    assert (
        strip_venv_from_command(".venv/bin/python exp/run_exp.py --seed 0")
        == "python exp/run_exp.py --seed 0"
    )


def test_normalize_planner_payload_rewrites_venv_setup() -> None:
    normalized = _normalize_planner_payload(
        {
            "plan_summary": "Plan",
            "objective": "Aim",
            "phases": [
                {
                    "phase_id": "setup",
                    "title": "Setup",
                    "run_template": (
                        "python -m venv --clear .venv && .venv/bin/pip install -U pip && "
                        ".venv/bin/pip install numpy"
                    ),
                    "matrix": [],
                },
                {
                    "phase_id": "smoke",
                    "title": "Smoke",
                    "run_template": "",
                    "matrix": [
                        {
                            "name": "row0",
                            "run_command": ".venv/bin/python exp/run_exp.py --seed 0",
                            "results_path": "results/p/smoke/0",
                        }
                    ],
                },
            ],
        }
    )
    assert normalized["phases"][0]["run_template"] == "pip install numpy"
    assert normalized["phases"][1]["matrix"][0]["run_command"] == "python exp/run_exp.py --seed 0"
    assert VENV_PLAN_REWRITE_NOTE in normalized["missing_context"]


def test_normalize_envelope_adds_warning() -> None:
    envelope = _normalize_envelope_dict(
        {
            "schema_version": "2.0",
            "agent": "planner",
            "status": "ok",
            "unknowns": [],
            "warnings": [],
            "payload": {
                "plan_summary": "Plan",
                "objective": "Aim",
                "phases": [
                    {
                        "phase_id": "setup",
                        "title": "Setup",
                        "run_template": "python -m venv .venv && .venv/bin/pip install x",
                        "matrix": [],
                    }
                ],
            },
        }
    )
    assert VENV_PLAN_REWRITE_WARNING in envelope["warnings"]


def test_annotate_results_summary_feasibility_when_undocumented(tmp_path: Path) -> None:
    envelope = AgentEnvelope[PlannerPayload](
        schema_version="2.0",
        agent="planner",
        status="ok",
        payload=PlannerPayload(
            plan_summary="Plan",
            objective="Aim",
            results_summary_path="results/demo/summary.json",
            phases=[PlanPhase(phase_id="setup", title="Setup")],
        ),
    )
    updated = _annotate_results_summary_feasibility(envelope, "demo", tmp_path)
    assert any("summary.json" in note for note in updated.payload.missing_context)
    assert any("feasibility" in warning for warning in updated.warnings)
