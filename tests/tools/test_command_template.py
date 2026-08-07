"""Tests for axes→matrix expansion and sequential run ids."""

from __future__ import annotations

from pathlib import Path

from src.state import PhaseRunSpec, PlanPhase
from src.tools.command_template import expand_phase_axes, phase_commands
from src.tools.run_ids import is_run_id, next_run_id


def test_next_run_id_sequences(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    assert next_run_id(runs) == "R1"
    (runs / "R1").mkdir()
    (runs / "R2").mkdir()
    (runs / "20260806T210552Z").mkdir()  # legacy ignored for numbering
    assert next_run_id(runs) == "R3"
    assert is_run_id("R12")
    assert not is_run_id("20260806T210552Z")


def test_expand_phase_axes_full_product() -> None:
    phase = PlanPhase(
        phase_id="synthetic",
        title="Synthetic",
        variables=["benchmark", "algorithm", "seed"],
        axes={
            "benchmark": ["lsq", "sim"],
            "algorithm": ["be-cbo", "cei"],
            "seed": [0, 1],
        },
        run_template=(
            "python exp/run_exp.py --fun FUN_NAME --algo ALGO_NAME "
            "--reg-type REGRESSOR --cls-type CLASSIFIER --log-path LOG_PATH"
        ),
        matrix=[
            PhaseRunSpec(
                name="example",
                variables={"benchmark": "lsq", "algorithm": "be-cbo", "seed": 0},
                run_command=(
                    "python exp/run_exp.py --fun lsq --algo be-cbo "
                    "--reg-type gp --cls-type de --log-path results/p/synthetic/lsq/gp_de/be-cbo/0 --seed 0"
                ),
                results_path="results/p/synthetic/lsq/gp_de/be-cbo/0",
            )
        ],
        results_path="results/p/synthetic",
    )
    rows = expand_phase_axes(phase, paper_id="p")
    assert len(rows) == 8  # 2×2×2
    commands = phase_commands(phase, paper_id="p", expand_axes=True)
    assert len(commands) == 8
    joined = "\n".join(cmd for cmd, _ in commands)
    assert "--fun sim" in joined
    assert "--algo cei" in joined
    assert "--seed 1" in joined
    assert "results/p/synthetic/sim/gp_de/cei/1" in joined

    # Default (Option B): matrix examples only — no axes expansion.
    default_commands = phase_commands(phase, paper_id="p")
    assert len(default_commands) == 1


def test_phase_commands_without_axes_keeps_matrix() -> None:
    phase = PlanPhase(
        phase_id="smoke",
        title="Smoke",
        matrix=[
            PhaseRunSpec(
                name="one",
                variables={"benchmark": "lsq"},
                run_command="python run.py --fun {benchmark}",
            )
        ],
    )
    commands = phase_commands(phase, paper_id="p")
    assert len(commands) == 1
    assert commands[0][0] == "python run.py --fun lsq"
