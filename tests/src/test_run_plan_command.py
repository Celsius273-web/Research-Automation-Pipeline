"""Tests for run-plan: load a saved plan without ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.main import build_parser
from src.plan_runner import paper_metadata_from_plan, run_saved_plan
from src.state import MetricsDocument, PaperMetadata


def test_run_plan_parser_requires_plan_repo_and_paper() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-plan",
            "--plan-path",
            "data/papers/synthetic_optimize/synthetic_optimize_plan.json",
            "--paper-id",
            "synthetic_optimize",
            "--repo-path",
            "benchmark/",
            "--non-interactive",
        ]
    )
    assert args.command == "run-plan"
    assert args.paper_id == "synthetic_optimize"
    assert args.non_interactive is True


def test_paper_metadata_from_plan_uses_stub_when_missing() -> None:
    paper = paper_metadata_from_plan({}, paper_id="synthetic_optimize")
    assert paper.paper_id == "synthetic_optimize"
    assert paper.title == "synthetic_optimize"
    assert paper.pdf_path == ""


def test_paper_metadata_from_plan_reads_embedded_paper() -> None:
    paper = paper_metadata_from_plan(
        {"paper": {"paper_id": "x", "title": "Hello", "pdf_path": "benchmark/papers/optimize.md"}},
        paper_id="synthetic_optimize",
    )
    assert paper == PaperMetadata(
        paper_id="synthetic_optimize",
        title="Hello",
        pdf_path="benchmark/papers/optimize.md",
    )


def test_checked_in_plans_load_as_planner_envelopes() -> None:
    from src.persistence import load_planner_envelope

    for paper_id in ("synthetic_optimize", "synthetic_graph"):
        path = Path(f"data/papers/{paper_id}/{paper_id}_plan.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        envelope = load_planner_envelope(payload)
        assert envelope.agent == "planner"
        assert envelope.payload.phases
        paper = paper_metadata_from_plan(payload, paper_id=paper_id)
        assert paper.paper_id == paper_id
        assert paper.title


def test_synthetic_graph_plan_documents_weighted_adj_shape() -> None:
    from src.persistence import load_planner_envelope

    path = Path("data/papers/synthetic_graph/synthetic_graph_plan.json")
    envelope = load_planner_envelope(json.loads(path.read_text(encoding="utf-8")))
    construct = next(phase for phase in envelope.payload.phases if phase.kind == "construct")
    spec = construct.specification
    example = spec["weighted_adj_example"]
    assert example == {"0": [[1, 2.0], [7, 12.0]], "1": [[0, 2.0], [2, 2.0]]}
    assert "not an edge list" in spec["weighted_adj_contract"].lower()
    dijkstra = spec["algorithm_specs"]["dijkstra"]
    assert "tuple" in dijkstra["output"].lower()
    assert "predecessor" in dijkstra["constraints"].lower()
    floyd = spec["algorithm_specs"]["floyd_warshall"]
    assert "neighbor, weight" in floyd["constraints"]
    assert "diagonal" in floyd["constraints"].lower()
    dfs = spec["algorithm_specs"]["dfs"]
    assert "traversal order" in dfs["output"].lower() or "visit order" in dfs["output"].lower()
    assert "list(set" in dfs["constraints"]
    kruskal = spec["algorithm_specs"]["kruskal"]
    assert "float" in kruskal["output"].lower()
    notes = " ".join(envelope.payload.engineer_notes).lower()
    assert "weighted_adj" in notes
    assert "predecessor" in notes
    assert "diagonal" in notes
    assert "visit order" in notes


def test_run_saved_plan_missing_plan_returns_error(tmp_path: Path) -> None:
    code = run_saved_plan(
        plan_path=str(tmp_path / "missing.json"),
        paper_id="demo",
        repo_path=str(tmp_path),
        non_interactive=True,
    )
    assert code == 1


def test_run_saved_plan_executes_engineer_then_reviewer(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "paper": {"paper_id": "demo", "title": "Demo", "pdf_path": ""},
                "plan_envelope": {
                    "schema_version": "2.0",
                    "agent": "planner",
                    "status": "ok",
                    "unknowns": [],
                    "warnings": [],
                    "payload": {
                        "plan_summary": "smoke",
                        "phases": [
                            {
                                "phase_id": "run",
                                "title": "Run",
                                "matrix": [
                                    {
                                        "name": "one",
                                        "run_command": "python -c 'print(1)'",
                                        "results_path": "out.json",
                                        "verify": ["exit_code:0"],
                                    }
                                ],
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "papers" / "demo" / "runs" / "R1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_status": "SUCCESS", "exit_code": 0, "metrics": []}),
        encoding="utf-8",
    )
    (run_dir / "engineer.log").write_text("ok\n", encoding="utf-8")

    import src.bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "PAPER_BUNDLES_DIR", tmp_path / "papers")

    fake_doc = MetricsDocument(run_status="SUCCESS", exit_code=0, metrics=[])
    with (
        patch("src.plan_runner.DockerExecutor"),
        patch("src.plan_runner.ExperimentRunner") as runner_cls,
    ):
        runner_cls.return_value.execute_paper.return_value = (fake_doc, run_dir)
        code = run_saved_plan(
            plan_path=str(plan_path),
            paper_id="demo",
            repo_path=str(repo),
            non_interactive=True,
        )

    assert code == 0
    assert (run_dir / "reviewer_report.json").exists()
    report = json.loads((run_dir / "reviewer_report.json").read_text(encoding="utf-8"))
    assert report["paper_id"] == "demo"
    assert "comparison_table" in report
