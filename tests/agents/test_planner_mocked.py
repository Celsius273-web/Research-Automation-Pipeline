from __future__ import annotations

import json
from pathlib import Path
from urllib import error

import src.agents.planner as planner_module
from src.agents.planner import (
    PaperPlanner,
    _apply_plan_verification,
    _apply_runnable_repo_contract,
    _collect_grounding_issues,
    _ensure_phases_from_builder,
    _extraction_to_analyst_dict,
    _merge_planned_actions,
    _normalize_planner_payload,
    _plan_misses_aim,
)
from src.planner_input import build_unified_planner_input
from src.state import (
    AgentEnvelope,
    PaperMetadata,
    PlannerInputContext,
    PlannerPayload,
    PlanPhase,
    PhaseRunSpec,
    PlanStep,
    ReportedResult,
    SectionExtraction,
    UnknownItem,
    project_phases_to_steps,
)
from src.tools.phase_builder import build_plan_phases


def _sample_context(tmp_path: Path | None = None) -> PlannerInputContext:
    bundle_path = str(tmp_path) if tmp_path is not None else None
    return PlannerInputContext(
        paper=PaperMetadata(
            paper_id="paper_1",
            title="Example Paper",
            pdf_path="data/papers/example.pdf",
        ),
        approved_extraction=SectionExtraction(
            research_question="How to optimize black-box functions?",
            methodology="Bayesian optimization",
            datasets_or_benchmarks=["BBOB"],
            variables=["acquisition function"],
            hyperparameters={"batch_size": "16", "hidden_layers": "[1, 2, 3]"},
            evaluation_metrics=["best regret"],
            reported_results=[
                ReportedResult(
                    benchmark="BBOB",
                    metric_name="best regret",
                    value="0.12",
                    source="Table 1",
                )
            ],
            notes="CPU only experiments.",
        ),
        runtime_constraints={"hardware": "cpu_only"},
        repo_context={
            "repo_url": "https://example.com/repo",
            "example_commands": ["python run.py --config baseline"],
            "build_system": "pip install -r requirements.txt",
            "has_code": True,
            "language": "python",
        },
        repo_setup_guide="pip install -r requirements.txt",
        hyperparameter_reference="Table 1: batch_size=16",
        paper_bundle_path=bundle_path,
    )


def _ok_envelope_content() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "agent": "planner",
        "status": "ok",
        "unknowns": [],
        "warnings": [],
        "payload": {
            "plan_summary": "Reproduce BO benchmarks from the paper aim.",
            "domain": "optimization",
            "objective": "How to optimize black-box functions under a limited budget.",
            "phases": [
                {
                    "phase_id": "setup",
                    "title": "Setup",
                    "goal": "Install deps",
                    "depends_on": [],
                    "variables": [],
                    "axes": {},
                    "run_template": "pip install -r requirements.txt",
                    "matrix": [],
                    "planned_actions": "Install then confirm entrypoint.",
                    "results_path": "results/paper_1",
                },
                {
                    "phase_id": "smoke",
                    "title": "Smoke",
                    "goal": "One baseline run",
                    "depends_on": ["setup"],
                    "variables": ["benchmark", "seed"],
                    "axes": {"benchmark": ["BBOB"], "seed": [0]},
                    "run_template": "python run.py --config baseline",
                    "matrix": [
                        {
                            "name": "bbob_smoke",
                            "variables": {"benchmark": "BBOB", "seed": 0},
                            "run_command": "python run.py --config baseline",
                            "code_refs": ["README.md", "run.py"],
                            "verify": ["exists:results/paper_1/summary.json"],
                            "results_path": "results/paper_1/smoke",
                            "metrics": ["best regret"],
                            "source": "repo",
                        }
                    ],
                    "planned_actions": "Run baseline once and verify summary path.",
                    "results_path": "results/paper_1/smoke",
                },
            ],
            "assumptions": [],
            "constraints": [],
            "missing_context": [],
            "verification_checks": [],
            "risks": [],
            "organization": ["setup → smoke"],
            "execution": ["Reuse python run.py for each benchmark."],
            "repo_usage": ["Use example command python run.py --config baseline"],
            "engineer_notes": ["CPU-only constraints apply."],
            "results_summary_path": "results/paper_1/summary.json",
        },
    }


def _bad_aim_envelope_content() -> dict[str, object]:
    content = _ok_envelope_content()
    content["status"] = "partial"
    content["unknowns"] = [
        {"field": "research_question", "reason": "missing", "severity": "high"}
    ]
    content["payload"]["objective"] = ""
    content["payload"]["plan_summary"] = ""
    return content


def test_planner_ollama_payload_uses_schema_and_writes_debug(monkeypatch, tmp_path: Path) -> None:
    planner = PaperPlanner()
    context = _sample_context(tmp_path)
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": json.dumps(_ok_envelope_content())}}).encode()

    def fake_urlopen(req, timeout=180):
        _ = timeout
        body = json.loads(req.data.decode("utf-8"))
        captured["format"] = body["format"]
        captured["num_predict"] = body["options"]["num_predict"]
        return _Response()

    monkeypatch.setattr(planner_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)

    out = planner._call_ollama_json(context)
    assert isinstance(out, AgentEnvelope)
    assert out.payload.phases
    assert captured["format"]["type"] == "object"
    assert set(captured["format"]["required"]) >= {
        "schema_version",
        "agent",
        "status",
        "payload",
    }
    assert captured["num_predict"] == 4096
    assert out.payload.results_summary_path == "results/paper_1/summary.json"
    assert (tmp_path / "planner_debug.md").exists()


def test_planner_soft_retries_when_aim_marked_unknown(monkeypatch, tmp_path: Path) -> None:
    planner = PaperPlanner()
    context = _sample_context(tmp_path)
    attempts = {"count": 0}
    user_prompts: list[str] = []

    class _Response:
        def __init__(self, content: dict[str, object]):
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": json.dumps(self._content)}}).encode("utf-8")

    def fake_urlopen(req, timeout=180):
        _ = timeout
        attempts["count"] += 1
        body = json.loads(req.data.decode("utf-8"))
        user_prompts.append(body["messages"][1]["content"])
        if attempts["count"] == 1:
            return _Response(_bad_aim_envelope_content())
        return _Response(_ok_envelope_content())

    monkeypatch.setattr(planner_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)

    out = planner._call_ollama_json(context)
    assert attempts["count"] == 2
    assert isinstance(out, AgentEnvelope)
    assert out.payload.objective
    assert "do not mark it unknown" in user_prompts[1]
    assert not _plan_misses_aim(
        context.approved_extraction.research_question,
        context.approved_extraction.methodology,
        out,
    )


def test_planner_build_plan_with_mocked_ollama_call(monkeypatch) -> None:
    planner = PaperPlanner()
    context = _sample_context()

    def fake_call(ctx: PlannerInputContext) -> AgentEnvelope[PlannerPayload]:
        assert ctx.paper.paper_id == "paper_1"
        return AgentEnvelope[PlannerPayload](
            schema_version="2.0",
            agent="planner",
            status="ok",
            unknowns=[],
            warnings=[],
            payload=PlannerPayload(
                plan_summary="Run baseline then BO variants.",
                domain="optimization",
                objective="Reproduce reported regret curves",
                phases=[PlanPhase(phase_id="setup", title="Set up environment")],
            ),
        )

    monkeypatch.setattr(planner, "_call_ollama_json", fake_call)
    out = planner.build_plan(context)
    assert out.payload.plan_summary
    assert out.payload.phases[0].phase_id == "setup"


def test_call_ollama_json_retries_after_urlerror(monkeypatch, tmp_path: Path) -> None:
    planner = PaperPlanner()
    context = _sample_context(tmp_path)
    attempts = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            payload = {"message": {"content": json.dumps(_ok_envelope_content())}}
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(_req, timeout=180):
        _ = timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise error.URLError("temporary network issue")
        return _Response()

    monkeypatch.setattr(planner_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)

    out = planner._call_ollama_json(context)
    assert out.payload.plan_summary == "Reproduce BO benchmarks from the paper aim."
    assert out.payload.phases[0].phase_id == "setup"
    assert attempts["count"] == 2


def test_planner_retry_names_missing_envelope_keys(monkeypatch, tmp_path: Path) -> None:
    planner = PaperPlanner()
    context = _sample_context(tmp_path)
    prompts: list[str] = []

    class _Response:
        def __init__(self, content: dict[str, object]):
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": json.dumps(self._content)}}).encode()

    def fake_urlopen(req, timeout=180):
        _ = timeout
        body = json.loads(req.data.decode("utf-8"))
        prompts.append(body["messages"][1]["content"])
        if len(prompts) == 1:
            return _Response({"summary": {"title": "Wrong response shape"}})
        return _Response(_ok_envelope_content())

    monkeypatch.setattr(planner_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)

    out = planner._call_ollama_json(context)

    assert out.payload.phases
    assert len(prompts) == 2
    assert "Your returned top-level keys were ['summary']" in prompts[1]
    assert "agent: Field required" in prompts[1]


def test_planner_parse_flattened_envelope() -> None:
    envelope = AgentEnvelope[PlannerPayload](
        schema_version="2.0",
        agent="planner",
        status="ok",
        unknowns=[],
        warnings=[],
        payload=PlannerPayload(
            plan_summary="Execute benchmark experiments",
            domain="bayesian_optimization",
            objective="Reproduce results",
            phases=[
                PlanPhase(
                    phase_id="smoke",
                    title="Run baseline",
                    goal="Execute baseline experiment",
                    run_template="python run.py",
                    matrix=[
                        PhaseRunSpec(
                            name="r1",
                            run_command="python run.py",
                            code_refs=["run.py"],
                            verify=["exists:results/paper_1/summary.json"],
                            results_path="results/paper_1/summary.json",
                        )
                    ],
                    results_path="results/paper_1/summary.json",
                )
            ],
            assumptions=["CPU-only"],
            constraints=["No external APIs"],
            organization=["Group by benchmark"],
            execution=["Reuse setup env"],
            repo_usage=["pip install -r requirements.txt"],
            engineer_notes=["Watch dependency pins"],
            results_summary_path="results/paper_1/summary.json",
        ),
    )

    assert envelope.payload.plan_summary == "Execute benchmark experiments"
    assert len(envelope.payload.phases) == 1
    assert envelope.payload.phases[0].phase_id == "smoke"
    steps = project_phases_to_steps(envelope.payload)
    assert steps[0].run_command == "python run.py"


def test_planner_normalize_nested_core_extensions_payload() -> None:
    payload = {
        "core": {
            "plan_summary": "Plan",
            "objective": "Aim",
            "phases": [
                {
                    "phase_id": "setup",
                    "title": "Setup",
                    "matrix": [],
                }
            ],
        },
        "extensions": {"engineer_notes": ["note"]},
    }
    normalized = _normalize_planner_payload(payload)
    assert normalized["plan_summary"] == "Plan"
    assert normalized["phases"][0]["phase_id"] == "setup"
    assert normalized["engineer_notes"] == ["note"]


def test_normalize_legacy_steps_matrix_into_phases() -> None:
    normalized = _normalize_planner_payload(
        {
            "plan_summary": "Legacy",
            "objective": "Aim",
            "steps": [
                {
                    "step_id": "s1",
                    "title": "Run",
                    "goal": "Go",
                    "run_command": "python run.py --config baseline",
                    "results_path": "results/paper_1/summary.json",
                }
            ],
            "experiment_matrix": [
                {
                    "name": "bbob",
                    "benchmarks": ["BBOB"],
                    "execution_pattern": "python run.py --config baseline",
                    "hyperparameters": {"batch_size": "16"},
                }
            ],
        }
    )
    assert normalized["phases"]
    assert normalized["phases"][0]["matrix"][0]["run_command"] == "python run.py --config baseline"
    assert "steps" not in normalized
    assert "experiment_matrix" not in normalized


def test_grounding_requires_phases_when_runnable() -> None:
    context = _sample_context()
    unified = build_unified_planner_input(context)
    empty = AgentEnvelope[PlannerPayload].model_validate(
        {
            "schema_version": "2.0",
            "agent": "planner",
            "status": "ok",
            "unknowns": [],
            "warnings": [],
            "payload": {
                "plan_summary": "Plan",
                "domain": "optimization",
                "objective": "Aim",
                "phases": [],
                "results_summary_path": "results/paper_1/summary.json",
            },
        }
    )
    issues = _collect_grounding_issues(empty, unified)
    assert any("phases is empty" in issue for issue in issues)


def test_ensure_phases_builds_dag_from_exploration() -> None:
    context = _sample_context()
    unified = build_unified_planner_input(context)
    thin = AgentEnvelope[PlannerPayload].model_validate(
        {
            "schema_version": "2.0",
            "agent": "planner",
            "status": "ok",
            "unknowns": [],
            "warnings": [],
            "payload": {
                "plan_summary": "Thin",
                "domain": "optimization",
                "objective": "Aim",
                "phases": [],
                "organization": [],
                "execution": [],
                "repo_usage": [],
                "results_summary_path": "results/paper_1/summary.json",
            },
        }
    )
    exploration = {
        "readme_file": "README.md",
        "readme_full": "support `gp` and `de`",
        "entrypoint_hints": ["exp/run_exp.py", "algorithms/__init__.py"],
        "example_commands": [
            "python exp/run_exp.py --fun FUN_NAME --algo ALGO_NAME --reg-type REGRESSOR "
            "--cls-type CLASSIFIER --log-path LOG_PATH"
        ],
        "registry_ids": {
            "functions_or_benchmarks": ["lsq", "tow", "3bar", "beam"],
            "algorithms_or_methods": ["be-cbo", "cei", "scbo-t-re"],
        },
        "source_excerpts": [],
        "experiment_candidates": [],
    }
    filled = _ensure_phases_from_builder(thin, unified, repo_exploration=exploration)
    ids = [phase.phase_id for phase in filled.payload.phases]
    assert ids[0] == "setup"
    assert "smoke" in ids
    assert "synthetic" in ids
    assert "real_world" in ids
    assert any(row.run_command for phase in filled.payload.phases for row in phase.matrix)
    remaining = _collect_grounding_issues(filled, unified, repo_exploration=exploration)
    assert remaining == []


def test_merge_planned_actions_keeps_scaffold_goals_when_axes_present() -> None:
    deterministic = [
        PlanPhase(
            phase_id="setup",
            title="Environment setup",
            goal="Install deps",
            planned_actions="pip install .",
        ),
        PlanPhase(
            phase_id="ablation_1_hidden_layers",
            title="Ablation: hidden_layers",
            goal="Sweep hidden_layers over Analyst values [1, 2, 3, 4] on all selected benchmarks with one seed (serial).",
            depends_on=["smoke"],
            variables=["hidden_layers", "benchmark", "seed"],
            axes={"hidden_layers": [1, 2, 3, 4], "benchmark": ["lsq"], "seed": [0]},
            matrix=[
                PhaseRunSpec(
                    name="ablation_row",
                    variables={"hidden_layers": 1, "benchmark": "lsq", "seed": 0},
                    run_command="python exp/run_exp.py --fun lsq",
                    code_refs=["exp/run_exp.py"],
                    verify=["exit_code:0"],
                )
            ],
            planned_actions="Do not invent values outside Analyst list [1, 2, 3, 4].",
        ),
    ]
    llm_phases = [
        PlanPhase(
            phase_id="setup",
            title="Bootstrapping",
            goal="LLM setup goal",
            planned_actions="LLM setup actions",
        ),
        PlanPhase(
            phase_id="ablation_1_hidden_layers",
            title="LLM Ablation Title",
            goal="Sweep hidden_layers over Analyst values [20, 40] on all selected benchmarks with one seed (serial).",
            variables=["hidden_layers"],
            axes={"hidden_layers": [20, 40]},
            matrix=[],
            planned_actions="Use invented values [20, 40].",
        ),
    ]
    merged = _merge_planned_actions(deterministic, llm_phases)
    by_id = {phase.phase_id: phase for phase in merged}
    assert by_id["setup"].goal == "LLM setup goal"
    assert by_id["setup"].planned_actions == "LLM setup actions"
    assert by_id["setup"].title == "Bootstrapping"
    assert by_id["ablation_1_hidden_layers"].title == "LLM Ablation Title"
    assert "[1, 2, 3, 4]" in by_id["ablation_1_hidden_layers"].goal
    assert "[20, 40]" not in by_id["ablation_1_hidden_layers"].goal
    assert by_id["ablation_1_hidden_layers"].planned_actions.startswith("Do not invent")
    assert by_id["ablation_1_hidden_layers"].axes["hidden_layers"] == [1, 2, 3, 4]


def test_apply_plan_verification_demotes_manual_edits(tmp_path: Path) -> None:
    (tmp_path / "Clustering.py").write_text(
        "from collections import OrderedDict\n"
        "tunables = OrderedDict([('dataset', ['cora']), ('method', ['mincut_pool'])])\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("Run Clustering.py\n", encoding="utf-8")
    envelope = AgentEnvelope[PlannerPayload].model_validate(
        {
            "schema_version": "2.0",
            "agent": "planner",
            "status": "ok",
            "unknowns": [],
            "warnings": [],
            "payload": {
                "plan_summary": "s",
                "domain": "d",
                "objective": "o",
                "phases": [
                    {
                        "phase_id": "smoke",
                        "title": "Smoke",
                        "goal": "g",
                        "matrix": [
                            {
                                "name": "smoke_row",
                                "variables": {"script": "Clustering.py"},
                                "run_command": "python Clustering.py",
                                "code_refs": ["Clustering.py"],
                                "verify": ["exit_code:0"],
                            }
                        ],
                    },
                    {
                        "phase_id": "experiments",
                        "title": "Experiments",
                        "goal": "g",
                        "variables": ["benchmark", "algorithm"],
                        "axes": {"benchmark": ["cora"], "algorithm": ["mincut_pool"]},
                        "matrix": [
                            {
                                "name": "exp_row",
                                "variables": {
                                    "benchmark": "cora",
                                    "algorithm": "mincut_pool",
                                },
                                "run_command": (
                                    "python Clustering.py  # set dataset=cora in script "
                                    "tunables OrderedDict"
                                ),
                                "code_refs": ["Clustering.py"],
                                "verify": ["exit_code:0"],
                            }
                        ],
                    },
                ],
                "results_summary_path": "results/spectral/summary.json",
            },
        }
    )
    updated = _apply_plan_verification(
        envelope,
        paper_id="spectral",
        repo_path=tmp_path,
        repo_exploration={
            "execution_surface": "script",
            "script_entrypoints": ["Clustering.py"],
            "repo_path": str(tmp_path),
        },
    )
    assert updated.status == "partial"
    by_id = {phase.phase_id: phase for phase in updated.payload.phases}
    assert by_id["smoke"].matrix
    assert by_id["experiments"].matrix
    assert any("demoted" in item or "stub" in item.lower() for item in updated.payload.missing_context + updated.warnings)
    assert any("verification" in item.lower() or "stub" in item.lower() for item in updated.warnings)


def test_missing_runnable_command_forces_partial_engineer_handoff() -> None:
    context = _sample_context()
    context.repo_context["example_commands"] = ["python3 -m venv env-pd"]
    unified = build_unified_planner_input(context)
    envelope = AgentEnvelope[PlannerPayload].model_validate(_ok_envelope_content())

    updated = _apply_runnable_repo_contract(envelope, unified)

    assert updated.status == "partial"
    ids = [phase.phase_id for phase in updated.payload.phases]
    assert ids[0] == "setup"
    assert "missing_context" in ids
    assert updated.payload.phases  # never empty when has_code
    assert any(
        "no grounded runnable surface" in item for item in updated.payload.missing_context
    )


def test_runnable_contract_keeps_script_and_native_phases() -> None:
    context = _sample_context()
    context.repo_context["example_commands"] = []
    unified = build_unified_planner_input(context)
    envelope = AgentEnvelope[PlannerPayload].model_validate(_ok_envelope_content())

    script_exploration = {
        "execution_surface": "script",
        "script_entrypoints": ["Clustering.py"],
        "example_commands": ["python Clustering.py"],
    }
    kept_script = _apply_runnable_repo_contract(
        envelope, unified, repo_exploration=script_exploration
    )
    assert kept_script.status == "ok"
    assert any(phase.phase_id == "smoke" for phase in kept_script.payload.phases)

    native_exploration = {
        "execution_surface": "native",
        "native_build": {
            "available": True,
            "files": ["CMakeLists.txt"],
            "commands": ["cmake -S . -B build && cmake --build build"],
        },
        "native_tests": ["test/graph_test.cpp"],
    }
    native_envelope = AgentEnvelope[PlannerPayload].model_validate(_ok_envelope_content())
    native_envelope = native_envelope.model_copy(
        update={
            "payload": native_envelope.payload.model_copy(
                update={
                    "phases": [
                        phase
                        for phase in build_plan_phases(
                            paper_id="paper_1",
                            build_system="cmake -S . -B build && cmake --build build",
                            exploration=native_exploration,
                            analyst=unified.analyst_output,
                        )
                    ]
                }
            )
        }
    )
    kept_native = _apply_runnable_repo_contract(
        native_envelope, unified, repo_exploration=native_exploration
    )
    assert kept_native.status == "ok"
    assert any(phase.phase_id == "native_smoke" for phase in kept_native.payload.phases)
    assert kept_native.payload.phases


def test_planner_retries_on_grounding_failure(monkeypatch, tmp_path: Path) -> None:
    planner = PaperPlanner()
    context = _sample_context(tmp_path)
    attempts = {"count": 0}

    bad = {
        "schema_version": "2.0",
        "agent": "planner",
        "status": "ok",
        "unknowns": [],
        "warnings": [],
        "payload": {
            "plan_summary": "Plan",
            "domain": "optimization",
            "objective": "Aim",
            "phases": [],
            "results_summary_path": "results/paper_1/summary.json",
        },
    }

    class _Response:
        def __init__(self, content: dict[str, object]):
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": json.dumps(self._content)}}).encode("utf-8")

    def fake_urlopen(_req, timeout=180):
        _ = timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _Response(bad)
        return _Response(_ok_envelope_content())

    monkeypatch.setattr(planner_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)

    out = planner._call_ollama_json(context)
    # Deterministic scaffolds run before grounding retry, so a thin LLM payload is
    # repaired without requiring a second model call when exploration can scaffold.
    assert attempts["count"] >= 1
    assert out.payload.phases
    assert out.payload.results_summary_path.startswith("results/paper_1/")


def test_extraction_to_analyst_dict_keeps_notes() -> None:
    extraction = SectionExtraction(
        research_question="Q?",
        methodology="M",
        notes="keep me",
    )
    dumped = _extraction_to_analyst_dict(extraction)
    assert dumped["notes"] == "keep me"
