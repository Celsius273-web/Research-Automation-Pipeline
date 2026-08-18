List of technical decisions # Simply for me the USER

## Ingestion / layout
- have a script to ingest files to ensure they fit project constraints
- each paper has a folder under `data/papers/<paper_id>/` (pdf, code, extraction JSON, plans, sections txt)
- use agent envelope to organize from strict JSON (planner/engineer/reviewer); Analyst stays flat `SectionExtraction`

## Analyst → Planner handoff
- Analyst persists full bundle (`by_section` + `merged`), not merged-only (so section evidence survives)
- Planner takes a fixed 4-key unified input: `analyst_output`, `repo_context`, `paper_context`, `flags`
- Flags are always re-derived on load (don’t trust stale flags in hand-written JSON)
- Soft-block: Planner must not `blocked` when RQ **or** methodology is present; force `partial` instead
- Entrypoints (1C): best-effort from README `example_commands`; else empty `run_command` + `partial` — do **not** invent `src.main` / fake CLIs
- Sparse Analyst (2B): block only when **both** RQ and methodology empty; flags drive routing (`has_research_question`, `has_methodology`)

## Analyst extraction quality
- Soft-fill RQ (B+C): if abstract RQ missing, soft-fill from `paper_overview` then methodology/title; mark `[inferred]` in notes; toolkit papers get `Toolkit paper: …`
- RQ ownership: fill `research_question` **only from abstract**; other sections leave it empty; merge prefers abstract
- New field `paper_overview`: 4–6 sentences (goals/aim/background from abstract, maybe intro/method); keep separate from `methodology` + `notes`
- Datasets: prefer matrix notation (`MNIST × {archs} × batch {sizes}`); finalize collapses aliases (`Name` vs `Name (4D)`) and junk phrases
- Hyperparams: canonical short keys (`learning_rate`, `optimizer`, …); finalize merges aliases when values agree
- Reported results: numeric gate (value must contain a digit); drop “see Table X” / qualitative figure fluff
- Results grounding (C): tables-only via PyMuPDF + caption reconstruction → `## Extracted Tables`; **skip OCR/vision** for now
- Experiments/appendix chunking (overlapping windows) so late tables aren’t truncated; `ANALYST_NUM_PREDICT` ≥ 2048
- Context: `MODEL_NUM_CTX=20480` for Ollama headroom; do **not** raise section char limits just because ctx is larger (latency/timeouts)
- Instrument Analyst failures: log `reason=timeout|empty_response|invalid_json|schema_validation|…` + `elapsed_ms`

## Planner flexibility
- Don’t block plans solely on missing RQ if methodology/overview exists
- Repo context enrichment: file_tree, readme_summary, build command, example_commands from README
- CLI: `plan --input-json PATH` for fixture/unified-input runs without re-extracting

## Still open / later
- Figure OCR/vision only if major gaps remain after tables-only
- Section-boundary fixes when “abstract” is huge (parser stuffing body into abstract)
- Further call-volume cuts (don’t duplicate tables into every appendix chunk) if timeouts persist


## Planner Issues
- Not “Planner can’t observe the BE-CBO repo.” It can; it still emitted empties.
- For HyperBO / Spectral / STAG, repo surface is the main blocker—sometimes correctly (HyperBO), sometimes because extraction is too strict.
- Prompting is too soft on required matrix/steps when runnable commands exist.
- Analyst noise + missing name mapping make concrete matrix rows harder.
- Updated prompt to explicitly order: README first, then tree/registries/excerpts, then fill matrix in Engineer-usable detail when runnable; stay empty/partial only when nothing is runnable.
- Force the llm to deep dive into repo - get key ideas.

## Planner phase DAG (current direction)
- Replace flat `steps` + global `experiment_matrix` with `payload.phases` DAG.
- Each phase: `depends_on`, compact `axes` + `variables` (only varying factors), `run_template`, small example `matrix` rows (`run_command`, `code_refs`, `verify`), `planned_actions`.
- Deterministic `phase_builder` scaffolds by `execution_surface`; LLM enriches notes; full cartesian expands at runtime.
- Multi-surface exploration (priority): `cli` → `script` → `library` → `native` → `config` → `container` → `artifact` → `unknown`.
  - `cli`: registry-backed BE-CBO path (setup → smoke → synthetic/real_world/ablations).
  - `script`: README/top-level `*.py` without registries (Spectral); scrape in-file tunables into axes.
  - `library`: tests/notebooks (HyperBO): setup → `library_smoke` → `reproduce_similar`.
  - `native`: CMake/Make + `*_test.cpp` (STAG): setup → `native_smoke` → `reproduce_similar`.
  - `unknown`: setup + `missing_context` — never silently empty `phases` when `has_code`.
- Runnable contract keeps script/native/config/container/library phases when evidence lists are non-empty.
- Engineer currently projects phases → PlanStep via `project_phases_to_steps` (iterate full DAG later).

## Smoke vs reproduce (what we’re doing now)
- **Aim of smoke:** prove the repo installs and core APIs work. Success = exit code 0 (unit/API validation). Smoke is **not** paper reproduction.
- Smoke may list Analyst paper metrics on matrix rows as *later targets*, but `planned_actions` must say clearly: smoke does **not** produce regret/eval counts. If a test happens to print numbers (eigenvalues, likelihood), capture them; otherwise smoke = API validation only.
- **Aim of `reproduce_similar`:** get as close as the local surface allows to paper-similar numbers — without inventing CLIs or fake data.
  - CLI/script surfaces: real experiment commands when grounded.
  - Library (HyperBO): often only a **demo-port gate**, not Table-X reproduction. Unit tests that look like “reproduce” get demoted; Planner writes `planner_stubs/port_demo_metrics.py` instead.
- Honesty over theater: better a short runnable scaffold + explicit fallback than a plan that pretends unit tests = paper regret curves.

## Library papers / HyperBO (current shape)
- Layout: `data/papers/<id>/` holds pdf + extraction + `code/` (clone) + sibling `planner_stubs/`; outputs under `results/<id>/`.
- Stub CLI (workspace root):  
  `python data/papers/<id>/planner_stubs/port_demo_metrics.py --repo-root data/papers/<id>/code --out results/<id>/reproduce_similar/demo`
- Stub writes `metrics.csv` with columns `metric_name,value,source,notes` — **status only** after nbconvert (not paper regret). If nbconvert fails: hand-port demo cells (e.g. Branin/synthetic BO loops) into the same CSV schema.
- `planned_actions` must spell that fallback so Engineer isn’t stuck at “port cells manually” with no how.
- Verification + repair (`plan_verification` → `plan_repair`): demote unverified / unit-test-as-reproduce rows; refill empty matrices with stubs or collapse ablations that lack CLI flags.
- Status for pretrained_gp_bo: good Engineer-ready **scaffold** (setup → smoke → demo gate → summarize). Full paper repro still needs external benchmarks + a real metric-producing port — recorded in `missing_context`, not faked in the matrix.

## Still rough / open on Planner
- LLM summary fields can still oversell (“reproduce paper-similar metrics”) while the matrix is honestly a demo gate — keep payload phases as source of truth.
- Empty `variables`/`axes` on single-path demo rows is OK; empty on phases that should vary factors is still a smell.
- Possible later split: repo-details call → experiment-matrix call → ordered run specs (instead of one soft LLM enrich).


the implementation_steps are a bit vague neeed to investigate what info is actually needed for implementation + execution - format the plan differently?
"variables": [] are empty - will cause issues
I have a few ideas. have a few different calls one for general repo details. next developing the experiment matrix: what Hyperparameters + variables are being tested = essentially determine what experiements are being run.
next needs to figure out with all this info what order is ideal for engineer to test

further call(s) will look at how each test needs to be run. provide an order with some details and where to reference so that the Engineer knows how to attack the problem. 

