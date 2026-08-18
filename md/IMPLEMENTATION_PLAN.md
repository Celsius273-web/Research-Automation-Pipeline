# Autonomous Research Assistant: Implementation Plan

## Technical overview

This project reproduces optimization papers automatically, using local LLMs on a 16GB M3 Mac. Five specialized agents handle paper extraction, planning, code adaptation, execution, and review. A separate discovery pipeline finds and ranks candidate papers before any agent work starts. All execution runs CPU only, since Docker on Apple Silicon has no access to the Metal GPU. All agents run fully local through Ollama, one model loaded at a time.

## Hardware and model constraints

A 16GB Mac gives roughly 11 to 11.5 GB of usable memory for model weights, after macOS and the runtime take their share. Do not use 14B models. A 14B model at Q4 needs 8 to 9 GB alone, leaving almost no room for context.

Model assignments:

NALYST_MODEL=qwen3.5:9b
PLANNER_MODEL=qwen3.5:9b
ENGINEER_MODEL=qwen2.5-coder:7b
REVIEWER_MODEL=qwen3.5:9b
REASONING_FALLBACK_MODEL=gpt-oss:20b

# LLM Generation Settings
MODEL_TEMPERATURE=0.1
MODEL_NUM_PREDICT=2048
PLANNER_NUM_PREDICT=4096
# Review Tolerances
REVIEW_MATCH_TOLERANCE_PCT=5.0
REVIEW_CLOSE_TOLERANCE_PCT=20.0

# Execution Settings
MAX_RETRY_ATTEMPTS=5
EXECUTOR_TIMEOUT_SECONDS=600

Load one model at a time. Never run two Ollama models in parallel.

## Domain and scope

Domain: CS papers that are easily replicatable given hardware constraints and runable in Docker.


Starter set: 10 to 15 papers with public code, selected through the discovery pipeline below.

## Agent architecture

Five agents, one job each, no overlap in responsibility.

1. **Paper Analyst.** Reads the paper and extracts the research question, methodology, datasets, variables, hyperparameters, and evaluation metrics. Splits extraction by paper section (abstract, method, experiments, hyperparameters, appendix) rather than one large pass. Human review checkpoint before hand off to Planner for v1.
2. **Planner.** Builds a step by step execution plan from the Paper Analyst's extraction.
3. **Engineer.** Adapts the paper's existing repository code according to the plan. Does not generate an implementation from scratch for v1.
4. **Executor.** Runs the code inside a CPU only Docker container, captures logs, and retries on failure.
5. **Reviewer.** Compares reproduced results to the paper's reported numbers and generates plots, tables, and a written report.

No master agent for v1. LangGraph's graph structure is the orchestrator for the core loop. Keep the Reviewer scoped to comparison and reporting only. Mixing high level decision making into the Reviewer blurs two different jobs and makes failures harder to trace. Add a separate Research Director agent later, as a stretch goal, once the core pipeline is proven, to propose follow up experiments such as sweeps or ablations.

## Retry loop

Executor and Engineer form the one loop in the core pipeline.

1. Executor runs the code and captures logs.
2. On failure, Executor packages the traceback and logs and hands them to Engineer.
3. Engineer patches the code.
4. Executor reruns.
5. Bounded at 5 attempts. Beyond that, mark the run failed and let me make decision to see if I can help. I dont want to go to report section yet because there are no results yet.

Distinguish compilation failures from runtime failures in the logs. A C++ build error and a Python traceback need different handling from Engineer.

## Multi-language execution

Reproduce each experiment in whatever language the paper's repository uses. Do not force a Python rewrite of a C, C++, or Fortran implementation. A rewrite risks introducing bugs that make results diverge from the paper for reasons unrelated to the method itself.

- Detect the language from repo files: setup.py or requirements.txt for Python, CMakeLists.txt or Makefile for C or C++, Cargo.toml for Rust.
- Build a minimal Docker base image per language, ARM64, CPU only.
- Keep Python as the orchestration layer only: the Engineer agent, logging, and comparison against the paper's reported numbers.
- Log which stage failed, build or runtime, so the retry loop routes the right context to Engineer.

## Tech stack decisions

- **Language.** Python for orchestration and all agents. Native language of each paper's repo for the experiment itself.
- **Containerization.** Docker, ARM64, CPU only base images.
- **Local LLM runtime.** Ollama.
- **Agent framework.** LangGraph.
- **Storage.** SQLite for v1. Move to Postgres only if the FastAPI phase happens.
- **Background jobs.** Skip Celery and Dramatiq for v1. Run the pipeline synchronously from the CLI, since a solo project with sequential model calls has no real concurrency need. If the FastAPI phase happens later, use Dramatiq. It carries less operational overhead than Celery for a solo maintainer.
- **Frontend.** Skip for v1. If built later, use React with Vite for an internal dashboard showing runs, logs, and reports. Reach for Next.js only if deploying the dashboard publicly with routing and hosting.
- **Paper parsing.** PyMuPDF.
- **Visualization.** Matplotlib, Pandas.

## Project directory structure

Supersedes the earlier sketch in DEV_PLAN.MD, which was a brainstorm rather than a locked layout. That sketch had three gaps: no home for the discovery pipeline below, no `data/` for papers, extractions, cloned repos, run logs, or reports, and a missing `executor.py` agent file, even though Executor is one of the five named agents, not just a Docker tool wrapper.

```
ResearchAssistant/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docker/                        # Phase 3+: one minimal base image per language
│   ├── python.Dockerfile
│   ├── cpp.Dockerfile
│   └── rust.Dockerfile
├── data/                          # gitignored (manifests excepted)
│   ├── papers/                    # input PDFs
│   ├── extractions/               # Paper Analyst output (Phase 1+)
│   ├── repos/                     # cloned paper repos, mounted into Docker (Phase 3+)
│   ├── runs/                      # per-run logs/attempts from Executor (Phase 3+)
│   ├── reports/                   # Reviewer plots/tables/reports (Phase 5+)
│   └── db.sqlite3                 # v1 storage (from Phase 2 on)
├── src/
│   ├── config.py                  # model routing, Ollama endpoint, paths
│   ├── state.py                   # LangGraph state schema, full pipeline shape
│   ├── db.py                      # SQLite schema/access (Phase 2+)
│   ├── graphs/
│   │   └── research_graph.py      # grows one node per phase, wired fully by Phase 6
│   ├── discovery/                 # separate pipeline, Chunks 1-6, own module
│   │   ├── clients/
│   │   │   ├── arxiv_client.py
│   │   │   ├── semantic_scholar_client.py
│   │   │   ├── openalex_client.py
│   │   │   └── crossref_client.py
│   │   ├── merge.py
│   │   ├── repo_linker.py
│   │   ├── classifier.py
│   │   ├── ranking.py
│   │   └── review_cli.py
│   ├── agents/
│   │   ├── analyst.py             # reasoning pool model (Phase 1)
│   │   ├── planner.py             # reasoning pool model (Phase 2)
│   │   ├── engineer.py            # coder model (Phase 3)
│   │   ├── executor.py            # no model, deterministic (Phase 3)
│   │   └── reviewer.py            # reasoning pool model (Phase 5)
│   ├── tools/
│   │   ├── pdf_parser.py          # PyMuPDF wrapper + section splitter (Phase 1)
│   │   ├── docker_executor.py     # Docker SDK sandbox interface (Phase 3)
│   │   └── language_detect.py     # repo language detection (Phase 3)
│   └── main.py                    # CLI entrypoint
├── scripts/                       # one-off utilities, not part of the package
│   ├── model_bakeoff.py
│   └── ollama_sanity_check.py
├── tests/
│   ├── agents/
│   ├── tools/
│   ├── discovery/
│   └── fixtures/                  # sample PDFs, mock Ollama responses
└── web/                           # stretch goal only, Phase 7, not created until then
```

Notes:

- `src/discovery/` sits next to `src/agents/`, not inside it. Discovery is "a separate discovery pipeline" that runs "before any agent work starts" (see Technical overview above), so it should not read like a sixth agent.
- `src/graphs/research_graph.py` is a single file that grows incrementally: Phase 1 wires just the Analyst, Phase 2 adds Planner, and so on through Phase 6. This avoids renaming or restructuring the graph module every phase.
- `agents/executor.py` was missing from the original sketch even though Executor is one of the five named agents above. `tools/docker_executor.py` stays as the low-level Docker SDK wrapper that `agents/executor.py` calls, the same low-level/orchestration split as `tools/pdf_parser.py` relating to `agents/analyst.py`.
- Directories are reserved now so later phases slot in without restructuring, but most subfolders (`docker/`, `data/repos`, `data/runs`, `data/reports`, `db.sqlite3`, `discovery/`, `web/`) are only physically created in the phase that first needs them, not up front.

## Paper discovery pipeline

Papers With Code was retired by Meta in July 2025. The site now redirects to Hugging Face's Trending Papers page, and the old code links no longer come from the original API. Do not build repo linking against Papers With Code. Use GitHub's search API directly, then confirm every match by hand.

**Chunk 1: API client layer.**
Goal: normalize arXiv, Semantic Scholar, OpenAlex, and Crossref into one schema.
Build four client modules: arxiv_client, semantic_scholar_client, openalex_client, crossref_client. Each returns records shaped as title, abstract, authors, year, venue, arxiv_id, doi, source name.
Test: query each client independently and confirm the output matches the common schema.

**Chunk 2: query and merge engine.**
Goal: run the full keyword list across all sources and produce one clean table.
Keywords: Bayesian Optimization, hyperparameter optimization, CMA-ES, evolutionary algorithms. Merge results. Dedupe by DOI first, then arXiv ID, then a fuzzy title match. Store in SQLite, not Postgres.
Test: confirm a paper appearing in two sources under different metadata collapses into one row.

**Chunk 3: repo linking.**
Goal: attach a candidate code repository to each paper, flagged by confidence.
Query GitHub's search API using paper title and first author name. Store the candidate link, the query, and a confidence field. Do not auto-accept anything.
Test: run against 15 known papers and check the hit rate against your own knowledge.

**Chunk 4: LLM classifier.**
Goal: label each paper using only the title and abstract.
Fields: requires_gpu, uses_deep_learning, benchmark_used (BBOB, COCO, synthetic test functions, real world dataset, none, other), estimated_runtime_class (minutes, hours, days), reproducibility_signal (0 to 1).
Test: hand-label 10 papers, run the classifier on the same 10, compare, tune the prompt until agreement holds.

**Chunk 5: scoring and ranking.**
Goal: combine classifier output and repo-linking output into one ranked list, using a plain weighted score, not another model call.
Weights: confirmed code counts most, no GPU requirement counts next, a recognized benchmark adds a bonus, recency adds a smaller share, reproducibility signal fills the remainder. Keep weights as named constants in one place.
Test: read the top 15 by hand. Adjust one weight at a time if the ranking feels off.

**Chunk 6: review and handoff.**
Goal: turn the ranked table into the approved set that feeds the Paper Analyst.
Build a CLI view showing score, GPU flag, benchmark type, and repo link status per paper. Accept or reject each one. Confirm or reject repo links marked low confidence.
Test: confirm the final approved file has 10 to 15 papers, each with a confirmed repo link and a GPU flag of false.

## Core pipeline build phases

- **Phase 0.** Domain and subfield locked above. Freeze model choices. Write the LangGraph state schema.
- **Phase 1.** Build Paper Analyst alone. Test against 3 papers by hand.
- **Phase 2.** Build Planner against the same papers.
- **Phase 3.** Build Engineer and a minimal CPU only Docker executor, with language detection from Chunk-level repo metadata.
- **Phase 4.** Add the bounded retry loop between Executor and Engineer so tweaks can be made.
- **Phase 5.** Build Reviewer: comparison tables, plots, report generation.
- **Phase 6.** Wire all five agents into one LangGraph graph. Run the full paper set end to end. Build the evaluation harness.
- **Phase 7 (stretch, no deadline).** FastAPI, Postgres, Dramatiq, React dashboard, expanded discovery to the rest of black-box optimization, multi-paper literature review, hypothesis generation, Research Director agent.

## Evaluation and metrics

Define the reproduction similarity metric, the recovery criteria, and the retry-counting method before running the full paper set. Do not report figures such as reproduction rate or accuracy until they come from real runs against the approved paper set. A defensible methodology matters more for a resume project than an impressive-sounding number with no method behind it.

## Environment Setup Best Practices

Docker runs for Engineer are ephemeral: each phase command typically starts a new container with the paper repo mounted read/write at `/workspace`. That mount is what persists across phases — not the container's system Python site-packages.

1. **Do not put `python -m venv` in Planner plans.** Plans describe *what* to install and run (`pip install …`, `python exp/run_exp.py …`). Embedding a venv create step in the plan mixes Docker persistence policy into paper-specific planning and is easy to get wrong.
2. **Bare `pip` / `python` in the plan; mounted `.venv` at runtime.** Engineer rewrites setup to `python -m venv --clear .venv && .venv/bin/pip …` and rewrites later `python` invocations to `.venv/bin/python`. The venv lives under the mounted repo (`code/.venv`), so it survives container teardown. Installing only into the container's system site-packages does **not** persist to the next phase container.
3. **Python version belongs in extraction / image selection, not an ad-hoc in-container venv invent.** If a repo needs Python 3.8 (e.g. old GPyTorch/BoTorch pins), record that in extraction notes and use the matching Docker base image. Do not try to invent a host-side venv outside the mount.
4. **Verify `results_summary_path` against real repo outputs.** The contract path is `results/{paper_id}/summary.json`. Many repos never write that file (BE-CBO writes per-run `.pkl` under `--log-path`). Grep the repo's main scripts or inspect prior artifacts; Planner should note gaps in `missing_context` so Engineer/Reviewer expectations stay honest.
5. **Run artifacts layout.** Engineer writes only under `data/papers/{paper_id}/runs/R{n}/`
   (`engineer.log`, `metrics.json` with `experiment_matrix` + captured metrics, later
   `reviewer_report.json`). Run ids are sequential (`R1`, `R2`, …), not timestamps. Pass
   that id as `--run-id` to Reviewer. Keep paper-repo `code/results/` — that is experiment
   output Engineer captures from. By default Engineer runs only the plan’s concrete
   `matrix` rows (fast). Set `ENGINEER_EXPAND_FULL_AXES=1` only when you intentionally want
   the full `axes` cartesian product (paper-scale / multi-hour).

