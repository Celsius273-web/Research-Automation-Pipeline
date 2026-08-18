# Quick Start Guide

A practical guide to running the Research Automation Pipeline: ingestion, agents, tests, and benchmarks.

---

## Setup

### Prerequisites
- Python 3.11+ (tested on 3.11.9)
- Docker (for experiment execution)
- Ollama running locally (for LLM inference)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Ingesting Papers

## Step 1 — Ingest a Paper

Copy a PDF into the project and register it. All files are automatically stored in `data/papers/{paper_id}/`: 
(I have stated paper id such that its easy to track what papers are what not a number combo.)
```bash
# PDF only
.venv/bin/python scripts/ingest_paper.py --pdf-path "/path/to/paper.pdf" --repo-url "https://github.com/staglibrary/stag" --title "STAG" --paper-id "STAG"
```

Ingestion stores the PDF and metadata in the database. The paper is now available for analysis.

### Verify Ingestion
```bash
python src/main.py analyze <paper_id> --non-interactive
```

If the paper was ingested successfully, analysis proceeds.

---

## Running the Pipeline

### Full Pipeline (All Phases: Analyze → Plan → Execute → Review)
```bash
python src/main.py run <paper_id> --non-interactive
```

**What happens:**
1. **Analyst**: Extracts methodology, datasets, metrics from the paper
2. **Planner**: Creates an experiment matrix and execution DAG
3. **Engineer**: Generates code from the plan
4. **Executor**: Runs experiments in Docker, captures metrics
5. **Reviewer**: Compares reported vs captured results

Output: `data/papers/<paper_id>/runs/R<N>/` with metrics and reports.

---

### Individual Phases

#### Phase 1: Analysis Only
```bash
python src/main.py analyze <paper_id> --non-interactive
```

Output: `data/extractions/<paper_id>.json`

#### Phase 1.5: Analysis + Planning
```bash
python src/main.py analyze <paper_id> --with-plan --non-interactive
```

Output:
- `data/extractions/<paper_id>.json`
- `data/plans/<paper_id>.json`

#### Phase 2: Planning (from Extraction)
```bash
python src/main.py plan --paper-id <paper_id> --non-interactive
```

Requires: An approved extraction file at `data/extractions/<paper_id>.json`

Output: `data/plans/<paper_id>.json`

#### Phase 3: Execution (from Plan)
```bash
python src/main.py execute --paper-id <paper_id> --non-interactive
```

Requires: An approved plan file at `data/plans/<paper_id>.json`

Output: `data/papers/<paper_id>/runs/R<N>/run_summary.json`

#### Phase 4: Review (from Run)
```bash
python src/main.py review --paper-id <paper_id>
```

Output: `data/papers/<paper_id>/runs/R<N>/reviewer_report.json`

---

## Benchmark Scenarios

### Optimization Benchmark
Runs **Random Search** and **Bayesian Optimization** on 5 functions (Sphere, Rastrigin, Ackley, Rosenbrock, Griewank).

```bash
python src/main.py run-plan \
  --plan-path data/papers/synthetic_optimize/synthetic_optimize_plan.json \
  --paper-id synthetic_optimize \
  --repo-path benchmark/ \
  --non-interactive
```

**Output:**
- `data/papers/synthetic_optimize/runs/R<N>/metrics.json`
- `data/papers/synthetic_optimize/runs/R<N>/reviewer_report.json`

**Expected results:**
- Bayesian Optimization: ≈0.004 final value on Sphere (Random Search: ≈8.8)
- Runtime: <2 minutes
- All 20 experiments succeed

---

### Graph Algorithm Benchmark
Generates **5 algorithms** from specs (DFS, BFS, Dijkstra, Floyd-Warshall, Kruskal) and validates against NetworkX.

**Setup graphs first:**
```bash
python benchmark/setup_graph.py
```

**Then run execution:**
```bash
python src/main.py run-plan \
  --plan-path data/papers/synthetic_graph/synthetic_graph_plan.json \
  --paper-id synthetic_graph \
  --repo-path benchmark/ \
  --non-interactive
```

**Output:**
- `data/papers/synthetic_graph/runs/R<N>/metrics.json`
- `data/papers/synthetic_graph/runs/R<N>/reviewer_report.json`

**Expected results:**
- DFS: All nodes visited ✓
- BFS: Level-order traversal, 100% match ✓
- Dijkstra: Shortest path distance 16, verified ✓
- Floyd-Warshall: All-pairs distance matrix, exact match ✓
- Kruskal: MST weight 17.0, confirmed ✓

---

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html
```

Coverage report: `htmlcov/index.html`

### Run Specific Test File
```bash
pytest tests/src/test_run_plan_command.py -v
```

### Run Tests for a Module
```bash
pytest tests/agents/ -v
pytest tests/tools/ -v
pytest tests/graphs/ -v
pytest tests/benchmark/ -v
```

### Run a Single Test
```bash
pytest tests/test_analyst.py::test_extract_from_pdf -v
```

### Test the Ingestion Flow
```bash
pytest tests/src/test_main_ingestion_enforcement.py -v
```

### Test Database
```bash
pytest tests/src/test_db.py -v
```

---

## Agent Details

| Agent | Phase | Role |
|-------|-------|------|
| **Analyst** | 1 | Extract methodology, datasets, metrics from PDF or spec |
| **Planner** | 2 | Create experiment matrix and execution DAG |
| **Engineer** | 3 | Generate code from the plan |
| **Executor** | 3 | Run experiments in Docker, capture outputs |
| **Reviewer** | 4 | Compare reported vs captured results, grade accuracy |

### Test Individual Agents
```bash
pytest tests/agents/test_analyst_mocked.py -v
pytest tests/agents/test_planner_mocked.py -v
pytest tests/agents/test_engineer_mocked.py -v
pytest tests/agents/test_reviewer_mocked.py -v
```

---

## Data Artifacts

All pipeline outputs are stored in `data/`:

```
data/
├── extractions/           # Analyst output JSON
├── plans/                 # Planner output JSON
└── papers/
    └── <paper_id>/
        ├── code/          # Ingested repository
        ├── pdf/           # Ingested PDF
        └── runs/
            └── R<N>/      # Run N artifacts
                ├── metrics.json          # Executor outputs
                ├── run_summary.json      # Run metadata
                └── reviewer_report.json  # Reviewer comparison
```

---

## Debugging

### View Extraction JSON
```bash
cat data/extractions/<paper_id>.json | python -m json.tool
```

### View Plan JSON
```bash
cat data/plans/<paper_id>.json | python -m json.tool
```

### View Metrics
```bash
cat data/papers/<paper_id>/runs/R1/metrics.json | python -m json.tool
```

### View Reviewer Report
```bash
cat data/papers/<paper_id>/runs/R1/reviewer_report.json | python -m json.tool
```

### Interactive Analysis (Pause for Human Review)
Omit `--non-interactive` to pause at each phase for manual approval:

```bash
python src/main.py analyze <paper_id>
```

When prompted, review the extraction and type `approve` or `reject`.

---

## Common Issues

### Paper Not Found
```
Paper '<paper_id>' not found. Please ingest the paper first using:
  python scripts/ingest_paper.py --pdf-path <pdf_path> --paper-id <paper_id>
```

**Solution:** Ingest the paper using the command shown.

### Extraction Not Approved
```
Extraction review is not approved. Planner cannot run.
```

**Solution:** Run analysis with `--non-interactive` or manually approve when prompted:
```bash
python src/main.py analyze <paper_id>
```

### Plan File Missing
```
Plan file does not exist: data/plans/<paper_id>.json
```

**Solution:** Run the Planner first:
```bash
python src/main.py plan --paper-id <paper_id> --non-interactive
```

### Docker Execution Failed
Ensure Docker is running:
```bash
docker ps
```

If Docker isn't running, start it first.

### Ollama Not Available
Ensure Ollama is running locally:
```bash
ollama serve
```

In another terminal, verify:
```bash
ollama list
```

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Ingest paper | `python scripts/ingest_paper.py --pdf-path <path> --paper-id <id>` |
| Full pipeline | `python src/main.py run <paper_id> --non-interactive` |
| Analysis only | `python src/main.py analyze <paper_id> --non-interactive` |
| Plan from extraction | `python src/main.py plan --paper-id <paper_id> --non-interactive` |
| Execute from plan | `python src/main.py execute --paper-id <paper_id> --non-interactive` |
| Review results | `python src/main.py review --paper-id <paper_id>` |
| Run opt benchmark | `python src/main.py run-plan --plan-path data/papers/synthetic_optimize/synthetic_optimize_plan.json --paper-id synthetic_optimize --repo-path benchmark/ --non-interactive` |
| Run graph benchmark | `python src/main.py run-plan --plan-path data/papers/synthetic_graph/synthetic_graph_plan.json --paper-id synthetic_graph --repo-path benchmark/ --non-interactive` |
| Run tests | `pytest` |
| Run tests with coverage | `pytest --cov=src --cov-report=html` |
| Test specific module | `pytest tests/agents/ -v` |

---

## Directory Structure

```
src/
├── config.py                    # All configuration & constants
├── state.py                     # Pydantic models & state schema
├── main.py                      # CLI entry point
├── persistence.py               # JSON I/O for artifacts
├── review_prompts.py            # Interactive checkpoints
├── pipeline_nodes.py            # Node factories for graph nodes
├── agents/
│   ├── analyst.py               # Extraction agent
│   ├── planner.py               # Planning agent
│   ├── engineer.py              # Code generation agent
│   ├── executor.py              # Execution orchestrator
│   └── reviewer.py              # Comparison & grading agent
├── graphs/
│   └── research_graph.py         # LangGraph phase builders
└── tools/
    ├── docker_executor.py        # Docker integration
    ├── language_detect.py        # Repo language detection
    ├── benchmark_expectations.py # Expected outputs for benchmarks
    └── ... (other helpers)

tests/                           # Mirrors src/ 1:1
benchmark/                       # Benchmark implementations
scripts/                         # Utility scripts
data/                            # Runtime artifacts (never committed)
```

---

## Performance Notes

- **Optimization Benchmark**: ~1-2 minutes
- **Graph Benchmark**: ~2-3 minutes  
- **Full Pipeline (Simple Paper)**: 5-15 minutes depending on complexity
- **Test Suite**: ~30-60 seconds

---

## Next Steps

1. **Ingest a paper** to test the Analyst
2. **Run `run-plan` on a benchmark** to verify the full system works
3. **Run the test suite** to ensure correctness
4. **Explore `data/`** to see pipeline outputs
5. **Read `TECHNICAL_DECISIONS.md`** for architecture context
