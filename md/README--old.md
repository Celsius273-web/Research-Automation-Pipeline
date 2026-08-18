# Autonomous Research Assistant

CLI-first local system to reproduce optimization papers with a phased LangGraph pipeline and PostgreSQL-backed paper ingestion.

## Features

- **Paper Ingestion Pipeline**: Single entry point for adding papers with PDF + optional GitHub repository
- **Multi-phase Analysis**: Paper Analyst → Planner → Engineer → Executor → Reviewer
- **Local LLM Support**: Ollama-based execution with configurable models for 16GB Mac systems
- **Docker Execution**: Sandboxed code execution for security
- **PostgreSQL Storage**: Centralized paper registry with metadata tracking

## System Requirements

- Python 3.11+
- PostgreSQL database
- Docker (for code execution)
- Ollama with configured models
- 16GB+ RAM (for local LLM inference)

# Model decisions

- Default reasoning pool model: `qwen3.5:9b` for Analyst, Planner, and Reviewer.
- Default coding model: `qwen2.5-coder:7b` for Engineer.
- Fallback reasoning model: `gpt-oss:20b` for targeted quality checks.
- `qwen2.5-coder:14b` is intentionally not in default routing due to memory pressure.

Rationale:
On a 16GB machine with roughly 11 to 11.5GB practical model-memory budget, the
9B reasoning model leaves safer context room than 12B and much more than 14B.
This supports long section extraction while avoiding swap-heavy runs. The 7B coder
model is the best fit for iterative code-edit loops under local constraints.


## Quick Start

1. **Environment Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Database Setup**:
   ```bash
   # Create PostgreSQL database
   createdb research_assistant
   
   # Copy and configure environment
   cp .env.example .env
   # Edit .env to set POSTGRES_DSN
   
   # Initialize database schema
   python scripts/ingest_paper.py --init-db
   ```

3. **Ingest Your First Paper**:
   ```bash
   # PDF only
   python scripts/ingest_paper.py --pdf-path "/path/to/paper.pdf"
   
   # PDF + GitHub repository
   python scripts/ingest_paper.py \
     --pdf-path "/path/to/paper.pdf" \
     --repo-url "https://github.com/author/repo"
   ```

4. **Run the Pipeline**:
   ```bash
   # Get the paper_id from ingestion output, then:
   python -m src.main analyze <paper_id> --with-plan
   python -m src.main execute --paper-id <paper_id> --with-review
   ```


5. **BenchMarking**
Analyst/Planner still require an **ingested PDF** in Postgres. The updated files are markdown (`benchmark/papers/optimize.md`, `benchmark/papers/graph.md`), so ingest will reject them as-is.

From the repo root, with venv + Ollama + Docker + Postgres up:

---

### 1. Turn the markdown into PDFs and ingest

```bash
pandoc benchmark/papers/optimize.md -o /tmp/optimize.pdf
pandoc benchmark/papers/graph.md -o /tmp/graph.pdf

python scripts/ingest_paper.py --init-db \
  --pdf-path /tmp/optimize.pdf \
  --paper-id synthetic_optimize \
  --title "Random Search vs Bayesian Optimization on Synthetic Benchmark Functions"

python scripts/ingest_paper.py \
  --pdf-path /tmp/graph.pdf \
  --paper-id synthetic_graph \
  --title "LLM-Generated Graph Algorithms"
```

`--init-db` only on the first ingest.

---

### 2. Analyst → Planner → Engineer

**Optimize**

```bash
python -m src.main analyst --paper-id synthetic_optimize --non-interactive
python -m src.main plan --paper-id synthetic_optimize --non-interactive
python -m src.main engineer --paper-id synthetic_optimize --repo-path benchmark/ --non-interactive
```

**Graph**

```bash
python -m src.main analyst --paper-id synthetic_graph --non-interactive
python -m src.main plan --paper-id synthetic_graph --non-interactive
python -m src.main engineer --paper-id synthetic_graph --repo-path benchmark/ --non-interactive

### Analysis + plan in 1 cmd
python -m src.main analyze synthetic_optimize --with-plan --non-interactive
python -m src.main analyze synthetic_graph --with-plan --non-interactive



python -m src.main engineer --paper-id synthetic_optimize --repo-path benchmark/ --non-interactive
python -m src.main engineer --paper-id synthetic_graph --repo-path benchmark/ --non-interactive

python -m src.main reviewer --paper-id synthetic_optimize --run-id R1 --non-interactive
python -m src.main reviewer --paper-id synthetic_graph --run-id R1 --non-interactive
```

---

### Skip Analyst/Planner (existing hand-written plans)

If you only want Engineer/Reviewer against the current plan JSONs:

```bash
python -m src.main run-plan \
  --plan-path data/papers/synthetic_optimize/synthetic_optimize_plan.json \
  --paper-id synthetic_optimize \
  --repo-path benchmark/ \
  --non-interactive

python -m src.main run-plan \
  --plan-path data/papers/synthetic_graph/synthetic_graph_plan.json \
  --paper-id synthetic_graph \
  --repo-path benchmark/ \
  --non-interactive
```

## Architecture

Papers are stored in per-paper bundles under `data/papers/<paper_id>/`:
- `paper.pdf` - Original paper
- `code/` - Cloned repository (if provided)
- `metadata.json` - Ingestion metadata with checksums

The pipeline phases generate artifacts in:
- `data/extractions/` - Analyst output
- `data/plans/` - Planner output  
- `data/runs/` - Executor logs and results
- `data/reports/` - Reviewer comparisons and reports

See [run.md](run.md) for detailed usage instructions.
