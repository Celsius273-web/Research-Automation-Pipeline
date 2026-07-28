# ResearchAssistant Usage

## Prerequisites

**Database setup (SQLite, run once):**
```bash
source /Users/school/ResearchAssistant/.venv/bin/activate 

.venv/bin/python -c "
from src.db import init_schema
init_schema()
print('Database initialized')
"
```
clear data base
psql postgresql://localhost/research_assistant -c "DELETE FROM papers;"

## Step 1 — Ingest a paper

Copy a PDF into the project and register it. All files are automatically stored in `data/papers/{paper_id}/`: 
(I have stated paper id such that its easy to track what papers are what not a number combo.)
```bash
# PDF only
.venv/bin/python scripts/ingest_paper.py --pdf-path "/path/to/paper.pdf" --repo-url "https://github.com/staglibrary/stag" --title "STAG" --paper-id "STAG"
```
cd /Users/school/ResearchAssistant

.venv/bin/python scripts/ingest_paper.py --pdf-path "/Users/school/Pre-trained_Gaussian_Processes_Bayesian_Optimization.pdf" --title "Pretrained Gaussian Processes Bayesian Optimization" --paper-id "pretrained_gp_bo" --repo-url "https://github.com/google-research/hyperbo/"

.venv/bin/python scripts/ingest_paper.py --pdf-path "/Users/school/2304.03170v1.pdf" --title "Sparse Topology Aware Graph Neural Networks" --paper-id "stag_sparse" --repo-url "https://github.com/staglibrary/stag"

.venv/bin/python scripts/ingest_paper.py --pdf-path "/Users/school/2402.07692v2.pdf" --title "Boundary Exploration Bayesian Optimization" --paper-id "boundary_exploration_bo" --repo-url "https://github.com/yunshengtian/BE-CBO"

.venv/bin/python scripts/ingest_paper.py --pdf-path "/Users/school/1907.00481v6.pdf" --title "Spectral Clustering Graph Neural Networks" --paper-id "spectral_clustering_gnn" --repo-url "https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling"
---

## Step 2 — Run Analyst (extraction)

.venv/bin/python -m src.main analyst --paper-id pretrained_gp_bo --non-interactive
.venv/bin/python -m src.main analyst --paper-id stag_sparse --non-interactive
.venv/bin/python -m src.main analyst --paper-id boundary_exploration_bo --non-interactive
.venv/bin/python -m src.main analyst --paper-id spectral_clustering_gnn --non-interactive

```bash
# Analyst only (interactive with review checkpoint)
.venv/bin/python -m src.main analyst --paper-id <paper_id>

# Analyst only (auto-approve, no review checkpoint)
.venv/bin/python -m src.main analyst --paper-id <paper_id> --non-interactive

# Analyst + Planner together (interactive with review checkpoints)
.venv/bin/python -m src.main analyst --paper-id <paper_id> --with-plan

# Analyst + Planner together (auto-approve all checkpoints)
.venv/bin/python -m src.main analyst --paper-id <paper_id> --with-plan --non-interactive


for id in boundary_exploration_bo pretrained_gp_bo spectral_clustering_gnn stag_sparse; do
  echo "=== ANALYST $id ==="
  .venv/bin/python -m src.main analyst --paper-id "$id" --non-interactive
done 
```
**Output files in `data/papers/<paper_id>/`:**

| File | Contents |
|---|---|
| `{paper_id}.json` | Machine-readable JSON with `by_section` + `merged` extraction |
| `{paper_id}_sections.txt` | Human-readable per-section breakdown |
"
---

## Step 3 — Run Planner (execution plan)

Requires an **approved extraction artifact** (run Analyst first).

```bash
# Planner only (interactive with review checkpoint)
.venv/bin/python -m src.main plan --paper-id <paper_id>

# Planner only (auto-approve)
.venv/bin/python -m src.main plan --paper-id <paper_id> --non-interactive

# Planner from explicit extraction path
.venv/bin/python -m src.main plan --extraction-path /path/to/extraction.json --non-interactive
```

**Output:** `data/papers/<paper_id>/{paper_id}_plan.json`

---

## Step 4 — Execute & review (Phase 3+)

```bash
# Execute plan + run reviewer
.venv/bin/python -m src.main execute --paper-id <paper_id> --with-review

# Execute against a different local repo
.venv/bin/python -m src.main execute --paper-id <paper_id> \
  --repo-path "/path/to/different/repo" --with-review
```

---

## Standalone commands

```bash
# Re-run reviewer on an existing run summary
.venv/bin/python -m src.main review --paper-id <paper_id>
```

---

## Already-ingested papers

| paper_id | Title |
|---|---|
| `pre_trained_gaussian_processes_bayesian_optimization` | Pre-trained Gaussian Processes for Bayesian Optimization |
| `p2p_replication` | P2P Search and Replication |

