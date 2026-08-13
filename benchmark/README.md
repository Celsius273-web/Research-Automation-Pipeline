# Synthetic benchmark projects

Two PoC projects for the research pipeline. Neither requires paper ingestion.

## Setup

From the repository root:

```bash
python benchmark/setup_graph.py
# writes benchmark/graphs.json (5 NetworkX-validated graphs + ground truth)

python benchmark/run_all_graphs.py
# writes benchmark/graph_tasks.json (5 Engineer tasks)
```

Install Executor dependencies inside the Docker-mounted `benchmark/` repo via the plan setup phase (`pip install -r requirements.txt`). Host-side, the same file is enough to generate graphs locally:

```bash
pip install -r benchmark/requirements.txt
```

## Run the pipeline (skip Analyst / Planner)

Plans live under `data/papers/{paper_id}/`. The working directory mounted into Docker is `benchmark/`.

Optimization (Random Search vs Bayesian Optimization):

```bash
python -m src.main run-plan \
  --plan-path data/papers/synthetic_optimize/synthetic_optimize_plan.json \
  --paper-id synthetic_optimize \
  --repo-path benchmark/ \
  --non-interactive
```

Graph algorithms (DFS, BFS, Dijkstra, Floyd-Warshall, Kruskal):

```bash
python -m src.main run-plan \
  --plan-path data/papers/synthetic_graph/synthetic_graph_plan.json \
  --paper-id synthetic_graph \
  --repo-path benchmark/ \
  --non-interactive
```

Each command writes:

```
data/papers/{paper_id}/runs/R{N}/
  engineer.log
  metrics.json
  reviewer_report.json
```

`run_id` is sequential (`R1`, `R2`, …), not a timestamp.

## What the Executor runs

- **optimize**: `python run_all.py` (5 functions × 2 optimizers × 3 seeds). Aggregated `CapturedMetric` rows go to `results/synthetic_optimize/metrics.json`.
- **graph**: one `python run_graph.py ...` command per algorithm. Per-algorithm JSON goes to `results/synthetic_graph/{algorithm}.json`.

Paper markdown (do not edit here; improved separately):

- `benchmark/papers/optimize.md`
- `benchmark/papers/graph.md`
