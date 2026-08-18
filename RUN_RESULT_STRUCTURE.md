# Run Result Structure

This document defines the standardized directory structure for test run results for Engineer + Reviewer validation.

## Directory Layout

```
runs/
  synthetic_graph/
    <run_id>/
      generated_code.py              # Engineer-generated graph algorithms module
      metrics.json                   # Aggregated test results (5 algorithms)
      reviewer_notes.json            # Reviewer validation report
      {algorithm_name}.json          # Individual algorithm result (e.g., dfs.json, bfs.json)
      run_metadata.json              # Execution metadata
  
  synthetic_optimize/
    <run_id>/
      generated_code.zip             # Engineer-generated code (run_all.py, benchmark.py, optimizer.py)
      metrics.json                   # Aggregated metrics (20 runs: 5 functions × 2 optimizers × 2 seeds)
      reviewer_notes.json            # Reviewer validation report
      {function}_{optimizer}_seed{N}.json  # Individual run result (e.g., sphere_random_seed0.json)
      run_metadata.json              # Execution metadata
```

## File Specifications

### generated_code.py (Graph)

**Format:** Python source code

**Content:**
- Five functions: `dfs`, `bfs`, `dijkstra`, `kruskal`, `topological_sort`
- Each function implements the algorithm as specified in `algorithm_specs` in the plan
- Include docstrings with input/output format

**Example:**
```python
def dfs(graph, start):
    """Depth-first search traversal.
    
    Args:
        graph: adjacency list dict {node: [neighbors]}
        start: starting node
    
    Returns:
        list of visited nodes in DFS order
    """
    # implementation
    pass
```

### generated_code.zip (Optimization)

**Format:** ZIP archive

**Contents:**
- `run_all.py`: Main orchestrator; loops over all 20 configs and invokes optimizers
- `benchmark.py`: Implementation of 5 benchmark functions (sphere, rastrigin, ackley, rosenbrock, griewank)
- `optimizer.py`: Wrappers for Random Search and Bayesian Optimization using scikit-optimize

**Interface:**
- `run_all.py` accepts CLI arguments: `--function`, `--optimizer`, `--seed`
- Each run outputs JSON with `simple_regret` and `final_value` metrics

### metrics.json (Aggregated)

**Format:** JSON

**Graph Schema:**
```json
{
  "paper_id": "synthetic_graph",
  "run_id": "<run_id>",
  "timestamp": "2026-08-14T12:30:00Z",
  "algorithms": 5,
  "summary": {
    "total_runs": 5,
    "passed": 4,
    "failed": 1,
    "pass_rate": 0.8
  },
  "results": [
    {
      "algorithm": "dfs",
      "graph": "simple_undirected",
      "output": "[0, 1, 3, 2]",
      "networkx_output": "[0, 1, 3, 2]",
      "match": true,
      "notes": "Correct traversal order"
    },
    {
      "algorithm": "bfs",
      "graph": "simple_undirected",
      "output": "[0, 1, 2, 3]",
      "networkx_output": "[0, 1, 2, 3]",
      "match": true,
      "notes": "Correct BFS order"
    },
    {
      "algorithm": "dijkstra",
      "graph": "weighted_shortest_path",
      "metric": "path_length_0_to_9",
      "value": 16.0,
      "expected": 16.0,
      "error_percent": 0.0,
      "within_tolerance": true
    },
    {
      "algorithm": "kruskal",
      "graph": "minimum_spanning_tree",
      "metric": "mst_weight",
      "value": 17.0,
      "expected": 17.0,
      "error_percent": 0.0,
      "within_tolerance": true
    },
    {
      "algorithm": "topological_sort",
      "graph": "dag",
      "output": "[5, 7, 3, 8, 11, 2, 9, 10]",
      "networkx_output": "[5, 7, 3, 8, 11, 2, 9, 10]",
      "is_valid_topological_order": true,
      "match": true
    }
  ]
}
```

**Optimization Schema:**
```json
{
  "paper_id": "synthetic_optimize",
  "run_id": "<run_id>",
  "timestamp": "2026-08-14T12:30:00Z",
  "total_runs": 20,
  "passed": 20,
  "failed": 0,
  "pass_rate": 1.0,
  "elapsed_minutes": 92,
  "summary_by_function": {
    "sphere": {
      "random_search": {
        "seed_0": {"simple_regret": 10.97, "final_value": 10.97},
        "seed_1": {"simple_regret": 11.23, "final_value": 11.23},
        "mean_simple_regret": 11.1
      },
      "bayesian_optimization": {
        "seed_0": {"simple_regret": 0.003, "final_value": 0.003},
        "seed_1": {"simple_regret": 0.005, "final_value": 0.005},
        "mean_simple_regret": 0.004
      }
    },
    "rastrigin": {...},
    "ackley": {...},
    "rosenbrock": {...},
    "griewank": {...}
  },
  "results": [
    {"function": "sphere", "optimizer": "random_search", "seed": 0, "simple_regret": 10.97, "final_value": 10.97, "status": "ok"},
    {"function": "sphere", "optimizer": "random_search", "seed": 1, "simple_regret": 11.23, "final_value": 11.23, "status": "ok"},
    {"function": "sphere", "optimizer": "bayesian_optimization", "seed": 0, "simple_regret": 0.003, "final_value": 0.003, "status": "ok"},
    {"function": "sphere", "optimizer": "bayesian_optimization", "seed": 1, "simple_regret": 0.005, "final_value": 0.005, "status": "ok"},
    ...
  ]
}
```

### reviewer_notes.json

**Format:** JSON

**Schema:**
```json
{
  "reviewer": "reviewer_agent",
  "review_timestamp": "2026-08-14T12:45:00Z",
  "paper_id": "synthetic_graph or synthetic_optimize",
  "run_id": "<run_id>",
  "status": "approved | needs_fixes | failed",
  "code_review": {
    "completeness": "Engineer provided all required implementations",
    "correctness": "5/5 algorithms produce correct outputs",
    "quality": "Code is readable and follows specifications",
    "issues": []
  },
  "test_results": {
    "total_tests": 5,
    "passed": 5,
    "failed": 0
  },
  "metrics_validation": {
    "all_within_tolerance": true,
    "tolerance_percent": 2.0
  },
  "recommendations": [
    "Engineer's code is production-ready.",
    "Dijkstra implementation handles negative weights correctly."
  ],
  "summary": "All tests passed. Engineer demonstrated strong understanding of graph algorithms."
}
```

### run_metadata.json

**Format:** JSON

**Schema:**
```json
{
  "run_id": "<uuid>",
  "paper_id": "synthetic_graph or synthetic_optimize",
  "created_at": "2026-08-14T12:00:00Z",
  "completed_at": "2026-08-14T12:45:00Z",
  "elapsed_seconds": 2700,
  "engineer_version": "claude-3.5-sonnet",
  "reviewer_version": "claude-3.5-sonnet",
  "executor_version": "executor-v1",
  "docker_image": "research-assistant:latest",
  "environment": {
    "python_version": "3.11",
    "numpy_version": "1.24.0",
    "scikit_optimize_version": "0.9.0",
    "networkx_version": "3.1"
  },
  "exit_code": 0,
  "phase_results": {
    "setup": {"status": "ok", "elapsed_seconds": 30},
    "engineer_code": {"status": "ok", "elapsed_seconds": 120},
    "test_execution": {"status": "ok", "elapsed_seconds": 150},
    "review": {"status": "ok", "elapsed_seconds": 1400}
  }
}
```

## Individual Result Files

Each algorithm/run produces an individual JSON result file for audit purposes.

**Graph Algorithm Example (dfs.json):**
```json
{
  "algorithm": "dfs",
  "graph": "simple_undirected",
  "exit_code": 0,
  "output": "[0, 1, 3, 2]",
  "networkx_reference": "[0, 1, 3, 2]",
  "match": true,
  "execution_time_ms": 2.5
}
```

**Optimization Example (sphere_bayesian_seed0.json):**
```json
{
  "function": "sphere",
  "optimizer": "bayesian_optimization",
  "seed": 0,
  "n_iterations": 50,
  "n_initial_points": 10,
  "exit_code": 0,
  "simple_regret": 0.003,
  "final_value": 0.003,
  "best_iteration": 47,
  "execution_time_seconds": 120,
  "convergence_history": [10.2, 8.5, 5.1, ..., 0.003]
}
```

## Summary

- **Graph tests:** 5 independent algorithm runs, each with individual result file + aggregated metrics.json
- **Optimization tests:** 20 independent function+optimizer+seed runs, each with individual result file + aggregated metrics.json
- **Code artifacts:** Captured in generated_code.py (graph) or generated_code.zip (optimization)
- **Reviewer notes:** Separate validation report with pass/fail decision
- **Metadata:** Timestamps, versions, environment info for reproducibility
