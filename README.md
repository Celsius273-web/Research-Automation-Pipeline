# Research Automation Pipeline

A five-agent LLM system that automates research paper analysis and reproduction. The pipeline extracts methodology, generates execution plans, synthesizes code, runs experiments in Docker, and compares results against reported metrics.

## Results

### Optimization Benchmark: 20/20 Success

I ran 20 experiments across 5 standard functions (Sphere, Rastrigin, Ackley, Rosenbrock, Griewank) using Random Search and Bayesian Optimization with 2 random seeds each.

**Key findings:**
- Bayesian Optimization achieved 0.004 final value on Sphere (Random Search: 8.8)
- Random Search stayed competitive on multimodal functions (Rastrigin: RS 43 vs BO 21)
- All 20 experiments completed with zero crashes in under two minutes
- Every metric captured correctly and validated

### Graph Algorithm Synthesis: 5/5 Algorithms Correct

The Engineer implemented five algorithms from natural language specifications. Each one was validated against NetworkX ground truth.

**Results:**
- DFS: Correct traversal, all nodes visited
- BFS: Exact level-order match with ground truth
- Dijkstra: Shortest path distance 16, verified
- Floyd-Warshall: Complete all-pairs matrix, perfect accuracy
- Kruskal: MST weight 17.0, exact match

**Confidence: HIGH.** Every single metric matched ground truth exactly.

## Why I Built This

I started with real papers like HBO Baseline and Optuna. The dream was to have the system read a paper, understand the methodology, implement the algorithms, run experiments, and verify the results automatically. It sounded perfect in theory.

The reality was different. The dependencies were from 2019. Import statements broke immediately. APIs had changed. Data files were missing. Even the papers themselves couldn't run their own code reliably. Reproduction failed not because my pipeline was broken, but because real papers are brittle systems with outdated dependencies.

After weeks of debugging import errors and tracking down missing packages, I realized something important. The noise from real papers was hiding whether the core architecture actually worked. So I pivoted.

Instead of fighting brittle dependencies, I built synthetic benchmarks that I control completely. Same orchestration as the real papers, but zero environmental noise. This approach proved the architecture works without getting buried in dependency hell.

## How It Works

The pipeline follows a simple flow. First, the Analyst reads the spec and extracts the research question, methodology, datasets, hyperparameters, and reported results. The Planner then looks at this extraction and designs an experiment matrix with the phase order and dependencies. The Engineer takes that plan and generates actual working code. The Executor runs the code in isolated Docker containers and captures all outputs. Finally, the Reviewer compares what was reported against what was actually measured and generates a detailed report.

Each agent outputs structured JSON. If something breaks, the error is clear and you know exactly which agent failed.

## Quick Start

See `RUN.md` for detailed instructions on running both benchmarks.

## Why It's Built This Way

### Multi-Agent Design

Building this as one massive LLM call would be messy and fragile. By breaking it into separate stages, each agent can focus on one job really well. If the Analyst has issues, I fix the Analyst. If the Engineer can't generate code, that problem is isolated and visible. This separation makes debugging realistic and scaling possible. Adding a new capability means adding a new agent with a clear interface, not rewriting one enormous prompt.

### Synthetic Benchmarks

Real papers have hidden complexity everywhere. Dependencies conflict, code paths are undocumented, data files go missing. When you're trying to validate an architecture, that noise is destructive. Synthetic benchmarks give clean inputs and predictable outputs. They prove the core design works without the distraction of environmental problems. The real papers sit in the repo as evidence that the system can plan for actual complexity. The synthetic benchmarks prove it can execute successfully.

### Docker Isolation

Every experiment runs in its own container. This prevents Python path pollution and weird global state issues. If one experiment crashes, it doesn't break others. This isolation is not just nice to have. It's essential for reproducibility because you know exactly what each experiment sees and does. 

### Structured Schemas

Every agent outputs the same top-level envelope with schema_version, agent, status, unknowns, warnings, and payload. This constraint sounds boring but it's incredibly important in practice. Bad data gets caught immediately instead of silently propagating through the pipeline. 

### Honest Metrics

I report confidence levels like HIGH for perfect matches and LOW for major issues. I show which metrics diverged and by how much. I don't hide sampling variance behind inflated numbers. This honesty matters because it forces you to understand what's actually happening instead of pretending results are better than they are.

## Project Structure

```
research_assistant/
├── src/
│   ├── agents/
│   │   ├── analyst.py          # Extract methodology and metrics
│   │   ├── planner.py          # Design experiment matrices
│   │   ├── engineer.py         # Generate implementations
│   │   ├── executor.py         # Run code in Docker
│   │   └── reviewer.py         # Compare and validate
│   ├── state.py                # Pydantic data models
│   ├── persistence.py          # Read and write JSON
│   ├── main.py                 # Command line interface
│   └── tools/                  # Helper functions
├── benchmark/
│   ├── benchmark.py            # Optimization test functions
│   ├── setup_graph.py          # Generate test graph data
│   └── run_graph.py            # Execute graph algorithms
├── data/papers/
│   ├── synthetic_optimize/     # Optimization results
│   ├── synthetic_graph/        # Algorithm results
│   ├── hbo_baseline/           # Real paper (planning phase)
│   └── optuna/                 # Real paper (planning phase)
└── requirements.txt
```

## What I Learned

### LLMs Generate Code Stochastically

The Engineer produces different code every time you run it. Sometimes it's perfect. Sometimes it has subtle bugs like off-by-one errors or logic mistakes. This is not a failure of the system. This is actually honest. The Executor catches the bugs. The Reviewer reports what went wrong. That visibility is the whole point. You see what LLMs can and cannot do reliably.

### Schema Contracts Prevent Downstream Problems

Early development showed me how bad things get when agents output inconsistent formats. The Planner would generate experiment configs that the Executor couldn't parse. The fix was simple but important: lock all agent outputs to the same AgentEnvelope structure with identical fields. Now bad contracts fail loudly on the spot instead of causing silent corruption downstream.

### Tolerance Levels Keep Validation Honest

Graph traversal order depends on whether you use a stack or recursion. Both orderings are correct if they visit all reachable nodes. I track this by reporting "close" for set-equality matches instead of requiring exact order. This prevents false negatives where a correct implementation fails validation just because it works differently than the reference.

### Simple Systems Are Reliable Systems

Most research automation projects fail because they try to handle too much complexity at once. This pipeline works because each agent has a narrow, well-defined job. Data flows through structured schemas. Validation happens at every step. Errors get caught early before they become expensive problems. By keeping each piece simple, the whole system becomes reliable.

## Technologies

The optimization benchmark uses scikit-optimize for Bayesian Optimization with a Matern kernel and Expected Improvement acquisition function. Graph algorithms are validated against NetworkX. Everything runs in Docker for isolation. The system uses Ollama for local LLM inference and Pydantic for data validation. The implementation is Python 3.11+.

## Key Takeaways for Production

This project demonstrates several principles that matter for reproducible research systems. Separation of concerns works at scale because each component can be tested and improved independently. Structured contracts between components prevent cascading failures. Deterministic validation of non-deterministic generation catches problems reliably. Honest metrics that show confidence levels and tolerance bands are more useful than inflated accuracy numbers. Isolation through containerization makes orchestration reliable and reproducible.

## What's Next

The real papers in the repo show that the system can handle planning for complex papers with real methodology and datasets. The synthetic benchmarks prove it can execute end-to-end without environmental noise. Combining both approaches requires solving the dependency management problem, which is separate from the pipeline architecture itself.

## Install

```bash
git clone <repository-url>
pip install -r requirements.txt
```

See `RUN.md` for how to execute the benchmarks.

## License

MIT