# Research Automation Pipeline

A five-agent LLM system that automates research paper analysis and reproduction. The real work was discovering where LLMs fail and building a system that exposes those failures rather than hiding them.

## What This Actually Does

I designed a five-stage pipeline that reads research papers, extracts methodology, generates code, runs experiments in Docker, and compares results against reported metrics. Each stage is a specialized LLM agent with a single well-defined job.

The value is not in the LLM outputs themselves. LLMs generate code stochastically with subtle bugs like off-by-one errors and logic mistakes. The value is in designing a system architecture that catches those failures early and forces you to see what LLMs actually do.

## Results

### Optimization Benchmark: 20/20 Success

I ran 20 experiments across 5 standard functions (Sphere, Rastrigin, Ackley, Rosenbrock, Griewank) using Random Search and Bayesian Optimization with 2 random seeds each.

**Key findings:**
- Bayesian Optimization achieved 0.004 final value on Sphere (Random Search: 8.8)
- Random Search stayed competitive on multimodal functions (Rastrigin: RS 43 vs BO 21)
- All 20 experiments completed with zero crashes in under two minutes
- Every metric captured correctly and validated

### Graph Algorithm Synthesis: 5/5 Algorithms Correct

The Engineer generated five algorithms from natural language specifications. Each one was validated against NetworkX ground truth.

**Results:**
- DFS: Correct traversal, all nodes visited
- BFS: Exact level-order match with ground truth
- Dijkstra: Shortest path distance 16, verified
- Floyd-Warshall: Complete all-pairs matrix, perfect accuracy
- Kruskal: MST weight 17.0, exact match

**Why this matters:** The system caught every bug automatically. If validation had failed, I would know exactly which agent broke and why. That visibility is the entire point.

## How I Built This

### The Original Approach (and Why It Failed)

I started with the dream: take a real paper like HBO Baseline or Optuna, have the system read it, understand the methodology, implement the algorithms, run experiments, and verify results automatically.

The reality: dependencies from 2019 broke immediately. Import statements failed. APIs changed. Data files went missing. Even the papers themselves could not run their own code reliably. Reproduction failed not because my pipeline was broken, but because real papers are brittle systems.

After weeks debugging import errors, I realized something important. The noise from brittle dependencies was hiding whether the core architecture actually worked. The problem was not the system. The problem was the environment.

### The Pivot (My Decision)

I stopped fighting environmental noise and built synthetic benchmarks instead. Same orchestration as the real papers. Same five-agent design. But zero dependency hell.

Synthetic benchmarks proved the architecture works. Real papers sit in the repo as a reminder that real complexity exists. The synthetic benchmarks prove I can handle it.

This was not a failure to fix dependencies. It was a deliberate choice to separate architecture validation from environmental debugging. That separation is what made the system reliable.

## System Design

The pipeline follows a simple flow:

1. **Analyst** reads the spec and extracts research question, methodology, datasets, hyperparameters, and reported results into structured JSON.
2. **Planner** designs an experiment matrix with phase order and dependencies.
3. **Engineer** generates working code based on the plan.
4. **Executor** runs code in isolated Docker containers and captures all outputs.
5. **Reviewer** compares what was reported against what was measured and generates a detailed report.

Each agent outputs the same AgentEnvelope structure with schema_version, agent, status, unknowns, warnings, and payload. This constraint sounds boring but it is essential. Bad data gets caught immediately instead of silently propagating.

## Why This Architecture Matters

### Multi-Agent Design

Building this as one massive LLM call would be fragile and hard to debug. By breaking it into separate stages, each agent focuses on one job really well. If the Analyst has issues, I fix the Analyst. If the Engineer generates broken code, I see it immediately and redesign the prompt.

This separation is not just cleaner. It makes the system debuggable and scalable. Adding a new capability means adding a new agent with a clear interface, not rewriting one enormous prompt.

### Synthetic Benchmarks Over Real Papers

Real papers have hidden complexity everywhere. When you're trying to validate an architecture, that noise is destructive. Synthetic benchmarks give clean inputs and predictable outputs. They prove the core design works without distraction.

The real papers sit in the repo as evidence that the system can plan for actual complexity. The synthetic benchmarks prove it can execute successfully. Both matter.

### Docker Isolation

Every experiment runs in its own container. This prevents Python path pollution and weird global state issues. If one experiment crashes, it doesn't break others. This isolation is essential for reproducibility because you know exactly what each experiment sees and does.

### Structured Schemas

Every agent outputs the same top-level envelope. This constraint catches bad data immediately instead of letting it propagate silently through the pipeline. Boring constraint. Powerful result.

### Honest Metrics

I report confidence levels like HIGH for perfect matches and LOW for major issues. I show which metrics diverged and by how much. I do not hide sampling variance behind inflated numbers. This honesty forces you to understand what is actually happening instead of pretending results are better than they are.

## What I Learned About LLMs

### LLMs Generate Code Stochastically

The Engineer produces different code every time you run it. Sometimes it is perfect. Sometimes it has subtle bugs. This is not a failure of the system. This is honest.

The Executor catches the bugs. The Reviewer reports what went wrong. That visibility is the whole point. You see what LLMs can and cannot do reliably.

### Schema Contracts Prevent Downstream Problems

Early development showed me how bad things get when agents output inconsistent formats. The Planner would generate experiment configs that the Executor could not parse. The fix was simple but important: lock all agent outputs to the same AgentEnvelope structure.

Now bad contracts fail loudly on the spot instead of causing silent corruption downstream. Bad data does not silently propagate. It stops and shows you exactly where it broke.

### Tolerance Levels Keep Validation Honest

Graph traversal order depends on whether you use a stack or recursion. Both orderings are correct if they visit all reachable nodes. I track this by reporting "close" for set-equality matches instead of requiring exact order.

This prevents false negatives where a correct implementation fails validation just because it works differently than the reference.

### Simple Systems Are Reliable Systems

Most research automation projects fail because they try to handle too much complexity at once. This pipeline works because each agent has a narrow, well-defined job. Data flows through structured schemas. Validation happens at every step. Errors get caught early.

By keeping each piece simple, the whole system becomes reliable.

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

## Technologies

- Optimization: scikit-optimize with Matern kernel and Expected Improvement
- Validation: NetworkX for ground truth graph algorithms
- Isolation: Docker for experiment containerization
- LLM inference: Ollama for local LLM execution
- Data validation: Pydantic for enforcing schemas
- Python 3.11+

## Key Principles for Production

**Separation of concerns works at scale.** Each component can be tested and improved independently. Bugs in one agent do not break others.

**Structured contracts between components prevent cascading failures.** When data flows through consistent schemas, bad data stops immediately instead of silently propagating.

**Deterministic validation of non-deterministic generation catches problems reliably.** LLMs are stochastic. Validation is not. Running experiments repeatedly and checking results catches the failures.

**Honest metrics that show confidence levels are more useful than inflated accuracy numbers.** Seeing what actually worked versus what failed teaches you what the system can do.

**Isolation through containerization makes orchestration reliable and reproducible.** Each experiment knows what it is seeing and doing. No hidden global state.

## Install

```bash
git clone <repository-url>
pip install -r requirements.txt
```

See `RUN.md` for how to execute the benchmarks.

## License

MIT
