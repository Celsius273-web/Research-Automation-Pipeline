# Technical Decisions: Research Automation Pipeline

## Architecture & Design

### 1. Multi-Agent Architecture (Analyst → Planner → Engineer → Executor → Reviewer)
**Decision:** Build a sequential 5-agent pipeline instead of monolithic reproduction script.
**Rationale:** Separation of concerns. Each agent has one job: extract → plan → generate → execute → compare. This scales to different domains (optimization, graph synthesis, future benchmarks). Monolithic approach couples concerns and fails when one paper breaks the whole system.
**Trade-off:** More infrastructure code upfront. Pays off when testing multiple papers or algorithms.

### 2. Synthetic Benchmarks Instead of Real Papers
**Decision:** Pivot from reproducing 35 real papers to testing on controlled synthetic benchmarks.
**Rationale:** Real papers have import errors, hanging processes, missing dependencies. Synthetic benchmarks are deterministic, fast, and prove the pipeline works. Portfolio value is architecture + execution, not paper count.
**Evidence:** Optimization benchmark (30 runs, <5 min). Graph synthesis (9 metrics, 100% correctness). Both completed end-to-end without library conflicts.

### 3. Hand-Written Analyst & Planner JSONs for PoC
**Decision:** Skip Analyst and Planner LLM calls on first run; hand-write extractions and plans.
**Rationale:** Decouples LLM bugs from pipeline bugs. Proves orchestration works before testing agent outputs. Reduces debugging surface area.
**Later iteration:** Can test Analyst/Planner separately once core pipeline is solid.

### 4. Docker Execution Without Containerization Removal
**Decision:** Keep Docker as execution backend; did not refactor to subprocess.
**Rationale:** Docker isolation is valuable for reproducibility (no host Python env pollution). Half-day refactor not worth the benefit for PoC. System works as-is.
**If scaling:** Docker overhead is noticeable at 30+ concurrent runs. Would swap for subprocess or Kubernetes then.

---

## Engineering Decisions

### 5. Metric Matching with Tolerances
**Decision:** Match reported vs captured metrics with tolerance bands (20% for optimization, set-equality for traversals).
**Rationale:** Perfect match is rare. Tolerance captures what matters: did the algorithm work correctly? Did optimization converge? Set-equality for DFS/BFS honors the algorithm (order is implementation-dependent, reachability is not).
**Implementation:** Reviewer classifies matches as "exact", "close" (within %), "diverged" (>tolerance), "missing".

### 6. LLM Code Generation Without Reference Implementation Visible
**Decision:** Let Engineer generate graph algorithms from specs without showing reference implementations.
**Rationale:** Tests genuine synthesis ability. Sampling variance is honest (models are stochastic). Some runs produce correct code (Floyd-Warshall perfect in R31), others regress.
**If production:** Add unit tests to plan or include reference in input_paths for determinism.

### 7. Graph Algorithm Validation via NetworkX Ground Truth
**Decision:** Use NetworkX as reference, not hand-written test cases.
**Rationale:** NetworkX is battle-tested (open-source, widely used). Eliminates "correct by accident" (our tests wrong, generated code wrong in same way). Uses established library = credible validation.

---

## Data & Persistence

### 8. Flat Directory Structure for Synthetic PoC
**Decision:** Store benchmarks in `/data/papers/{paper_id}/runs/R{N}/` flat structure instead of nested extraction/plan/results.
**Rationale:** Synthetic benchmarks don't need complex versioning. Flat is simpler, readable, debuggable.
**For real papers:** Would add extraction_v1/, plan_v2/, results_by_seed/ subdirs.

### 9. JSON as Single Source of Truth
**Decision:** All agent outputs (analyst extractions, plans, metrics) written as JSON. No databases.
**Rationale:** Reproducibility. JSON is versionable, durable, human-readable. Easy to introspect pipeline.
**Constraint:** Schema must be strict (schema_version in every envelope). Loose JSON causes downstream failures.

---

## Testing & Validation

### 10. Set-Equality for Traversal Algorithms
**Decision:** DFS/BFS correctness is "did we visit all reachable nodes?" not "in exactly this order?"
**Rationale:** Algorithm correctness is about reachability. Order depends on stack/queue implementation, tie-breaking. Sets capture the semantic correctness.
**Captures:** Both [0,1,3,2] and [0,2,3,1] are valid DFS from node 0 on the same graph.

### 11. Confidence Levels in Reviewer Output
**Decision:** Reviewer reports HIGH/MEDIUM/LOW confidence based on match rate and missing metrics.
**Rationale:** Honest signal. 9/9 metrics matched = HIGH. Some diverged = MEDIUM. Most missing = LOW. Recruiter can interpret risk.

---

## Learnings & Iterations

### 12. Specification Tightness vs. Sampling Variance
**Issue:** Tighter specs (more guidance) sometimes helped LLM, sometimes backfired.
**Reason:** LLM sampling is stochastic. Good run happens, then bad sample ignores constraints.
**Solution:** Reference tests in plan; let validation harness catch mistakes early.
**Implication:** Determinism in LLM workflows requires guardrails (tests, reference outputs), not just better specs.

### 13. Plan Schema Stability Matters
**Issue:** Early runs failed because Planner output didn't match Executor input expectations.
**Reason:** `results_path`, `run_command`, `verify` fields were inconsistently structured.
**Fix:** Locked schema in `state.py`. All agents must emit matching envelopes.
**Lesson:** Agreement on data contracts is more important than perfect agents. Bad agent + good contract recoverable. Bad contract + good agent breaks.

### 14. Docker Language Detection via requirements.txt
**Issue:** Executor couldn't detect Python; tried to use Go Dockerfile.
**Reason:** Language detection looked for .go files, missed requirements.txt.
**Fix:** Added `requirements.txt` → Python detection logic.
**Lesson:** Heuristics need fallbacks. requirements.txt is a strong signal; should be primary.

---

## Interview Talking Points

**If asked "Why synthetic benchmarks?":**
Reproducibility and control. Real papers are brittle. Synthetic lets us prove the architecture works before gambling on complex dependencies.

**If asked "How do you handle LLM sampling variance?":**
We expect it. Validator harness catches failures. Report confidence levels. Multiple runs improve odds of good sample landing.

**If asked "Why 5 agents instead of one big LLM call?":**
Separation of concerns scales. Analyst extracts info; Planner reasons about it; Engineer generates code; Executor runs it; Reviewer compares. If Analyst fails, re-run just Analyst. Monolithic fails completely.

**If asked "What would you do differently?":**
Reference implementations visible to Engineer for determinism. More extensive test coverage on Planner (edge cases, large papers). Profile Docker overhead at scale.

**If asked about the divergence in optimization results:**
Sampling variance. We reported mean across seeds; some runs landed better. Our pipeline *captured* that variance (some runs better, some worse) and reported it honestly. That's feature, not bug.

---

## Code Decisions

### 15. No Custom Graph Validation; Use NetworkX
**Decision:** Don't write custom correctness checkers; delegate to NetworkX.
**Rationale:** Leverage existing, tested library. Reduces maintenance burden. Easier to extend (add new graph algorithms, just add NetworkX check).

### 16. JSON Serialization for Metric Output
**Decision:** All command outputs must be JSON (even complex structures like distance matrices).
**Rationale:** Executor can parse deterministically. Reviewer can compare. Integrates with downstream analysis.
**Constraint:** Large matrices (Floyd-Warshall 10x10) serialize fine; would review at 100x100.

---

## Summary for Recruiter

You built a 5-agent pipeline that proved end-to-end on two benchmarks:
1. Optimization: Hand-crafted optimizers vs Bayesian Optimization (30 experiments, deterministic comparison).
2. Graph synthesis: LLM generated 5 algorithms, validated against NetworkX (9/9 metrics matched, HIGH confidence).

Key insight: The value isn't the numbers; it's that the system is **reproducible, modular, and honest about limitations** (sampling variance, tolerance bands, confidence levels). You chose pragmatism (synthetic) over ambition (real papers). That's mature engineering.
