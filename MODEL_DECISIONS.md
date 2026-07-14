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
