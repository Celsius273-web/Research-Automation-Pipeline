
import json
import os
import numpy as np

results_dir = "results/synthetic_benchmark"
functions = ["sphere", "rastrigin", "ackley", "rosenbrock", "griewank"]
optimizers = ["random_search", "bayesian_optimization"]
seeds = [0, 1, 2]

extracted_results = {}

for func in functions:
    extracted_results[func] = {}
    for opt in optimizers:
        regrets = []
        for seed in seeds:
            filename = f"{func}_{opt}_s{seed}.json"
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                for item in data:
                    if item["metric_name"] == "simple_regret":
                        regrets.append(item["value"])
                        break
        extracted_results[func][opt] = np.mean(regrets)

# Format for paper.md
# [SPHERE_RS], [SPHERE_BO], etc.
markdown_placeholders = {}
for func in functions:
    for opt in optimizers:
        key = f"[{func.upper()}_{opt.replace('random_search', 'RS').replace('bayesian_optimization', 'BO')}]"
        markdown_placeholders[key] = f"{extracted_results[func][opt]:.4f}"

# Read paper.md
with open("benchmark/paper.md", "r") as f:
    paper_content = f.read()

# Replace placeholders
for placeholder, value in markdown_placeholders.items():
    paper_content = paper_content.replace(placeholder, value)

with open("benchmark/paper.md", "w") as f:
    f.write(paper_content)

print("Paper.md updated successfully with benchmark results.")
