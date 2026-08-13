"""Run the synthetic optimization matrix and write aggregated CapturedMetric JSON."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).resolve().parent
CONFIG = {
    "functions": ["sphere", "rastrigin", "ackley", "rosenbrock", "griewank"],
    "optimizers": ["random_search", "bayesian_optimization"],
    "seeds": [0, 1, 2],
    "n_iter": 50,
    "output_dir": BENCHMARK_DIR / "results" / "synthetic_optimize",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(per_run: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in per_run:
        if "value" not in row:
            continue
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        key = (str(row.get("benchmark", "")), str(row.get("algorithm", "")), str(row.get("metric_name", "")))
        grouped[key].append(value)
    aggregated = []
    for (benchmark, algorithm, metric_name), values in sorted(grouped.items()):
        aggregated.append(
            {
                "benchmark": benchmark,
                "algorithm": algorithm,
                "metric_name": metric_name,
                "value": _mean(values),
                "source": "run_all.py",
            }
        )
    return aggregated


def run_all() -> list[dict]:
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run: list[dict] = []
    combos = list(product(CONFIG["functions"], CONFIG["optimizers"], CONFIG["seeds"]))
    for index, (func, opt, seed) in enumerate(combos, start=1):
        out_file = output_dir / f"{func}_{opt}_s{seed}.json"
        cmd = [
            sys.executable,
            str(BENCHMARK_DIR / "benchmark.py"),
            "--func",
            func,
            "--optimizer",
            opt,
            "--seed",
            str(seed),
            "--n_iter",
            str(CONFIG["n_iter"]),
            "--out",
            str(out_file),
        ]
        logger.info("[%s/%s] Running: %s", index, len(combos), " ".join(cmd))
        result = subprocess.run(cmd, cwd=BENCHMARK_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Command failed for %s %s seed=%s: %s", func, opt, seed, result.stderr.strip())
            continue
        if result.stdout.strip():
            logger.info(result.stdout.strip())
        if out_file.exists():
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                per_run.extend(item for item in payload if isinstance(item, dict))
    summary = _aggregate(per_run)
    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_all()


if __name__ == "__main__":
    main()
