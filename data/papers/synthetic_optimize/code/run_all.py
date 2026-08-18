#!/usr/bin/env python
"""Reference optimization CLI for synthetic_optimize plan."""

from __future__ import annotations

import argparse
import json

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

FUNCTIONS = {
    "sphere": {"dim": 5, "bounds": (-5.12, 5.12)},
    "rastrigin": {"dim": 5, "bounds": (-5.12, 5.12)},
    "ackley": {"dim": 2, "bounds": (-5, 5)},
    "rosenbrock": {"dim": 5, "bounds": (-2, 2)},
    "griewank": {"dim": 5, "bounds": (-600, 600)},
}


def sphere(x: list[float] | np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sum(x**2))


def rastrigin(x: list[float] | np.ndarray) -> float:
    x = np.asarray(x)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: list[float] | np.ndarray) -> float:
    x = np.asarray(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x**2)))
        - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x)))
        + 20
        + np.e
    )


def rosenbrock(x: list[float] | np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def griewank(x: list[float] | np.ndarray) -> float:
    x = np.asarray(x).flatten()
    return float(1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))))


def random_search(
    func,
    dim: int,
    bounds: tuple[float, float],
    n_iterations: int = 50,
    seed: int | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    low, high = bounds
    best_value = float("inf")
    for _ in range(n_iterations):
        x = rng.uniform(low, high, size=dim)
        value = float(func(x))
        if value < best_value:
            best_value = value
    return best_value


def bayesian_optimization(
    func,
    dim: int,
    bounds: tuple[float, float],
    n_calls: int = 50,
    n_initial_points: int = 10,
    seed: int | None = None,
) -> float:
    low, high = bounds
    space = [Real(low, high) for _ in range(dim)]
    result = gp_minimize(
        func,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=seed,
    )
    return float(result.fun)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimization experiments.")
    parser.add_argument("--function", required=True, choices=sorted(FUNCTIONS))
    parser.add_argument(
        "--optimizer",
        required=True,
        choices=["random_search", "bayesian_optimization"],
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    spec = FUNCTIONS[args.function]
    func = globals()[args.function]
    if args.optimizer == "random_search":
        final_value = random_search(func, spec["dim"], spec["bounds"], seed=args.seed)
    else:
        final_value = bayesian_optimization(func, spec["dim"], spec["bounds"], seed=args.seed)

    print(
        json.dumps(
            {
                "function": args.function,
                "optimizer": args.optimizer,
                "seed": args.seed,
                "simple_regret": final_value,
                "final_value": final_value,
            }
        )
    )


if __name__ == "__main__":
    main()
