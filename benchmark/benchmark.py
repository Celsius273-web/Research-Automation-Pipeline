import numpy as np

def ackley(x):
    x = np.asarray(x)
    arg1 = -0.2 * np.sqrt(0.5 * np.sum(x**2))
    arg2 = 0.5 * np.sum(np.cos(2. * np.pi * x))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
from skopt import gp_minimize
from skopt.space import Real
import argparse
import json
import logging
import os

# --- Test Functions ---
def sphere(x):
    x = np.asarray(x)
    return np.sum(x**2)

def rastrigin(x):
    x = np.asarray(x)
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley_2d(x):
    x = np.asarray(x)
    if len(x) != 2:
        raise ValueError("Ackley function is implemented for 2 dimensions only.")
    return ackley(x)

def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def griewank(x):
    x = np.asarray(x).flatten()
    s = np.sum(x**2 / 4000)
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return s - p + 1

# --- Optimizers ---
def random_search(func, domain, n_iter, seed):
    np.random.seed(seed)
    best_value = np.inf
    best_x = None
    regret_curve = []

    for _ in range(n_iter):
        x = [np.random.uniform(low, high) for low, high in domain]
        value = func(np.array(x))
        if value < best_value:
            best_value = value
            best_x = x
        regret_curve.append(best_value)
    return best_value, regret_curve, best_x

def bayesian_optimization(func, domain, n_iter, seed):
    space = [Real(low, high) for low, high in domain]
    
    res = gp_minimize(
        func,
        space,
        n_calls=n_iter,
        random_state=seed,
        verbose=False
    )
    
    best_value = res.fun
    regret_curve = np.minimum.accumulate(res.func_vals)
    best_x = res.x
    
    return best_value, regret_curve.tolist(), best_x

# --- Main Logic ---
def run_benchmark(func_name, optimizer_name, seed, n_iter, out_path):
    # Define functions and their domains
    functions = {
        "sphere": {"func": sphere, "domain": [(-5.12, 5.12)] * 5},
        "rastrigin": {"func": rastrigin, "domain": [(-5.12, 5.12)] * 5},
        "ackley": {"func": ackley_2d, "domain": [(-5, 5)] * 2},
        "rosenbrock": {"func": rosenbrock, "domain": [(-2, 2)] * 5},
        "griewank": {"func": griewank, "domain": [(-600, 600)] * 5},
    }
    
    optimizers = {
        "random_search": random_search,
        "bayesian_optimization": bayesian_optimization,
    }

    func_info = functions[func_name]
    optimizer_func = optimizers[optimizer_name]

    final_value, regret_curve, _ = optimizer_func(func_info["func"], func_info["domain"], n_iter, seed)

    simple_regret = final_value  # Global minimum is 0
    auc_regret = np.sum(regret_curve) / n_iter

    # Prepare output
    output_data = [
        {"benchmark": func_name, "algorithm": optimizer_name, "metric_name": "final_value",
         "value": final_value, "source": "benchmark.py"},
        {"benchmark": func_name, "algorithm": optimizer_name, "metric_name": "simple_regret",
         "value": simple_regret, "source": "benchmark.py"},
        {"benchmark": func_name, "algorithm": optimizer_name, "metric_name": "auc_regret",
         "value": auc_regret, "source": "benchmark.py"}
    ]

    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logging.info("Done: %s %s seed=%s final=%s", func_name, optimizer_name, seed, final_value)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run optimization benchmarks.")
    parser.add_argument("--func", type=str, required=True,
                        choices=["sphere", "rastrigin", "ackley", "rosenbrock", "griewank"],
                        help="Function to optimize.")
    parser.add_argument("--optimizer", type=str, required=True,
                        choices=["random_search", "bayesian_optimization"],
                        help="Optimizer to use.")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for reproducibility.")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="Number of iterations/evaluations.")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CapturedMetric JSON path (default: results/<func>_<optimizer>_s<seed>.json).",
    )
    args = parser.parse_args()

    if args.out:
        out_path = args.out
    else:
        output_dir = "results"
        output_filename = f"{args.func}_{args.optimizer}_s{args.seed}.json"
        out_path = os.path.join(output_dir, output_filename)

    run_benchmark(args.func, args.optimizer, args.seed, args.n_iter, out_path)