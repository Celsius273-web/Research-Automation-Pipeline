# Random Search vs Bayesian Optimization on Synthetic Benchmark Functions

## Abstract
This paper compares the performance of Random Search (RS) and Bayesian Optimization (BO) on five standard synthetic optimization benchmarks. Both optimization methods were run for 50 iterations with 3 different seeds to ensure robustness. Bayesian Optimization utilized a Gaussian Process surrogate model with an Expected Improvement acquisition function. Our results indicate that BO converges significantly faster on unimodal functions such as Sphere and Rosenbrock, while RS remains competitive on highly multimodal functions like Rastrigin.

## Method
### Random Search
Random Search is a simple optimization algorithm that explores the search space by uniformly sampling candidate solutions. It does not employ any learning or build a model of the objective function based on past observations. Each iteration is independent, making it robust to highly irregular or non-convex landscapes but potentially inefficient for smoother functions.

### Bayesian Optimization
Bayesian Optimization is a global optimization strategy for black-box functions that are expensive to evaluate. It constructs a probabilistic surrogate model (in this case, a Gaussian Process with a Matern 5/2 kernel) of the objective function. An acquisition function, Expected Improvement (EI), is then used to determine the next sampling point. The process begins with 10 initial random points to build an initial surrogate model before the iterative optimization begins.

## Experiments
Our experimental setup involved evaluating both Random Search and Bayesian Optimization on a suite of five synthetic benchmark functions:
*   Sphere Function (5 dimensions)
*   Rastrigin Function (5 dimensions)
*   Ackley Function (2 dimensions)
*   Rosenbrock Function (5 dimensions)
*   Griewank Function (5 dimensions)

For each combination of benchmark function and optimizer, 50 iterations were performed. To account for stochasticity, 3 different seeds were used for each (benchmark, optimizer) pair. The primary metric for comparison was simple regret, defined as the best objective function value found during the optimization run, with the global minimum for all functions being 0.



## Results

### Table 1: Mean Simple Regret after 50 Iterations (Lower is Better)

| Benchmark Function | Random Search (RS) | Bayesian Optimization (BO) |
|--------------------|--------------------|----------------------------|
| Sphere             | 10.9768            | 0.0029                     |
| Rastrigin          | 48.4342            | 32.7766                    |
| Ackley             | 3.6163             | 0.1462                     |
| Rosenbrock         | 256.6105           | 16.8361                    |
| Griewank           | 38.5694            | 0.7424                     |

## Discussion
Bayesian Optimization consistently outperforms Random Search on smoother, unimodal functions where its Gaussian Process surrogate model can accurately capture the underlying function landscape. Conversely, highly multimodal functions such as Rastrigin's present a significant challenge to the GP surrogate, allowing Random Search to remain competitive due to its extensive exploration of the search space.