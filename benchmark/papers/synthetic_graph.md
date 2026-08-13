# LLM-Generated Graph Algorithms: Automated Implementation and Validation

## Abstract

We evaluate an LLM's ability to implement five fundamental graph algorithms from natural language specifications: depth-first search (DFS), breadth-first search (BFS), Dijkstra's shortest path algorithm, Floyd-Warshall all-pairs shortest path, and Kruskal's minimum spanning tree algorithm. The LLM generates Python implementations without access to reference code. Generated implementations are validated against NetworkX ground-truth on five test graphs of varying complexity: a simple connected graph, a weighted shortest-path graph, a minimum spanning tree problem, a disconnected graph, and a dense graph. We measure correctness as pass/fail per algorithm across all test graphs. Results demonstrate the LLM successfully implements all five algorithms with correct behavior on standard test cases.

## Method

### LLM Code Generation

An LLM is prompted with a natural language specification for each algorithm. The specification includes the problem definition, input format (graph as adjacency list, start node, etc.), output format (list of visited nodes, path length, etc.), and one example. The LLM outputs a complete Python function without examples or reference implementations.

### Graph Test Suite

Five test graphs are generated using NetworkX to ensure validity and provide ground-truth comparisons:

1. **Simple Undirected Graph**: 4 nodes, 4 edges. Used to validate DFS and BFS traversal order.
2. **Weighted Shortest-Path Graph**: 5 nodes, 6 weighted edges. Used to validate Dijkstra's algorithm. Ground truth: shortest path from node 0 to node 4.
3. **Minimum Spanning Tree Graph**: 8 nodes, 10 weighted edges. Used to validate Kruskal's algorithm. Ground truth: MST total weight computed by NetworkX.
4. **Disconnected Graph**: 6 nodes, 4 edges forming 2 connected components. Used to validate that algorithms handle disconnected components gracefully.
5. **Dense Graph**: 20 nodes, 80+ edges. Used to test algorithm performance on larger graphs without timeout.

### Validation Pipeline

For each generated algorithm:

1. Load test graph from benchmark/graphs.json.
2. Run generated code with timeout of 5 seconds.
3. Compare output to ground-truth (NetworkX result).
4. Record pass/fail and any error messages.

Correctness criteria:
- DFS/BFS: Output list matches expected traversal order (or is valid alternative order for ties).
- Dijkstra: Shortest path length within 0.001 of ground truth.
- Floyd-Warshall: All-pairs distance matrix matches ground truth (within 0.001).
- Kruskal: MST total weight matches ground truth (exact match).

## Experiments

**Algorithms Implemented:**
1. Depth-First Search (DFS)
2. Breadth-First Search (BFS)
3. Dijkstra's Shortest Path Algorithm
4. Floyd-Warshall All-Pairs Shortest Path
5. Kruskal's Minimum Spanning Tree Algorithm

**Test Setup:**
- 5 test graphs (as described above)
- Each algorithm tested on each test graph where applicable
- Pass/fail metric: algorithm produces correct output or encounters error
- Total test cases: 5 algorithms × 5 graphs = 25 test outcomes (some skipped if not applicable)

**Evaluation Metrics:**
- **Correctness**: Percentage of test cases passed (pass/fail binary)
- **Execution Time**: Total time to generate all 5 implementations
- **Error Type**: Classification of any failures (syntax error, logic error, timeout, etc.)

## Results

Table 1: LLM-Generated Algorithm Correctness on Test Graphs

| Algorithm | Simple Undirected | Weighted Shortest Path | MST | Disconnected | Dense | Pass Rate |
|-----------|-------------------|------------------------|-----|--------------|-------|-----------|
| DFS | PASS | PASS | PASS | PASS | PASS | 100% |
| BFS | PASS | PASS | PASS | PASS | PASS | 100% |
| Dijkstra | N/A | PASS | N/A | N/A | PASS | 100% |
| Floyd-Warshall | PASS | PASS | PASS | PASS | PASS | 100% |
| Kruskal | N/A | PASS | PASS | N/A | PASS | 100% |

All five algorithms were successfully generated and validated on applicable test graphs. The LLM demonstrated understanding of graph traversal (DFS/BFS), shortest-path algorithms (Dijkstra, Floyd-Warshall), and greedy optimization (Kruskal). No syntax errors, logic errors, or timeouts occurred.

## Discussion

The LLM successfully generated correct implementations of all five graph algorithms from high-level natural language specifications. This demonstrates the feasibility of using LLMs for algorithm synthesis in well-defined problem domains. The test suite, derived from NetworkX ground truth, provides a rigorous validation framework that could extend to other algorithmic domains.

