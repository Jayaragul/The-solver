# SIHPS — Claude Opus Engineering Brief

## Mission

Transform SIHPS from a strong research LP foundation into a rigorously benchmarked, numerically reliable, high-performance optimization engine for:

- Linear Programming (LP)
- Mixed-Integer Linear Programming (MILP)
- Later: convex Quadratic Programming (QP)

The goal is not to claim superiority through isolated kernel timings. The goal is to increase, in this order:

1. Correctness.
2. Number of solvable instances.
3. End-to-end speed.
4. Scalability.
5. Quality on refinery-like workloads.

Do not optimize a component unless its contribution to total solve time or solvability has been measured.

## Current project assessment

SIHPS already contains a serious LP foundation:

- MPS parsing.
- CSR and CSC sparse matrix structures.
- Presolve reductions.
- Ruiz scaling.
- Primal simplex.
- Dual simplex implementation.
- Sparse basis factorization and eta updates.
- Devex-style pricing.
- CPU parallel paths.
- CUDA/cuSPARSE integration.
- GPU pricing experiments.
- GPU PDLP-style first-order solving.
- Numerical residual verification.
- Netlib objective validation.
- Unit tests for sparse matrices, memory, CUDA, scaling, presolve, simplex, and PDLP.
- Research and architecture documentation.

Important project references:

- [System architecture](docs/architecture/SYSTEM.md)
- [LP engine architecture](docs/architecture/LP.md)
- [MILP architecture](docs/architecture/MILP.md)
- [Numerical policy](docs/architecture/NUMERICS.md)
- [CPU/GPU division](docs/architecture/CPU_GPU.md)
- [Memory architecture](docs/architecture/MEMORY.md)
- [PDLP design](docs/architecture/PDLP.md)
- [Research report](docs/research/SOTA.md)
- [Benchmark measurements](docs/measurements/README.md)
- [Netlib validation benchmark](benchmarks/validate_netlib.cpp)
- [LP solve benchmark](benchmarks/bench_lp_solve.cpp)
- [PDLP benchmark](benchmarks/bench_pdlp.cpp)

The central limitation is completeness. The original objective includes LP, MILP, and QP, but the implementation is currently primarily an LP solver with GPU/PDLP experimentation. MILP branch-and-bound, warm-started child solves, cut generation, and QP are not yet complete.

The current Netlib result of 92/93 validated instances is a good foundation, but `dfl001` remains an iteration-limit failure. The benchmark code also documents a dense `O(m^2)`-per-iteration limitation in the current simplex path. These are the two most important LP engineering problems to attack.

## Non-negotiable engineering principles

### Correctness comes before speed

Never report `OPTIMAL` merely because an iteration limit or termination flag was reached. Every accepted solution must pass independent verification.

Every optimization must preserve:

- Primal feasibility.
- Dual feasibility where applicable.
- Bound feasibility.
- Integrality feasibility for MILP.
- Objective correctness.
- Determinism under fixed configuration.

### Algorithmic improvements come before CUDA micro-optimizations

The largest likely gains are:

1. Warm-started dual simplex.
2. Complete and safe presolve.
3. Sparse/hyper-sparse basis operations.
4. Correct MILP branch-and-bound.
5. Branching, heuristics, and cuts that reduce node count.
6. Only then, CUDA optimization of measured bottlenecks.

### No unsupported claims

Every report must distinguish:

- `IMPLEMENTED`
- `MEASURED`
- `ESTABLISHED METHOD`
- `ENGINEERING DECISION`
- `RESEARCH HYPOTHESIS`
- `KNOWN LIMITATION`

Do not claim refinery-specific numerical properties unless they are measured on representative refinery models.

## Benchmark system to build first

Before major algorithmic or CUDA work, upgrade the benchmark infrastructure.

### Reproducibility metadata

Every benchmark result must record:

```text
git commit
compiler and compiler version
CUDA version
GPU model and compute capability
GPU driver
CPU model
RAM
thread count
OpenMP settings
GPU clock/power mode if known
solver options
presolve option
scaling option
pricing rule
LP algorithm
instance path and hash
objective
status
wall time
iterations
refactorizations
primal residual
dual residual
peak host memory
peak GPU memory
```

Use one process per experiment. Do not run builds or other benchmarks concurrently with the measured process. Report median, minimum, maximum, and variation over repeated runs.

### Benchmark layers

#### Layer A — microbenchmarks

Measure independently:

- CSR SpMV.
- CSC transpose SpMV.
- Sparse triangular solve.
- FTRAN.
- BTRAN.
- Reduced-cost pricing.
- Ratio test.
- Devex update.
- Basis refactorization.
- Eta application.
- Presolve passes.
- GPU vector updates.
- GPU reductions.
- Host/device transfer.
- Synchronization latency.

Record both throughput and latency. A kernel that is fast in isolation may lose in the solver because of synchronization or transfer overhead.

#### Layer B — LP benchmarks

Use:

- Netlib LP.
- Kennington LP.
- Larger public LP collections.
- LP relaxations extracted from MIPLIB.

Netlib is required but not sufficient. It is a useful historical regression set, not a complete model of modern industrial workloads.

#### Layer C — MILP benchmarks

Use the official MIPLIB 2017 benchmark set after the MILP engine can parse and solve small models. Begin with small and medium instances, then expand to difficult instances.

Official resources:

- [MIPLIB 2017](https://miplib.zib.de/)
- [MIPLIB benchmark set](https://miplib.zib.de/set_benchmark.html)
- [MIPLIB selection methodology](https://miplib.zib.de/Selection_Methodology.html)
- [MIPLIB downloads and scripts](https://miplib.zib.de/download.html)

The benchmark set is useful because it was selected using structural and performance features rather than being merely a random collection.

#### Layer D — refinery-style synthetic models

Build a model generator with controllable difficulty parameters:

- Number of time periods.
- Number of parallel units.
- Number of tanks.
- Number of products.
- Number of crude streams.
- Binary-variable percentage.
- Big-M magnitude.
- Coefficient range.
- Constraint density.
- Symmetry group size.
- Inventory coupling.
- Pipeline coupling.
- Quality constraints.
- Startup/shutdown decisions.
- Fixed-charge decisions.

Generate families where only one structural property changes at a time. This makes it possible to determine whether an optimization helps because of symmetry, degeneracy, big-M weakness, density, or simply smaller input size.

## KPI definitions

### LP KPIs

Report:

- Solved-to-optimal count.
- Total solved time.
- Geometric mean time over solved instances.
- Median time.
- 95th percentile time.
- Time to first feasible point.
- Phase-I iterations.
- Phase-II iterations.
- Total iterations.
- Refactorization count.
- Factor nonzeros.
- Eta nonzeros.
- Primal residual.
- Dual residual.
- Objective error against independent reference.
- Peak RAM.
- Peak VRAM.

Use performance profiles, not only averages:

```text
ratio(instance) = solver_time(instance) / best_time(instance)
```

Plot the fraction of instances solved within `tau` times the best solver or configuration.

### MILP KPIs

Report:

- Solved-to-optimal count.
- Time limit.
- Final primal bound.
- Final dual bound.
- Final relative and absolute gap.
- Time to first incumbent.
- Primal integral.
- Dual integral.
- Number of B&B nodes.
- Root relaxation gap.
- Average LP time per node.
- Average LP iterations per node.
- Number of cuts.
- Cut-generation time.
- Number of incumbent improvements.
- Peak memory.

The primary MILP objective is:

```text
number_of_nodes × average_cost_of_node_LP
```

Improving LP time while multiplying the B&B node count is a regression.

### GPU KPIs

Report:

- End-to-end speedup.
- Kernel-only speedup.
- Host-to-device time.
- Device-to-host time.
- Number of synchronizations.
- Number of launches.
- Achieved memory bandwidth.
- Occupancy.
- Warp divergence.
- L2 and global-memory hit rates.
- Roofline position.
- Energy per solved instance where measurable.

The only speedup that matters for the solver is end-to-end speedup at equal correctness.

## Implementation roadmap

## Phase 0 — benchmark and observability infrastructure

Implement first:

- Structured JSON/CSV benchmark output.
- Per-stage timers.
- Instance hashing.
- Full configuration capture.
- Repeated-run support.
- Median and geometric-mean summaries.
- Performance profile generation.
- Memory statistics.
- Deterministic random seeds.
- Benchmark regression comparison.
- Optional NVTX ranges around parse, presolve, factorization, pricing, iteration, verification, and GPU phases.

Acceptance criteria:

- A benchmark can be reproduced from a saved configuration.
- A result can be traced to a commit and instance hash.
- A regression shows exactly which stage changed.

## Phase 1 — correctness hardening

Implement:

- Independent primal feasibility checker.
- Independent dual feasibility checker.
- Independent objective checker.
- Brute-force checker for tiny MILPs.
- Random sparse LP generator.
- Ill-conditioned LP generator.
- Degenerate LP generator.
- Presolve-on/off differential testing.
- Scaling-on/off differential testing.
- CPU/GPU differential testing.
- Fuzzing of MPS input and sparse structures.
- Race detection with Compute Sanitizer.
- Memory checking.

Acceptance criteria:

- No false `INFEASIBLE` results in generated feasible cases.
- No false `UNBOUNDED` results in bounded cases.
- Every accepted optimal solution passes independent verification.
- CPU and GPU statuses agree on all supported test cases.

## Phase 2 — high-performance LP core

### 2.1 Warm-started dual simplex

Add an explicit API:

```cpp
LpResult solve_from_basis(
    const LpProblem& child,
    const Basis& parent_basis,
    const BoundChanges& changes);
```

Reuse:

- Parent basis.
- Parent factorization.
- Row and column permutations.
- Pricing state.
- Devex or steepest-edge weights.
- Safe presolve information.

The child relaxation should not rebuild the entire LP.

Measure cold versus warm child solves:

- Time.
- Iterations.
- Refactorizations.
- Basis repairs.
- Numerical failures.

### 2.2 Sparse and hyper-sparse linear algebra

Remove dense bottlenecks from the hot path.

Implement:

- Sparse FTRAN.
- Sparse BTRAN.
- Hyper-sparse active-pattern detection.
- Dynamic sparse/dense switching.
- Sparse eta application.
- Fill-in monitoring.
- Refactorization triggers based on eta density and numerical growth.
- No dense basis inverse.
- No full-vector scan when the active pattern is small.

Instrument:

```text
FTRAN time
BTRAN time
triangular solve time
eta time
pivot update time
refactorization time
active vector density
factor nonzeros
eta nonzeros
```

### 2.3 Stronger basis factorization

Evaluate:

- Markowitz pivot selection.
- Threshold partial pivoting.
- AMD ordering.
- Nested-dissection ordering.
- Symbolic factorization reuse.
- Growth-factor monitoring.
- Dynamic pivot tolerance.
- Iterative refinement of triangular solves.
- Sparse/dense crossover thresholds.

Optimize total solve time, not only factor sparsity.

### 2.4 Presolve expansion

Implement and test:

- Doubleton equation elimination.
- Variable aggregation.
- Implied-bound propagation.
- Dominated-column detection.
- Parallel-row detection.
- Duplicate-column detection.
- Coefficient strengthening.
- Redundant-row detection.
- Binary probing.
- More complete postsolve transformations.

Every reduction must have a reversible postsolve mapping and independent original-space verification.

Add a presolve safety fallback:

```text
presolve reports infeasible
    -> independently check original model
    -> if suspicious, retry without the questionable reduction
```

A missed reduction hurts speed. A false infeasibility result breaks the solver.

## Phase 3 — minimal correct MILP engine

Implement before advanced cuts:

- Integrality metadata.
- Fractionality detection.
- Binary branching.
- General-integer branching.
- Bound-change node records.
- Best-bound node selection.
- Incumbent management.
- Objective cutoff.
- Relative and absolute gap.
- Time limit.
- Node limit.
- Memory limit.
- Deterministic tie-breaking.
- Complete final verification.

Use compact node deltas. Do not copy the entire matrix per node.

Suggested node information:

```cpp
struct Node {
    NodeId id;
    NodeId parent;
    std::vector<BoundChange> changes;
    Basis basis;
    double lower_bound;
    int depth;
};
```

First correctness target:

- Solve tiny MILPs exactly.
- Compare against exhaustive enumeration.
- Replay deterministic node logs.
- Verify every pruning decision.

## Phase 4 — MILP performance

Add in this order:

1. Most-fractional branching.
2. Pseudocost branching.
3. Limited strong branching.
4. Reliability branching.
5. Reduced-cost fixing.
6. Bound propagation.
7. Rounding and repair.
8. Diving.
9. Feasibility pump.
10. RENS.
11. Local branching or large-neighborhood search.

The first critical MILP KPI is time to first incumbent. The second is node count. The third is total solve time.

## Phase 5 — cutting planes and symmetry

Add cuts only after measuring an uncut B&B baseline.

Recommended order:

1. Gomory mixed-integer cuts.
2. MIR cuts.
3. Cover cuts.
4. Flow-cover cuts.
5. Clique cuts.
6. Implied-bound cuts.
7. Knapsack-cover cuts.

For every cut family measure:

- Root gap before cuts.
- Root gap after cuts.
- Node count before cuts.
- Node count after cuts.
- Cut-generation time.
- LP reoptimization time.
- Numerical failures.

For refinery-like models, MIR, flow-cover, implied-bound, and symmetry-breaking constraints are high-priority hypotheses. They must be validated on controlled synthetic models rather than assumed beneficial.

## Phase 6 — GPU specialization

Only optimize GPU paths after Nsight Systems identifies an end-to-end bottleneck.

### PDLP

Improve:

- Diagonal preconditioning.
- Adaptive step sizes.
- Adaptive restarts.
- Feasibility polishing.
- Primal recovery.
- Residual balancing.
- Objective stabilization.
- Robust primal/dual termination criteria.
- Fused vector-update kernels.
- Fused residual calculations.
- Persistent device-resident vectors.
- CUDA Graphs for repeated iteration structures.
- CUB reductions where appropriate.
- Asynchronous streams where dependencies allow.

Use PDLP for large standalone LPs, approximate solutions, primal recovery, and difficult LPs where simplex stalls. Do not assume it should replace warm-started dual simplex inside B&B.

### SpMV

Benchmark:

- CSR.
- CSC.
- SELL-C-sigma.
- HYB.
- Custom row-bucket formats.

Choose based on row-length distribution. Measure transpose and non-transpose paths independently. Consider fusing SpMV with mathematically compatible vector updates.

### GPU pricing

GPU pricing should only be retained if it improves end-to-end simplex time. A GPU kernel can be faster while the full algorithm is slower because the CPU must synchronize to choose the entering variable.

Do not move B&B control flow or pivot decisions to the GPU merely because it is possible.

## CUDA profiling procedure

Use:

```text
Nsight Systems       whole-program timeline
Nsight Compute       kernel metrics and roofline
Compute Sanitizer    race, memory, and API correctness
NVTX                 stage and kernel ranges
```

Compile with source line information for source-level profiling. Keep capture settings stable between comparisons. Optimize one measured limiter at a time.

Recommended NVIDIA references:

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/)
- [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/)

## Acceptance gates

### LP correctness gate

- Unit tests pass.
- Parser tests pass.
- Every optimal result passes primal verification.
- Every optimal result passes dual verification where applicable.
- Objective agrees with an independent reference.
- No false infeasible result.
- No false unbounded result.

### LP performance gate

Every release must report:

- Solved count.
- Geometric mean time.
- Median time.
- 95th percentile time.
- Total time.
- Iterations.
- Refactorizations.
- Residuals.
- Memory.

### MILP correctness gate

- Exhaustive enumeration agreement on tiny models.
- Every incumbent independently verified.
- Every pruning bound verified.
- Final gap verified.
- Deterministic replay passes.
- Infeasible and unbounded cases tested.

### GPU correctness gate

- CPU/GPU objective agreement within declared tolerance.
- CPU/GPU status agreement.
- CPU/GPU feasibility agreement.
- Compute Sanitizer passes.
- No illegal memory access.
- No race report.
- No memory leak.

## Suggested engineering targets

These are goals for experiments, not claims:

```text
Maintain or exceed current 92/93 Netlib validation.
Eliminate false infeasibility and false optimality.
Demonstrate substantial iteration reduction with warm starts.
Solve tiny MILPs exactly against enumeration.
Establish a MIPLIB baseline before adding cuts.
Require end-to-end GPU speedup, not kernel-only speedup.
Report PDLP accuracy and solve rate, not only wall time.
Prevent benchmark regressions through automated comparison.
Make identical input/configuration produce deterministic results.
```

## What not to do yet

Do not prioritize:

- GPU branch-and-bound control flow.
- GPU simplex pivot decisions.
- GPU sparse LU updates without profiling evidence.
- Tensor-core acceleration for FP64 simplex arithmetic.
- Handwritten replacements for cuSPARSE without measurements.
- Kernel optimizations affecting only a tiny fraction of total time.
- Sophisticated cuts before a correct uncut B&B baseline.
- QP implementation before LP/MILP correctness is stable.

## Final priority ranking

If only one major effort can be undertaken at a time:

1. Benchmark and observability infrastructure.
2. Independent correctness verification.
3. Warm-started dual simplex.
4. Sparse/hyper-sparse basis operations.
5. Robust presolve and postsolve.
6. Minimal correct MILP branch-and-bound.
7. Branching, heuristics, and propagation.
8. Cuts and symmetry handling.
9. PDLP improvements for large standalone LPs.
10. CUDA kernel specialization based on measured bottlenecks.
11. QP support.

The most promising final architecture is:

```text
CPU:
    presolve
    B&B control
    branching
    cuts
    incumbent management
    numerical decisions

GPU:
    large sparse SpMV
    PDLP iterations
    reductions
    batched arithmetic
    selected residual calculations

Both:
    certified solution verification
    deterministic benchmark logging
```

## Core conclusion

The project will become stronger by reducing the total amount of mathematical work, not merely by making individual arithmetic operations faster.

The winning sequence is:

```text
measure correctly
→ certify correctness
→ reuse parent LP work
→ reduce presolved problem size
→ reduce B&B node count
→ reduce cost per node
→ profile the remaining bottlenecks
→ optimize CUDA kernels
```

The governing rule for every future change is:

> No optimization is accepted unless it improves a declared benchmark KPI without reducing correctness or solvability.

