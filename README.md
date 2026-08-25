# SANKHYA

SANKHYA is a sovereign mathematical-optimization core. The production path is
native C with CUDA kernels; no Python runtime is required by the solver.

## Current native implementation

- C-compatible sparse CSC model ownership, validation, matrix-vector products,
  and transpose products in `c/`.
- Native C MPS reader for LP/MILP rows, bounds, integer markers, ranges, and
  objective coefficients.
- Native QPS second pass for `QUADOBJ`, `QMATRIX`, and `QSECTION`, storing a
  symmetric sparse Hessian in the public C model. The general-Hessian solver
  is handled by a native CPU PDHG prototype for continuous convex QPs; it has
  primal/step convergence checks but no QP dual optimality certificate yet.
- C dense partial-pivot LU API in `c/`, intended as the first basis/KKT solve
  primitive.
- Native sparse-LU factorization with threshold pivoting, FTRAN/BTRAN, and
  product-form basis updates; its regression test checks residuals through
  `n=1200` and validates singular-matrix rejection.
- Native bounded-variable revised simplex for continuous LPs, with a two-phase
  feasibility procedure, Harris ratio test, periodic basis refactorization,
  and an independently checked primal-dual certificate.
- Native power-of-two row/column equilibration before simplex iterations, with
  exact unscaling of primal and dual output. This stabilizes coefficient ranges
  encountered in the Netlib corpus without changing the public model.
- CUDA double-precision CSR SpMV and fused AXPY kernels.
- Persistent device-resident CSR storage for repeated iterations.
- CUDA primal-dual hybrid-gradient (PDHG) prototype for bounded continuous LP:
  `min c'x` subject to `row_lower <= A*x <= row_upper` and variable bounds.
- CUDA PDHG specialization for convex *diagonal* QPs: `min 0.5 x'Dx + c'x`,
  where `D` is nonnegative diagonal, with the same linear and bound constraints.
  General sparse-Hessian QPS input is handled by the native CPU path, not CUDA.
- Native public `sk_solve` dispatch for continuous LP and sparse convex QP
  models, plus exact-simplex MILP branch-and-bound with bound propagation,
  pseudocost branching, rounding heuristics, and best-bound backtracking.
  It reports `optimal` only after the proven bound closes the requested gap.
  Cuts, conflict analysis, parallel search, and MIQP remain future work.
- Native interval presolve detects contradictory column bounds and row-activity
  intervals in `O(nnz + rows + columns)` before solving or branching.
- CMake targets for native C smoke tests and optional CUDA smoke tests.

## Evidence status — 25 August 2026

| Item | Evidence | Status |
|---|---|---|
| CUDA toolkit | `nvcc` 13.3 is installed | confirmed |
| GPU | GeForce RTX 3050 Laptop GPU, 4 GiB, driver 592.82 | confirmed |
| Host compiler | MSVC 19.44 via Visual Studio Build Tools developer environment | confirmed |
| CMake | bundled with Visual Studio Build Tools; native and CUDA configurations pass | confirmed |
| Native C smoke tests | model, LU, MPS parser, and verifier compiled and passed | passed |
| Sparse-LU regression | FTRAN/BTRAN through `n=1200`, 12 eta updates, singular rejection | passed |
| CUDA sparse-operator smoke test | compiled for `sm_86` and passed on RTX 3050 Laptop GPU | passed |
| CUDA PDHG LP smoke test | one-variable LP compiled and solved on GPU | passed |
| Native end-to-end CLI | MPS parse → GPU solve → independent C verification | passed |
| Native revised-simplex CLI | AFIRO: 16 iterations, exact published objective to `6.14e-12` relative error, certified | passed |
| Netlib 5-second sweep | 25/31 LPs optimal with independent residual checks; 6 declared time limits, no false optimum | recorded |
| Netlib / HiGHS comparison | 46/50 solutions agree with isolated HiGHS 1.11; 4 limits/stalls retained in the record | recorded |
| Netlib AFIRO | default norm-scaled CUDA PDHG: objective error `5.54e-8`; primal verification at `1e-5` | passed |
| HiGHS comparison on AFIRO | isolated HiGHS 1.15.1 baseline recorded | passed |
| CUDA diagonal-QP smoke test | known solution `x=2`, objective `-4` | passed |
| Native sparse-QP smoke test | QPS parse → constrained QP solve → independent quadratic objective check | passed |
| Native continuous-LP smoke solve | `min x`, `x >= 1` returns `x=1`, objective `1` | passed |
| Native MILP branch-and-bound smoke | fractional binary knapsack branches to `(1,0)`, objective `-2`, zero proven gap | passed |
| Native presolve smoke | proves `x >= 2` infeasible under `0 <= x <= 1` before iterations | passed |
| Full CMake test suite | 17/17 C and CUDA smoke tests passed | passed |
| MIPLIB results | no dataset/run recorded yet | not claimed |
| QPLIB results | no dataset/run recorded yet | not claimed |

The repository contains no benchmark score that has not actually been measured.

## Build once the toolchain is available

```text
cmake -S . -B build/native -DSANKHYA_BUILD_TESTS=ON -DSANKHYA_ENABLE_CUDA=OFF
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure

cmake -S . -B build/cuda -DSANKHYA_BUILD_TESTS=ON -DSANKHYA_ENABLE_CUDA=ON
cmake --build build/cuda --config Release
ctest --test-dir build/cuda -C Release --output-on-failure
```

CUDA compilation requires `nvcc` plus a supported host C++ compiler. On this
machine, compile for `sm_86`; the default PTX path is incompatible with the
installed driver. GPU timings must use the persistent CSR API and be compared
with a same-precision CPU implementation on the same input and thread policy.

For this laptop, build the native CUDA CLI with:

```text
scripts\build_windows_sm86.bat
build\native\sankhya_pdhg.exe bench\smoke\unit_lp.mps --iterations 5000 --tolerance 1e-5
```

The verified smoke result is `x[X]=1`, objective `1`, and zero primal violation.

The native CPU CLI loads MPS/QPS and exercises the public LP/QP/MILP dispatch:

```text
build\native\sankhya_solve.exe bench\smoke\unit_lp.mps --iterations 200000
```

It prints primal and MIP-gap diagnostics. On QPs it also reports KKT residuals;
on the legacy PDHG LP path it reports `lp_pdhg_primal_only`, not an LP dual
certificate. For certified MILP runs, use the native exact-simplex driver:

```text
build\native\sk_mip.exe bench\smoke\knapsack.mps --expect -20
```

It emits the incumbent, best bound, gap, integrality residual, node count, and
LP-relaxation statistics. A `gap_limit`, node limit, or time limit is never
presented as a proof of optimality.

For a certified continuous-LP solve using the revised simplex path:

```text
build\native\sk_lp.exe bench\smoke\unit_lp.mps
```

It reports the independently recomputed primal infeasibility, dual
infeasibility, and LP gap. The current simplex implementation is deliberately
limited to continuous LPs; QP and MILP remain on their separately documented
paths.

The first external benchmark record is [AFIRO](bench/results/AFIRO.md). Its
default path uses a deterministic CPU power-iteration estimate of `||A||_2` to
select the CUDA PDHG primal/dual steps. It remains an approximate solution with
an independent primal-feasibility check, not an LP optimality certificate.

The native revised-simplex prefix sweep is recorded in
[Netlib-31](bench/results/NETLIB_31.md). It uses a uniform five-second limit
and lists every solved and time-limited instance.

The broader, isolated HiGHS comparison is stored as machine-readable measured
data in [netlib_lp.json](bench/results/netlib_lp.json): 50 original Netlib MPS
models with objective, residual, status, iteration, and timing fields for both
solvers. It records 46 agreeing optima; `cycle`, `d6cube`, `dfl001`, and
`modszk1` are explicitly retained as current simplex limits/stalls.

## Benchmark contract

The benchmark protocol is defined in [`bench/PROTOCOL.md`](bench/PROTOCOL.md).
Netlib is the correctness suite; Mittelmann is for larger LP timing; QPLIB is
for convex QP; MIPLIB is for MILP. Every claimed LP optimum must pass an
independent certificate. Results must include objective, residuals, status,
iterations/nodes, wall time, memory, machine, compiler, and dataset hashes.

## Roadmap

1. Make the native C model and LU smoke tests build and pass.
2. Validate CUDA SpMV/transpose-SpMV and the PDHG LP prototype.
3. Extend the native MPS reader with QPS support, sparse Hessian storage, and independent certificate checking.
4. Extend the revised-simplex LP path with full presolve/postsolve and broader
   Netlib regression coverage.
5. Generalize the diagonal-QP prototype to convex sparse-QP KKT/interior-point solves.
6. Add validated cuts, conflict analysis, and parallel node search to the
   exact-simplex MILP branch-and-bound path.
7. Run frozen Netlib/Mittelmann/QPLIB/MIPLIB comparisons against an isolated
   open-source baseline and publish the complete result set.
