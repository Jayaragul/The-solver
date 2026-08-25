# SANKHYA

SANKHYA is a sovereign mathematical-optimization core. The production path is
native C with CUDA kernels; no Python runtime is required by the solver.

## Current native implementation

- C-compatible sparse CSC model ownership, validation, matrix-vector products,
  and transpose products in `c/`.
- Native C MPS reader for LP/MILP rows, bounds, integer markers, ranges, and
  objective coefficients.
- Native QPS second pass for `QUADOBJ`, `QMATRIX`, and `QSECTION`, storing a
  symmetric sparse Hessian in the public C model. Continuous convex QPs use
  native CPU PDHG, with a CUDA path for diagonal and general sparse Hessians;
  CUDA check points require primal feasibility, iterate stability, and a
  projected KKT residual. First-order paths still do not claim a full QP dual
  optimality certificate.
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
- CUDA PDHG specialization for convex QPs: diagonal terms use an implicit
  proximal update, while general sparse Hessians use a resident CUDA CSR
  product under the same linear and bound constraints. Its host-visible
  checkpoints independently evaluate the projected primal and row-dual KKT
  inclusions before reporting convergence.
- Native CPU QP dispatch has an exact closed-form path for unconstrained
  nonnegative diagonal Hessians; it returns an independently verified solution
  without entering the iterative PDHG loop.
- Small bound-free QPs with all-equality rows use a guarded dense KKT solve;
  singular or rank-deficient KKT systems fall back, while bounded or inequality
  models remain on verified PDHG.
- Small unconstrained QPs with a general sparse Hessian also use the guarded
  dense KKT path (`Qx = -c`); larger models retain sparse PDHG.
- Small non-convex Hessians are rejected as unsupported rather than being
  mislabeled optimal from a stationary-point residual.
- The CUDA QP CLI applies the same small-Hessian PSD/symmetry admission check
  before running its approximate GPU first-order path.
- Native public `sk_solve` dispatch for continuous LP and sparse convex QP
  models, plus exact-simplex MILP branch-and-bound with bound propagation,
  pseudocost branching, incumbent-driven objective bound propagation, rounding
  heuristics, and best-bound backtracking.
  It reports `optimal` only after the proven bound closes the requested gap.
  A guarded MIQP slice is also available for small positive-semidefinite
  sparse Q with at most twelve total variables/rows; its node relaxations are
  solved by exhaustive active-set KKT certification (with the closed form
  retained for separable bounds-only nodes). Larger MIQP remains future work.
  Validated binary cover cuts are generated for suitable positive knapsack
  rows; conflict analysis and parallel search remain future work.
- QP PDHG uses diagonal row/column norm preconditioning so sparse models with
  uneven constraint or Hessian scales do not inherit one globally throttled
  step size.
- When the Hessian is diagonal, the PDHG primal update treats that quadratic
  term implicitly (a diagonal proximal step); general sparse Hessians retain
  the explicit fallback, and diagonal Hessian products avoid a repeated sparse
  traversal inside the iteration loop.
- When available, native QP transpose-SpMV uses OpenMP; `sk_qp --threads N`
  selects the worker count while preserving a serial fallback.
- Small convex QPs receive a bounded dense active-set KKT polish; its output is
  accepted only when the independent verifier confirms a strict KKT improvement.
- Native interval presolve detects contradictory column bounds and row-activity
  intervals in `O(nnz + rows + columns)` before solving or branching. The
  revised-simplex path also applies non-mutating singleton-row bound tightening
  before basis construction, avoiding needless Phase I pivots for simple
  implied bounds.
- MILP limit reports retain the active node's valid relaxation bound, so a
  timeout or node limit cannot accidentally overstate the proven dual bound.
- Every MILP incumbent is independently rechecked for row bounds, variable
  bounds, and integrality before it can affect pruning or the final gap.
- CMake targets for native C smoke tests and optional CUDA smoke tests.
- Native `sankhya_qp_cuda` accepts continuous QPS models, using an implicit
  diagonal or resident sparse-Hessian product on the GPU, and independently
  verifies the returned primal objective and feasibility residual.

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
| CUDA QP CLI | DPKLO1 diagonal and TINYQP off-diagonal QPS solved on RTX 3050; objectives independently verified | recorded |
| CUDA queued-kernel benchmark | AFIRO median 2.286 s vs. 2.753 s synchronized; identical objective/residual | recorded |
| Native end-to-end CLI | MPS parse → GPU solve → independent C verification | passed |
| Native revised-simplex CLI | AFIRO: 16 iterations, exact published objective to `6.14e-12` relative error, certified | passed |
| Netlib 5-second sweep | 25/31 LPs optimal with independent residual checks; 6 declared time limits, no false optimum | recorded |
| Netlib / HiGHS comparison | 47/50 solutions agree with isolated HiGHS 1.11; 3 limits/stalls retained in the record | recorded |
| Netlib AFIRO | default norm-scaled CUDA PDHG: objective error `5.54e-8`; primal verification at `1e-5` | passed |
| HiGHS comparison on AFIRO | isolated HiGHS 1.15.1 baseline recorded | passed |
| CUDA diagonal-QP smoke test | known solution `x=2`, objective `-4` | passed |
| Native sparse-QP smoke test | QPS parse → constrained QP solve → independent quadratic objective check | passed |
| Native general-QP active-set smoke | off-diagonal convex Hessian with an active inequality; exact KKT objective `-2.25` | passed |
| Native QP Hessian guard smoke | deliberately nonsymmetric Q is rejected as unsupported | passed |
| Native JSONL benchmark harness | one command dispatches LP/QP/MILP files and reports provenance, status, residuals, bounds, iterations/nodes, and timing; smoke covers all three | passed |
| Native QP CLI regression | `sk_qp` solves `tiny_qp.qps`, objective `-4`, independent KKT check | passed |
| Native continuous-LP smoke solve | `min x`, `x >= 1` returns `x=1`, objective `1` | passed |
| Native MILP branch-and-bound/cut smoke | binary row receives a validated cover cut, reaches `(1,0)`, objective `-2`, and closes the proven gap; controlled ablation is 0 versus 3 nodes | passed |
| Native diagonal-MIQP branch-and-bound smoke | small row-constrained separable convex integer QP reaches `(1,1)`, objective `-4`, zero proven gap | passed |
| Native general-MIQP branch-and-bound smoke | small off-diagonal PSD integer QP reaches objective `-2`, zero proven gap | passed |
| Native presolve smoke | proves `x >= 2` infeasible under `0 <= x <= 1`; simplex singleton tightening starts `x >= 2` at its implied bound | passed |
| Full CMake test suite | 32/32 C and CUDA smoke tests passed | passed |
| MIPLIB results | classic five-instance native record; 3/5 proven optimal | recorded |
| MIPLIB / HiGHS comparison | frozen five-instance comparison with explicit gap-tolerance semantics | recorded |
| Maros–Mészáros QP results | ten-instance native baseline; one proven optimal, limits retained | recorded |
| QPLIB results | no QPLIB dataset/run recorded yet | not claimed |

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

For a native QPS run with independent KKT diagnostics, use:

```text
build\native\sk_qp.exe bench\smoke\tiny_qp.qps --expect -4
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
solvers. It records 47 agreeing optima; `cycle`, `d6cube`, and `dfl001` remain
explicitly retained as current simplex limits/stalls.

The focused anti-degeneracy follow-up is [NETLIB_DEGENERACY](bench/results/NETLIB_DEGENERACY.md).
It records `cycle` and `modszk1` reaching independently measured HiGHS optima
with the revised default refactor interval, while retaining two limits.

The current native MILP record is [MIPLIB classic 5](bench/results/MIPLIB_CLASSIC_5.md).
With validated cover cuts it proves 3/5 small official MIPLIB instances within
a fixed 20-second limit and retains the two time-limited cases with valid dual
bounds.

The focused cover-cut ablation is recorded in
[MILP_COVER_CUT](bench/results/MILP_COVER_CUT.md); it measures the bundled
binary smoke instance with cuts enabled and disabled under the same limit.

The first external QP baseline is [Maros–Mészáros QP 10](bench/results/MAROS_MESZAROS_QP_10.md).
It records one proven optimum and retains nine QP limits; QPLIB remains an
explicitly unclaimed future benchmark.

The measured native CPU parallel check is [QP_THREADS](bench/results/QP_THREADS.md);
it records identical objective/residual results across thread counts and shows
the adaptive sparse threshold keeping eight threads within 0.9% of serial on a
fixed sparse-QP workload.

## Benchmark contract

The benchmark protocol is defined in [`bench/PROTOCOL.md`](bench/PROTOCOL.md).
Netlib is the correctness suite; Mittelmann is for larger LP timing; QPLIB is
for convex QP; MIPLIB is for MILP. Every claimed LP optimum must pass an
independent certificate. Results must include objective, residuals, status,
iterations/nodes, wall time, memory, machine, compiler, and dataset hashes.

For a native JSONL run over a frozen file list, use `sk_bench`; it has no
Python dependency and emits one machine-readable record per instance:

```text
build\native\sk_bench.exe --time-limit 5 bench\smoke\unit_lp.mps bench\smoke\tiny_qp.qps
```

## Roadmap

1. Make the native C model and LU smoke tests build and pass.
2. Validate CUDA SpMV/transpose-SpMV and the PDHG LP prototype.
3. Extend the native MPS reader with QPS support, sparse Hessian storage, and independent certificate checking.
4. Extend the revised-simplex LP path with full presolve/postsolve and broader
   Netlib regression coverage.
5. Extend the guarded diagonal/equality-QP KKT paths to broader convex sparse-QP
   KKT/interior-point solves.
6. Extend the guarded MIQP branch-and-bound slice to larger certified convex
   sparse-QP node relaxations, then add conflict analysis and parallel node
   search around the validated cover-cut foundation.
7. Run frozen Netlib/Mittelmann/QPLIB/MIPLIB comparisons against an isolated
   open-source baseline and publish the complete result set.
