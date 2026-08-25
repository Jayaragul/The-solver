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
- CUDA double-precision CSR SpMV and fused AXPY kernels.
- Persistent device-resident CSR storage for repeated iterations.
- CUDA primal-dual hybrid-gradient (PDHG) prototype for bounded continuous LP:
  `min c'x` subject to `row_lower <= A*x <= row_upper` and variable bounds.
- CUDA PDHG specialization for convex *diagonal* QPs: `min 0.5 x'Dx + c'x`,
  where `D` is nonnegative diagonal, with the same linear and bound constraints.
  General sparse-Hessian QPS input is handled by the native CPU path, not CUDA.
- Native public `sk_solve` dispatch for continuous LP and sparse convex QP
  models, plus bounded-MILP depth-first branch-and-bound over LP relaxations.
  The MILP path has no cuts, heuristics, parallel search, or exact LP bound
  certificate, so it reports `gap_limit` rather than optimality.
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
| Netlib AFIRO | default norm-scaled CUDA PDHG: objective error `5.54e-8`; primal verification at `1e-5` | passed |
| HiGHS comparison on AFIRO | isolated HiGHS 1.15.1 baseline recorded | passed |
| CUDA diagonal-QP smoke test | known solution `x=2`, objective `-4` | passed |
| Native sparse-QP smoke test | QPS parse → constrained QP solve → independent quadratic objective check | passed |
| Native continuous-LP smoke solve | `min x`, `x >= 1` returns `x=1`, objective `1` | passed |
| Native MILP branch-and-bound smoke | fractional binary relaxation branches to `(1,1)`, objective `2` | passed |
| Native presolve smoke | proves `x >= 2` infeasible under `0 <= x <= 1` before iterations | passed |
| Full CMake test suite | 12/12 C and CUDA smoke tests passed | passed |
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
on LP/MILP the current PDHG path reports `lp_pdhg_primal_only`, not an LP dual
certificate. A `gap_limit` MILP status is a feasible incumbent from the current
first-order branch-and-bound baseline, not a commercial-solver-grade optimality
proof.

The first external benchmark record is [AFIRO](bench/results/AFIRO.md). Its
default path uses a deterministic CPU power-iteration estimate of `||A||_2` to
select the CUDA PDHG primal/dual steps. It remains an approximate solution with
an independent primal-feasibility check, not an LP optimality certificate.

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
4. Add revised simplex and presolve/postsolve for robust LP solving.
5. Generalize the diagonal-QP prototype to convex sparse-QP KKT/interior-point solves.
6. Add MILP branch-and-bound, then validated cuts and parallel node search.
7. Run frozen Netlib/Mittelmann/QPLIB/MIPLIB comparisons against an isolated
   open-source baseline and publish the complete result set.
