# CUDA sparse-QP path

Date: 2026-08-25

The native CUDA executable accepts continuous QPS models. Diagonal,
nonnegative Hessians use an implicit proximal update; general sparse Hessians
use a resident CUDA CSR product. Both use native row/column mass
preconditioning with 0.9 damping. The canonical C solver CSC
matrices are converted to resident CUDA CSR matrices, and the returned
objective and primal residual are independently recomputed by the native
verifier. A CUDA convergence status additionally requires primal feasibility,
iterate stability, and the projected primal/row-dual KKT residual to be at
the requested tolerance; the output reports that residual explicitly.

The QP driver does not expose unused scalar `tau` and `sigma` values. It
passes the actual finite, positive diagonal primal and dual steps to the CUDA
solver and records `qp_step_policy=diagonal_row_column_mass` at launch.

## Protocol

```text
build\\cmake-cuda\\cuda\\sankhya_qp_cuda.exe \\
  bench\\data\\qp_test_problems\\QPS_Files\\DPKLO1.QPS \\
  --iterations 100000 --tolerance 1e-5
```

`sankhya_qp_cuda` also accepts `--time-limit S`; it reports status `2` when
the bounded GPU run stops on that wall-clock limit.

Measured on the local NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.3):

| Instance | Hessian | Status | Iterations | Objective | Primal inf | KKT inf | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| DPKLO1 | diagonal, 77 entries | converged | 300 | 0.370096213395 | 2.429e-07 | 2.429e-07 | 0.027539 |
| TINYQP | full sparse, 4 entries | converged | 200 | -4 | 0.000e+00 | 2.220e-16 | 0.014195 |
| QPCBOEI1 | diagonal, 384 entries | iteration limit | 100000 | 11283944.7713 | 2.190e+00 | 2.190e+00 | 8.568071 |
| QSCAGR7 | full sparse, 42 entries | iteration limit | 100000 | 26866191.8662 | 1.296e-02 | 3.310e-02 | 9.224650 |

The printed DPKLO1 objective agrees with the native CPU optimum record
(`0.370096217114`) to the displayed precision; TINYQP independently reaches
the known objective `-4`. These are accelerator capability records, not a
claim that the first-order GPU path replaces the certificate-bearing CPU
simplex/KKT engines.

## Fixed five-second GPU check

The following use `--iterations 1000000 --time-limit 5 --tolerance 1e-5` on
the same RTX 3050. A status of `2` is a retained time limit, not an optimality
claim.

| Instance | Status | Iterations | Objective at stop | Primal inf | KKT inf | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| DPKLO1 | converged | 300 | 0.370096213395 | 2.429e-07 | 2.429e-07 | 0.024062 |
| QPCBOEI1 | time limit | 59300 | 11273921.6363 | 2.481e+00 | 2.481e+00 | 5.004864 |
| QSCAGR7 | time limit | 61200 | 26865821.5958 | 4.319e-02 | 5.823e-02 | 5.017126 |
