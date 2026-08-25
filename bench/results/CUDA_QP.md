# CUDA sparse-QP path

Date: 2026-08-25

The native CUDA executable accepts continuous QPS models. Diagonal,
nonnegative Hessians use an implicit proximal update; general sparse Hessians
use a resident CUDA CSR product. Both use native row/column mass
preconditioning. The canonical C solver CSC
matrices are converted to resident CUDA CSR matrices, and the returned
objective and primal residual are independently recomputed by the native
verifier.

For general sparse Hessians, automatic steps retain the same `tau*sigma`
coupling product as the LP path but use an asymmetric pair (`tau=0.25/||A||`,
`sigma=1.0/||A||`) so explicit Q*x curvature does not suppress dual progress.

## Protocol

```text
build\\cmake-cuda\\cuda\\sankhya_qp_cuda.exe \\
  bench\\data\\qp_test_problems\\QPS_Files\\DPKLO1.QPS \\
  --iterations 100000 --tolerance 1e-5
```

Measured on the local NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.3):

| Instance | Hessian | Status | Iterations | Objective | Primal inf | Seconds |
|---|---|---:|---:|---:|---:|---:|
| DPKLO1 | diagonal, 77 entries | converged | 300 | 0.370096213395 | 2.429e-07 | 0.020251 |
| TINYQP | full sparse, 4 entries | converged | 200 | -4 | 0.000e+00 | 0.017349 |
| QPCBOEI1 | diagonal, 384 entries | iteration limit | 100000 | 11283944.7713 | 2.190e+00 | 8.351472 |
| QSCAGR7 | full sparse, 42 entries | iteration limit | 100000 | 26866191.8662 | 1.296e-02 | 6.058142 |

The printed DPKLO1 objective agrees with the native CPU optimum record
(`0.370096217114`) to the displayed precision; TINYQP independently reaches
the known objective `-4`. These are accelerator capability records, not a
claim that the first-order GPU path replaces the certificate-bearing CPU
simplex/KKT engines.
