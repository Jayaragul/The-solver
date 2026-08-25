# CUDA diagonal-QP path

Date: 2026-08-25

The native CUDA executable now accepts continuous QPS models whose Hessian is
diagonal and nonnegative. It converts the canonical C solver CSC matrix to a
resident CUDA CSR matrix, runs diagonal-QP PDHG, and independently recomputes
the objective and primal residual with the native verifier. General sparse
Hessians remain on the CPU QP path.

## Protocol

```text
build\\cmake-cuda\\cuda\\sankhya_qp_cuda.exe \\
  bench\\data\\qp_test_problems\\QPS_Files\\DPKLO1.QPS \\
  --iterations 100000 --tolerance 1e-5
```

Measured on the local NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.3):

| Instance | Hessian | Status | Iterations | Objective | Primal inf | Seconds |
|---|---|---:|---:|---:|---:|---:|
| DPKLO1 | diagonal, 77 entries | converged | 8200 | 0.37009621712 | 6.067e-06 | 0.340967 |

The printed objective agrees with the native CPU optimum record
(`0.370096217114`) to the displayed precision. The run is an accelerator
capability record, not a claim that the first-order GPU path replaces the
certificate-bearing CPU simplex/KKT engines.
