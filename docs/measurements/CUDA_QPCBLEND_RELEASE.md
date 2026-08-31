# Release CUDA QP: QPCBLEND

Date: 2026-09-01

This is a larger sparse-QP validation after correcting the Release CUDA
architecture cache to `sm_86;sm_89`. It exercises the device-resident sparse
Hessian/SpMV path, not the tiny exact shortcut.

```text
build\cmake-cuda-release\cuda\sankhya_qp_cuda.exe \
  bench\data\qp_test_problems\QPS_Files\QPCBLEND.QPS \
  --iterations 100000 --tolerance 1e-5
```

Measured output:

| Status | Iterations | Objective | Primal/KKT residual | Solve seconds | Independent check |
|---|---:|---:|---:|---:|---|
| iteration limit (`1`) | 100,000 | -0.00821668807709 | 6.528e-05 | 9.049160 | passed |

The GPU path is numerically stable and the independent CPU verifier accepts
the returned point as feasible. The result is deliberately retained as an
iteration limit rather than being mislabeled optimal; the residual is above
the requested tolerance. This complements the certified CPU QP record and
demonstrates that the corrected Release binary actually executes the larger
CUDA workload.
