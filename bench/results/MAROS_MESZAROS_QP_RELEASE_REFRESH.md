# Maros–Mészáros QP release refresh

Date: 2026-08-31

This is a measured refresh of three representative files from the fixed
Maros–Mészáros subset. It uses the optimized native CUDA-enabled build; the
solver remains C/CUDA (no Python solver path).

## Protocol

```text
build\cmake-cuda-release\sk_qp.exe <instance>.QPS --time-limit 5 --iter-limit 300000 --quiet
```

## Results

| Instance | Rows | Cols | A nnz | Q nnz | Status | Objective | Primal inf | Dual inf | Complementarity | Iterations | Solve seconds |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| HS35MOD | 1 | 3 | 3 | 7 | optimal | 0.250000000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0 | 0.000040 |
| DPKLO1 | 77 | 133 | 1575 | 77 | optimal | 0.370096217114 | 1.066e-14 | 5.225e-16 | 6.044e-16 | 0 | 0.000727 |
| QPCBLEND | 74 | 83 | 491 | 83 | iteration_limit | -0.00804972449615 | 6.460e-05 | 5.513e+00 | 4.090e-05 | 300001 | 0.287862 |

The two optimal records satisfy the reported KKT residual checks. QPCBLEND is
explicitly retained as a limit: its primal residual is small, but the dual
residual is not yet a certificate. This refresh is evidence of a faster
optimized build and guarded exact paths, not a claim of complete QP coverage.
The full ten-instance baseline remains in
`MAROS_MESZAROS_QP_10.md`.
