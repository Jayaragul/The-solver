# Maros–Mészáros QP release refresh

Date: 2026-08-31

This is a measured refresh of all ten files in the fixed Maros–Mészáros
subset. It uses the optimized native CUDA-enabled build; the solver remains
C/CUDA (no Python solver path).

## Protocol

```text
build\cmake-cuda-release\sk_qp.exe <instance>.QPS --time-limit 5 --iter-limit 300000 --quiet
```

## Results

| Instance | Rows | Cols | A nnz | Q nnz | Status | Objective | Primal inf | Dual inf | Complementarity | Iterations | Solve seconds |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| HS35MOD | 1 | 3 | 3 | 7 | optimal | 0.250000000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0 | 0.000020 |
| QPCBLEND | 74 | 83 | 491 | 83 | iteration_limit | -0.00804972449615 | 6.460e-05 | 5.513e+00 | 4.090e-05 | 300001 | 0.277567 |
| QPCBOEI1 | 351 | 384 | 3485 | 384 | iteration_limit | 11319462.5301 | 1.372e+00 | 6.035e+04 | 7.315e-03 | 300001 | 1.898077 |
| QPCBOEI2 | 166 | 143 | 1196 | 143 | iteration_limit | 7272905.9469 | 3.580e+01 | 1.191e+04 | 2.777e-02 | 300001 | 0.597870 |
| QSCAGR7 | 129 | 140 | 420 | 42 | iteration_limit | 26865948.4968 | 6.196e-05 | 4.684e+04 | 7.327e-08 | 300001 | 0.367826 |
| QSCFXM1 | 330 | 457 | 2589 | 1410 | iteration_limit | 16881464.2816 | 1.455e-02 | 9.437e+04 | 9.760e-06 | 300001 | 1.926684 |
| QSCSD1 | 77 | 760 | 2388 | 1436 | iteration_limit | 8.66666609937 | 7.145e-08 | 6.667e+00 | 1.975e-08 | 63600 | 0.426377 |
| QSHIP04S | 402 | 1458 | 4352 | 98 | iteration_limit | 2402498.7773 | 2.082e-01 | 8.760e+03 | 5.411e-04 | 300001 | 4.847712 |
| QSTANDAT | 359 | 1075 | 3031 | 1470 | iteration_limit | 6411.83838668 | 4.403e-08 | 2.444e+02 | 2.342e-08 | 80300 | 0.726605 |
| DPKLO1 | 77 | 133 | 1575 | 77 | optimal | 0.370096217114 | 1.066e-14 | 5.225e-16 | 6.044e-16 | 0 | 0.000694 |

The two optimal records satisfy the reported KKT residual checks. Every other
record is explicitly retained as an iteration limit: several have small primal
residuals but large dual residuals, so they are not certificates. This refresh
is evidence of an optimized build and guarded exact paths, not a claim of
complete QP coverage.
The full ten-instance baseline remains in
`MAROS_MESZAROS_QP_10.md`.
