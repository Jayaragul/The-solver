# Maros–Mészáros QP subset — native QP baseline

Date: 2026-08-25

This is a fixed ten-instance subset of the public Maros–Mészáros convex-QP
collection. Inputs are kept outside version control under
`bench/data/qp_test_problems`; the native QPS reader and `sk_qp` driver are
fully C-based. The run is a capability baseline, not a competitive ranking.

## Protocol

```text
build\cmake-cuda\sk_qp.exe <instance>.QPS \
  --time-limit 5 --iter-limit 300000 --quiet
```

## Results

| Instance | Rows | Cols | A nnz | Q nnz | Status | Objective at stop | Primal inf | Iterations | Seconds |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| HS35MOD | 1 | 3 | 3 | 7 | optimal | 0.250000000000 | 0.000e+00 | 140 | 0.000059 |
| QPCBLEND | 74 | 83 | 491 | 83 | iteration limit | -0.00804972479292 | 6.460e-05 | 300001 | 1.218975 |
| QPCBOEI1 | 351 | 384 | 3485 | 384 | time limit | 11291429.1758 | 1.957e+00 | 145712 | 5.000152 |
| QPCBOEI2 | 166 | 143 | 1196 | 143 | iteration limit | 7272905.86727 | 3.580e+01 | 300001 | 2.613583 |
| QSCAGR7 | 129 | 140 | 420 | 42 | iteration limit | 26865948.4968 | 6.196e-05 | 300001 | 1.394502 |
| QSCFXM1 | 330 | 457 | 2589 | 1410 | time limit | 16607544.1827 | 4.101e-01 | 161884 | 5.000190 |
| QSCSD1 | 77 | 760 | 2388 | 1436 | iteration limit | 8.66666609937 | 7.145e-08 | 63600 | 1.589729 |
| QSHIP04S | 402 | 1458 | 4352 | 98 | time limit | 1835189.20025 | 3.889e+00 | 89953 | 5.000416 |
| QSTANDAT | 359 | 1075 | 3031 | 1470 | iteration limit | 6411.83838668 | 4.403e-08 | 80300 | 2.711014 |
| DPKLO1 | 77 | 133 | 1575 | 77 | optimal | 0.370096217114 | 3.553e-15 | 340 | 0.007381 |

`HS35MOD` and `DPKLO1` are claimed optimal in this baseline. The remaining rows are
explicitly retained as limits, motivating the next QP work: an active-set or
interior-point KKT path for sparse convex QPs, followed by a full benchmark
comparison against an established open-source solver.
