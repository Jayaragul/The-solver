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
| HS35MOD | 1 | 3 | 3 | 7 | optimal | 0.250000000001 | 0.000e+00 | 240 | 0.000045 |
| QPCBLEND | 74 | 83 | 491 | 83 | iteration limit | -0.00819459391024 | 6.415e-05 | 300001 | 1.287026 |
| QPCBOEI1 | 351 | 384 | 3485 | 384 | time limit | 11280178.8777 | 2.285e+00 | 153994 | 5.000206 |
| QPCBOEI2 | 166 | 143 | 1196 | 143 | iteration limit | 7259986.92418 | 3.853e+01 | 300001 | 2.807193 |
| QSCAGR7 | 129 | 140 | 420 | 42 | iteration limit | 26866050.4482 | 2.784e-03 | 300001 | 1.557780 |
| QSCFXM1 | 330 | 457 | 2589 | 1410 | time limit | 14748016.1427 | 3.336e+00 | 150722 | 5.000202 |
| QSCSD1 | 77 | 760 | 2388 | 1436 | time limit | 8.66675600998 | 1.110e-05 | 139522 | 5.000158 |
| QSHIP04S | 402 | 1458 | 4352 | 98 | time limit | 1688444.93546 | 4.267e+00 | 86286 | 5.000344 |
| QSTANDAT | 359 | 1075 | 3031 | 1470 | time limit | 6411.84098591 | 5.113e-04 | 107285 | 5.000294 |
| DPKLO1 | 77 | 133 | 1575 | 77 | iteration limit | 0.370096215357 | 8.059e-08 | 600 | 0.006561 |

Only `HS35MOD` is claimed optimal in this baseline. The remaining rows are
explicitly retained as limits, motivating the next QP work: an active-set or
interior-point KKT path for sparse convex QPs, followed by a full benchmark
comparison against an established open-source solver.
