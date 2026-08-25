# Maros–Mészáros QP subset — native QP baseline

Date: 2026-08-25

This is a fixed ten-instance subset of the public Maros–Mészáros convex-QP
collection. Inputs are kept outside version control under
`bench/data/qp_test_problems`; the native QPS reader and `sk_qp` driver are
fully C-based. The run is a capability baseline, not a competitive ranking.
Diagonal Hessians use an implicit proximal PDHG update; general sparse
Hessians retain the explicit update.

## Protocol

```text
build\cmake-cuda\sk_qp.exe <instance>.QPS \
  --time-limit 5 --iter-limit 300000 --quiet
```

## Results

| Instance | Rows | Cols | A nnz | Q nnz | Status | Objective at stop | Primal inf | Iterations | Seconds |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| HS35MOD | 1 | 3 | 3 | 7 | optimal | 0.250000000000 | 0.000e+00 | 140 | 0.000059 |
| QPCBLEND | 74 | 83 | 491 | 83 | iteration limit | -0.00804972449615 | 6.460e-05 | 300001 | 1.020988 |
| QPCBOEI1 | 351 | 384 | 3485 | 384 | time limit | 11291429.1758 | 1.957e+00 | 145712 | 5.000152 |
| QPCBOEI2 | 166 | 143 | 1196 | 143 | iteration limit | 7272905.86727 | 3.580e+01 | 300001 | 2.613583 |
| QSCAGR7 | 129 | 140 | 420 | 42 | iteration limit | 26865948.4968 | 6.196e-05 | 300001 | 1.264215 |
| QSCFXM1 | 330 | 457 | 2589 | 1410 | time limit | 16607544.1827 | 4.101e-01 | 161884 | 5.000190 |
| QSCSD1 | 77 | 760 | 2388 | 1436 | iteration limit | 8.66666609937 | 7.145e-08 | 63600 | 1.420627 |
| QSHIP04S | 402 | 1458 | 4352 | 98 | time limit | 1835189.20025 | 3.889e+00 | 89953 | 5.000416 |
| QSTANDAT | 359 | 1075 | 3031 | 1470 | iteration limit | 6411.83838668 | 4.403e-08 | 80300 | 2.711014 |
| DPKLO1 | 77 | 133 | 1575 | 77 | optimal | 0.370096217114 | 3.553e-15 | 340 | 0.007381 |

`HS35MOD` and `DPKLO1` are claimed optimal in this baseline. The remaining rows are
explicitly retained as limits, motivating the next QP work: extending the
guarded KKT paths to sparse convex QPs, followed by a full benchmark comparison
against an established open-source solver.

## Input hashes

SHA-256 (lowercase):

```text
HS35MOD.QPS  d7bdecfb4c425d5fbb6e3ba4321a7193bbec0e40c4aedf79d6d5f5739c5e5023
QPCBLEND.QPS 45aff6b9792885d2fe23e4971061fcce41fb35df683a1323c7b9241e4e78cbf9
QPCBOEI1.QPS 0d4e39f9458bb7b1ec45a44b63eef65411f1b013c8803ad6d43ef9b86e015428
QPCBOEI2.QPS 4d6a8c31327b8f40e66cfd564ae6d1957b287701005ece47fa4f9746c0f2dc00
QSCAGR7.QPS  6f7d5957423ca8b0e17af7fac8f3d29cfc877512aab148db78c1c843d55d3f29
QSCFXM1.QPS  6ee5617212fee6f47fb5ce65b950b03097a01f62cd93bb90dc5a096cfc8f6fa7
QSCSD1.QPS   cd45e4b1f0ba1c97e86b6206247b5d83da63941ee5d84f677452e2f5ca9c46b1
QSHIP04S.QPS 219ea5310b164982a99d48ed2ec88b72985c10bab5a355e91329c963686cdb2f
QSTANDAT.QPS 289f34fe700e68dd14db6c3ccf4ff835d9eac1dedf7b330bf7866f85eb612fa7
DPKLO1.QPS   7a49c5707a22c8fc45f7ad74c848d7a3fa872ea98d6fb8356d253686e0a61c16
```
