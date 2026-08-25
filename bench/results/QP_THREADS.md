# Native QP OpenMP scaling check

Date: 2026-08-25

This is a fixed-work measurement of the optional OpenMP transpose-SpMV path.
It is intentionally a throughput check, not an optimality claim: `QSTANDAT`
does not converge within this iteration budget.

## Protocol

```text
build\cmake-cuda\sk_qp.exe bench\data\qp_test_problems\QPS_Files\QSTANDAT.QPS \
  --iter-limit 50000 --threads N --quiet
```

| Threads | Iterations | Solve seconds | Objective | Primal inf |
|---:|---:|---:|---:|---:|
| 1 | 50001 | 1.602874 | 6411.84001457 | 6.366e-04 |
| 2 | 50001 | 1.656842 | 6411.84001457 | 6.366e-04 |
| 4 | 50001 | 1.606249 | 6411.84001457 | 6.366e-04 |
| 8 | 50001 | 1.616658 | 6411.84001457 | 6.366e-04 |

The adaptive sparse threshold keeps the eight-thread run within 0.9% of the
serial run on this laptop, removing the earlier small-workload regression.
This does not claim positive scaling for every QP; thread count remains a
benchmark-controlled option rather than a universal default.

Input SHA-256 (`QSTANDAT.QPS`):

```text
289f34fe700e68dd14db6c3ccf4ff835d9eac1dedf7b330bf7866f85eb612fa7
```
