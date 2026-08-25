# MIPLIB classic subset — native MILP record

Date: 2026-08-25

This is a small, independently reproducible MILP check using five official
classic MIPLIB instances. The input files are downloaded by
`bench/scripts/fetch_miplib.sh` into the ignored `bench/data/miplib` directory;
third-party inputs are intentionally not committed to the repository.

## Protocol

```text
build\cmake-cuda\sk_mip.exe bench\data\miplib\mps\<name>.mps \
  --time-limit 20 --quiet
```

The run uses the native revised-simplex relaxation, bound propagation,
pseudocost branching, best-bound backtracking, and the exact MILP gap test.
An `optimal` row means the native dual bound closed the gap; a time limit is
reported as a limit and is not treated as a failure or an optimality claim.

## Results

| Instance | Rows | Cols | Integer vars | Status | Objective | Dual bound | Nodes | LP solves | Seconds |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| p0033 | 16 | 33 | 33 | optimal | 3089 | 3089 | 796 | 807 | 0.083547 |
| bell5 | 91 | 104 | 58 | time limit | — | 8608417.94651 | 15871 | 16112 | 20.018210 |
| stein27 | 118 | 27 | 27 | optimal | 18 | 18 | 14116 | 14226 | 3.490471 |
| flugpl | 18 | 18 | 11 | optimal | 1201500 | 1201500 | 374 | 381 | 0.031429 |
| set1ch | 492 | 712 | 240 | time limit | — | 35118.1098485 | 631 | 643 | 20.000373 |

Summary: 3/5 instances were proven optimal within the fixed limit; the other
two retained valid root dual bounds but no incumbent was found. These results
are a capability record, not a performance ranking. A commercial/open-source
comparison is maintained separately in the Netlib/HiGHS record.
