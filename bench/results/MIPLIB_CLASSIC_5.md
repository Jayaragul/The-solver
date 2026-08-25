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

## Input hashes

SHA-256 (lowercase):

```text
p0033.mps  8ccff819023237c79ef32e238a5da9348725ce9a4425d48888baf3a0b3b42628
bell5.mps  d4846d7ea889d482e32842832e599ca6d0dc04fd8ea3ea58f49a52b436859dd5
stein27.mps b8889994eda0853d71bad6ba9bf45755d4e59f33a13a3f43008e1ba681738c5b
flugpl.mps ed4b1fcb07d26b513027721d61344c2c4371cb506bf66098df328047c183b0c0
set1ch.mps d001258f10bf0970215fe857b4bd0a425a736cc018ce763a4ec533f924bf2867
```
