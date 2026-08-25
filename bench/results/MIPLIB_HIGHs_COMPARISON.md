# MIPLIB classic 5 — SANKHYA/HiGHS comparison

Date: 2026-08-25

This is a controlled comparison on the same Windows x86_64 machine and the
same five MIPLIB MPS files. It is a capability/timing record, not a claim that
the solvers use equivalent algorithms.

## Protocol and provenance

SANKHYA:

```text
build\cmake-cuda\sk_bench.exe bench\data\miplib\mps\<name>.mps --time-limit 20
```

HiGHS:

```text
highs.exe --model_file bench\data\miplib\mps\<name>.mps --time_limit 20
```

- Baseline: official [HiGHS releases](https://github.com/ERGO-Code/HiGHS/releases)
  Windows x86_64 static MIT package, version `1.14.0`, githash `7df0786`.
- Downloaded archive SHA-256:
  `F094CA3A436B7F1B801D51A8C0D7EE7896510AAC27C17716B857F96EB2F96A47`.
- HiGHS reports `Optimal` when its relative MIP gap is within its default
  `0.01%` tolerance. SANKHYA reports `optimal` only when its requested gap is
  closed at the native bound, so the status labels are not interchangeable.
- Times below are solver-reported seconds from one run; they are not a
  statistically powered performance ranking.

## Results

| Instance | SANKHYA status | SANKHYA objective | SANKHYA dual bound | SANKHYA s | HiGHS status | HiGHS primal | HiGHS dual | HiGHS s |
|---|---|---:|---:|---:|---|---:|---:|---:|
| p0033 | optimal | 3089 | 3089 | 0.093923 | Optimal | 3089 | 3089 | 0.03 |
| bell5 | time limit | — | 8608417.946508 | 20.021214 | Optimal* | 8966406.49152 | 8965510.24893 | 0.43 |
| stein27 | optimal | 18 | 18 | 4.070301 | Optimal | 18 | 18 | 0.68 |
| flugpl | optimal | 1201500 | 1201500 | 0.044780 | Optimal | 1201500 | 1201500 | 0.09 |
| set1ch | time limit | 101030 | 35681.989552 | 20.000517 | Optimal* | 54537.75 | 54534.90893 | 0.40 |

`*` HiGHS stopped with a nonzero gap that was within its default tolerance;
these are not exact zero-gap proofs. SANKHYA's two time-limited rows retain
valid bounds; set1ch's incumbent is independently checked but is not claimed
optimal or directly comparable as a speed result.
