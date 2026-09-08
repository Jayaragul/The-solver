# MIPLIB 2017 small subset — production pseudocost refresh

Date: 2026-09-09

This is a fresh native Release run of the checked-in 19-instance MIPLIB 2017
subset after the bounded binary-slack repair heuristic was integrated. It uses
the production settings: pseudocost branching, warm-started node relaxations,
serial LP execution, no optional GMI/RENS/coefficient-rounding/feasibility-pump
experiments, and the default rounding heuristic.

## Protocol

```text
build\\cmake-cuda-release\\benchmarks\\bench_miplib.exe \\
  data\\miplib2017_small data\\miplib2017_small\\miplib2017-v36.solu "" \\
  10 pseudocost on serial 1 off 50000 off off off on 0
```

The run was executed on the recorded RTX 3050 laptop host. The MILP proof path
uses certified CPU simplex relaxations; GPU availability is recorded for
provenance but does not turn an approximate GPU point into a proof bound.

## Summary

| Outcome | Count |
|---|---:|
| Exact certified reference match | 9/19 |
| Certified solver results | 9/19 |
| Time/node-limit results | 10/19 |

Exact certified instances were `22433`, `23588`, `dcmulti`, `enlight4`,
`flugpl`, `gr4x6`, `khb05250`, `neos859080`, and `p0201`.

The hard `markshare2` run produced incumbent `189` with best bound `0` in
10.061 seconds. This is an incumbent-quality improvement over the earlier
production record (`231`), not an optimality claim. `gen-ip002`, `gen-ip054`,
`pk1`, and the remaining time-limited cases retain explicit gaps.

The complete machine-readable output is retained in
[`MIPLIB_PSEUDOCOST_PRODUCTION_REFRESH_10S.txt`](MIPLIB_PSEUDOCOST_PRODUCTION_REFRESH_10S.txt).
