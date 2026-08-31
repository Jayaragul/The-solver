# MIPLIB 2017 small subset — native Release MILP record

Date: 2026-08-31

This record is a complete 19-instance run of the checked-in MIPLIB subset on
the native MSVC/CUDA Release build. It uses the production warm-start default,
reliability branching, and automatic parallel-probe gating. The reference
objectives are read from `data/miplib2017_small/miplib2017-v36.solu`.

## Protocol

```text
build/cmake-cuda-release/benchmarks/bench_miplib.exe \
  data/miplib2017_small data/miplib2017_small/miplib2017-v36.solu "" 60 reliability
```

Every `OPTIMAL` or `INFEASIBLE` result below is certified by the solver's
bound closure. A `TIME_LIMIT` is not counted as a solved instance, even when
its incumbent happens to match the reference objective.

## Summary

| Outcome | Count |
|---|---:|
| Exact certified reference match | 9/19 |
| Certified infeasible/optimal results | 9/19 |
| Time-limit or non-exact result | 10/19 |

Exact certified instances were `22433`, `23588`, `blend2`, `dcmulti`,
`enlight4`, `flugpl`, `gr4x6`, `gt2`, and `p0201`. The remaining records are
retained below as honest gaps for the next MILP work: `gen-ip002` and
`neos859080` reached reference incumbents without closing the gap, while the
other time-limited cases still need stronger cuts, propagation, or search.

## Per-instance result

| Instance | Status | Ours | Reference | Nodes | LPs | Seconds | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 22433 | OPTIMAL | 21477 | 21477 | 47 | 259 | 7.301 | EXACT |
| 23588 | OPTIMAL | 8090 | 8090 | 1071 | 1294 | 9.164 | EXACT |
| blend2 | OPTIMAL | 7.598985 | 7.598985 | 7859 | 5679 | 17.694 | EXACT |
| bppc4-08 | TIME_LIMIT | — | 53 | 473 | 4197 | 60.010 | LIMIT |
| dcmulti | OPTIMAL | 188182 | 188182 | 9411 | 7081 | 11.762 | EXACT |
| enlight4 | INFEASIBLE | — | — | 651 | 829 | 0.075 | EXACT(inf) |
| flugpl | OPTIMAL | 1201500 | 1201500 | 5233 | 5432 | 0.285 | EXACT |
| gen-ip002 | TIME_LIMIT | -4783.7333916 | -4783.733392 | 248861 | 249023 | 60.284 | INCUMBENT_ONLY |
| gen-ip054 | TIME_LIMIT | 6857.649055344 | 6840.96564179 | 355572 | 355928 | 60.458 | MISMATCH |
| gr4x6 | OPTIMAL | 202.35 | 202.35 | 125 | 175 | 0.035 | EXACT |
| gt2 | OPTIMAL | 21166 | 21166 | 25675 | 13485 | 1.529 | EXACT |
| khb05250 | TIME_LIMIT | 122024332 | 106940226 | 454 | 3849 | 60.103 | MISMATCH |
| mad | TIME_LIMIT | — | 0.0268 | 77552 | 78458 | 60.070 | LIMIT |
| markshare2 | TIME_LIMIT | 231 | 1 | 842826 | 843092 | 60.690 | MISMATCH |
| neos5 | TIME_LIMIT | 16.0625 | 15 | 1164 | 1596 | 60.349 | MISMATCH |
| neos859080 | TIME_LIMIT | — | — | 53725 | 54405 | 60.038 | INCUMBENT_ONLY |
| noswot | TIME_LIMIT | — | -41.00000885 | 142756 | 143419 | 60.095 | LIMIT |
| p0201 | OPTIMAL | 7615 | 7615 | 1419 | 1634 | 5.644 | EXACT |
| pk1 | TIME_LIMIT | 44 | 11 | 155523 | 155771 | 60.208 | MISMATCH |

The wall-time and CPU-time columns are native Windows process measurements;
the runner also reports CUDA visibility and device memory for each process.
