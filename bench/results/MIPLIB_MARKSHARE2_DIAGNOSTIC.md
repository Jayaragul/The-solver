# MIPLIB `markshare2` diagnostic

Date: 2026-09-01

This record isolates the current MILP benchmark gap on `markshare2`, a small
binary equality model from MIPLIB 2017. The run uses the optimized native
C++/CUDA build with CPU-resident branch-and-bound control and serial LP
relaxations.

## Protocol

```text
build\cmake-cuda-release\benchmarks\bench_miplib.exe \
  data\miplib2017_small data\miplib2017_small\miplib2017-v36.solu \
  markshare2 15 reliability on serial
```

## Measured result

| Instance | Reference optimum | Status | Incumbent | Best bound | Relative gap | Nodes | LP relaxations | Cuts | Time (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| markshare2 | 1 | TIME_LIMIT | 231 | 0 | 0.99568966 | 190,513 | 190,779 | 2 | 15.346 |

The incumbent is integral and passes the feasibility gate; the bound is also
valid. The failure is therefore search/heuristic strength, not an invalid
certificate or numerical integrality error. The model has seven equalities and
60 binary variables with a zero root bound, so a stronger feasibility-pump or
repair heuristic is the next justified experiment. GPU pricing is not a
credible fix for this instance because branch-and-bound control and the
combinatorial repair remain CPU-resident by design.
