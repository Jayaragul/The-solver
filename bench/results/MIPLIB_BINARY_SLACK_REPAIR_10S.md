# Binary-slack repair heuristic

Date: 2026-09-08

This is a controlled native Release comparison for the weak-LP `markshare2`
family. The new heuristic is a deterministic, structure-gated tabu repair
search over binary columns in equality rows with nonnegative coefficients and
unit-cost continuous slacks. It only proposes an incumbent; node bounds and
the optimality gate remain unchanged.

## Protocol

```text
build\cmake-cuda-release\benchmarks\bench_miplib.exe \
  data\miplib2017_small data\miplib2017_small\miplib2017-v36.solu \
  markshare2 10 pseudocost on serial 3 off 50000 off off off on 0.0
```

## Result

| Configuration | Reference | Incumbent | Best bound | Gap | Median wall | Median nodes |
|---|---:|---:|---:|---:|---:|---:|
| Previous production heuristic set | 1 | 231 | 0 | 0.99568966 | 10.082 s | 117,307 |
| Binary-slack repair enabled | 1 | 189 | 0 | 0.99473684 | 10.062 s | 217,382 |

The heuristic improves the incumbent by 18.2% (231 to 189) but increases the
search work because it consumes a bounded root budget and does not strengthen
the dual bound. It is therefore an incumbent-quality improvement, not an
optimality or throughput claim. `markshare2` remains an explicit time limit.

Raw stdout is retained in
[`MIPLIB_BINARY_SLACK_REPAIR_10S.txt`](MIPLIB_BINARY_SLACK_REPAIR_10S.txt).
