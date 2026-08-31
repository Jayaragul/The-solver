# MIPLIB `pk1` diagnostic

Date: 2026-09-01

`pk1` is a second hard small MIPLIB 2017 instance used to check whether the
`markshare2` gap is isolated. This run used the optimized native C++/CUDA
build, CPU-resident branch-and-bound, serial LP relaxations, and a 30-second
limit.

```text
build\cmake-cuda-release\benchmarks\bench_miplib.exe \
  data\miplib2017_small data\miplib2017_small\miplib2017-v36.solu \
  pk1 30 reliability on serial
```

| Instance | Reference optimum | Status | Incumbent | Best bound | Relative gap | Nodes | LP relaxations | Cuts | Time (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pk1 | 11 | TIME_LIMIT | 21 | 6.24716105 | 0.67058359 | 53,030 | 53,247 | 0 | 30.065 |

The incumbent is integral and the lower bound is valid, but the gap remains
large. Together with the `markshare2` record, this confirms that the current
limitation is broad combinatorial search/feasibility quality on hard small
binary models, not a single parser or numerical failure.
