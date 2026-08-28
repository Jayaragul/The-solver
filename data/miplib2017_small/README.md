# MIPLIB 2017 small benchmark subset

This directory contains five small instances downloaded from the official
MIPLIB 2017 benchmark set, plus `miplib2017-v36.solu`, the current official
solution/reference file used by the benchmark runner.

Instances selected for coverage:

- `gen-ip002`: general integer, published optimum `-4783.733392`
- `gen-ip054`: general integer with variable bounds, published optimum `6840.96564179`
- `markshare2`: mixed-binary integer-knapsack, published optimum `1`
- `pk1`: mixed-binary, published optimum `11`
- `neos859080`: infeasible mixed-integer instance

Sources:

- https://miplib.zib.de/set_benchmark.html
- https://miplib.zib.de/download.html

Run from the repository root:

```text
build/benchmarks/bench_miplib data/miplib2017_small data/miplib2017_small/miplib2017-v36.solu
```

The runner uses a 60-second limit per instance and distinguishes a certified
optimal result from an incumbent that merely matches the published objective.
