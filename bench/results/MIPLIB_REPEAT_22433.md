# Repeated MIPLIB measurement

The native MIPLIB runner now accepts an optional final repetition count after
the existing parallel-mode argument. Each repetition constructs and solves
the model independently. Wall and CPU seconds are summarized by the median;
RSS is the maximum observed peak working set. Status or incumbent-objective
changes set `REPEAT_VARIANCE`, and the aggregate is never treated as an
optimality certificate.

Smoke record, Release CUDA preset, `22433`, 5-second limit, reliability
branching, warm starts, serial mode, three repetitions:

```text
time_limit_seconds=5 branching_rule=reliability warm_start=on parallel_mode=serial repetitions=3
22433 TIME_LIMIT ours=0 reference=21477 seconds=5.031 cpu_s=4.656 RSS_MB=183.293 verdict=MISMATCH [REPEAT_VARIANCE]
```

At least one repetition found no incumbent while another run had a different
outcome, so the runner surfaced the instability instead of presenting one
selected run as representative. The raw stdout is retained in
[`MIPLIB_REPEAT_22433.txt`](MIPLIB_REPEAT_22433.txt).

Reproduce:

```text
cmake --build --preset cuda-release --target bench_miplib
build\cmake-cuda-release\benchmarks\bench_miplib.exe data\miplib2017_small data\miplib2017_small\miplib2017-v36.solu 22433 5 reliability on serial 3
```
