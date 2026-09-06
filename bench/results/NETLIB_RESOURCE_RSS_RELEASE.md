# Windows Netlib resource capture

Date: 2026-09-06. Release CUDA preset, native MSVC runner, one-instance
directory containing `afiro.mps`. Model SHA-256:
`04992B87E57E57C1C417C96B846833BFA12E055EEAA623F043C45FC773B5BE41`.

The LP benchmark completed with zero skipped instances and zero CPU/GPU
objective mismatches. It reported:

```text
Host peak RSS: 112.4 MB before -> 123.2 MB after (delta 10.8 MB)
GPU free VRAM: 3.46 GB before -> 3.46 GB after (of 4.29 GB total)
```

The same Windows peak-working-set implementation is now used by
`bench_lp_solve`, `validate_netlib`, and `bench_miplib`. `Psapi` is linked
only on Windows; Linux continues to use `getrusage`. The RSS value is peak
working set, not a complete allocator or committed-virtual-memory profile.

Reproduce with a directory containing only the model:

```text
cmake --build --preset cuda-release --target bench_lp_solve
build\cmake-cuda-release\benchmarks\bench_lp_solve.exe <one-model-directory>
```
